# -*- coding: utf-8 -*-
"""
PredictSales.py (재귀 예측/다개월 예측 지원)

사용법:
  py -u PredictSales.py <brandId> <start_YYYY-MM> <end_YYYY-MM> [--store <STORE_ID>]

예)
  py -u PredictSales.py obong 2026-01 2026-05
  py -u PredictSales.py obong 2026-01 2026-05 --store obong_a00025

동작 개요:
- 월별로 getPredictSalesData_01 데이터를 조회
- STORE별 히스토리(최대 12개월)를 기반으로 SALES_LAG_1/SALES_LAG_12/AVG_SALES_3M 를 파이썬에서 주입
- 해당 월 예측 결과(PRED_SALES)를 히스토리에 append -> 다음 월 피처로 사용(재귀)
- 비용모델 입력 SALES는 row별 PRED_SALES로 주입
- 저장은 save_predict_sales_data.sql의 bind 변수명과 payload 키가 일치해야 함
  (기본은 :pred_sales, :pred_cogs, ... 형태를 가정)
"""

import os
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import deque

import numpy as np
import pandas as pd
from joblib import load

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comm.sql_loader import SqlRepo
from comm.dataBase.DBOciConnect import DBOciConnect


# -------------------------
# 날짜/월 유틸
# -------------------------
def _norm_yyyymm(s: str) -> str:
    s = str(s).strip()
    s = s.replace("/", "-").replace(".", "-")
    digits = "".join([c for c in s if c.isdigit()])
    if len(digits) < 6:
        raise ValueError(f"Invalid yyyymm: {s}")
    y = int(digits[:4])
    m = int(digits[4:6])
    return f"{y:04d}-{m:02d}"


def add_months(yyyymm: str, n: int) -> str:
    yyyymm = _norm_yyyymm(yyyymm)
    y = int(yyyymm[:4])
    m = int(yyyymm[5:7])
    m2 = m + n
    y += (m2 - 1) // 12
    m2 = (m2 - 1) % 12 + 1
    return f"{y:04d}-{m2:02d}"


def month_range(start_yyyymm: str, end_yyyymm: str):
    start_yyyymm = _norm_yyyymm(start_yyyymm)
    end_yyyymm = _norm_yyyymm(end_yyyymm)
    out = []
    cur = start_yyyymm
    while True:
        out.append(cur)
        if cur == end_yyyymm:
            break
        cur = add_months(cur, 1)
    return out


def yyyymm_to_time_idx(yyyymm: str) -> float:
    """YYYYMM or YYYY-MM -> (year*12+month) index"""
    if yyyymm is None or (isinstance(yyyymm, float) and np.isnan(yyyymm)):
        return np.nan
    s = str(yyyymm)
    s = "".join([c for c in s if c.isdigit()])
    if len(s) < 6:
        return np.nan
    y = int(s[:4])
    m = int(s[4:6])
    return y * 12 + m


# -------------------------
# 안전 변환/클린
# -------------------------
def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, float, int, np.integer)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return None
        return float(s)
    except Exception:
        return None


def ensure_required_cols(df: pd.DataFrame, cols: list[str], fill_value=0):
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
    return df


def _get_pipeline_expected_cols(pipeline, cat_name="cat", num_name="num"):
    prep = pipeline.named_steps.get("prep")
    if prep is None:
        return [], []
    cat_cols, num_cols = [], []
    for name, trans, cols in getattr(prep, "transformers_", []):
        if name == cat_name:
            cat_cols = list(cols)
        elif name == num_name:
            num_cols = list(cols)
    return cat_cols, num_cols


def _clean_X(X: pd.DataFrame, cat_cols: list[str], num_cols: list[str], tag: str):
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("string")
            X[c] = X[c].fillna("UNKNOWN")
            X[c] = X[c].replace({"": "UNKNOWN", "nan": "UNKNOWN", "None": "UNKNOWN"})
            X[c] = X[c].astype(str)

    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        counts = {c: int(X[c].isna().sum()) for c in nan_cols}
        print(f"[WARN] {tag}: NaN 남음:", counts)
        X = X.fillna(0)
    return X


# -------------------------
# DB에서 히스토리/기준치 로드
# -------------------------
def load_sales_history(conn, brand_id: str, end_yyyymm: str, store_filter: str | None = None) -> pd.DataFrame:
    """
    end_yyyymm(포함)까지 STORE별 SALES 히스토리 조회
    - 최소 12개월 확보가 이상적(없어도 됨)
    """
    sql = """
    SELECT STORE_ID, YYYYMM, SALES
      FROM TBL_EXP_PREDICT_SALES_INFO
     WHERE BRAND_ID = :brand_id
       AND YYYYMM <= :end_yyyymm
    """
    params = {"brand_id": brand_id, "end_yyyymm": _norm_yyyymm(end_yyyymm)}
    if store_filter:
        sql += " AND STORE_ID = :store_id"
        params["store_id"] = store_filter
    sql += " ORDER BY STORE_ID, YYYYMM"
    cur = conn.cursor()
    cur.execute(sql, params)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


def load_baseline_by_month(conn, brand_id: str, store_filter: str | None = None) -> dict[int, float]:
    """
    신규매장 대체값용: 브랜드+월(MONTH) 중앙값(median) 매출
    """
    sql = """
    SELECT TO_NUMBER(SUBSTR(YYYYMM, 6, 2)) AS MONTH,
           MEDIAN(SALES) AS MED_SALES
      FROM TBL_EXP_PREDICT_SALES_INFO
     WHERE BRAND_ID = :brand_id
       AND SALES IS NOT NULL
     GROUP BY TO_NUMBER(SUBSTR(YYYYMM, 6, 2))
    """
    params = {"brand_id": brand_id}
    if store_filter:
        # store_filter는 신규매장 1개만 예측할 때도 baseline은 "브랜드 전체"가 더 안정적이라
        # 기본은 브랜드 전체 기준치를 씁니다.
        pass

    cur = conn.cursor()
    cur.execute(sql, params)
    out = {}
    for m, med_sales in cur.fetchall():
        try:
            out[int(m)] = float(med_sales) if med_sales is not None else 0.0
        except Exception:
            out[int(m)] = 0.0
    return out


def build_store_hist_deque(hist_df: pd.DataFrame, maxlen: int = 12) -> dict[str, deque]:
    """
    STORE별 최근 maxlen개월 SALES를 deque로 구성
    """
    store_hist: dict[str, deque] = {}
    if hist_df is None or hist_df.empty:
        return store_hist

    hist_df = hist_df.copy()
    hist_df["STORE_ID"] = hist_df["STORE_ID"].astype(str)
    hist_df["YYYYMM"] = hist_df["YYYYMM"].astype(str)
    hist_df["SALES"] = pd.to_numeric(hist_df["SALES"], errors="coerce")

    hist_df = hist_df.sort_values(["STORE_ID", "YYYYMM"])
    for sid, g in hist_df.groupby("STORE_ID", sort=False):
        vals = g["SALES"].dropna().astype(float).tolist()
        store_hist[sid] = deque(vals[-maxlen:], maxlen=maxlen)
    return store_hist


def compute_recursive_features_for_row(store_id: str, month_int: int,
                                       store_hist: dict[str, deque],
                                       baseline_by_month: dict[int, float]):
    """
    신규매장/히스토리 부족 대응 포함:
    - lag_1: 있으면 사용, 없으면 baseline(month) 사용
    - avg_3m: 있으면 있는 만큼 평균, 없으면 baseline(month)
    - lag_12: 12개월 없으면 baseline(month)
    """
    base = float(baseline_by_month.get(int(month_int), 0.0))
    dq = store_hist.get(store_id)

    if dq is None or len(dq) == 0:
        return base, base, base  # lag1, lag12, avg3

    lag1 = float(dq[-1]) if len(dq) >= 1 else base
    if len(dq) >= 3:
        avg3 = float(np.mean(list(dq)[-3:]))
    else:
        avg3 = float(np.mean(list(dq))) if len(dq) > 0 else base

    lag12 = float(dq[0]) if len(dq) >= 12 else base  # deque는 최근 12개만 들고 있으므로 len==12이면 dq[0]이 12개월 전
    return lag1, lag12, avg3


# -------------------------
# SQL bind 추출(디버그)
# -------------------------
def extract_binds(sql_text: str) -> list[str]:
    return sorted(set(re.findall(r":([A-Za-z_][A-Za-z0-9_]*)", sql_text)))


# -------------------------
# main
# -------------------------
def main():
    start = datetime.now()
    print("시작시간 ==>", start)

    # args
    if len(sys.argv) < 4:
        print("사용법: py -u PredictSales.py <brandId> <start_YYYY-MM> <end_YYYY-MM> [--store <STORE_ID>]")
        print("예)   : py -u PredictSales.py obong 2026-01 2026-05")
        sys.exit(1)

    brand_id = sys.argv[1]
    start_yyyymm = _norm_yyyymm(sys.argv[2])
    end_yyyymm = _norm_yyyymm(sys.argv[3])

    store_filter = None
    if "--store" in sys.argv:
        i = sys.argv.index("--store")
        if i + 1 < len(sys.argv):
            store_filter = sys.argv[i + 1]

    print("brandId     ==>", brand_id)
    print("start_yyyymm==>", start_yyyymm)
    print("end_yyyymm  ==>", end_yyyymm)
    if store_filter:
        print("store_filter==>", store_filter)

    BASE_DIR = Path(__file__).parent.parent
    SQL_REPO = SqlRepo(BASE_DIR / "sql")
    db = DBOciConnect(BASE_DIR, SQL_REPO)
    model_dir = BASE_DIR / "model"

    # 모델 로드
    def _pick_model(name: str):
        p_final = model_dir / f"{brand_id}_{name}_model_FINAL.pkl"
        p_base = model_dir / f"{brand_id}_{name}_model.pkl"
        return p_final if p_final.exists() else p_base

    paths = {
        "sales": _pick_model("sales"),
        "cogs": _pick_model("cogs"),
        "labor": _pick_model("labor"),
        "sell": _pick_model("sell"),
        "admin": _pick_model("admin"),
        "other": _pick_model("other"),
    }
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"모델 파일이 없습니다: {p}")

    sales_model = load(paths["sales"])
    cogs_model = load(paths["cogs"])
    labor_model = load(paths["labor"])
    sell_model = load(paths["sell"])
    admin_model = load(paths["admin"])
    other_model = load(paths["other"])

    print("[MODEL] loaded:")
    for k, p in paths.items():
        print(f" - {k}: {p}")

    months = month_range(start_yyyymm, end_yyyymm)
    prev_month_for_hist = add_months(start_yyyymm, -1)

    # 저장 SQL
    save_sql = SQL_REPO.get("save_predict_sales_data")
    binds = extract_binds(save_sql)
    print("[DEBUG] save_predict_sales_data binds:", binds)

    # 히스토리/기준치 준비
    with db.get_connection() as conn:
        baseline_by_month = load_baseline_by_month(conn, brand_id, store_filter=None)
        hist_df = load_sales_history(conn, brand_id, prev_month_for_hist, store_filter=store_filter)
    store_hist = build_store_hist_deque(hist_df, maxlen=12)

    print(f"[INFO] baseline_by_month size: {len(baseline_by_month)}")
    print(f"[INFO] history rows: {len(hist_df)} / store_hist: {len(store_hist)}")

    # 월별 재귀 예측
    all_saved = 0
    for yyyymm in months:
        # 1) 입력 데이터 조회 (그 달의 기본 피처)
        with db.get_connection() as conn:
            params = {"brand_id": brand_id, "yyyymm": yyyymm}
            cur = conn.cursor()
            cur.execute(SQL_REPO.get("getPredictSalesData_01"), params)
            cols = [d[0] for d in cur.description]
            rows_raw = cur.fetchall()

        df = pd.DataFrame(rows_raw, columns=cols)
        if df.empty:
            print(f"[WARN] {yyyymm}: 조회 결과가 없습니다. (getPredictSalesData_01) -> 이 달 예측 스킵")
            continue

        # store filter (SQL에서 이미 필터링 안 했다면 여기서 필터)
        if store_filter and "STORE_ID" in df.columns:
            df = df[df["STORE_ID"].astype(str) == str(store_filter)].copy()
            if df.empty:
                print(f"[WARN] {yyyymm}: store_filter 적용 후 데이터가 없습니다. -> 스킵")
                continue

        print(f"\n[MONTH] {yyyymm} / rows={len(df)}")

        # 2) 타입/전처리
        for c in ["BRAND_ID", "STORE_ID", "YYYYMM"]:
            if c in df.columns:
                df[c] = df[c].astype(str)

        if "YEAR" in df.columns:
            df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce")
        if "MONTH" in df.columns:
            df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce")

        if "TIME_IDX" not in df.columns and "YYYYMM" in df.columns:
            df["TIME_IDX"] = df["YYYYMM"].apply(yyyymm_to_time_idx)

        numeric_cols = [
            "GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI",
            "STORE_STATE_OPEN", "STORE_STATE_CLOSE", "STORE_STATE_ALL",
            "SALES", "COGS_AMT", "LABOR_COST", "SELL_EXPENSE", "ADMIN_EXPENSE", "OTHER_GA_EXP",
            "TIME_IDX", "SALES_LAG_1", "SALES_LAG_12", "AVG_SALES_3M",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # 3) 재귀 피처 주입 (SQL 값이 있더라도 여기서 덮어씀)
        if "STORE_ID" not in df.columns or "MONTH" not in df.columns:
            raise RuntimeError("입력 df에 STORE_ID 또는 MONTH가 없습니다. getPredictSalesData_01 쿼리를 확인하세요.")

        lag1_list, lag12_list, avg3_list = [], [], []
        for sid, m in zip(df["STORE_ID"].astype(str), df["MONTH"].fillna(0).astype(int)):
            l1, l12, a3 = compute_recursive_features_for_row(sid, int(m), store_hist, baseline_by_month)
            lag1_list.append(l1)
            lag12_list.append(l12)
            avg3_list.append(a3)

        df["SALES_LAG_1"] = lag1_list
        df["SALES_LAG_12"] = lag12_list
        df["AVG_SALES_3M"] = avg3_list

        # 4) 매출 예측
        sales_cat, sales_num = _get_pipeline_expected_cols(sales_model)
        df = ensure_required_cols(df, sales_cat, fill_value="UNKNOWN")
        df = ensure_required_cols(df, sales_num, fill_value=0)

        X_sales = df[sales_cat + sales_num].copy()
        X_sales = _clean_X(X_sales, sales_cat, sales_num, tag=f"X_sales_{yyyymm}")
        df["PRED_SALES"] = sales_model.predict(X_sales)

        # 5) 비용 예측(입력은 cost 모델의 컬럼에 맞추고, SALES는 예측매출로 주입)
        cost_cat, cost_num = _get_pipeline_expected_cols(cogs_model)
        df = ensure_required_cols(df, cost_cat, fill_value="UNKNOWN")
        df = ensure_required_cols(df, cost_num, fill_value=0)

        X_cost = df[cost_cat + cost_num].copy()
        X_cost = _clean_X(X_cost, cost_cat, cost_num, tag=f"X_cost_{yyyymm}")

        if "SALES" in X_cost.columns:
            X_cost["SALES"] = pd.to_numeric(df["PRED_SALES"], errors="coerce").fillna(0).astype(float)

        X_cost = X_cost.replace([np.inf, -np.inf], np.nan).fillna(0)

        df["PRED_COGS_AMT"] = cogs_model.predict(X_cost)
        df["PRED_LABOR_COST"] = labor_model.predict(X_cost)
        df["PRED_SELL_EXPENSE"] = sell_model.predict(X_cost)
        df["PRED_ADMIN_EXPENSE"] = admin_model.predict(X_cost)
        df["PRED_OTHER_GA_EXP"] = other_model.predict(X_cost)

        df["PRED_COST_SUM"] = (
            df["PRED_COGS_AMT"] + df["PRED_LABOR_COST"] + df["PRED_SELL_EXPENSE"] +
            df["PRED_ADMIN_EXPENSE"] + df["PRED_OTHER_GA_EXP"]
        )
        df["PRED_OP_PROFIT"] = df["PRED_SALES"] - df["PRED_COST_SUM"]
        df["PRED_OP_MARGIN"] = np.where(
            np.abs(df["PRED_SALES"]) < 1e-9,
            0.0,
            df["PRED_OP_PROFIT"] / df["PRED_SALES"]
        )

        # 미리보기
        preview_cols = [c for c in [
            "BRAND_ID", "STORE_ID", "YYYYMM",
            "SALES_LAG_1", "SALES_LAG_12", "AVG_SALES_3M",
            "PRED_SALES", "PRED_COST_SUM", "PRED_OP_PROFIT", "PRED_OP_MARGIN"
        ] if c in df.columns]
        print("[DEBUG] Prediction preview (head):")
        print(df[preview_cols].head(10).to_string(index=False))

        # 6) 저장 payload (SQL bind명과 일치하도록 pred_* 사용)
        payload = []
        for _, r in df.iterrows():
            payload.append({
                "brand_id": r.get("BRAND_ID", brand_id),
                "store_id": r.get("STORE_ID"),
                "yyyymm": r.get("YYYYMM"),

                "pred_sales": _safe_float(r.get("PRED_SALES")),
                "pred_cogs": _safe_float(r.get("PRED_COGS_AMT")),
                "pred_labor": _safe_float(r.get("PRED_LABOR_COST")),
                "pred_sell": _safe_float(r.get("PRED_SELL_EXPENSE")),
                "pred_admin": _safe_float(r.get("PRED_ADMIN_EXPENSE")),
                "pred_other": _safe_float(r.get("PRED_OTHER_GA_EXP")),

                "pred_cost_sum": _safe_float(r.get("PRED_COST_SUM")),
                "pred_op_profit": _safe_float(r.get("PRED_OP_PROFIT")),
                "pred_op_margin": _safe_float(r.get("PRED_OP_MARGIN")),
            })

        # (선택) SQL이 요구하는 bind가 payload에 없으면 즉시 알려줌
        missing = [b for b in binds if b not in payload[0]]
        if missing:
            raise RuntimeError(
                f"save_predict_sales_data.sql에 필요한 bind가 payload에 없습니다: {missing}\n"
                f"payload keys={list(payload[0].keys())}\n"
                f"binds={binds}"
            )

        with db.get_connection() as conn:
            cur = conn.cursor()
            cur.executemany(save_sql, payload)
            conn.commit()
            print(f"[DB] {yyyymm} 저장 완료: {len(payload)} rows")
            all_saved += len(payload)

        # 7) 재귀 업데이트: 이번 달 예측매출을 다음 달 lag/avg에 반영
        for sid, pred_sales in zip(df["STORE_ID"].astype(str), df["PRED_SALES"].astype(float)):
            if sid not in store_hist:
                store_hist[sid] = deque([], maxlen=12)
            store_hist[sid].append(float(pred_sales))

    end = datetime.now()
    print(f"\n[FINISH] 총 저장 rows={all_saved}")
    print("종료시간 ==>", end)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[ERROR] 실행 중 예외가 발생했습니다:", repr(e))
        traceback.print_exc()
        sys.stdout.flush()
        raise
