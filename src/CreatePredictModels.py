
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump

from comm.sql_loader import SqlRepo
from comm.dataBase.DBOciConnect import DBOciConnect

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


# =========================
# 0) 공통 유틸
# =========================
def yyyymm_to_time_idx(yyyymm: str) -> float:
    """YYYYMM or YYYY-MM -> (year*12 + month) numeric index for trend learning."""
    if yyyymm is None or (isinstance(yyyymm, float) and np.isnan(yyyymm)):
        return np.nan
    s = str(yyyymm)
    s = "".join([c for c in s if c.isdigit()])  # keep digits only
    if len(s) < 6:
        return np.nan
    y = int(s[:4])
    m = int(s[4:6])
    return y * 12 + m


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    # MAPE: avoid division by 0
    denom = np.where(np.abs(y_true) < 1e-9, np.nan, np.abs(y_true))
    mape = float(np.nanmean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {"MAE": mae, "RMSE": rmse, "MAPE(%)": mape}


def print_metrics(title: str, metrics: dict) -> None:
    msg = ", ".join([f"{k}={v:,.4f}" for k, v in metrics.items()])
    print(f"[EVAL] {title}: {msg}")


def add_sales_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    ✔ 전월/전년동월 Lag 피처 추가
    - 매장별(STORE_ID)로 SALES의 lag(1), lag(12) 생성
    - TIME_IDX(월 인덱스)를 기준으로 정렬 후 shift
    """
    if "STORE_ID" not in df.columns or "TIME_IDX" not in df.columns or "SALES" not in df.columns:
        return df

    df = df.sort_values(["STORE_ID", "TIME_IDX"]).copy()
    g = df.groupby("STORE_ID", sort=False)

    df["SALES_LAG_1"] = g["SALES"].shift(1)
    df["SALES_LAG_12"] = g["SALES"].shift(12)

    # (선택) 변화량도 같이 넣고 싶으면 아래 주석 해제
    # df["SALES_LAG_DIFF_1"] = df["SALES"] - df["SALES_LAG_1"]
    # df["SALES_LAG_DIFF_12"] = df["SALES"] - df["SALES_LAG_12"]

    return df


def get_feature_cols(macro_vars: list[str], use_lag: bool):
    """
    모델 입력 컬럼 정의
    - SALES : macro + YEAR + TIME_IDX + AVG_SALES_3M (+ lag)
    - COST  : macro + YEAR + TIME_IDX + SALES + AVG_SALES_3M (+ lag)
    """
    base_sales_num = macro_vars + ["YEAR", "TIME_IDX", "AVG_SALES_3M"]
    base_cost_num  = macro_vars + ["YEAR", "TIME_IDX", "SALES", "AVG_SALES_3M"]

    if use_lag:
        lag_cols = ["SALES_LAG_1", "SALES_LAG_12"]
        sales_num_cols = base_sales_num + lag_cols
        cost_num_cols = base_cost_num + lag_cols
    else:
        sales_num_cols = base_sales_num
        cost_num_cols = base_cost_num

    sales_cat_cols = ["STORE_ID", "MONTH"]
    cost_cat_cols = ["STORE_ID", "MONTH"]

    return sales_num_cols, sales_cat_cols, cost_num_cols, cost_cat_cols


def build_preprocessors(macro_vars: list[str], use_lag: bool):
    """Build preprocessors for SALES and COST models."""
    sales_num_cols, sales_cat_cols, cost_num_cols, cost_cat_cols = get_feature_cols(macro_vars, use_lag)

    sales_preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), sales_cat_cols),
            ("num", "passthrough", sales_num_cols),
        ],
        remainder="drop",
    )

    cost_preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cost_cat_cols),
            ("num", "passthrough", cost_num_cols),
        ],
        remainder="drop",
    )

    return sales_num_cols, sales_cat_cols, sales_preprocess, cost_num_cols, cost_cat_cols, cost_preprocess


def fit_models(df_train: pd.DataFrame, macro_vars: list[str], use_lag: bool):
    """
    모델 학습
    - SALES 1개 + COST 5개
    """
    (sales_num_cols, sales_cat_cols, sales_preprocess,
     cost_num_cols, cost_cat_cols, cost_preprocess) = build_preprocessors(macro_vars, use_lag)

    # 모델은 동일하게 RandomForest를 유지(추후 XGBoost 등으로 교체 가능)
    sales_model = Pipeline(steps=[
        ("prep", sales_preprocess),
        ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
    ])

    def _fit_cost_model(y: pd.Series):
        model = Pipeline(steps=[
            ("prep", cost_preprocess),
            ("rf", RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
        ])
        X = df_train[cost_num_cols + cost_cat_cols]
        return model.fit(X, y)

    # 1) SALES model
    X_sales = df_train[sales_num_cols + sales_cat_cols]
    y_sales = df_train["SALES"]
    sales_model.fit(X_sales, y_sales)

    # 2) COST models (training uses ACTUAL SALES)
    cogs_model = _fit_cost_model(df_train["COGS_AMT"])
    labor_model = _fit_cost_model(df_train["LABOR_COST"])
    sell_model = _fit_cost_model(df_train["SELL_EXPENSE"])
    admin_model = _fit_cost_model(df_train["ADMIN_EXPENSE"])
    other_model = _fit_cost_model(df_train["OTHER_GA_EXP"])

    return {
        "sales_model": sales_model,
        "cogs_model": cogs_model,
        "labor_model": labor_model,
        "sell_model": sell_model,
        "admin_model": admin_model,
        "other_model": other_model,
        "feature_meta": {
            "macro_vars": macro_vars,
            "use_lag": use_lag,
            "sales_num_cols": sales_num_cols,
            "sales_cat_cols": sales_cat_cols,
            "cost_num_cols": cost_num_cols,
            "cost_cat_cols": cost_cat_cols,
        }
    }


def evaluate_models(models: dict, df_test: pd.DataFrame, export_dir: Path | None = None, tag: str = "") -> dict:
    """
    ✔ 평가/검증
    - COST는 예측 SALES 사용 (운영 방식 동일)
    ✔ 매장별 오차 리포트 자동 생성 (CSV)
    """
    meta = models["feature_meta"]
    sales_num_cols = meta["sales_num_cols"]
    sales_cat_cols = meta["sales_cat_cols"]
    cost_num_cols = meta["cost_num_cols"]
    cost_cat_cols = meta["cost_cat_cols"]

    sales_model = models["sales_model"]

    # --- (1) SALES evaluation ---
    X_sales_test = df_test[sales_num_cols + sales_cat_cols]
    y_sales_true = df_test["SALES"].to_numpy()
    y_sales_pred = sales_model.predict(X_sales_test)

    overall = {}
    overall["SALES"] = regression_metrics(y_sales_true, y_sales_pred)
    print_metrics("SALES", overall["SALES"])

    # --- (2) COST evaluation using *predicted* SALES ---
    X_cost_test = df_test[cost_num_cols + cost_cat_cols].copy()
    X_cost_test["SALES"] = y_sales_pred  # 핵심: 예측 SALES 사용

    targets = {
        "COGS_AMT": (models["cogs_model"], df_test["COGS_AMT"].to_numpy()),
        "LABOR_COST": (models["labor_model"], df_test["LABOR_COST"].to_numpy()),
        "SELL_EXPENSE": (models["sell_model"], df_test["SELL_EXPENSE"].to_numpy()),
        "ADMIN_EXPENSE": (models["admin_model"], df_test["ADMIN_EXPENSE"].to_numpy()),
        "OTHER_GA_EXP": (models["other_model"], df_test["OTHER_GA_EXP"].to_numpy()),
    }

    # 상세 예측 프레임(리포트용)
    base_cols = ["BRAND_ID", "STORE_ID", "YYYYMM"]
    base_cols = [c for c in base_cols if c in df_test.columns]
    pred_df = df_test[base_cols].copy()
    pred_df["SALES_TRUE"] = y_sales_true
    pred_df["SALES_PRED"] = y_sales_pred

    for name, (m, y_true) in targets.items():
        y_pred = m.predict(X_cost_test)
        overall[name] = regression_metrics(y_true, y_pred)
        print_metrics(name, overall[name])
        pred_df[f"{name}_TRUE"] = y_true
        pred_df[f"{name}_PRED"] = y_pred

    # Preview
    print("\n[DEBUG] Prediction preview (head):")
    print(pred_df.head(10).to_string(index=False))

    # ----------------------------
    # 매장별 오차 리포트 자동 생성
    # ----------------------------
    if export_dir is not None:
        export_dir.mkdir(parents=True, exist_ok=True)

        def _store_report_for_pair(true_col: str, pred_col: str, metric_prefix: str):
            tmp = pred_df[base_cols + [true_col, pred_col]].copy()
            tmp["ABS_ERR"] = (tmp[true_col] - tmp[pred_col]).abs()
            denom = tmp[true_col].abs().replace({0: np.nan})
            tmp["APE"] = (tmp["ABS_ERR"] / denom) * 100.0
            g = tmp.groupby("STORE_ID", dropna=False)

            rep = g.agg(
                N=("ABS_ERR", "size"),
                MAE=("ABS_ERR", "mean"),
                MAPE=("APE", "mean"),
                TRUE_SUM=(true_col, "sum"),
                PRED_SUM=(pred_col, "sum"),
            ).reset_index()

            rep["BIAS_SUM"] = rep["PRED_SUM"] - rep["TRUE_SUM"]  # +면 과대예측
            rep["TAG"] = tag
            rep["TARGET"] = metric_prefix
            return rep

        reports = []
        reports.append(_store_report_for_pair("SALES_TRUE", "SALES_PRED", "SALES"))

        for k in targets.keys():
            reports.append(_store_report_for_pair(f"{k}_TRUE", f"{k}_PRED", k))

        store_report = pd.concat(reports, ignore_index=True)
        store_report = store_report.sort_values(["TARGET", "MAPE"], ascending=[True, False])

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = export_dir / f"{tag}_store_error_report_{ts}.csv"
        store_report.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[REPORT] 매장별 오차 리포트 저장: {out_path}")

        # 전체 예측 결과도 함께 저장(원하면 BI툴에서 바로 확인)
        out_pred = export_dir / f"{tag}_predictions_{ts}.csv"
        pred_df.to_csv(out_pred, index=False, encoding="utf-8-sig")
        print(f"[REPORT] 예측 결과 저장: {out_pred}")

    return overall


def compare_before_after(df_train: pd.DataFrame, df_test: pd.DataFrame, macro_vars: list[str], export_dir: Path):
    """
    ✔ 정확도 개선 전/후 비교 구조
    - BEFORE: 기존 피처 (no lag)
    - AFTER : lag 피처 포함
    - 동일 테스트셋에서 metric 비교 표 출력 + CSV 저장
    """
    print("\n========== (BEFORE) 기존 피처로 학습/평가 ==========")
    m_before = fit_models(df_train, macro_vars, use_lag=False)
    before_metrics = evaluate_models(m_before, df_test, export_dir=export_dir, tag="BEFORE")

    print("\n========== (AFTER) Lag 피처 포함 학습/평가 ==========")
    m_after = fit_models(df_train, macro_vars, use_lag=True)
    after_metrics = evaluate_models(m_after, df_test, export_dir=export_dir, tag="AFTER")

    # 비교 테이블
    rows = []
    for target in ["SALES", "COGS_AMT", "LABOR_COST", "SELL_EXPENSE", "ADMIN_EXPENSE", "OTHER_GA_EXP"]:
        b = before_metrics.get(target, {})
        a = after_metrics.get(target, {})
        rows.append({
            "TARGET": target,
            "BEFORE_MAE": b.get("MAE", np.nan),
            "AFTER_MAE": a.get("MAE", np.nan),
            "MAE_IMPROVE(%)": (1.0 - (a.get("MAE", np.nan) / b.get("MAE", np.nan))) * 100.0 if b.get("MAE", np.nan) not in [0, np.nan] else np.nan,
            "BEFORE_RMSE": b.get("RMSE", np.nan),
            "AFTER_RMSE": a.get("RMSE", np.nan),
            "RMSE_IMPROVE(%)": (1.0 - (a.get("RMSE", np.nan) / b.get("RMSE", np.nan))) * 100.0 if b.get("RMSE", np.nan) not in [0, np.nan] else np.nan,
            "BEFORE_MAPE": b.get("MAPE(%)", np.nan),
            "AFTER_MAPE": a.get("MAPE(%)", np.nan),
            "MAPE_IMPROVE(%)": (1.0 - (a.get("MAPE(%)", np.nan) / b.get("MAPE(%)", np.nan))) * 100.0 if b.get("MAPE(%)", np.nan) not in [0, np.nan] else np.nan,
        })
    cmp = pd.DataFrame(rows)
    print("\n========== BEFORE vs AFTER 비교 ==========")
    with pd.option_context("display.max_rows", 50, "display.max_columns", 50, "display.width", 200):
        print(cmp.to_string(index=False))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_cmp = export_dir / f"compare_before_after_{ts}.csv"
    cmp.to_csv(out_cmp, index=False, encoding="utf-8-sig")
    print(f"[REPORT] BEFORE/AFTER 비교 저장: {out_cmp}")

    return m_before, m_after, cmp


# =========================
# main
# =========================
def main():
    start_time = datetime.now()
    print("시작시간 ==>", start_time)

    if len(sys.argv) < 2:
        print("사용법: py -u CreatePredictModels.py <brandId>")
        sys.exit(1)

    brandId = sys.argv[1]
    print("brandId ==>", brandId)

    BASE_DIR = Path(__file__).parent.parent
    SQL_REPO = SqlRepo(BASE_DIR / "sql")
    db = DBOciConnect(BASE_DIR, SQL_REPO)

    yyyymm = None

    # 1) 데이터 로드
    with db.get_connection() as conn:
        params = {"brand_id": brandId, "yyyymm": yyyymm}
        cur = conn.cursor()
        cur.execute(SQL_REPO.get("getPredictSalesData"), params)
        columns = [d[0] for d in cur.description]
        rows_raw = cur.fetchall()
        print(f"[DEBUG] SQL 결과 건수: {len(rows_raw)}")

    rows = [dict(zip(columns, r)) for r in rows_raw]
    df = pd.DataFrame(rows, columns=columns)

    # 2) 타입/결측 정리
    if "STORE_ID" in df.columns:
        df["STORE_ID"] = df["STORE_ID"].astype(str)

    for col in ["YEAR", "MONTH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "YYYYMM" in df.columns:
        df["TIME_IDX"] = df["YYYYMM"].apply(yyyymm_to_time_idx)
    else:
        df["TIME_IDX"] = np.nan

    numeric_candidates = [
        "GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI",
        "STORE_STATE_OPEN", "STORE_STATE_CLOSE", "STORE_STATE_ALL",
        "SALES", "COGS_AMT", "LABOR_COST", "SELL_EXPENSE", "ADMIN_EXPENSE", "OTHER_GA_EXP",
        "TIME_IDX", "AVG_SALES_3M", "SALES_LAG_1" 
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort
    sort_cols = [c for c in ["TIME_IDX", "STORE_ID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # 핵심 타깃 결측 제거
    required_targets = [c for c in ["SALES", "COGS_AMT", "LABOR_COST", "SELL_EXPENSE", "ADMIN_EXPENSE", "OTHER_GA_EXP"] if c in df.columns]
    print("[DEBUG] target NA counts before dropna:")
    print(df[required_targets].isna().sum())
    df = df.dropna(subset=required_targets)

    # 매크로 변수 보정
    macro_all = [
        "GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI",
        "STORE_STATE_OPEN", "STORE_STATE_CLOSE", "STORE_STATE_ALL",
    ]
    macro_vars = [c for c in macro_all if c in df.columns]
    if macro_vars:
        df[macro_vars] = df[macro_vars].ffill().bfill()

    required_features = [c for c in ["YEAR", "MONTH", "TIME_IDX", "STORE_ID", "AVG_SALES_3M"] if c in df.columns]
    df = df.dropna(subset=required_features)

    # 3) Lag 피처 생성
    df = add_sales_lag_features(df)

    # 비교를 공정하게: lag가 존재하는 행만 공통 사용
    lag_needed = ["SALES_LAG_1", "SALES_LAG_12"]
    if all(c in df.columns for c in lag_needed):
        df_common = df.dropna(subset=lag_needed).copy()
    else:
        df_common = df.copy()

    print(f"[INFO] Loaded rows (after cleanup): {len(df_common)} (brand_id={brandId})")

    if "YYYYMM" not in df_common.columns:
        print("[ERROR] YYYYMM 컬럼이 없습니다. 시간기반 검증을 할 수 없습니다.")
        sys.exit(2)

    # 4) Time-based split
    unique_months = sorted(df_common["YYYYMM"].astype(str).unique())
    test_n = 3 if len(unique_months) >= 8 else 1
    test_months = set(unique_months[-test_n:])

    df_train = df_common[~df_common["YYYYMM"].astype(str).isin(test_months)].copy()
    df_test = df_common[df_common["YYYYMM"].astype(str).isin(test_months)].copy()

    print(f"[INFO] Train months: {len(unique_months) - test_n}, Test months: {test_n}")
    print(f"[INFO] Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    if len(df_train) < 50 or len(df_test) < 10:
        print("[WARN] Train/Test split이 불안정할 수 있습니다. 데이터 기간/건수를 늘리는 것을 권장합니다.")

    # 5) 평가/검증 + 비교
    model_dir = BASE_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    # (A) BEFORE vs AFTER 비교 + 리포트 생성
    m_before, m_after, _cmp = compare_before_after(df_train, df_test, macro_vars, export_dir=model_dir)

    do_final = ("--final" in sys.argv)
    # ... (Test 평가까지 수행)

    if do_final:
        print("\n========== 전체 데이터로 재학습 후 모델 저장 (AFTER=Lag 포함) ==========")
        models_full = fit_models(df_common, macro_vars, use_lag=True)
        # dump...
    else:
        print("\n[SKIP] 최종 전체 재학습/저장은 생략했습니다. (--final 옵션을 주면 수행)")
        return
    paths = {
        "sales": model_dir / f"{brandId}_sales_model.pkl",
        "cogs": model_dir / f"{brandId}_cogs_model.pkl",
        "labor": model_dir / f"{brandId}_labor_model.pkl",
        "sell": model_dir / f"{brandId}_sell_model.pkl",
        "admin": model_dir / f"{brandId}_admin_model.pkl",
        "other": model_dir / f"{brandId}_other_model.pkl",

        # 참고용(BASE)
        "sales_base": model_dir / f"{brandId}_sales_model_BASE.pkl",
        "cogs_base": model_dir / f"{brandId}_cogs_model_BASE.pkl",
        "labor_base": model_dir / f"{brandId}_labor_model_BASE.pkl",
        "sell_base": model_dir / f"{brandId}_sell_model_BASE.pkl",
        "admin_base": model_dir / f"{brandId}_admin_model_BASE.pkl",
        "other_base": model_dir / f"{brandId}_other_model_BASE.pkl",
    }
    
    # AFTER 저장
    dump(models_full["sales_model"], paths["sales"])
    dump(models_full["cogs_model"], paths["cogs"])
    dump(models_full["labor_model"], paths["labor"])
    dump(models_full["sell_model"], paths["sell"])
    dump(models_full["admin_model"], paths["admin"])
    dump(models_full["other_model"], paths["other"])

    # BEFORE 참고용 저장
    if "sales_base" in paths:

        dump(m_before["sales_model"], paths["sales_base"])
    if "cogs_base" in paths:

        dump(m_before["cogs_model"], paths["cogs_base"])
    if "labor_base" in paths:

        dump(m_before["labor_model"], paths["labor_base"])
    if "sell_base" in paths:

        dump(m_before["sell_model"], paths["sell_base"])
    if "admin_base" in paths:

        dump(m_before["admin_model"], paths["admin_base"])
    if "other_base" in paths:

        dump(m_before["other_model"], paths["other_base"])

    print("모델 저장 완료:")
    for k, p in paths.items():
        print(f" - {k}: {p}")

    end_time = datetime.now()
    print("종료시간 ==>", end_time)


if __name__ == "__main__":
    # 어떤 예외가 나더라도 PowerShell에서 조용히 종료되지 않도록 강제 출력
    try:
        main()
    except Exception as e:
        import traceback
        print("\n[ERROR] 실행 중 예외가 발생했습니다:", repr(e))
        traceback.print_exc()
        sys.stdout.flush()
        raise
