"""CreatePredictModels.py

예측 모델 생성 스크립트.

※ PowerShell에서 "아무 로그도 안 나오고 멈춤" 현상이 있을 때를 대비해
  - 스크립트 진입(ENTRY) 로그
  - import 단계별 로그
  - 예외 발생 시 즉시 출력
을 추가했습니다.

가능하면 아래처럼 unbuffered(-u)로 실행하세요.
  python -u .\CreatePredictModels.py <brandId>
"""

import os
import sys
import time

# --- ENTRY LOG (imports 보다 먼저 출력되어야 원인 파악이 됩니다) ---
try:
    print("=== ENTRY ===", __file__)
    print("exe:", sys.executable)
    print("cwd:", os.getcwd())
    print("argv:", sys.argv)
    sys.stdout.flush()
except Exception:
    # 출력 자체가 실패하는 극단 케이스 방지
    pass

# 프로젝트 루트(상위 1단계)를 import path에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _import_step(msg: str) -> None:
    """import 중 멈추는 지점을 찾기 위한 헬퍼."""
    print(f"[IMPORT] {msg}")
    sys.stdout.flush()
    time.sleep(0.01)

_import_step("pathlib/datetime")
from pathlib import Path
from datetime import datetime

_import_step("numpy/pandas/joblib")
import numpy as np
import pandas as pd
from joblib import dump

_import_step("project modules: comm.sql_loader / DBOciConnect")
from comm.sql_loader import SqlRepo
from comm.dataBase.DBOciConnect import DBOciConnect

_import_step("sklearn")
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


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


def build_preprocessors(macro_vars: list[str]):
    """Build preprocessors for SALES and COST models."""
    # SALES features
    sales_num_cols = macro_vars + ["YEAR", "TIME_IDX"]
    sales_cat_cols = ["STORE_ID", "MONTH"]

    sales_preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), sales_cat_cols),
            ("num", "passthrough", sales_num_cols),
        ],
        remainder="drop",
    )

    # COST features (include SALES)
    cost_num_cols = macro_vars + ["YEAR", "TIME_IDX", "SALES"]
    cost_cat_cols = ["STORE_ID", "MONTH"]

    cost_preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cost_cat_cols),
            ("num", "passthrough", cost_num_cols),
        ],
        remainder="drop",
    )

    return (sales_num_cols, sales_cat_cols, sales_preprocess,
            cost_num_cols, cost_cat_cols, cost_preprocess)


def fit_models(df_train: pd.DataFrame, macro_vars: list[str]):
    (sales_num_cols, sales_cat_cols, sales_preprocess,
     cost_num_cols, cost_cat_cols, cost_preprocess) = build_preprocessors(macro_vars)

    sales_model = Pipeline(steps=[
        ("prep", sales_preprocess),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
    ])

    def _fit_cost_model(y: pd.Series):
        model = Pipeline(steps=[
            ("prep", cost_preprocess),
            ("rf", RandomForestRegressor(n_estimators=300, random_state=42)),
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
            "sales_num_cols": sales_num_cols,
            "sales_cat_cols": sales_cat_cols,
            "cost_num_cols": cost_num_cols,
            "cost_cat_cols": cost_cat_cols,
        }
    }


def evaluate_models(models: dict, df_test: pd.DataFrame) -> None:
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
    print_metrics("SALES", regression_metrics(y_sales_true, y_sales_pred))

    # --- (2) COST evaluation using *predicted* SALES ---
    # Mimic production: COST model input SALES must be SALES_pred
    X_cost_test = df_test[cost_num_cols + cost_cat_cols].copy()
    X_cost_test["SALES"] = y_sales_pred

    targets = {
        "COGS_AMT": (models["cogs_model"], df_test["COGS_AMT"].to_numpy()),
        "LABOR_COST": (models["labor_model"], df_test["LABOR_COST"].to_numpy()),
        "SELL_EXPENSE": (models["sell_model"], df_test["SELL_EXPENSE"].to_numpy()),
        "ADMIN_EXPENSE": (models["admin_model"], df_test["ADMIN_EXPENSE"].to_numpy()),
        "OTHER_GA_EXP": (models["other_model"], df_test["OTHER_GA_EXP"].to_numpy()),
    }

    for name, (m, y_true) in targets.items():
        y_pred = m.predict(X_cost_test)
        print_metrics(name, regression_metrics(y_true, y_pred))

    # Optional: show sample
    preview = df_test[["BRAND_ID", "STORE_ID", "YYYYMM"]].copy() if "BRAND_ID" in df_test.columns else df_test[["STORE_ID", "YYYYMM"]].copy()
    preview["SALES_TRUE"] = y_sales_true
    preview["SALES_PRED"] = y_sales_pred
    print("\n[DEBUG] Prediction preview (head):")
    print(preview.head(10).to_string(index=False))


def main():
    start_time = datetime.now()
    print("시작시간 ==>", start_time)
    sys.stdout.flush()

    if len(sys.argv) < 2:
        print("사용법: python CreatePredictModels_v2.py <brandId>")
        sys.exit(1)

    brandId = sys.argv[1]
    print("brandId ==>", brandId)

    BASE_DIR = Path(__file__).parent.parent
    SQL_REPO = SqlRepo(BASE_DIR / "sql")
    db = DBOciConnect(BASE_DIR, SQL_REPO)

    yyyymm = None

    with db.get_connection() as conn:
        params = {"brand_id": brandId, "yyyymm": yyyymm}
        cur = conn.cursor()
        cur.execute(SQL_REPO.get("getPredictSalesData"), params)
        columns = [d[0] for d in cur.description]
        rows_raw = cur.fetchall()
        print(f"[DEBUG] SQL 결과 건수: {len(rows_raw)}")

    rows = [dict(zip(columns, r)) for r in rows_raw]
    df = pd.DataFrame(rows, columns=columns)

    # ------------------ cleanup / typing ------------------
    # Ensure categorical
    if "STORE_ID" in df.columns:
        df["STORE_ID"] = df["STORE_ID"].astype(str)

    # YEAR, MONTH numeric
    for col in ["YEAR", "MONTH"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # TIME_IDX
    if "YYYYMM" in df.columns:
        df["TIME_IDX"] = df["YYYYMM"].apply(yyyymm_to_time_idx)
    else:
        df["TIME_IDX"] = np.nan

    # Candidate numeric columns
    numeric_candidates = [
        "GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI",
        "STORE_STATE_OPEN", "STORE_STATE_CLOSE", "STORE_STATE_ALL",
        "SALES", "COGS_AMT", "LABOR_COST", "SELL_EXPENSE", "ADMIN_EXPENSE", "OTHER_GA_EXP",
        "TIME_IDX"
    ]

    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort
    sort_cols = [c for c in ["YYYYMM", "STORE_ID"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # ------------------ missing handling ------------------
    # 핵심 타깃(매출/비용) 결측만 제거
    required_targets = [c for c in ["SALES", "COGS_AMT", "LABOR_COST", "SELL_EXPENSE", "ADMIN_EXPENSE", "OTHER_GA_EXP"] if c in df.columns]

    print("[DEBUG] target NA counts before dropna:")
    print(df[required_targets].isna().sum())

    df = df.dropna(subset=required_targets)

    # 매크로 변수는 결측이 있으면 ffill/bfill로 완화 (월단위 데이터가 띄엄띄엄일 때 데이터 손실 방지)
    macro_all = [
        "GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI",
        "STORE_STATE_OPEN", "STORE_STATE_CLOSE", "STORE_STATE_ALL",
    ]
    macro_vars = [c for c in macro_all if c in df.columns]
    if macro_vars:
        df[macro_vars] = df[macro_vars].sort_index().ffill().bfill()

    # YEAR/MONTH/TIME_IDX also required
    required_features = [c for c in ["YEAR", "MONTH", "TIME_IDX", "STORE_ID"] if c in df.columns]
    df = df.dropna(subset=required_features)

    print(f"[INFO] Loaded rows (after cleanup): {len(df)} (brand_id={brandId})")

    if len(df) < 50:
        print("[WARN] 학습 데이터가 매우 적습니다. 예측 성능이 불안정할 수 있습니다.")

    # ------------------ time-based split ------------------
    if "YYYYMM" not in df.columns:
        print("[ERROR] YYYYMM 컬럼이 없습니다. 시간기반 검증을 할 수 없습니다.")
        sys.exit(2)

    unique_months = sorted(df["YYYYMM"].astype(str).unique())

    # 테스트 기간: 최근 3개월(가능하면). 데이터가 적으면 최근 1개월.
    test_n = 3 if len(unique_months) >= 8 else 1
    test_months = set(unique_months[-test_n:])

    df_train = df[~df["YYYYMM"].astype(str).isin(test_months)].copy()
    df_test = df[df["YYYYMM"].astype(str).isin(test_months)].copy()

    print(f"[INFO] Train months: {len(unique_months) - test_n}, Test months: {test_n}")
    print(f"[INFO] Train rows: {len(df_train)}, Test rows: {len(df_test)}")

    if len(df_train) < 10 or len(df_test) < 5:
        print("[WARN] Train/Test split이 불안정합니다. 데이터 기간/건수를 늘리는 것을 권장합니다.")

    # ------------------ (2) Train + Evaluate ------------------
    models = fit_models(df_train, macro_vars)
    print("\n========== 평가/검증 (Test 기간, COST는 예측 SALES 사용) ==========")
    evaluate_models(models, df_test)

    # ------------------ Refit on FULL data for final saving ------------------
    print("\n========== 전체 데이터로 재학습 후 모델 저장 ==========")
    models_full = fit_models(df, macro_vars)

    model_dir = BASE_DIR / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "sales": model_dir / f"{brandId}_sales_model.pkl",
        "cogs": model_dir / f"{brandId}_cogs_model.pkl",
        "labor": model_dir / f"{brandId}_labor_model.pkl",
        "sell": model_dir / f"{brandId}_sell_model.pkl",
        "admin": model_dir / f"{brandId}_admin_model.pkl",
        "other": model_dir / f"{brandId}_other_model.pkl",
    }

    dump(models_full["sales_model"], paths["sales"])
    dump(models_full["cogs_model"], paths["cogs"])
    dump(models_full["labor_model"], paths["labor"])
    dump(models_full["sell_model"], paths["sell"])
    dump(models_full["admin_model"], paths["admin"])
    dump(models_full["other_model"], paths["other"])

    print("모델 저장 완료:")
    for k, p in paths.items():
        print(f" - {k}: {p}")

    end_time = datetime.now()
    print("종료시간 ==>", end_time)
    sys.stdout.flush()


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
