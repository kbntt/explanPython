import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from joblib import dump

from comm.props_loader import load_properties
from comm.sql_loader import SqlRepo
from comm.dataBase.DBConnect import DBConnect
from comm.dataBase.DBOciConnect import DBOciConnect
from comm.util.DateUtils import parseYearMonth

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

start_time = datetime.now()
print("시작시간 ==>", start_time)
# ========= 1) Oracle DB 접속 정보 =========
BASE_DIR   = Path(__file__).parent.parent  # 프로젝트 루트로 변경
SQL_REPO   = SqlRepo(BASE_DIR / "sql")  # ← 쿼리 폴더 지정
db         = DBOciConnect(BASE_DIR,SQL_REPO)

# ========= 2) rows를 컬럼명과 함께 dict로 변환 =========
with db.get_connection() as conn:
    brandId    = "obong"
    params     = {"brand_id": brandId}
    cur = conn.cursor()
    cur.execute(SQL_REPO.get("getPredictSalesData"), params)
    columns = [desc[0] for desc in cur.description]
    rows_raw = cur.fetchall()

# 0건 방지: rows를 dict로 바꾸되, DataFrame은 columns를 유지
rows = [dict(zip(columns, row)) for row in rows_raw]
df = pd.DataFrame(rows, columns=columns)  # ← 0건이어도 컬럼 유지

# 결과가 0건이면 종료(또는 경고만 하고 스킵)
if df.empty:
    print("[경고] 쿼리 결과가 0건입니다. 바인드 변수명 또는 WHERE 조건을 확인하세요.")
    # 디버깅 힌트: SQL 안의 바인드명이 :brand_id 인지 (:BRAND_ID 등 대소문자 포함) 정확히 확인
    # 필요 시 여기서 sys.exit(0) 또는 예외 발생
    raise SystemExit

# YEAR, MONTH 파생 + 숫자형 캐스팅
for row in rows:
    y, m = parseYearMonth(row.get("YYYYMM"))

    row["YEAR"]  = y
    row["MONTH"] = m
    # 숫자형 보정: Decimal/str → float/int
    for k in ["GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI"]:
        v = row.get(k)
        if v is not None:
            try: row[k] = float(v)
            except: pass
    if row.get("SALES") is not None:
        try: row["SALES"] = int(float(row["SALES"]))
        except: pass

# ========= 3) DataFrame 구성 & 정제 =========
df = pd.DataFrame(rows)

# 학습에 필요한 컬럼만 유지
num_cols   = ["GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI"]
cat_cols   = ["STORE_ID"]
date_cols  = ["YEAR", "MONTH"]
target_col = "SALES"

needed_cols = cat_cols + date_cols + num_cols + [target_col]
df = df[needed_cols].dropna()  # YEAR/MONTH 파싱 실패·결측 제거

# (선택) YEAR, MONTH는 정수형 보장
df["YEAR"]  = df["YEAR"].astype(int)
df["MONTH"] = df["MONTH"].astype(int)

# ========= 4) Train / Test 분리 (무작위 분할; 필요시 시계열 분할로 교체 가능) =========
X = df[cat_cols + date_cols + num_cols]
y = df[target_col].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ========= 5) 전처리 파이프라인 + 모델 =========
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), date_cols + num_cols),  # YEAR/MONTH 포함 스케일링
    ],
    remainder="drop",
)

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([
    ("prep", preprocess),
    ("model", model)
])

# ========= 6) 학습 =========
pipe.fit(X_train, y_train)

# 평가
pred_test = pipe.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2  = r2_score(y_test, pred_test)
print(f"Test MAE: {mae:,.0f}")
print(f"Test R2 : {r2:.4f}")

end_time = datetime.now()
print("종료시간 ==>", end_time)

# ========= 7) 전체 (STORE_ID, YEAR, MONTH) 기준 예측 =========
# 요구사항: YEAR,MONTH 별로 STORE_ID의 SALES 예측
pred_all = pipe.predict(X)
out = df[["STORE_ID", "YEAR", "MONTH"]].copy()
out["PRED_SALES"] = np.round(pred_all).astype(int)
out["ACTUAL_SALES"] = df[target_col].astype(int)

# 보기 좋게 정렬
out = out.sort_values(["STORE_ID", "YEAR", "MONTH"]).reset_index(drop=True)

# ========= 8) 결과 저장 =========
fileName = f"{brandId}_pred_sales_results"
model_dir = BASE_DIR / "model"
csv_path = model_dir /  f"{fileName}.csv"
out.to_csv(csv_path, index=False, encoding="utf-8-sig")
print(f"예측 결과 저장: {csv_path}")

# ========= 9) 모델 저장 =========
model_path = model_dir / f"{fileName}.pkl"
dump(pipe, model_path)
print(f"모델 저장 완료: {model_path}")
