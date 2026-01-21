import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from pathlib import Path
from joblib import load
from datetime import datetime
from comm.sql_loader import SqlRepo
from comm.dataBase.DBConnect import DBConnect
from comm.dataBase.DBOciConnect import DBOciConnect
from dateutil.relativedelta import relativedelta

# 생성된 모델로 예측데이타 만들기

start_time = datetime.now()
print("시작시간 ==>", start_time)
# 저장된 모델 로드

# 커맨드라인에서 brandId 받기
if len(sys.argv) < 2:
    print("사용법: python PredictCostRatio.py <brandId>")
    sys.exit(1)
brandId = sys.argv[1]
fileName = f"{brandId}_pred_cost_ratio_results"
BASE_DIR   = Path(__file__).parent.parent
model_dir = BASE_DIR / "model"
model     = load(model_dir / f"{fileName}.pkl")


# ========= 1) Oracle DB 접속 정보 =========
SQL_REPO   = SqlRepo(BASE_DIR / "sql")  # ← 쿼리 폴더 지정
db         = DBOciConnect(BASE_DIR, SQL_REPO)

today = datetime.today()
one_month_ago = today - relativedelta(months=1)# 한 달 전 날짜

# 원하는 형식으로 출력
yyyymm  = one_month_ago.strftime("%Y-%m")
print(yyyymm)

params     = {"brand_id": brandId, "yyyymm": yyyymm}

# ========= 2) rows를 컬럼명과 함께 dict로 변환 =========
with db.get_connection() as conn:
    cur = conn.cursor()
    cur.execute(SQL_REPO.get("getPredictSalesData_01"), params)
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]
    print(SQL_REPO.get("getPredictSalesData_01"))

# 조회 결과가 없을 때 방어
if not rows:
    raise ValueError(f"조회 결과가 없습니다. brand_id={brandId}, yyyymm={yyyymm}")

df = pd.DataFrame(rows)

# ========= 3) 모델 입력 컬럼 정렬 =========
# 모델이 학습 시 사용한 컬럼을 그대로 쓰는 게 가장 안전합니다.
# scikit-learn Pipeline을 저장했다면 feature_names_in_이 존재할 수 있습니다.
if hasattr(model, "feature_names_in_"):
    feature_cols = list(model.feature_names_in_)
    print(f"모델 학습 시 사용한 피처 컬럼: {feature_cols}")   
else:
    # 일반적인 피처 셋 (필요시 여기 조정)
    feature_cols = ["STORE_ID", 'YEAR', 'MONTH', "GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI"]
    print(f"기본 피처 컬럼 사용: {feature_cols}")

missing = [c for c in feature_cols if c not in df.columns]
if missing:
    raise KeyError(f"필요한 피처 컬럼이 없습니다: {missing}")

# ========= 4) 형 변환 (학습과 동일하게 맞추기) =========

num_cols = [c for c in ["GDP","INFLATION_RATE","UNEMPLOYMENT","INTEREST_RATE","CCSI"]
            if c in feature_cols]
for c in num_cols:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# 누락/NaN 있으면 제거(또는 적절히 대체)
X = df[feature_cols].dropna().copy()

if X.empty:
    raise ValueError("피처에 유효한 값이 없습니다. NaN/형 변환을 확인하세요.")


# ========= 5) 예측 =========
pred = model.predict(X)

# 원본 df에 정렬된 행들의 인덱스 기준으로 예측치 병합
df_pred = X.copy()
df_pred["COST_RATIO"] = pred


# ========= 6) 결과 확인/활용 =========
# 필요한 컬럼만 보기 좋게
out_cols = [c for c in ["BRAND_ID","STORE_ID", 'YEAR', 'MONTH'
                       ,"GDP","INFLATION_RATE","UNEMPLOYMENT","INTEREST_RATE","CCSI"]
            if c in df_pred.columns] + ["COST_RATIO"]

print(df_pred[out_cols].head(20))  # 상위 20건 미리보기


# (선택) CSV로 저장
save_path = BASE_DIR / f"{brandId}_pred_cost_ratio_{yyyymm}.csv"
df_pred[out_cols].to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"예측 결과 저장: {save_path}")

# 예측 결과 DB 저장
insert_sql = SQL_REPO.get("update_predict_sales_data")
with db.get_connection() as conn:
    with conn.cursor() as cur:
        for _, row in df_pred.iterrows():
            params = {
                "brand_id": row.get("BRAND_ID", brandId),
                "store_id": row["STORE_ID"],
                "yyyymm": f"{int(row['YEAR']):04d}-{int(row['MONTH']):02d}",
                "cost_ratio": row["COST_RATIO"]
            }
            cur.execute(insert_sql, params)
        conn.commit()
print("예측 결과를 DB에 저장 완료.")

# 예측


end_time = datetime.now()
print("종료시간 ==>", end_time)
