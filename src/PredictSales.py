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
    print("사용법: python PredictSales.py <brandId> [yyyymm: YYYY-MM]")
    sys.exit(1)
brandId = sys.argv[1]

# ========= 1) Oracle DB 접속 정보 =========
BASE_DIR   = Path(__file__).parent.parent  # 프로젝트 루트로 변경
SQL_REPO   = SqlRepo(BASE_DIR / "sql")  # ← 쿼리 폴더 지정
db         = DBOciConnect(BASE_DIR, SQL_REPO)

today = datetime.today()
one_month_ago = today - relativedelta(months=1)# 한 달 전 날짜

# 원하는 형식으로 출력

# yyyymm이 주어지지 않으면 None -> SQL에서 필터 미적용
yyyymm = one_month_ago.strftime("%Y-%m")

print("yyyymm =>", yyyymm)
if len(sys.argv) >= 3:
    yyyymm = sys.argv[2]  # 예: '2025-08'

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

# 간단 출력 (앞부분만)
print("\n[조회 결과 DataFrame: 상위 5건]")
print(df.head())

# 1) 타입 보정
if "STORE_ID" in df.columns:
    df["STORE_ID"] = df["STORE_ID"].astype(str)

for col in ["YEAR", "MONTH"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

macro_vars = ["GDP", "INFLATION_RATE", "UNEMPLOYMENT", "INTEREST_RATE", "CCSI"]
for col in macro_vars + ["SALES","COGS_AMT","LABOR_COST","SELL_EXPENSE","ADMIN_EXPENSE","OTHER_GA_EXP"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 2) 피처 세트 (학습 시 설계와 동일하게)
sales_num_cols = [c for c in (macro_vars + ["YEAR"]) if c in df.columns]
sales_cat_cols = [c for c in ["STORE_ID","MONTH"] if c in df.columns]


fileAdmin = f"{brandId}_admin_model.pkl"
fileCogs = f"{brandId}_cogs_model.pkl"
fileLabor = f"{brandId}_labor_model.pkl"
fileOther = f"{brandId}_other_model.pkl"
fileSales = f"{brandId}_sales_model.pkl"
fileSell = f"{brandId}_sell_model.pkl"

BASE_DIR   = Path(__file__).parent.parent  # 프로젝트 루트로 변경
model_dir = BASE_DIR / "model"
modelAdmin     = load(model_dir / f"{fileAdmin}")
modelCogs     = load(model_dir / f"{fileCogs}")
modelLabor     = load(model_dir / f"{fileLabor}")
modelOther     = load(model_dir / f"{fileOther}")
modelSales     = load(model_dir / f"{fileSales}")
modelSell     = load(model_dir / f"{fileSell}")


# 3) 매출 예측
X_sales = df[sales_num_cols + sales_cat_cols]
df["PRED_SALES"] = modelSales.predict(X_sales)

# 4) 비용 예측 (예측 매출 사용)
cost_num_cols = [c for c in (macro_vars + ["YEAR", "SALES"]) if c in df.columns]
cost_cat_cols = sales_cat_cols[:]

cost_input = df.copy()
cost_input["SALES"] = cost_input["PRED_SALES"]  # 예측 매출 주입
X_cost = cost_input[cost_num_cols + cost_cat_cols]

df["PRED_COGS"]   = modelCogs.predict(X_cost)
df["PRED_LABOR"]  = modelLabor.predict(X_cost)
df["PRED_SELL"]   = modelSell.predict(X_cost)
df["PRED_ADMIN"]  = modelAdmin.predict(X_cost)
df["PRED_OTHER"]  = modelOther.predict(X_cost)

# 5) 음수/NaN 방어
for c in ["PRED_SALES","PRED_COGS","PRED_LABOR","PRED_SELL","PRED_ADMIN","PRED_OTHER"]:
    df[c] = df[c].fillna(0)
    df[c] = df[c].clip(lower=0)

# 6) 영업이익/이익률
df["PRED_COST_SUM"]     = df["PRED_COGS"] + df["PRED_LABOR"] + df["PRED_SELL"] + df["PRED_ADMIN"] + df["PRED_OTHER"]
df["PRED_OPER_PROFIT"]  = df["PRED_SALES"] - df["PRED_COST_SUM"]
df["PRED_OP_MARGIN"]    = (df["PRED_OPER_PROFIT"] / df["PRED_SALES"]).where(df["PRED_SALES"] != 0, 0.0)

# 7) 출력
out_cols = [
    "BRAND_ID","STORE_ID","YEAR","MONTH",
    "PRED_SALES","PRED_COGS","PRED_LABOR","PRED_SELL","PRED_ADMIN","PRED_OTHER",
    "PRED_COST_SUM","PRED_OPER_PROFIT","PRED_OP_MARGIN"
]
out_cols = [c for c in out_cols if c in df.columns]

print("\n[예측 결과 상위 10건]")
#print(df[out_cols].head(10).to_string(index=False))

# 퍼센트 보기 좋게 예시 출력(선택)
if "PRED_OP_MARGIN" in df.columns:
    tmp = df[out_cols].copy()
    tmp["PRED_OP_MARGIN"] = (tmp["PRED_OP_MARGIN"] * 100).round(2)
    #print("\n[예측 결과 (OP_MARGIN % 포맷, 상위 10건)]")
    #print(tmp.head(10).to_string(index=False))

# ==================================
# 8) 예측 결과 DB 저장
# ==================================

# 예측 결과 DB 저장
insert_sql = SQL_REPO.get("save_predict_sales_data")
with db.get_connection() as conn:
    with conn.cursor() as cur:
        batch_params = []
        for _, row in df[out_cols].iterrows():
            batch_params.append({
                "brand_id": row.get("BRAND_ID", brandId),
                "store_id": row["STORE_ID"],
                "yyyymm": f"{int(row['YEAR']):04d}-{int(row['MONTH']):02d}",
                "pred_sales": float(row["PRED_SALES"]),
                "pred_cogs": float(row["PRED_COGS"]),
                "pred_labor": float(row["PRED_LABOR"]),
                "pred_sell": float(row["PRED_SELL"]),
                "pred_admin": float(row["PRED_ADMIN"]),
                "pred_other": float(row["PRED_OTHER"]),
                "pred_cost_sum": float(row["PRED_COST_SUM"]),
                "pred_op_profit": float(row["PRED_OPER_PROFIT"]),
                "pred_op_margin": float(row["PRED_OP_MARGIN"]),
            })
        cur.executemany(insert_sql, batch_params)
        conn.commit()
print("예측 결과를 DB에 저장 완료.")



end_time = datetime.now()
print("종료시간 ==>", end_time)


