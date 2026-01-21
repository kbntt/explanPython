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

# ========= 1) Oracle DB 접속 정보 =========
BASE_DIR   = Path(__file__).parent.parent  # 프로젝트 루트로 변경
SQL_REPO   = SqlRepo(BASE_DIR / "sql")  # ← 쿼리 폴더 지정
PROPS_PATH = BASE_DIR / "application.properties"

dbOci  = DBOciConnect(BASE_DIR, SQL_REPO)
db     = DBConnect(BASE_DIR, SQL_REPO)

# ========= 2) 데이타 조회 =========
params = {"brand_id": "obong"}
with db.get_connection() as conn:
    cur = conn.cursor()
    cur.execute(SQL_REPO.get("getCopyData_01"), params)
    columns = [desc[0] for desc in cur.description]
    rows = [dict(zip(columns, row)) for row in cur.fetchall()]

# 조회 결과가 없을 때 방어
if not rows:
    raise ValueError(f"조회 결과가 없습니다. ")

df = pd.DataFrame(rows)

# 결과 DB 저장
insert_sql = SQL_REPO.get("save_predict_sales_info")
with dbOci.get_connection() as conn:
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            params = {
                "BRAND_ID": row["BRAND_ID"],
                "STORE_ID": row["STORE_ID"],
                "YYYYMM": row["YYYYMM"],
                "COST_RATIO": float(row["COST_RATIO"])
            }
            cur.execute(insert_sql, params)
        conn.commit()
print("예측 결과를 DB에 저장 완료.")
end_time = datetime.now()
print("종료시간 ==>", end_time)
