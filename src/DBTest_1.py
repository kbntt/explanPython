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

# ========= 1) Oracle DB 접속 정보 =========
BASE_DIR   = Path(__file__).parent.parent  # 프로젝트 루트로 변경
SQL_REPO   = SqlRepo(BASE_DIR / "sql")  # ← 쿼리 폴더 지정
db         = DBOciConnect(BASE_DIR, SQL_REPO)
print("BASE_DIR =>", BASE_DIR)
brandId    = "obong"
params     = {"brand_id": brandId}

# yyyymm이 주어지지 않으면 None -> SQL에서 필터 미적용
today = datetime.today()
one_month_ago = today - relativedelta(months=1)# 한 달 전 날짜

# 원하는 형식으로 출력

# yyyymm이 주어지지 않으면 None -> SQL에서 필터 미적용
yyyymm = one_month_ago.strftime("%Y-%m")

# ========= 2) rows를 컬럼명과 함께 dict로 변환 =========

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

# YEAR, MONTH 파생 + 숫자형 캐스팅
for row in rows:
    storeNm = row.get("STORE_ID")      
    print("storeNm:",storeNm)
end_time = datetime.now()
print("종료시간 ==>", end_time)
print("===================================================")
