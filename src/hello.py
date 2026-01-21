import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from joblib import load
import pandas as pd
import oracledb
from pathlib import Path
from comm.props_loader import load_properties
from comm.sql_loader import SqlRepo
from comm.dataBase.DBConnect import DBConnect

BASE_DIR   = Path(__file__).parent.parent  # 프로젝트 루트로 변경
PROPS_PATH = BASE_DIR / "application.properties"
SQL_REPO   = SqlRepo(BASE_DIR / "sql")  # ← 쿼리 폴더 지정

# ----------------- (2) 오라클 조회 -----------------

db = DBConnect(PROPS_PATH, SQL_REPO)
params = {"brand_id": "obong"}
rows = db.execute("store.select_tb_rotation", params)

# 파이썬 모델 예측 =====================================
""" """ 
# 모델 로드
model = load("./regression_model.pkl")

# 학습 당시 컬럼명 확인(있으면 그걸 사용)
feat_names = getattr(model, "feature_names_in_", None)

# TB_ROTATION 값을 모델 입력으로 사용
if rows:
    tb_rotation_vals = [val for (val,) in rows]
    feat_names = getattr(model, "feature_names_in_", None)
    X = pd.DataFrame(tb_rotation_vals, columns=feat_names) if feat_names is not None else [[v] for v in tb_rotation_vals]

    y = model.predict(X)
    print(y)

    # 예측 결과를 DB에 저장 (파일에서 쿼리 로드)
    insert_sql = SQL_REPO.get("store.insert_test_ai_result")
    for val in y:
        with db.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, {"value": float(val)})
            conn.commit()
        print(f"Inserted value {val} into TEST_AI_RESULT.")
else:
    print("예측에 사용할 TB_ROTATION 값이 없습니다.")

# 파이썬 모델 예측 #####################################
