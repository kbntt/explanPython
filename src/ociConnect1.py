import os, oracledb
from pathlib import Path

# 1) 전자지갑 폴더(압축 해제된 폴더) 지정
WALLET_DIR = Path(r"D:\OCI\Wallet_REALDB")  # ← 본인 지갑 폴더로 변경
os.environ["TNS_ADMIN"] = str(WALLET_DIR)     # tnsnames.ora, sqlnet.ora 위치

# 2) TNS alias (tnsnames.ora에 있는 이름 사용)
DSN = "realdb_high"  # pythondb_high / pythondb_tp 등

print("======================================")

# 3) DB 사용자
DB_USER = "OBNG"
DB_PASSWORD = "Dnsdud932152"

def run_app():
    pool = oracledb.create_pool(
        user=DB_USER,
        password=DB_PASSWORD,
        dsn=DSN,
        config_dir=str(WALLET_DIR),
        wallet_location=str(WALLET_DIR),
        wallet_password=DB_PASSWORD,
        min=1, max=4, increment=1
    )
    with pool.acquire() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT STORE_NM
                FROM TBL_EXP_STORE_MST
                WHERE BRAND_ID = :brand_id
                ORDER BY STORE_NM
            """, brand_id='obong')

            # ① 모두 가져와 출력
            rows = cur.fetchall()
            print(f"총 {len(rows)}건")
            for i, (store_nm,) in enumerate(rows, 1):
                print(f"{i}. {store_nm}")

            # --- 대량일 때 메모리 아끼려면 ② 방식 사용 ---
            # while True:
            #     chunk = cur.fetchmany(1000)
            #     if not chunk:
            #         break
            #     for (store_nm,) in chunk:
            #         print(store_nm)

if __name__ == "__main__":
    run_app()
