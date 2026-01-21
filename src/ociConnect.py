
import os
import oracledb
from pathlib import Path

class OCIDBConnect:
    def __init__(self, wallet_dir, dsn, user, password, min=1, max=4, increment=1):
        self.wallet_dir = Path(wallet_dir)
        os.environ["TNS_ADMIN"] = str(self.wallet_dir)
        self.dsn = dsn
        self.user = user
        self.password = password
        self.pool = oracledb.create_pool(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            config_dir=str(self.wallet_dir),
            wallet_location=str(self.wallet_dir),
            wallet_password=self.password,
            min=min, max=max, increment=increment
        )

    def execute(self, sql, params=None):
        with self.pool.acquire() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                return cur.fetchall()

    def get_connection(self):
        return self.pool.acquire()

# 사용 예시
if __name__ == "__main__":
    WALLET_DIR = r"D:\OCI\Wallet_REALDB"
    DSN = "realdb_high"
    DB_USER = "OBNG"
    DB_PASSWORD = "Dnsdud932152"
    db = OCIDBConnect(WALLET_DIR, DSN, DB_USER, DB_PASSWORD)
    result = db.execute("SELECT * FROM TBL_EXP_STORE_MST WHERE BRAND_ID = 'obong' ORDER BY STORE_NM ")
    print("Connected! result:", result[0][0])
