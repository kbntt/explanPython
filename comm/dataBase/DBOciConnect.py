
import os
import oracledb
from pathlib import Path
from comm.props_loader import load_properties

class DBOciConnect:
    def __init__(self, baseDir, sql_repo, min=1, max=4, increment=1):
        props = load_properties(baseDir / "application.properties")
        wallet_dir = Path(props.get("oci.wallet_dir"))
        print("wallet_dir =>", wallet_dir)

        self.wallet_dir = Path(props.get("oci.wallet_dir"))
        os.environ["TNS_ADMIN"] = str(self.wallet_dir)
        self.dsn = props.get("oci.dsn")
        self.user = props.get("oci.user")
        self.password = props.get("oci.password")
        self.sql_repo = sql_repo
        self.pool = oracledb.create_pool(
            user=self.user,
            password=self.password,
            dsn=self.dsn,
            config_dir=str(self.wallet_dir),
            wallet_location=str(self.wallet_dir),
            wallet_password=self.password,
            min=min, max=max, increment=increment
        )

    def execute(self, sql_key, params=None):
        sql = self.sql_repo.get(sql_key)
        with self.pool.acquire() as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                return cur.fetchall()

    def get_connection(self):
        return self.pool.acquire()
