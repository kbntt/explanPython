import oracledb
from comm.props_loader import load_properties, getHost, getPort, getUser, getPassword, getSid, getServiceName

class DBConnect:
    def __init__(self, props_path, sql_repo):
        
        props = load_properties(baseDir / "application.properties")
        
        self.props = load_properties(props_path)
        self.host = getHost(self.props)
        self.port = getPort(self.props)
        self.user = getUser(self.props)
        self.password = getPassword(self.props)
        self.sid = getSid(self.props)
        self.service_name = getServiceName(self.props)
        self.sql_repo = sql_repo
        self.dsn = self._make_dsn()

    def _make_dsn(self):
        if self.service_name:
            return oracledb.makedsn(host=self.host, port=self.port, service_name=self.service_name)
        elif self.sid:
            return oracledb.makedsn(host=self.host, port=self.port, sid=self.sid)
        else:
            raise ValueError("application.properties에 datasource.sid 또는 datasource.service_name을 설정하세요.")

    def execute(self, sql_key, params=None):
        sql = self.sql_repo.get(sql_key)
        with oracledb.connect(user=self.user, password=self.password, dsn=self.dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params or {})
                return cur.fetchall()

    def get_connection(self):
        return oracledb.connect(user=self.user, password=self.password, dsn=self.dsn)
