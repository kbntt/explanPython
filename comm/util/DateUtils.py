
from datetime import datetime

def parseYearMonth(val):
    """
    문자열(YYYYMM, YYYY-MM, May-21 등)을 받아
    (year, month) 튜플로 변환
    """
    if val is None:
        return None, None

    s = str(val).strip()
    # 지원할 포맷 정의
    formats = ("%Y-%m", "%b-%y", "%b-%Y", "%Y%m")

    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.year, dt.month
        except ValueError:
            continue

    # 파싱 실패
    return None, None