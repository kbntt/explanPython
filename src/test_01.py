from datetime import datetime
from dateutil.relativedelta import relativedelta

# 현재 날짜
today = datetime.today()

# 한 달 전 날짜
one_month_ago = today - relativedelta(months=1)

# 원하는 형식으로 출력
formatted_date = one_month_ago.strftime("%Y-%m")
print(formatted_date)

# 현재 날짜
today = datetime.today()

# 한 달 전 날짜
result_month = today - relativedelta(months=2)

format_date = result_month.strftime("%Y-%m-%d %H:%M:%S")
print(format_date)
