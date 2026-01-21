import sys

# 커맨드라인 인자가 제대로 들어왔는지 체크
if len(sys.argv) < 2:
    print("사용법: python cmdInput.py <값>")
    sys.exit(1)

value = sys.argv[1]
print("입력받은 값:", value)