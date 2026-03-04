from math import comb
line = input().strip()
n, m, q = map(int, line.split())

i = 0
while i < q:
    line = input().strip()
    x1, y1, x2, y2 = map(int, line.split())
    dx = abs(x1 - x2)
    dy = abs(y1 - y2)
    print(comb(dx + dy ,dx))
    i += 1
