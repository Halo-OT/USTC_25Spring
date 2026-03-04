data = []
line = input().strip()
N, M = map(int, line.split())

for i in range(N):    
    line = input().strip()
    if not line:  # 遇到空行则停止            
        break
    a, b = map(int, line.split())
    data.append((a, b))
    
data.sort()

total_cost = 0
remaining = N

for pi, ai in data:
    if remaining <= 0:
        break
    if ai <= remaining:
        total_cost += pi * ai
        remaining -= ai
    else:
        total_cost += pi * remaining
        remaining = 0
        break

print(total_cost)