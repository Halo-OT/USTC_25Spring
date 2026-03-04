line = input().strip()
a, b = map(str, line.split())
result = 0
for i in a:
    for j in b:
        result += int(i) * int(j)
print(result)