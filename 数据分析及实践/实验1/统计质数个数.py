from math import sqrt

def Is_Prime(n):
    flag = True
    if n == 1:
        return False
    i = 2
    while i <= sqrt(n) + 1 and i < n:
        if n % i == 0:
            flag = False
            break
        i += 1

    return flag

count = 0
i = 2
n = int(input())

while i <= n:
    if Is_Prime(i):
        count += 1
    i += 1

print(count)
