def count_1(n):
    count = 0
    while n != 0:
        d = n % 2
        n = n // 2
        if d == 1:
            count += 1
    return count

n = int(input())
print(count_1(n))