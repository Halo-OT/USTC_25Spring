def Fibonacci(n):
    while n >= 0:
        if n == 1:
            return 1
        elif n == 0:
            return 0
        else:
            return Fibonacci(n-1) + Fibonacci(n-2)

n = int(input())
print(Fibonacci(n))

    
    