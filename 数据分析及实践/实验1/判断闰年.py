n = int(input())
flag = False

if n % 100 == 0:
    if n % 400 == 0:
        flag = True
else:
    if n % 4 == 0:
        flag = True

if flag:
    print("yes")
else :
    print("no")
