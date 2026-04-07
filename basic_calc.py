x,y=map(int,input().split())
a=str(input("Enter s for sum,b for subtract,m for multiply and d for division :"))
if a=="s":
    print(x+y)
elif a=="b" :
    print(x-y)
elif a=="m":
    print(x*y)
elif a=="d" :
    print(x/y)
else :
    print("Wrong string")            
    