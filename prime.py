count = 0
p = int(input("prime time: "))
list = []
for i in range(2,p):
    count = 0
    for j in range(2,i):
        if i%j==0:
            count=count+1
    if count==0:
        list.append(i);
print(list)
