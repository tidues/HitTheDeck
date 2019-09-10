dic = {}
for i in range(1,1001):
    dic[i] = i

for i in dic:
    if i % 3 == 0:
        dic[i] = i * 2

for i in dic:
    if i % 5 == 0:
        dic[i] = i * 3

for i in dic:
    if i % 15 == 0:
        dic[i] = i * 10

vals = list(dic.values())
print(vals[:15])
print(sum(vals))

