ls = [-5, -19, 0, -20, -11, 12, 27, -16, -2, -2, 23, 0, -3, 4, 7, -1, -28, 18, 21, 17, -23, 9, 2, -19, 8]

length = 5

maxval = -float('inf')
maxls = None

for i in range(len(ls)-4):
    tmpls = ls[i:i+5]
    if sum(tmpls) >= maxval:
        maxls = tmpls
        maxval = sum(tmpls)

print(maxval)
print(maxls)
    

