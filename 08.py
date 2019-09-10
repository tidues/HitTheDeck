a = 5000
b = 50000
num = 0
vals=[]
for x in range(2,300):
    for y in range(2,15):
        z = x ** y
        if z not in vals:
            if z >= a and z <= b:
                vals.append(z)
                num += 1
        if z > b:
            break

print(num)
        

