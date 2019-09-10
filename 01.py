num = 0
for i in range(2,5000):
    if i % 3 == 0 or i % 7 == 0 or i % 11 == 0:
        num += 1
print(num)

