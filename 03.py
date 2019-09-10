a = 2
b = 5

idx = 2

while idx < 40:
    c = a + b
    a = b
    b = c
    idx += 1

print(b)
