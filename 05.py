def str2code(text):
    codes = [ord(x) for x in text]
    for idx, code in enumerate(codes):
        if code < ord('G'):
            codes[idx] += 100
    codes.sort()
    for idx, code in enumerate(codes):
        if code >= 100:
            codes[idx] -= 100
    text = [chr(x) for x in codes]
    res = ''
    for char in text:
        res += char

    return res

val = 'TECHCHALLENGECODINGSPEEDRUN'

print(str2code(val))

GGHHILLNNNOPRSTUACCCDDEEEEE


