import random as rd
from progress import Progress


def genc():
    val = rd.random()
    if val < 0.752:
        return 'c', rd.randint(1,80)
    elif val < 0.952:
        return 'r', rd.randint(1,40)
    elif val < 0.992:
        return 'e', rd.randint(1,20)
    else:
        return 'l', rd.randint(1,10)

def genr():
    val = rd.random()
    if val < 0.76:
        return 'r', rd.randint(1,40)
    elif val < 0.96:
        return 'e', rd.randint(1,20)
    else:
        return 'l', rd.randint(1,10)


def openpack():
    res = []
    for i in range(4):
        res.append(genc())
    res.append(genr())
    return res

def gethand(req):
    dic = {}
    dic['c'] = list(range(1,81))
    dic['r'] = list(range(1,41))
    dic['e'] = list(range(1,21))
    dic['l'] = list(range(1,11))
    cards = {}
    for key in dic.keys():
        for val in dic[key]:
            cards[key,val] = 0
    iters = 0
    while True:
        res = openpack()
        iters += 1
        for card in res:
            cards[card] += 1
        if testhand(cards, req):
            return iters
        
def testhand(cards, req):
    for key in req.keys():
        if cards[key] < req[key]:
            return False
    return True

def exphands(req, rounds=20000):
    iters = []
    prog = Progress(total_iter=rounds)
    for i in range(rounds):
        tmp = gethand(req)
        iters.append(tmp)
        prog.count()
    return sum(iters)/(len(iters) * 1.0)

        
reqkeys = {
        'c': ([7, 9, 12, 45, 61, 78], 2),
        'r': ([3, 6, 20, 34], 2),
        'e': ([5, 11, 19], 2),
        'l': ([1, 4, 8, 10], 1)
        }
reqkeys = {
        'c': ([7, 9], 2),
        'r': ([3, 6], 1),
        'e': ([5, 11], 1),
        'l': ([1, 4], 1)
        }
req = {}
for key in reqkeys:
    vals, cond  = reqkeys[key]
    for val in vals:
        req[key, val] = cond

print(exphands(req,rounds=100000)) 
