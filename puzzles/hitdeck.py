import random as rd


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
    for i in range(rounds):
        tmp = gethand(req)
        iters.append(tmp)
    return sum(iters)/(len(iters) * 1.0)

        

req = {}
clst = [7,9,12,45,61,78]
rlst = [3,6,20,34]
elst = [5,11,19]
llst = [1,4,8,10]
for c in clst:
    req['c', c] = 2
for r in rlst:
    req['r', r] = 2
for e in elst:
    req['e', e] = 2
for l in llst:
    req['l', l] = 1

print(exphands(req,rounds=10000)) 
