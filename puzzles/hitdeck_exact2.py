import random as rd
import numpy as np
import operator as op
from functools import reduce
import scipy.sparse.linalg as lin
from progress import Progress

######################################################
# start: help functions
######################################################


# return ordered set for B_curl(a,b)
# where a is the dimension of the tuple, and k is the sum
def B_curl(n, k):
    tmp = B_curl_h(n, k)
    return list(map(tuple, tmp))


def B_curl_h(n, k):
    if n == 1:
        return [[k]]

    if k == 0:
        return list(map(lambda xs: addhd(0, xs), B_curl_h(n - 1, 0)))
    
    # for each possible value, get one
    res = []
    for i in range(k + 1):
        res += list(map(lambda xs: addhd(i, xs), B_curl_h(n - 1, k - i)))

    return res


# the number of elements in B_curl
def B_curl_num(n, k):
    if n == 1:
        return 1

    if k == 0:
        return 1
    
    res = 0
    for i in range(k + 1):
        res += B_curl_num(n - 1, k - i)

    return res


# return ordered set for A_curl(a,b)
def A_curl(n, k):
    res = []
    for k0 in range(k + 1):
        res += B_curl(n, k0)

    return res


# return ordered set for A_curl(a,b)
def A_curl_num(n, k):
    res = 0
    for k0 in range(k + 1):
        res += B_curl_num(n, k0)

    return res


# function to fast add head to a list
def addhd(hd, xs):
    xs.insert(0, hd)
    return xs


# choose func
def ncr(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer / denom


# cartesian product
def cartesian_bi(ls1, ls2):
    res = []
    for val1 in ls1:
        val1 = cartesian_term_convert(val1)
        for val2 in ls2:
            val2 = cartesian_term_convert(val2)
            res.append(tuple(val1 + val2))
    return res


# cartesian product list of lists
def cartesian_ls(ls2dim):
    if len(ls2dim) == 1:
        tmp = ls2dim[0]
        return list(map(lambda x: tuple(cartesian_term_convert(x)), tmp))
    elif len(ls2dim) == 2:
        return cartesian_bi(ls2dim[0], ls2dim[1])
    else:
        sp = int(len(ls2dim) / 2.0) + 1
        ls1 = ls2dim[:sp]
        ls2 = ls2dim[sp:]
        lsa = cartesian_ls(ls1)
        lsb = cartesian_ls(ls2)
        return cartesian_bi(lsa, lsb)


# convert cartesian term
def cartesian_term_convert(val):
    if 'tuple' in str(type(val)):
        res = list(val)
        return list(val)
    else:
        res = [val]
    return res

######################################################
# end: help functions
######################################################

# the class of state
class StateType:
    def __init__(self, type_name, total_cards, max_repetition, reqvec):
        self.type = type_name
        self.total_cards = total_cards
        self.max_rep = max_repetition
        self.reqvec = reqvec


class State:
    def __init__(self, state_type, state_vec):
        self.stype = state_type
        self.type = self.stype.type
        self.vec = self.init_vec(state_vec)
        if self.satisfy(self.stype.reqvec):
            self.vec = self.init_vec(self.stype.reqvec)

    def init_vec(self, vec):
        if len(vec) != self.stype.max_rep:
            raise Exception('error: vector length incorrect')
        ls = list(vec)
        last_entry = self.stype.total_cards - sum(ls)
        ls.append(last_entry)
        return tuple(ls)

    def satisfy(self, vec):
        vec1 = self.vec
        if len(vec) == self.stype.max_rep:
            vec2 = tuple(list(vec) + [0])
        else:
            vec2 = vec
        res = True
        diff = 0
        for i in range(len(vec1)):
            val1 = vec1[i] + diff
            val2 = vec2[i]
            if val1 < val2:
                res = False
                break
            else:
                diff = val1 - val2
        return res

    def __eq__(self, other):
        if self.type != other.type:
            return None
        if self.vec == other.vec:
            return True
        else:
            return False

    def __repr__(self):
        return self.type + str(self.vec)


reqkeys = {
        'c': ([7, 9, 12, 45, 61, 78], 2, 80),
        'r': ([3, 6, 20, 34], 2, 40),
        'e': ([5, 11, 19], 2, 20),
        'l': ([1, 4, 8, 10], 1, 10)
        }
# generate all states
def markov_states_gen(reqkeys):
    for key in reqkeys:
        max_rep = reqkeys[1]
        total_cards = reqkeys[2]
        firstval = len(reqkeys[0])
        reqvec = [firstval]
        for i in range(max_rep - 1):
            reqvec.append(0)
        reqvec = tuple(reqvec)
        stype = StateType(key, total_cards, max_rep, reqvec)

    



ctype = StateType('c', 80, 2)
s0 = State(ctype, (0,0))
s1 = State(ctype, (5,0))
s2 = State(ctype, (4,2))
print(s0)
print(s1)
print(s2)
v1 = (4,1)
v2 = (4,2)
print(s1.satisfy(v1))
print(s1.satisfy(v2))


# generate all states







# convert the list into a vector
# knum the number of cards
def get_prob(lst, reqs, prob_vec, knum):
    prob = 1.0
    prop_in_reqs = 0
    for idx, key in enumerate(reqs):
        choose_num = lst[idx]
        prop_in_reqs += prob_vec[key]
        if choose_num > 0:
            prob *= ncr(knum, choose_num) * (prob_vec[key] ** choose_num)
            knum -= choose_num
    prob *= (1-prop_in_reqs) ** knum
    return prob


# find all vectors given a distribution
def vecprob(reqs, prob_vec, card_num):
    vects = A_curl(len(reqs), card_num)
    v_dist = {}
    total_prob = 0
    for v in vects:
        prob_v = get_prob(v, reqs, prob_vec, card_num)
        if prob_v > 0:
            total_prob += prob_v
            v_dist[v] = prob_v
    # change the probability of the zero vector
    #v0 = tuple([0] * len(reqs))
    #v_dist[v0] = 2 - total_prob
    return v_dist


# add vectors
def vecadd(v1, v2):
    lst = [v1[i] + v2[i] for i in range(len(v1))]
    return tuple(lst)


# calc all the possible vectors with prob
def vecprob_combined(reqs, prob_vec_c, prob_vec_r):
    v_dist_c = vecprob(reqs, prob_vec_c, 4)
    v_dist_r = vecprob(reqs, prob_vec_r, 1)
    # combine two vectors
    v_dist = {}
    for v1 in v_dist_c:
        for v2 in v_dist_r:
            key = vecadd(v1, v2)
            if key in v_dist:
                v_dist[key] += v_dist_c[v1] * v_dist_r[v2]
            else:
                v_dist[key] = v_dist_c[v1] * v_dist_r[v2]
    return v_dist


# test if satisfy the condition
def state_mod(vec, reqs):
    vec = list(vec)
    reqvec = list(reqs.values())
    satisfied = True
    for i in range(len(vec)):
        if vec[i] > reqvec[i]:
            vec[i] = reqvec[i]
        elif vec[i] < reqvec[i]:
            satisfied = False
    if satisfied:
        state = 'done'
    else:
        state = tuple(vec)
    return state


# generate all the states 
def markov_states(reqs):
    print('generating markov chain states...')
    values = list(reqs.values())
    ls2dim = list(map(lambda x: list(range(0, x + 1)), values))
    states = cartesian_ls(ls2dim)
    return states


# idx to key
def idx2key(idx, reqs):
    keys = list(reqs.keys())
    return keys[idx]


# get probability
def get_p(v1, v2, reqs, one_step_vecs, state_f):
    v2_greater = True
    total_diff = 0
    c_diff = 0
    diffvec = []
    for i in range(len(v1)):
        v2_greater = v2_greater and v2[i] >= v1[i]
        total_diff += v2[i] - v1[i]
        diffvec.append(v2[i] - v1[i])
        key = idx2key(i, reqs)
        if key[0] == 'c':
            c_diff += v2[i] - v1[i]
    if v1 == state_f:
        if v1 == v2:
            res = 1.0
        else:
            res = 0.0
    elif v2_greater and total_diff <= 5 and c_diff <= 4:
        # calc prob of diffvec
        res = 0.0
        for vec in one_step_vecs:
            s1 = vecadd(v1, vec)
            s1 = state_mod(s1, reqs)
            if s1 == v2:
                res += one_step_vecs[vec]
    else:
        res = 0.0
    return res


# def markov_chain_gen(reqs, prob_vec_c, prob_vec_r):
#     v0 = tuple([0] * len(reqs))
#     states = set([])
#     new_states = [v0]
#     old_states = []
#     one_step_vecs = vecprob_combined(reqs, prob_vec_c, prob_vec_r)
#     trans_prob = {}
#     iters = 1
#     for key in reqs:
#         iters *= reqs[key] + 1
#     # print(iters)
#     prog = Progress(total_iter=iters)
#     while len(new_states) > 0:
#         s0 = new_states.pop(0)
#         old_states.append(s0)
#         if s0 == 'done':
#             s1 = 'done'
#             trans_prob[s0, s1] = 1.0
#         else:
#             for vec in one_step_vecs:
#                 s1 = vecadd(s0, vec)
#                 # first check if s1 satisfy the requirement
#                 s1 = state_mod(s1, reqs)
#                 if (s0, s1) not in trans_prob:
#                     trans_prob[s0, s1] = one_step_vecs[vec]
#                 else:
#                     trans_prob[s0, s1] += one_step_vecs[vec]
#                 # save s1 to new_states if not in the old_states
#                 if s1 not in old_states and s1 not in new_states:
#                     new_states.append(s1)
#                 # update all states
#                 oldlen = len(states)
#                 states.add(s1)
#                 newlen = len(states)
#                 if newlen > oldlen:
#                     prog.count()
#     return states, trans_prob


# solve for the 1st passage time
def solve_psg(states, reqs, prob_vec_c, prob_vec_r):
    reqlen = len(reqs)
    one_step_vecs = vecprob_combined(reqs, prob_vec_c, prob_vec_r)
    a = []
    b = []
    idx = 0
    key2idx = {}
    j = states[-1]
    states1 = [s for s in states if s != j]
    print('creating matrix...')
    prog = Progress(total_iter=len(states1) ** 2)
    for i in states1:
        key2idx[i] = idx
        idx += 1
        row = []
        for k in states1:
            if k == i:
                row.append(1.0 - get_p(i, i, reqs, one_step_vecs, j))
            else:
                row.append(-1.0 * get_p(i, k, reqs, one_step_vecs, j))
            prog.count()
        a.append(row.copy())
    b = [1] * len(states1)
    a = np.array(a)
    b = np.array(b)
    print('solving linear system...')
    x, info = lin.bicgstab(a, b)
    # print(x)
    print('solve info: ', info)
    # x = np.linalg.solve(a,b)
    v0 = tuple([0] * reqlen)
    return x[key2idx[v0]]
            

# create the required vector
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
reqs = {}
for key in reqkeys:
    vals, req = reqkeys[key]
    for val in vals:
        reqs[key, val] = req

# vector probabilities
allkeys = {}
allkeys['c'] = list(range(1, 81))
allkeys['r'] = list(range(1, 41))
allkeys['e'] = list(range(1, 21))
allkeys['l'] = list(range(1, 11))
prob_c = {'c': 0.752, 'r': 0.2, 'e': 0.04, 'l': 0.008}
prob_r = {'c': 0, 'r': 0.76, 'e': 0.2, 'l': 0.04}
prob_vec_c = {}
prob_vec_r = {}
for key in allkeys:
    for idx in allkeys[key]:
        prob_vec_c[key, idx] = prob_c[key] / len(allkeys[key])
        prob_vec_r[key, idx] = prob_r[key] / len(allkeys[key])

# solve
states = markov_states(reqs)
# print(states)
print(solve_psg(states, reqs, prob_vec_c, prob_vec_r))
        
