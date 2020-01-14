import itertools as itr
import operator as op
import random


def powerset(s):
    return map(frozenset, itr.chain.from_iterable(itr.combinations(s, r) for r in range(len(s)+1)))


def powerset_monotone(s, predicate):
    return (ss for ss in powerset(s) if predicate(ss))


def random_max(d: dict):
    max_val = max(d.items(), key=op.itemgetter(1))[1]
    eligible_keys = [key for key, val in d.items() if val == max_val]
    return random.choice(eligible_keys)


def write_list(l, filename):
    with open(filename, 'w') as f:
        for item in l:
            f.write('%s\n' % item)


def read_list(filename) -> list:
    l = []
    with open(filename) as f:
        for line in f.readlines():
            l.append(line.rstrip())
    return l
