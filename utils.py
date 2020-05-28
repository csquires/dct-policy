import itertools as itr
import operator as op
import random


def powerset(s):
    """
    Return all subsets of `s`, order by increasing size.
    """
    return map(frozenset, itr.chain.from_iterable(itr.combinations(s, r) for r in range(len(s)+1)))


def powerset_predicate(s, predicate):
    # any_satisfy = True
    # for r in range(len(s)+1):
    #     if any_satisfy:
    #         for ss in itr.combinations(s, r):
    #             if predicate(ss):
    #                 any_satisfy = True
    #                 yield frozenset(ss)
    return (ss for ss in powerset(s) if predicate(ss))


def random_max(d: dict, minimize=False):
    """
    Return a random key amongst the items in a dictionary with the maximum value.
    """
    if minimize:
        optimal_val = min(d.items(), key=op.itemgetter(1))[1]
    else:
        optimal_val = max(d.items(), key=op.itemgetter(1))[1]
    eligible_keys = [key for key, val in d.items() if val == optimal_val]
    return random.choice(eligible_keys)
