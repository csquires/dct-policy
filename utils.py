import itertools as itr


def powerset(s):
    return map(frozenset, itr.chain.from_iterable(itr.combinations(s, r) for r in range(len(s)+1)))


def powerset_monotone(s, predicate):
    return (ss for ss in powerset(s) if predicate(ss))
