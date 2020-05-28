a = {(1, 2): 3, (2, 3): 4}

backward = {(j, i): label for (i, j), label in a.items()}
import itertools as itr


for (i, j), label in itr.chain(a.items(), backward.items()):
    if i == 1:
        backward.pop((j, i))
    print((i, j))

