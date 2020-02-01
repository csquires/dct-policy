import causaldag as cd
import operator as op
import itertools as itr
import random
import networkx as nx


def partition(lst, s):
    res = [[]]
    i = 0
    for el in lst:
        if el in s and len(res[i]) > 0:
            i += 1
            res.append([])
        if el not in s:
            res[i].append(el)
    if not res[-1]:
        del res[-1]
    return res


def new_optset(d):
    cliques = nx.chordal_graph_cliques(d.to_nx().to_undirected())
    cliques = sorted(cliques, key=len, reverse=True)
    topsort = d.topological_sort()
    sorted_cliques = [[node for node in topsort if node in c] for c in cliques]
    intervened_nodes = set()
    for c in sorted_cliques:
        for section in partition(c, intervened_nodes):
            intervened_nodes.update(section[-2::-2])

    return intervened_nodes


def pattern2graph(pattern):
    arcs = set()
    tail_counter = max(pattern, key=op.itemgetter(0))[0]
    for head, tail in pattern:
        clique = [*range(head), *range(tail_counter, tail_counter+tail)]
        arcs.update(itr.combinations(clique, 2))
        tail_counter += tail
    return cd.DAG(arcs=arcs)


def random_pattern(min_head=1, max_head=6, min_tail=1, max_tail=4, min_cliques=2, max_cliques=5):
    head = random.randint(min_head, max_head)
    pattern = [(head, random.randint(min_tail, max_tail)), (head, random.randint(min_tail, max_tail))]
    for i in range(random.randint(min_cliques, max_cliques) - 2):
        pattern.append([random.randint(1, head), random.randint(min_tail, max_tail)])
    return pattern


ngraphs = 10
# patterns = [random_pattern() for _ in range(ngraphs)]
patterns = [[(4, 1), (4, 2)]]
graphs = [pattern2graph(p) for p in patterns]
optsets1 = [d.optimal_fully_orienting_interventions() for d in graphs]
optsets2 = [d.optimal_fully_orienting_interventions(new=True) for d in graphs]
optsets3 = [new_optset(d) for d in graphs]
print(optsets1)
print(optsets2)
print(optsets3)
