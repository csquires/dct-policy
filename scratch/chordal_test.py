import networkx as nx
import itertools as itr

g = nx.Graph()
g.add_edges_from([(1, 2), (1, 4), (2, 4), (2, 5), (4, 5), (1, 5)])
g.add_edges_from([(2, 3), (2, 6), (3, 5), (3, 6), (5, 6)])
g.add_edges_from([(4, 7), (4, 8), (5, 7), (5, 8), (7, 8)])
g.add_edges_from([(5, 9), (6, 8), (6, 9), (8, 9)])
g.add_edge(2, 8)

print(nx.is_chordal(g))
print(nx.chordal_graph_cliques(g))


def make_annoying(n):
    g = nx.Graph()
    g.add_edges_from(itr.combinations(range(n), 2))  # largest intersection
    for i in range(n, 2*n):
        g.add_edges_from((i, k) for k in range(i-n+1))
    g.add_edges_from((2*n, k) for k in range(n))
    return g


g = make_annoying(3)

print(nx.is_chordal(g))
print(nx.chordal_graph_cliques(g))


g = nx.Graph()
g.add_edges_from([
    (2, 1),
    (2, 4),
    (2, 3),
    (2, 5),
    (2, 7),
    (7, 4),
    (7, 6),
    (7, 8),
    (7, 5),
    (4, 1),
    (4, 6),
    (5, 3),
    (5, 8),
])