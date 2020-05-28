import networkx as nx
import itertools as itr

cliques = [(1, 3, 4, 6), (1, 2, 3, 4), (2, 3, 4, 5)]
edges = set(itr.chain.from_iterable((itr.combinations(c, 2) for c in cliques)))
g = nx.Graph()
g.add_edges_from(edges)
print(nx.chordal_graph_cliques(g))
