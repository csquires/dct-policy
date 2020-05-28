import causaldag as cd
import networkx as nx
from directed_chordal_utils import tree_plus
from mixed_graph import LabelledMixedGraph
import random
import numpy as np
from time import time

random.seed(633)

dags = [cd.DAG.from_nx(d) for d in tree_plus(30, 2, 5, 50)]

start = time()
dcts = [LabelledMixedGraph.from_nx(d.directed_clique_tree()) for d in dags]
print(time() - start)

start = time()
dcts2 = [LabelledMixedGraph.from_nx(d.directed_clique_tree2(verbose=False)) for d in dags]
print(time() - start)

# dcgs = [get_directed_clique_graph(d) for d in dags]
total_weights = np.array([sum([len(label) for label in (*dct.directed.values(), *dct.bidirected.values())]) for dct in dcts])
total_weights2 = np.array([sum([len(label) for label in (*dct.directed.values(), *dct.bidirected.values())]) for dct in dcts2])
print(all(nx.is_tree(dct.to_nx().to_undirected()) for dct in dcts2))
print(np.alltrue(total_weights == total_weights2))
