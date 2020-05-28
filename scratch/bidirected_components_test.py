import causaldag as cd
import itertools as itr
from directed_chordal_utils import tree_plus
from graph_utils import get_directed_clique_graph
import networkx as nx
import random
from tqdm import tqdm

random.seed(192842)

nnodes = 30
ngraphs = 1000
dags = [cd.DAG.from_nx(d) for d in tree_plus(nnodes, 2, 5, ngraphs=ngraphs)]
dcgs = [get_directed_clique_graph(d) for d in dags]

for dag, dcg in tqdm(zip(dags, dcgs), total=ngraphs):
    dcg_ = dcg.copy()
    dcg_.remove_all_directed()
    comps = list(nx.connected_components(dcg_.to_nx().to_undirected()))
    for comp in comps:
        for c1, c2 in itr.combinations(comp, 2):
            for child in dcg.children_of(c1) & dcg.children_of(c2) - set(comp):
                if (c1 & child) != (c2 & child):
                    print(c1, c2, child)
                    raise RuntimeError
            # for p in dcg.parents_of(c1) & dcg.parents_of(c2):
            #     if (c1 & p) != (c2 & p):
            #         print(c1, c2, p)
            #         raise RuntimeError


