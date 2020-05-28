import causaldag as cd
import itertools as itr
from random_graphs import tree_plus
import networkx as nx
import random
from mixed_graph import LabelledMixedGraph
from graph_utils import get_directed_clique_graph
from tqdm import tqdm

random.seed(82)
nnodes = 20
ngraphs = 1000
dags = [cd.DAG.from_nx(d) for d in tree_plus(nnodes, 2, 5, ngraphs)]
dcts = [d.directed_clique_tree() for d in dags]
dcgs = [get_directed_clique_graph(d) for d in dags]

for dag, dct, dcg in tqdm(zip(dags, dcts, dcgs), total=ngraphs):
    dct_ = LabelledMixedGraph.from_nx(dct)
    dcg_nx = dcg.to_nx()
    for c in dct.nodes:
        parents = dcg.parents_of(c)
        if parents:
            s = min((len(p & c) for p in parents))
            smallest_parents = {p for p in parents if len(p & c) == s}
            for s, p in itr.product(smallest_parents, parents-smallest_parents):
                if not nx.has_path(dcg_nx, s, p):
                    raise RuntimeError

