import causaldag as cd
from utils import powerset
from random_graphs import tree_plus
from tqdm import tqdm
import random

random.seed(185723)

nnodes = 15
ngraphs = 10
dags = [cd.DAG.from_nx(d) for d in tree_plus(nnodes, 2, 10, ngraphs)]
cpdags = [d.cpdag() for d in dags]

for dag, cpdag in tqdm(zip(dags, cpdags), total=ngraphs):
    sdct = dag.simplified_directed_clique_tree()
    sdct_nodes = list(sdct.nodes)
    sdct_components = [frozenset.union(*b) for b in sdct_nodes]
    parents = [list(sdct.predecessors(b)) for b in sdct_nodes]
    parents = [frozenset.union(*p[0]) if p else set() for p in parents]
    sdct_residuals = [frozenset.union(*b) - p for b, p in zip(sdct, parents)]
    print(sdct_residuals)

    sdct_subgraphs = [dag.induced_subgraph(r) for r in sdct_residuals]
    subgraph_cpdags = [s.cpdag() for s in sdct_subgraphs]

    for interventions in powerset(set(range(nnodes))):
        if cpdag.interventional_cpdag(dag, interventions).num_edges == 0:
            for subgraph, subgraph_cpdag in zip(sdct_subgraphs, subgraph_cpdags):
                if subgraph_cpdag.interventional_cpdag(subgraph, interventions & subgraph.nodes).num_edges != 0:
                    raise RuntimeError
