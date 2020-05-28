import causaldag as cd
import itertools as itr
from random_graphs import tree_plus
import random

random.seed(8852)
nnodes = 40
ngraphs = 1000
dags = [cd.DAG.from_nx(d) for d in tree_plus(nnodes, 10, 20, ngraphs)]
residuals_list = [d.residuals() for d in dags]
r_essential_graphs = [d.residual_essential_graph() for d in dags]
r_essential_graphs_ = [r.copy() for r in r_essential_graphs]
for r in r_essential_graphs_:
    r.to_complete_pdag()

residuals_full = []
for dag, residuals in zip(dags, residuals_list):
    arcs = set()
    for r1, r2 in itr.combinations(residuals, 2):
        arcs.update(dag._arcs & (set(itr.product(r1, r2)) | set(itr.product(r2, r1))))
    g = cd.PDAG(arcs=arcs, edges=dag.arcs-arcs)
    residuals_full.append(g)
    print(dag.num_arcs)
    print(g.num_adjacencies)

for ix, r, r_, rf in zip(range(ngraphs), r_essential_graphs, r_essential_graphs_, residuals_full):
    print(r == r_)
    if r != rf:
        raise RuntimeError

# arcs = {(1, 2), (1, 3), (1, 4), (1, 6), (1, 8), (1, 10), (2, 6), (3, 9), (4, 2), (4, 3), (4, 6), (5, 0), (5, 1),
#         (6, 3), (9, 7), (10, 2), (10, 4), (10, 6), (10, 11)}
# dag = cd.DAG(arcs=arcs)
# dag.directed_clique_tree(verbose=True)
