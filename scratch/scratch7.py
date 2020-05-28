import causaldag as cd
from mixed_graph import LabelledMixedGraph

d = cd.DAG(arcs={(2, 1), (2, 3), (3, 1), (3, 4), (5, 2), (2, 6), (6, 3), (2, 4)})
print(d.vstructures())
dct = LabelledMixedGraph.from_nx(d.directed_clique_tree())
print(dct)
