from submit.dag_loader import DagSampler
from submit.plot_results_vary_nnodes2 import plot_results_vary_nodes

algs = {
    'random',
    'dct',
    'coloring',
}
nnodes_list = [100, 150, 200, 250, 300]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.HAIRBALL_PLUS,
    dict(degree=3, e_min=2, e_max=6),
    algorithms=algs,
    overwrite=False
)


