from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = {
    'random',
    'dct',
    'coloring',
}
nnodes_list = [100, 200, 300, 400, 500]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.HAIRBALL_PLUS,
    dict(degree=4, e_min=2, e_max=5),
    algorithms=algs,
    overwrite=False
)


