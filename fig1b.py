from dag_loader import DagSampler
from plot_results_vary_nnodes import plot_results_vary_nodes

algs = {
    'random',
    'dct',
    'coloring',
    'opt_single',
    'greedy_minmax',
    'greedy_entropy'
}
nnodes_list = [8, 10, 12, 14]
plot_results_vary_nodes(
    nnodes_list,
    100,
    DagSampler.SHANMUGAM,
    dict(density=.1),
    algorithms=algs,
    overwrite=True
)


