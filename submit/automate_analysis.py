from submit.plot_results_vary_density import plot_results_vary_density
from submit.plot_results_vary_nodes import plot_results_vary_nodes
from submit.dag_loader import DagSampler

ngraphs = 100

# fig1
nnodes_list = [100, 150, 200, 250, 300]
sampler = DagSampler.HAIRBALL_PLUS
other_params = dict(e_min=2, e_max=5, degree=4)
algorithms = {
    'random',
    'dct',
    # 'opt_single',
    'coloring',
    # 'greedy_minmax',
    # 'greedy_entropy'
}
plot_results_vary_nodes(nnodes_list, ngraphs, sampler, other_params, algorithms)

# fig2
nnodes_list = [20, 30, 40]
sampler = DagSampler.HAIRBALL_PLUS
other_params = dict(e_min=2, e_max=5, degree=4)

algorithms = {
    'random',
    'dct',
    'opt_single',
    'coloring',
    'greedy_minmax',
    'greedy_entropy'
}
plot_results_vary_nodes(nnodes_list, ngraphs, sampler, other_params, algorithms)

# figure 3
nnodes = 30
density_list = [.1, .2, .3, .4]
sampler = DagSampler.SHANMUGAM
algorithms = {
    'random',
    'dct',
    # 'opt_single',
    'coloring',
    # 'greedy_minmax',
    # 'greedy_entropy'
}
plot_results_vary_density(nnodes, density_list, ngraphs, sampler, dict(), algorithms)
