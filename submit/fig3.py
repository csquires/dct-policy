from submit.dag_loader import DagSampler
from submit.plot_results_vary_density import plot_results_vary_density

nnodes = 20
density_list = [.1, .2]
ngraphs = 100
sampler = DagSampler.ERDOS
algorithms = {
    'random',
    'dct',
    'opt_single',
    'coloring',
    'greedy_minmax',
    'greedy_entropy'
}
plot_results_vary_density(nnodes, density_list, ngraphs, sampler, dict(), algorithms)

