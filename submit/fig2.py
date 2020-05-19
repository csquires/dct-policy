from submit.dag_loader import DagSampler
import seaborn as sns
from submit.plot_results_vary_nodes import plot_results_vary_nodes
sns.set()

ngraphs = 100
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
