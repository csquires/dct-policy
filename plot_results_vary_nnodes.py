from dag_loader import DagSampler
import matplotlib.pyplot as plt
import seaborn as sns
from config import FIGURE_FOLDER, POLICY2COLOR, POLICY2LABEL
import os
import random
import ipdb
from result_getter import ResultGetter
sns.set()

OVERWRITE_ALL = True


def plot_results_vary_nodes(
        nnodes_list: list,
        ngraphs: int,
        sampler: DagSampler,
        other_params: dict,
        algorithms: set,
        overwrite=False
):
    random.seed(98625472)
    os.makedirs('figures', exist_ok=True)

    rg = ResultGetter(
        algorithms,
        nnodes_list,
        sampler,
        other_params_list=[other_params],
        ngraphs=ngraphs,
    )
    res_df = rg.get_results(overwrite=overwrite)
    mean_ratios = res_df.groupby(level=['alg', 'nnodes'])['regret_ratio'].mean()
    max_ratios = res_df.groupby(level=['alg', 'nnodes'])['regret_ratio'].max()
    average_times = res_df.groupby(level=['alg', 'nnodes'])['time'].mean()

    plt.clf()
    for alg in algorithms:
        plt.plot(nnodes_list, mean_ratios[mean_ratios.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg])

    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Competitive Ratio')
    plt.xticks(nnodes_list)
    plt.legend()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'avgregret_sampler={sampler},nnodes_list={nnodes_list},{other_params_str}.png'))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'avgregret'))

    plt.clf()
    for alg in algorithms:
        plt.plot(nnodes_list, max_ratios[max_ratios.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg])

    plt.xlabel('Number of Nodes')
    plt.ylabel('Maximum Competitive Ratio')
    plt.legend()
    plt.xticks(nnodes_list)
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'maxregret_sampler={sampler},nnodes_list={nnodes_list},{other_params_str}.png'))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'maxregret'))

    plt.clf()
    for alg in algorithms:
        plt.plot(nnodes_list, average_times[average_times.index.get_level_values('alg') == alg], color=POLICY2COLOR[alg], label=POLICY2LABEL[alg])

    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Computation Time')
    plt.legend()
    plt.yscale('log')
    plt.xticks(nnodes_list)
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'times_sampler={sampler},nnodes_list={nnodes_list},{other_params_str}.png'))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'times'))

    ipdb.set_trace()
