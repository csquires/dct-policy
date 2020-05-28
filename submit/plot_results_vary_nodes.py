from submit.alg_runner import AlgRunner
from submit.dag_loader import DagLoader, DagSampler
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from submit.config import FIGURE_FOLDER, POLICY2COLOR
import os
import networkx as nx
from p_tqdm import p_map
import random
import ipdb
sns.set()

OVERWRITE_ALL = True


def plot_results_vary_nodes(
        nnodes_list: list,
        ngraphs: int,
        sampler: DagSampler,
        other_params: dict,
        algorithms: set
):
    random.seed(98625472)

    random_results = np.zeros([len(nnodes_list), ngraphs])
    dct_results = np.zeros([len(nnodes_list), ngraphs])
    opt_single_results = np.zeros([len(nnodes_list), ngraphs])
    coloring_results = np.zeros([len(nnodes_list), ngraphs])
    greedy_minmax_results = np.zeros([len(nnodes_list), ngraphs])
    greedy_entropy_results = np.zeros([len(nnodes_list), ngraphs])

    verification_optimal = np.zeros([len(nnodes_list), ngraphs])
    clique_sizes = np.zeros([len(nnodes_list), ngraphs])
    num_cliques = np.zeros([len(nnodes_list), ngraphs])
    dags_array = np.zeros([len(nnodes_list), ngraphs], dtype=object)
    num_arcs = np.zeros([len(nnodes_list), ngraphs])

    for i, nnodes in enumerate(nnodes_list):
        dl = DagLoader(nnodes, ngraphs, sampler, {**other_params, 'nnodes': nnodes}, comparable_edges=True)
        dags = dl.get_dags()
        cliques_list = p_map(lambda d: nx.chordal_graph_cliques(d.to_nx().to_undirected()), dags)
        clique_sizes[i] = [max(len(c) for c in cliques) for cliques in cliques_list]
        num_cliques[i] = [len(cliques) for cliques in cliques_list]
        dags_array[i] = dags
        num_arcs[i] = [d.num_arcs for d in dags]

        verification_optimal[i] = dl.get_verification_optimal_ivs()
        if 'random' in algorithms:
            ar = AlgRunner('random', dl)
            random_results[i] = ar.get_alg_results(overwrite=OVERWRITE_ALL)
        if 'dct' in algorithms:
            ar = AlgRunner('dct', dl)
            dct_results[i] = ar.get_alg_results(overwrite=OVERWRITE_ALL)
        if 'opt_single' in algorithms:
            ar = AlgRunner('opt_single', dl)
            opt_single_results[i] = ar.get_alg_results(overwrite=OVERWRITE_ALL)
        if 'coloring' in algorithms:
            ar = AlgRunner('coloring', dl)
            coloring_results[i] = ar.get_alg_results(overwrite=OVERWRITE_ALL)
        if 'greedy_minmax' in algorithms:
            ar = AlgRunner('greedy_minmax', dl)
            greedy_minmax_results[i] = ar.get_alg_results(overwrite=OVERWRITE_ALL)
        if 'greedy_entropy' in algorithms:
            ar = AlgRunner('greedy_entropy', dl)
            greedy_entropy_results[i] = ar.get_alg_results(overwrite=OVERWRITE_ALL)

    print(num_arcs.mean(axis=1))
    log_num_cliques = np.ceil(np.log2(num_cliques))
    bounds = 3 * log_num_cliques + 2
    plt.clf()
    if 'random' in algorithms:
        plt.plot(nnodes_list, (random_results/verification_optimal).mean(axis=1), color=POLICY2COLOR['random'], label='ND-Random')
    if 'dct' in algorithms:
        plt.plot(nnodes_list, (dct_results/verification_optimal).mean(axis=1), color=POLICY2COLOR['dct'], label='DCT')
    if 'opt_single' in algorithms:
        plt.plot(nnodes_list, (opt_single_results/verification_optimal).mean(axis=1), color=POLICY2COLOR['opt_single'], label='OptSingle')
    if 'coloring' in algorithms:
        plt.plot(nnodes_list, (coloring_results/verification_optimal).mean(axis=1), color=POLICY2COLOR['coloring'], label='Coloring')
    if 'greedy_minmax' in algorithms:
        plt.plot(nnodes_list, (greedy_minmax_results/verification_optimal).mean(axis=1), color=POLICY2COLOR['greedy_minmax'], label='Greedy Minmax')
    if 'greedy_entropy' in algorithms:
        plt.plot(nnodes_list, (greedy_minmax_results/verification_optimal).mean(axis=1), color=POLICY2COLOR['greedy_entropy'], label='Greedy Entropy')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Competitive Ratio')
    plt.legend()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'avgregret_sampler={sampler},nnodes_list={nnodes_list},{other_params_str}.png'))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'avgregret'))

    plt.clf()
    if 'random' in algorithms:
        plt.plot(nnodes_list, (random_results/verification_optimal).max(axis=1), color=POLICY2COLOR['random'], label='ND-Random')
    if 'dct' in algorithms:
        plt.plot(nnodes_list, (dct_results/verification_optimal).max(axis=1), color=POLICY2COLOR['dct'], label='DCT')
    if 'opt_single' in algorithms:
        plt.plot(nnodes_list, (opt_single_results/verification_optimal).max(axis=1), color=POLICY2COLOR['opt_single'], label='OptSingle')
    if 'coloring' in algorithms:
        plt.plot(nnodes_list, (coloring_results/verification_optimal).max(axis=1), color=POLICY2COLOR['coloring'], label='Coloring')
    if 'greedy_minmax' in algorithms:
        plt.plot(nnodes_list, (greedy_minmax_results/verification_optimal).max(axis=1), color=POLICY2COLOR['greedy_minmax'], label='Greedy Minmax')
    if 'greedy_entropy' in algorithms:
        plt.plot(nnodes_list, (greedy_entropy_results/verification_optimal).max(axis=1), color=POLICY2COLOR['greedy_entropy'], label='Greedy Entropy')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Maximum Competitive Ratio')
    plt.legend()
    other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'maxregret_sampler={sampler},nnodes_list={nnodes_list},{other_params_str}.png'))
    plt.savefig(os.path.join(FIGURE_FOLDER, f'maxregret'))

    ipdb.set_trace()
