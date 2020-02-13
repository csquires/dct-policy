from submit.alg_runner import AlgRunner
from submit.dag_loader import DagLoader, DagSampler
import itertools as itr
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from submit.config import FIGURE_FOLDER
import os
import networkx as nx
from p_tqdm import p_map
sns.set()

ngraphs = 100
nnodes_list = [100, 150, 200, 250, 300]
RANDOM = True
DCT = True
OPT_SINGLE = False
sampler = DagSampler.TREE_OF_CLIQUES
other_params = dict(min_clique_size=5, max_clique_size=10, degree=10)

random_results = np.zeros([len(nnodes_list), ngraphs])
dct_results = np.zeros([len(nnodes_list), ngraphs])
opt_single_results = np.zeros([len(nnodes_list), ngraphs])
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
    if RANDOM:
        ar = AlgRunner('random', dl)
        random_results[i] = ar.get_alg_results()
    if DCT:
        ar = AlgRunner('dct', dl)
        dct_results[i] = ar.get_alg_results(overwrite=True)
    if OPT_SINGLE:
        ar = AlgRunner('opt_single', dl)
        opt_single_results[i] = ar.get_alg_results()

print(num_arcs.mean(axis=1))
log_num_cliques = np.ceil(np.log2(num_cliques))
bounds = 3 * log_num_cliques + 2
plt.clf()
if RANDOM:
    plt.plot(nnodes_list, (random_results/verification_optimal).mean(axis=1), label='ND-Random')
if DCT:
    plt.plot(nnodes_list, (dct_results/verification_optimal).mean(axis=1), label='DCT')
if OPT_SINGLE:
    plt.plot(nnodes_list, (opt_single_results/verification_optimal).mean(axis=1), label='OptSingle')
plt.xlabel('Number of Nodes')
plt.ylabel('Average Competitive Ratio')
plt.legend()
other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
plt.savefig(os.path.join(FIGURE_FOLDER, f'avgregret_sampler={sampler},{other_params_str}.png'))
plt.savefig(os.path.join(FIGURE_FOLDER, f'avgregret'))


plt.clf()
if RANDOM:
    plt.plot(nnodes_list, (random_results/verification_optimal).max(axis=1), label='ND-Random')
if DCT:
    plt.plot(nnodes_list, (dct_results/verification_optimal).max(axis=1), label='DCT')
if OPT_SINGLE:
    plt.plot(nnodes_list, (opt_single_results/verification_optimal).max(axis=1), label='OptSingle')
plt.xlabel('Number of Nodes')
plt.ylabel('Maximum Competitive Ratio')
plt.legend()
other_params_str = ','.join((f"{k}={v}" for k, v in other_params.items()))
plt.savefig(os.path.join(FIGURE_FOLDER, f'maxregret_sampler={sampler},{other_params_str}.png'))
plt.savefig(os.path.join(FIGURE_FOLDER, f'maxregret'))
