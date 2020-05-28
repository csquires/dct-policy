import os
from dag_loader import DagLoader, DagSampler
from dct_policy import dct_policy
from baseline_policies import random_policy, max_degree_policy, opt_single_policy, coloring_policy, greedy_minmax_policy, greedy_entropy_policy
import numpy as np
from tqdm import tqdm
from p_tqdm import p_map
from time import time
from multiprocessing import cpu_count
import random


ALG_DICT = {
    'dct': dct_policy,
    'random': random_policy,
    'max_degree': max_degree_policy,
    'opt_single': opt_single_policy,
    'coloring': coloring_policy,
    'greedy_minmax': greedy_minmax_policy,
    'greedy_entropy': greedy_entropy_policy
}


class AlgRunner:
    def __init__(self, alg: str, dag_loader: DagLoader):
        self.alg = alg
        self.dag_loader = dag_loader

    @property
    def alg_folder(self):
        return os.path.join(self.dag_loader.dag_folder, 'results', f'alg={self.alg}')

    def get_alg_results(self, overwrite=False, validate=True, multithread=True):
        random.seed(9859787)
        result_filename = os.path.join(self.alg_folder, f'num_nodes_list.npy')
        time_result_filename = os.path.join(self.alg_folder, f'times_list.npy')
        print(self.alg_folder)
        if overwrite or not os.path.exists(self.alg_folder):
            dags = self.dag_loader.get_dags(overwrite=overwrite)
            os.makedirs(self.alg_folder, exist_ok=True)

            def run_alg(tup):
                ix, dag = tup

                start = time()
                intervened_nodes = ALG_DICT[self.alg](dag)
                time_taken = time() - start

                if validate:
                    cpdag = dag.interventional_cpdag([{node} for node in intervened_nodes], cpdag=dag.cpdag())
                    if cpdag.num_edges > 0:
                        print(f"**************** BROKEN")
                        print(f"ix={ix}, alg={self.alg}, num intervened = {len(intervened_nodes)}, num edges={cpdag.num_edges}")
                        raise RuntimeError
                # write_list(intervened_nodes, os.path.join(self.alg_folder, f'nodes{ix}.txt'))
                return len(intervened_nodes), time_taken

            if multithread:
                print(f'[AlgRunner.get_alg_results] Running {self.alg} on {cpu_count()-1} cores')
                num_nodes_list, times_list = zip(*p_map(run_alg, list(enumerate(dags))))
            else:
                print(f'[AlgRunner.get_alg_results] Running {self.alg} on 1 core')
                num_nodes_list, times_list = zip(*list(tqdm((run_alg(dag) for dag in dags), total=len(dags))))

            np.save(result_filename, np.array(num_nodes_list))
            np.save(time_result_filename, np.array(times_list))
            return np.array(num_nodes_list), np.array(times_list)
        else:
            return np.load(result_filename), np.load(time_result_filename)

    def specific_dag(self, ix, verbose=False):
        dag = self.dag_loader.get_dags()[ix]
        intervened_nodes = ALG_DICT[self.alg](dag, verbose=verbose)
        print(intervened_nodes)
        cpdag = dag.interventional_cpdag([{node} for node in intervened_nodes], cpdag=dag.cpdag())
        cpdag.to_complete_pdag()
        print(cpdag.edges)


if __name__ == '__main__':
    import random

    # nnodes = 18
    nnodes = 100
    random.seed(8128)
    # dl = DagLoader(nnodes, 10, DagSampler.TREE_PLUS, dict(e_min=2, e_max=5), comparable_edges=True)
    dl = DagLoader(nnodes, 10, DagSampler.HAIRBALL_PLUS, dict(num_layers=5, degree=3, e_min=2, e_max=5), comparable_edges=True)
    dl.get_dags(overwrite=True)
    ar_random = AlgRunner('random', dl)
    ar_dct = AlgRunner('dct', dl)

    RUN_ALL = True
    if RUN_ALL:
        results_random = ar_random.get_alg_results(overwrite=True)
        results_dct = ar_dct.get_alg_results(overwrite=True)
        clique_sizes = dl.max_clique_sizes()
        num_cliques = dl.num_cliques()
        optimal_ivs = dl.get_verification_optimal_ivs()
        bound = np.ceil(np.log2(num_cliques)) * clique_sizes + 2*optimal_ivs

        print("Number of cliques")
        print(num_cliques)

        print("Clique sizes")
        print(clique_sizes)

        print("Verification optimal")
        print(optimal_ivs)

        print("Bound")
        print(bound)

        print(np.where(bound < nnodes))
        above_bound = results_dct > bound
        print(np.where(above_bound))
        print(np.mean(results_random))
        print(np.mean(results_dct))

    # ix = 27
    # d = dl.get_dags()[ix]
    # dct = d.directed_clique_tree()
    # dcg = get_directed_clique_graph(d)
    # dct_ = LabelledMixedGraph.from_nx(dct)
    # ar_dct.specific_dag(ix, verbose=True)
    # ar_random.specific_dag(ix)


