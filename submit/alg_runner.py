import os
from submit.dag_loader import DagLoader
from intervention_policy import dct_policy
from submit.baselines import random_policy, max_degree_policy
from utils import write_list
import numpy as np
from tqdm import tqdm
import random


ALG_DICT = {
    'dct': dct_policy,
    'random': random_policy,
    'max_degree': max_degree_policy
}


class AlgRunner:
    def __init__(self, alg: str, dag_loader: DagLoader):
        self.alg = alg
        self.dag_loader = dag_loader

    @property
    def alg_folder(self):
        return os.path.join(self.dag_loader.dag_folder, 'results', f'alg={self.alg}')

    def get_alg_results(self, overwrite=False, validate=True):
        random.seed(9859787)
        result_filename = os.path.join(self.alg_folder, f'num_nodes_list.npy')
        if overwrite or not os.path.exists(self.alg_folder):
            dags = self.dag_loader.get_dags()
            os.makedirs(self.alg_folder, exist_ok=True)
            num_nodes_list = []
            for ix, dag in tqdm(enumerate(dags), total=len(dags)):
                intervened_nodes = ALG_DICT[self.alg](dag)
                if validate:
                    cpdag = dag.interventional_cpdag([{node} for node in intervened_nodes], cpdag=dag.cpdag())
                    if cpdag.num_edges > 0:
                        print(f"{ix}, alg={self.alg}, num intervened = {len(intervened_nodes)}, num edges={cpdag.num_edges}")
                write_list(intervened_nodes, os.path.join(self.alg_folder, f'nodes{ix}.txt'))
                num_nodes_list.append(len(intervened_nodes))
            np.save(result_filename, np.array(num_nodes_list))
            return np.array(num_nodes_list)
        else:
            return np.load(result_filename)

    def specific_dag(self, ix):
        dag = self.dag_loader.get_dags()[ix]
        intervened_nodes = ALG_DICT[self.alg](dag, verbose=False)
        print(intervened_nodes)
        cpdag = dag.interventional_cpdag([{node} for node in intervened_nodes], cpdag=dag.cpdag())
        cpdag.to_complete_pdag()
        print(cpdag.edges)


if __name__ == '__main__':
    from mixed_graph import LabelledMixedGraph

    nnodes = 100
    dl = DagLoader(nnodes, 4, 100)
    dl.get_dags(overwrite=True)
    ar_random = AlgRunner('random', dl)
    ar_dct = AlgRunner('dct', dl)
    results_random = ar_random.get_alg_results(overwrite=True)
    results_dct = ar_dct.get_alg_results(overwrite=True)
    clique_sizes = dl.max_clique_sizes()
    num_cliques = dl.num_cliques()
    optimal_ivs = dl.get_verification_optimal_ivs()
    bound = np.ceil(np.log2(num_cliques)) * clique_sizes + 3*optimal_ivs
    print(bound)
    print(np.where(bound < nnodes))
    above_bound = results_dct > bound
    print(np.where(above_bound))
    print(np.mean(results_random))
    print(np.mean(results_dct))

    ix = 81
    ar_dct.specific_dag(ix)
    ar_random.specific_dag(ix)
    d = dl.get_dags()[ix]
    dct = d.directed_clique_tree()
    dct_ = LabelledMixedGraph.from_nx(dct)
