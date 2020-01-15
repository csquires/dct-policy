import os
from submit.dag_loader import DagLoader
from intervention_policy import dct_policy
from submit.baselines import random_policy, max_degree_policy
from utils import write_list
import numpy as np


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

    def get_alg_results(self, overwrite=False):
        result_filename = os.path.join(self.alg_folder, f'num_nodes_list.npy')
        if overwrite or not os.path.exists(self.alg_folder):
            print('here')
            dags = self.dag_loader.get_dags()
            os.makedirs(self.alg_folder, exist_ok=True)
            num_nodes_list = []
            for ix, dag in enumerate(dags):
                intervened_nodes = ALG_DICT[self.alg](dag)
                write_list(intervened_nodes, os.path.join(self.alg_folder, f'nodes{ix}.txt'))
                num_nodes_list.append(len(intervened_nodes))
            np.save(result_filename, np.array(num_nodes_list))
            return np.array(num_nodes_list)
        else:
            print(os.path.exists(self.alg_folder))
            return np.load(result_filename)


if __name__ == '__main__':
    dl = DagLoader(5, 2, 10)
    ar_random = AlgRunner('random', dl)
    ar_dct = AlgRunner('dct', dl)
    # results_random = ar_random.get_alg_results()
    results_dct = ar_dct.get_alg_results(overwrite=True)

