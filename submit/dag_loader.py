import os
from submit.config import DATA_FOLDER
from directed_chordal_utils import random_chordal_graph2
import numpy as np
from causaldag import DAG
from utils import write_list, read_list
import networkx as nx
from tqdm import tqdm


class DagLoader:
    def __init__(self, nnodes: int, density: int, num_dags: int):
        self.nnodes = nnodes
        self.density = density
        self.num_dags = num_dags

    @property
    def dag_folder(self):
        return os.path.join(DATA_FOLDER, f'nnodes={self.nnodes},density={self.density},num_dags={self.num_dags}')

    @property
    def dag_filenames(self):
        return [os.path.join(self.dag_folder, 'dags', f'dag{i}.npy') for i in range(self.num_dags)]

    def get_dags(self, overwrite=False):
        if overwrite or not os.path.exists(self.dag_folder):
            dags = [DAG.from_nx(d) for d in random_chordal_graph2(self.nnodes, self.density, self.num_dags)]
            if any(len(d.vstructures()) > 0 for d in dags):
                print([len(d.vstructures()) for d in dags])
                raise ValueError("DAG has v-structures")
            os.makedirs(os.path.join(self.dag_folder, 'dags'), exist_ok=True)
            for dag, filename in zip(dags, self.dag_filenames):
                np.save(filename, dag.to_amat()[0])
        else:
            dags = [DAG.from_amat(np.load(filename)) for filename in self.dag_filenames]
        return dags

    def get_verification_optimal_ivs(self):
        filename = os.path.join(self.dag_folder, 'optimal_num_interventions.txt')
        if not os.path.exists(filename):
            optimal_ivs = list(tqdm(
                (len(dag.optimal_fully_orienting_interventions()) for dag in self.get_dags()),
                total=self.num_dags
            ))
            write_list(optimal_ivs, filename)
        else:
            optimal_ivs = read_list(filename)
        return optimal_ivs

    def max_clique_sizes(self):
        clique_numbers = np.array([
            max(map(len, nx.chordal_graph_cliques(dag.to_nx().to_undirected())))
            for dag in self.get_dags()
        ])
        return clique_numbers

    def num_cliques(self):
        return np.array([len(nx.chordal_graph_cliques(dag.to_nx().to_undirected())) for dag in self.get_dags()])


if __name__ == '__main__':
    dl = DagLoader(10, 2, 10)
    # dl.get_dags(overwrite=True)
    ds = dl.get_dags()




