import os
from submit.config import DATA_FOLDER
from random_graphs import random_chordal_graph2, tree_plus, hairball_plus, tree_of_cliques
import numpy as np
from causaldag import DAG
from utils import write_list, read_list
import networkx as nx
from tqdm import tqdm
from enum import Enum
from chordal_utils import get_directed_clique_graph
from mixed_graph import LabelledMixedGraph


class DagSampler(Enum):
    CHORDAL2 = 1
    TREE_PLUS = 2
    HAIRBALL_PLUS = 3
    TREE_OF_CLIQUES = 4


class DagLoader:
    def __init__(self, nnodes: int, num_dags: int, sampler: DagSampler, other_params: dict, comparable_edges=False):
        self.nnodes = nnodes
        self.other_params = other_params
        self.num_dags = num_dags
        self.sampler = sampler
        self.comparable_edges = comparable_edges

    @property
    def dag_folder(self):
        other_params = ','.join([f"{key}={value}" for key, value in self.other_params.items()])
        return os.path.join(DATA_FOLDER, f'sampler={self.sampler.name},nnodes={self.nnodes},num_dags={self.num_dags},{other_params}')

    @property
    def dag_filenames(self):
        return [os.path.join(self.dag_folder, 'dags', f'dag{i}.npy') for i in range(self.num_dags)]

    def get_dags(self, overwrite=False):
        if overwrite or not os.path.exists(self.dag_folder):
            dags = []
            counter = 0
            while len(dags) < self.num_dags:
                counter += 1
                if counter > 100:
                    raise RuntimeError('change parameters, not getting incomparable graphs')
                if self.sampler == DagSampler.CHORDAL2:
                    d = DAG.from_nx(random_chordal_graph2(self.nnodes, self.other_params['density']))
                elif self.sampler == DagSampler.TREE_PLUS:
                    d = DAG.from_nx(tree_plus(self.nnodes, self.other_params['e_min'], self.other_params['e_max']))
                elif self.sampler == DagSampler.HAIRBALL_PLUS:
                    d = DAG.from_nx(hairball_plus(
                        self.other_params['degree'],
                        self.other_params['e_min'],
                        self.other_params['e_max'],
                        num_layers=self.other_params.get('num_layers'),
                        nnodes=self.other_params.get('nnodes')
                    ))
                elif self.sampler == DagSampler.TREE_OF_CLIQUES:
                    d = DAG.from_nx(tree_of_cliques(
                        self.other_params['degree'],
                        self.other_params['min_clique_size'],
                        self.other_params['max_clique_size'],
                        nnodes=self.other_params.get('nnodes')
                    ))
                else:
                    raise ValueError
                if self.comparable_edges or get_directed_clique_graph(d) == LabelledMixedGraph.from_nx(d.directed_clique_tree()):
                    counter = 0
                    dags.append(d)
                    print(len(dags))
            if any(len(d.vstructures()) > 0 for d in dags):
                print([len(d.vstructures()) for d in dags])
                raise ValueError("DAG has v-structures")
            for d in dags:
                d_nx = d.to_nx().to_undirected()
                if not nx.is_chordal(d_nx):
                    raise RuntimeError
                if not nx.is_connected(d_nx):
                    raise RuntimeError
            os.makedirs(os.path.join(self.dag_folder, 'dags'), exist_ok=True)
            for dag, filename in zip(dags, self.dag_filenames):
                np.save(filename, dag.to_amat()[0])
        else:
            dags = [DAG.from_amat(np.load(filename)) for filename in self.dag_filenames]
        return dags

    def get_verification_optimal_ivs(self):
        filename = os.path.join(self.dag_folder, 'optimal_num_interventions.txt')
        if not os.path.exists(filename):
            optimal_ivs = np.array(list(tqdm(
                (len(dag.optimal_fully_orienting_interventions(new=True)) for dag in self.get_dags()),
                total=self.num_dags
            )))
            np.savetxt(filename, optimal_ivs)
        else:
            optimal_ivs = np.loadtxt(filename)
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




