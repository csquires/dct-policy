import os
from submit.config import DATA_FOLDER
from directed_chordal_utils import random_chordal_graph2
import numpy as np
from causaldag import DAG


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
            os.makedirs(os.path.join(self.dag_folder, 'dags'), exist_ok=True)
            for dag, filename in zip(dags, self.dag_filenames):
                np.save(filename, dag.to_amat()[0])
        else:
            dags = [DAG.from_amat(np.load(filename)) for filename in self.dag_filenames]
        return dags


if __name__ == '__main__':
    dl = DagLoader(10, 2, 10)
    # dl.get_dags(overwrite=True)
    ds = dl.get_dags()




