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

    def get_dags(self, overwrite=False):
        folder = os.path.join(DATA_FOLDER, f'nnodes={self.nnodes},density={self.density},num_dags={self.num_dags}')
        if overwrite or not os.path.exists(folder):
            dags = random_chordal_graph2(self.nnodes, self.density, self.num_dags)
            filenames = [os.path.join(folder, f'dag{i}.npy') for i in range(self.num_dags)]
            os.makedirs(folder, exist_ok=True)
            for dag, filename in zip(dags, filenames):
                np.save(filename, DAG.from_nx(dag).to_amat()[0])
        else:
            dags = [DAG.from_amat(np.load(os.path.join(folder, f'dag{i}.npy'))) for i in range(self.num_dags)]
        return dags


if __name__ == '__main__':
    dl = DagLoader(10, 2, 10)
    # dl.get_dags(overwrite=True)
    ds = dl.get_dags()




