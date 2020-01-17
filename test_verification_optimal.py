from directed_chordal_utils import random_chordal_graph2
import numpy as np
from causaldag import DAG
import random

random.seed(9834285)
dags = [DAG.from_nx(d) for d in random_chordal_graph2(10, 3, 100)]
old_vo = np.array([len(d.optimal_fully_orienting_interventions(new=False)) for d in dags])
new_vo = np.array([len(d.optimal_fully_orienting_interventions(new=True)) for d in dags])

d = dags[0]
d.optimal_fully_orienting_interventions(new=True, verbose=True)
