from directed_chordal_utils import random_chordal_graph2
import numpy as np
from causaldag import DAG
import random
from tqdm import tqdm

random.seed(9834285)
ngraphs = 100
dags = [DAG.from_nx(d) for d in random_chordal_graph2(20, 3, ngraphs)]
old_vo = list(tqdm((len(d.optimal_fully_orienting_interventions(new=False)) for d in dags), total=ngraphs))
old_vo = np.array(old_vo)
new_vo = list(tqdm((len(d.optimal_fully_orienting_interventions(new=True)) for d in dags), total=ngraphs))
new_vo = np.array(new_vo)
match = old_vo == new_vo
print(all(match))
