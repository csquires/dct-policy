import causaldag as cd
from baseline_policies import opt_single_policy
from random_graphs import hairball_plus
import random
from line_profiler import LineProfiler

lp = LineProfiler()

random.seed(633)

# dags = [cd.DAG.from_nx(d) for d in tree_plus(100, 2, 5, 2)]
dags = [cd.DAG.from_nx(d) for d in hairball_plus(4, 2, 5, 2, nnodes=100)]


def run_policy():
    for dag in dags:
        opt_single_policy(dag)


lp.add_function(cd.PDAG.to_complete_pdag)
lp.runcall(run_policy)
lp.print_stats()

