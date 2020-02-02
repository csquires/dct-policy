from causaldag import DAG, PDAG
import random
from utils import random_max
import numpy as np
import itertools as itr
from utils import powerset_monotone


def is_clique(cpdag, s):
    return all(cpdag.has_edge(i, j) for i, j in itr.combinations(s, 2))


def pick_opt_single(cpdag: PDAG):
    minmax = float('inf')
    minmax_node = None
    for node in cpdag.nodes - cpdag.dominated_nodes:
        max_edges_left = -float('inf')
        for p in powerset_monotone(cpdag.undirected_neighbors_of(node), lambda c: is_clique(cpdag, c)):
            cpdag_ = cpdag.copy()
            cpdag_.assign_parents(node, p)
            max_edges_left = max(max_edges_left, cpdag_.num_edges)

        if max_edges_left < minmax:
            minmax = max_edges_left
            minmax_node = node

    return minmax_node


def opt_single_policy(dag: DAG) -> set:
    intervened_nodes = set()
    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        node = pick_opt_single(current_cpdag)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes


def random_policy(dag: DAG, verbose: bool = False) -> set:
    intervened_nodes = set()

    current_cpdag = dag.cpdag()

    while current_cpdag.num_arcs != dag.num_arcs:
        if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")
        eligible_nodes = list(current_cpdag.nodes - current_cpdag.dominated_nodes)
        node = random.choice(eligible_nodes)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})

    return intervened_nodes


def max_degree_policy(dag: DAG, verbose: bool = False) -> set:
    intervened_nodes = set()

    nodes2degrees = dag.nodes
    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")
        nodes2degrees = {node: current_cpdag.undirected_degree_of(node) for node in nodes2degrees}
        node = random_max(nodes2degrees)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})

    return intervened_nodes


if __name__ == '__main__':
    import causaldag as cd

    dags = cd.rand.directed_erdos(20, .8, 100)
    random_list = [random_policy(d, verbose=False) for d in dags]
    random_lens = [len(nodes) for nodes in random_list]
    md_list = [max_degree_policy(d, verbose=True) for d in dags]
    md_lens = [len(nodes) for nodes in md_list]

    print(np.mean(random_lens))
    print(np.mean(md_lens))
