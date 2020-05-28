from causaldag import DAG, PDAG
from utils import random_max


def pick_greedy_minmax_size(cpdag: PDAG, verbose=False):
    eligible_nodes = cpdag.nodes - cpdag.dominated_nodes
    node2max_size = {node: -float('inf') for node in eligible_nodes}
    for node in eligible_nodes:
        for parents in cpdag.possible_parents(node):
            cpdag_ = cpdag.copy()
            cpdag_.assign_parents(node, parents)
            mec_size = cpdag_.mec_size()
            node2max_size[node] = max(mec_size, node2max_size[node])
    if verbose: print(node2max_size)
    return random_max(node2max_size, minimize=True)


def greedy_minmax_policy(dag: DAG, verbose=False) -> set:
    """
    He and Geng 2008 first policy: greedily pick nodes based on a minmax criterion for the size of the Markov
    equivalence class after intervention.
    """
    intervened_nodes = set()
    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        node = pick_greedy_minmax_size(current_cpdag, verbose=verbose)
        if verbose: print(f"Picked {node}")
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes
