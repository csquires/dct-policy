from causaldag import DAG, PDAG
from utils import random_max


def pick_opt_single(cpdag: PDAG):
    """
    Pick the single-node intervention in `cpdag` which orients the maximum number of edges in the worst case.
    """
    eligible_nodes = cpdag.nodes - cpdag.dominated_nodes
    node2max = {node: float('-inf') for node in eligible_nodes}
    for node in eligible_nodes:
        for p in cpdag.possible_parents(node):
            cpdag_ = cpdag.copy()
            cpdag_.assign_parents(node, p)
            node2max[node] = max(node2max[node], cpdag_.num_edges)

    return random_max(node2max, minimize=True)


def opt_single_policy(dag: DAG) -> set:
    """
    Until the DAG is oriented, pick interventions according to Equation (1) of
    Hauser and Buhlmann 2012: "Two Optimal Strategies for Active Learning of Causal Models
    from Interventional Data". This policy picks the node which orients the maximum number of edges in the worst
    case.
    """
    intervened_nodes = set()
    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        node = pick_opt_single(current_cpdag)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes
