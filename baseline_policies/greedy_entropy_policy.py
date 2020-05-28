from causaldag import DAG, PDAG
from utils import random_max
from scipy.stats import entropy


def pick_greedy_entropy(cpdag: PDAG, verbose=False):
    eligible_nodes = cpdag.nodes - cpdag.dominated_nodes
    node2entropy = dict()
    for node in eligible_nodes:
        partition_sizes = []
        for parents in cpdag.possible_parents(node):
            cpdag_ = cpdag.copy()
            cpdag_.assign_parents(node, parents)
            partition_sizes.append(cpdag_.mec_size())
        node2entropy[node] = entropy(partition_sizes)
    if verbose: print(node2entropy)
    return random_max(node2entropy)


def greedy_entropy_policy(dag: DAG, verbose=False) -> set:
    """
    He and Geng 2008 second policy: greedily pick nodes based on maximizing entropy of the distribution over I-MECs.
    """
    intervened_nodes = set()
    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        node = pick_greedy_entropy(current_cpdag, verbose=verbose)
        if verbose: print(f"Picked {node}")
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes


if __name__ == '__main__':
    # dag = hairball_plus(nnodes=10, degree=4, e_min=2, e_max=5, ngraphs=1)
    # dag = DAG.from_nx(dag)
    dag = DAG(arcs={(1, 2), (3, 2), (1, 3), (3, 4), (2, 4), (4, 5)})
    a = greedy_entropy_policy(dag, verbose=True)
