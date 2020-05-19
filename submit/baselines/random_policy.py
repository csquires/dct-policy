from causaldag import DAG
import random


def random_policy(dag: DAG, verbose: bool = False) -> set:
    """
    Until the DAG is oriented, pick interventions randomly amongst non-dominated nodes.
    """
    intervened_nodes = set()

    current_cpdag = dag.cpdag()

    while current_cpdag.num_arcs != dag.num_arcs:
        if verbose: print(f"Remaining edges: {current_cpdag.num_edges}")
        eligible_nodes = list(current_cpdag.nodes - current_cpdag.dominated_nodes)
        node = random.choice(eligible_nodes)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})

    return intervened_nodes
