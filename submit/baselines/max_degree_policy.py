from causaldag import DAG
import random


def max_degree_policy(dag: DAG, verbose: bool = False) -> set:
    """
    Until the DAG is oriented, pick interventions according to the node with the greatest
    number of incident undirected edges.
    """
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

