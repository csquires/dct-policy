import networkx as nx
from causaldag import DAG


def induced_forest(graph: nx.Graph, coloring: dict, color1, color2):
    """
    Return the forest induced by taking only nodes of `color1` and `color2` from `coloring`.
    """
    forest = nx.Graph()
    forest.add_nodes_from(graph.nodes)
    forest.add_edges_from({(i, j) for i, j in graph.edges if {coloring[i], coloring[j]} == {color1, color2}})
    return forest


def worst_case_subtree(tree: nx.Graph, root) -> set:
    if tree.number_of_nodes() == 1:
        return set()
    tree_ = tree.copy()
    tree_.remove_node(root)
    subtrees = nx.connected_components(tree_)
    return max(subtrees, key=lambda t: len(t))


def score_node(graph: nx.Graph, coloring: dict, v, verbose=False):
    colors = set(coloring.values())
    forests = [induced_forest(graph, coloring, coloring[v], color) for color in colors if color != coloring[v]]
    trees_containing_v = [forest.subgraph(nx.node_connected_component(forest, v)) for forest in forests]
    if verbose:
        print(f"Trees containing node {v} for each color: {[tree.edges for tree in trees_containing_v]}")
    wc_subtrees = [worst_case_subtree(tree, v) for tree in trees_containing_v]
    if verbose:
        print(f"Worst case subtrees for node {v}: {wc_subtrees}")
    wc_edges_learned = [
        tree.number_of_nodes() - len(wc_subtree)
        for tree, wc_subtree in zip(trees_containing_v, wc_subtrees)
    ]
    if verbose:
        print(f"Worst case number of edges learned for {v}: {wc_edges_learned}")
    return sum(wc_edges_learned)


def pick_coloring_policy_node(graph: nx.Graph, verbose=False):
    coloring = nx.greedy_color(graph)
    node_scores = {node: score_node(graph, coloring, node, verbose=verbose) for node in graph.nodes}
    return max(node_scores.keys(), key=lambda k: node_scores[k])


def coloring_policy(dag: DAG, verbose: bool = False) -> set:
    intervened_nodes = set()

    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        undirected_portions = current_cpdag.copy()
        undirected_portions.remove_all_arcs()
        undirected_portions = undirected_portions.to_nx()

        if verbose: print('=========')
        node = pick_coloring_policy_node(undirected_portions, verbose=verbose)
        if verbose: print(f'Picked {node}')
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes
