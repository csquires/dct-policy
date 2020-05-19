from causaldag import DAG, PDAG
import random
from utils import random_max
import numpy as np
import itertools as itr
import networkx as nx
from utils import powerset_predicate


def is_clique(cpdag, s):
    """
    Check if every pair of nodes in s is adjacent in `cpdag`.
    """
    return all(cpdag.has_edge(i, j) for i, j in itr.combinations(s, 2))


def pick_opt_single(cpdag: PDAG):
    """
    Pick the single-node intervention in `cpdag` which orients the maximum number of edges in the worst case.
    """
    minmax = float('inf')
    minmax_node = None
    for node in cpdag.nodes - cpdag.dominated_nodes:
        max_edges_left = -float('inf')
        for p in powerset_predicate(cpdag.undirected_neighbors_of(node), lambda c: is_clique(cpdag, c)):
            cpdag_ = cpdag.copy()
            cpdag_.assign_parents(node, p)
            max_edges_left = max(max_edges_left, cpdag_.num_edges)

        if max_edges_left < minmax:
            minmax = max_edges_left
            minmax_node = node

    return minmax_node


def opt_single_policy(dag: DAG) -> set:
    """
    Until the DAG is oriented, pick interventions according to Equation (1) of
    Hauser and Buhlmann 2012: "Two Optimal Strategies for Active Learning of Causal Models
    from Interventional Data"
    """
    intervened_nodes = set()
    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        node = pick_opt_single(current_cpdag)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes


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


def maximum_cardinality_search(graph: nx.Graph) -> list:
    """
    Deprecated: use networkx greedy coloring
    """
    graph = graph.copy()
    weights = {v: 0 for v in graph.nodes}
    order = []
    for i in range(graph.number_of_nodes()):
        max_weight = max(weights.values())
        max_weight_nodes = [v for v, w in weights.items() if w == max_weight]
        selected_node = random.choice(max_weight_nodes)
        print(len(weights), graph.number_of_nodes())
        print(graph.nodes)
        for nbr in graph.neighbors(selected_node):
            weights[nbr] += 1
        del weights[selected_node]
        graph.remove_node(selected_node)
        order.append(selected_node)
    return order


def create_greedy_coloring(graph: nx.Graph) -> dict:
    """
    DEPRECATED: use networkx greedy coloring.
    """
    order = reversed(maximum_cardinality_search(graph))
    assigned_colors = dict()
    for node in order:
        nbr_colors = {assigned_colors[nbr] for nbr in set(graph.neighbors(node)) & set(assigned_colors.keys())}
        assigned_colors[node] = next(color for color in itr.count() if color not in nbr_colors)
    return assigned_colors


def induced_forest(graph: nx.Graph, coloring: dict, color1, color2):
    """
    Return the forest induced by taking only nodes of `color1` and `color2` from `coloring`.

    Parameters
    ----------
    graph
    coloring
    color1
    color2

    Returns
    -------

    """
    forest = nx.Graph()
    forest.add_nodes_from(graph.nodes)
    forest.add_edges_from({(i, j) for i, j in graph.edges if {coloring[i], coloring[j]} == {color1, color2}})
    return forest


def worst_case_subtree(tree: nx.Graph, root):
    tree_ = tree.copy()
    tree_.remove_node(root)
    subtrees = nx.connected_components(tree)
    return max(subtrees, key=lambda t: len(t))


def score_node(graph: nx.Graph, coloring: dict, v):
    colors = set(coloring.values())
    forests = [induced_forest(graph, coloring, coloring[v], color) for color in colors if color != coloring[v]]
    trees_containing_v = [forest.subgraph(nx.node_connected_component(forest, v)) for forest in forests]
    wc_subtrees = [worst_case_subtree(tree, v) for tree in trees_containing_v]
    wc_edges_learned = [
        tree.number_of_nodes() - len(wc_subtree)
        for tree, wc_subtree in zip(trees_containing_v, wc_subtrees)
    ]
    return sum(wc_edges_learned)


def pick_coloring_policy_node(graph: nx.Graph):
    coloring = nx.greedy_color(graph)
    node_scores = {node: score_node(graph, coloring, node) for node in graph.nodes}
    return max(node_scores.keys(), key=lambda k: node_scores[k])


def coloring_policy(dag: DAG, verbose: bool = False) -> set:
    intervened_nodes = set()

    current_cpdag = dag.cpdag()
    while current_cpdag.num_arcs != dag.num_arcs:
        undirected_portions = current_cpdag.copy()
        undirected_portions.remove_all_arcs()
        undirected_portions = undirected_portions.to_nx()

        node = pick_coloring_policy_node(undirected_portions)
        intervened_nodes.add(node)
        current_cpdag = current_cpdag.interventional_cpdag(dag, {node})
    return intervened_nodes


if __name__ == '__main__':
    import causaldag as cd

    # dags = cd.rand.directed_erdos(20, .8, 100)
    # random_list = [random_policy(d, verbose=False) for d in dags]
    # random_lens = [len(nodes) for nodes in random_list]
    # md_list = [max_degree_policy(d, verbose=True) for d in dags]
    # md_lens = [len(nodes) for nodes in md_list]
    #
    # print(np.mean(random_lens))
    # print(np.mean(md_lens))

    from random_graphs import hairball_plus
    from tqdm import tqdm

    ngraphs = 5
    dags = hairball_plus(nnodes=40, degree=4, e_min=2, e_max=5, ngraphs=ngraphs)
    dags = [DAG.from_nx(d) for d in dags]
    # gs = [d.to_undirected() for d in dags]
    a = list(tqdm((coloring_policy(dag) for dag in dags), total=ngraphs))


