import networkx as nx
import itertools as itr
import random
from mixed_graph import LabelledMixedGraph
from causaldag import UndirectedGraph, DAG


def get_directed_clique_graph(dag: DAG) -> LabelledMixedGraph:
    ug = UndirectedGraph(nodes=dag._nodes, edges=dag._arcs)
    ug.add_edges_from(dag.arcs)
    clique_graph = get_clique_graph(ug)
    directed_edges = dict()
    bidirected_edges = dict()
    for (c1, c2), label in clique_graph.undirected.items():
        arrowhead_c1 = all(dag.has_arc(c12_, c1_) for c12_, c1_ in itr.product(c1 & c2, c1 - c2))
        if not arrowhead_c1:
            directed_edges[(c1, c2)] = label
        else:
            arrowhead_c2 = all(dag.has_arc(c12_, c2_) for c12_, c2_ in itr.product(c1 & c2, c2 - c1))
            if arrowhead_c2:
                bidirected_edges[frozenset({c1, c2})] = label
            else:
                directed_edges[(c2, c1)] = label
    return LabelledMixedGraph(directed=directed_edges, bidirected=bidirected_edges)


def get_clique_graph(ug: UndirectedGraph) -> LabelledMixedGraph:
    if not isinstance(ug, UndirectedGraph):
        raise ValueError("Not an UndirectedGraph")
    clique_tree = get_clique_tree(ug)
    ug = ug.to_nx()
    cliques = nx.chordal_graph_cliques(ug)

    # === C1-C2 is in the clique graph iff c1&c2 is the label of some edge on the path between them
    paths = dict(nx.all_pairs_shortest_path(clique_tree.to_nx()))
    clique_path_edges = {
        frozenset({c1, c2}): {paths[c1][c2][i] & paths[c1][c2][i+1] for i in range(len(paths[c1][c2])-1)}
        for c1, c2 in itr.combinations(cliques, 2)
    }
    clique_graph = nx.Graph()
    clique_graph.add_edges_from([
        (c1, c2, dict(label=c1 & c2)) for c1, c2 in itr.combinations(cliques, 2)
        if c1 & c2 in clique_path_edges[frozenset({c1, c2})]
    ])
    return LabelledMixedGraph.from_nx(clique_graph)


def get_clique_tree(ug: UndirectedGraph) -> LabelledMixedGraph:
    if not isinstance(ug, UndirectedGraph):
        raise ValueError("Not an UndirectedGraph")
    cliques = nx.chordal_graph_cliques(ug.to_nx())
    clique_intersection_graph = nx.Graph()
    clique_intersection_graph.add_nodes_from(cliques)
    clique_intersection_graph.add_edges_from(
        [(c1, c2, dict(weight=len(c1 & c2))) for c1, c2 in itr.combinations(cliques, 2) if c1 & c2]
    )
    clique_tree = nx.maximum_spanning_tree(clique_intersection_graph)
    nx.set_edge_attributes(clique_tree, {(c1, c2): c1 & c2 for c1, c2 in clique_tree.edges()}, name='label')
    return LabelledMixedGraph.from_nx(clique_tree)


def edge_neighbors(graph) -> set:
    e_nbrs = set()
    for node in graph.nodes():
        incident_edges = [frozenset({node, nbr}) for nbr in graph.neighbors(node)]
        e_nbrs.update({frozenset({edge1, edge2}) for edge1, edge2 in itr.combinations(incident_edges, 2)})
    return e_nbrs


def get_induced_chordal(clique_tree):
    induced_graph = clique_tree.copy()

    checked_edge_neighbors = set()
    unchecked_edge_neighbors = edge_neighbors(induced_graph)
    while unchecked_edge_neighbors:
        for edge1, edge2 in unchecked_edge_neighbors:
            label1 = induced_graph.get_edge_data(*edge1)['label']
            label2 = induced_graph.get_edge_data(*edge2)['label']
            shared_node = edge1 & edge2
            diff_node1 = list(edge1 - shared_node)[0]
            diff_node2 = list(edge2 - shared_node)[0]
            if label1 <= label2:
                # print('adding', diff_node1, diff_node2)
                induced_graph.add_edge(diff_node1, diff_node2, label=label1)
            elif label2 < label1:
                # print('adding', diff_node1, diff_node2)
                induced_graph.add_edge(diff_node1, diff_node2, label=label2)
        checked_edge_neighbors.update(unchecked_edge_neighbors)
        unchecked_edge_neighbors = edge_neighbors(induced_graph) - checked_edge_neighbors
    return induced_graph


def get_tree_centroid(tree: UndirectedGraph, verbose=True):
    tree = tree.to_nx()
    nnodes = tree.number_of_nodes()

    candidate_nodes = list(tree.nodes())
    while True:
        v = random.choice(candidate_nodes)
        if verbose: print(f"Finding candidate centroid: picked {v}")
        forest = tree.copy()
        forest.remove_node(v)
        subtrees = [list(s) for s in nx.connected_components(forest)]
        max_subtree = max(subtrees, key=lambda s: len(s))

        if len(max_subtree) <= nnodes/2:
            return v
        candidate_nodes = max_subtree


if __name__ == '__main__':
    g = nx.Graph()
    g.add_edges_from(set(itr.combinations(range(5), 2)))
    t1 = get_clique_tree(g)

    g = nx.Graph()
    g.add_edges_from({(1, 2), (2, 3), (2, 4), (3, 4), (4, 5), (2, 5)})
    t2 = get_clique_tree(g)
    e2 = edge_neighbors(t2)
    c2 = get_induced_chordal(t2)
    # centroid = get_tree_centroid(t2)

    g = nx.balanced_tree(3, 3)
    centroid = get_tree_centroid(g)
    forest = g.copy()
    forest.remove_node(centroid)
    print(g.number_of_nodes())
    print(max(len(s) for s in nx.connected_components(forest)))

    g = nx.Graph()
    cs = [[1, 2], [2, 3, 4], [2, 4, 5]]
    g.add_edges_from(set(itr.chain.from_iterable((itr.combinations(c, 2)) for c in cs)))
    ct = get_clique_tree(g)
    cg = get_clique_graph(g)

