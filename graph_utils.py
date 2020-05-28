import operator as op
import networkx as nx
import itertools as itr
import random
from mixed_graph import LabelledMixedGraph
from causaldag import UndirectedGraph, DAG


def get_directed_clique_graph(dag: DAG, clique_graph: LabelledMixedGraph=None) -> LabelledMixedGraph:
    """
    Return the *directed* clique graph of a chordal graph.
    """
    ug = UndirectedGraph(nodes=dag._nodes, edges=dag._arcs)
    ug.add_edges_from(dag.arcs)
    clique_graph = get_clique_graph(ug) if clique_graph is None else clique_graph
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


def get_clique_graph(ug: UndirectedGraph, cliques=None) -> LabelledMixedGraph:
    """
    Compute the *undirected* clique graph of a chordal graph, i.e., the union of all clique trees.
    """
    if not isinstance(ug, UndirectedGraph):
        raise ValueError("Not an UndirectedGraph")
    clique_tree = get_clique_tree(ug, cliques=cliques)
    ug = ug.to_nx()
    cliques = nx.chordal_graph_cliques(ug) if cliques is None else cliques

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


def get_clique_tree(ug: UndirectedGraph, cliques=None) -> LabelledMixedGraph:
    """
    Compute an *unidirected* clique tree for a chordal graph by finding a max-weight spanning tree of the clique intersection graph.
    """
    if not isinstance(ug, UndirectedGraph):
        raise ValueError("Not an UndirectedGraph")
    cliques = nx.chordal_graph_cliques(ug.to_nx()) if cliques is None else cliques
    clique_intersection_graph = nx.Graph()
    clique_intersection_graph.add_nodes_from(cliques)
    clique_intersection_graph.add_edges_from(
        [(c1, c2, dict(weight=len(c1 & c2))) for c1, c2 in itr.combinations(cliques, 2) if c1 & c2]
    )
    clique_tree = nx.maximum_spanning_tree(clique_intersection_graph)
    nx.set_edge_attributes(clique_tree, {(c1, c2): c1 & c2 for c1, c2 in clique_tree.edges()}, name='label')
    return LabelledMixedGraph.from_nx(clique_tree)


def edge_neighbors(graph) -> set:
    """
    Return all pairs of "edge neighbors" in a graph, i.e. edges that share a vertex.
    """
    e_nbrs = set()
    for node in graph.nodes():
        incident_edges = [frozenset({node, nbr}) for nbr in graph.neighbors(node)]
        e_nbrs.update({frozenset({edge1, edge2}) for edge1, edge2 in itr.combinations(incident_edges, 2)})
    return e_nbrs


# def get_induced_chordal(clique_tree):
#     induced_graph = clique_tree.copy()
#
#     checked_edge_neighbors = set()
#     unchecked_edge_neighbors = edge_neighbors(induced_graph)
#     while unchecked_edge_neighbors:
#         for edge1, edge2 in unchecked_edge_neighbors:
#             label1 = induced_graph.get_edge_data(*edge1)['label']
#             label2 = induced_graph.get_edge_data(*edge2)['label']
#             shared_node = edge1 & edge2
#             diff_node1 = list(edge1 - shared_node)[0]
#             diff_node2 = list(edge2 - shared_node)[0]
#             if label1 <= label2:
#                 # print('adding', diff_node1, diff_node2)
#                 induced_graph.add_edge(diff_node1, diff_node2, label=label1)
#             elif label2 < label1:
#                 # print('adding', diff_node1, diff_node2)
#                 induced_graph.add_edge(diff_node1, diff_node2, label=label2)
#         checked_edge_neighbors.update(unchecked_edge_neighbors)
#         unchecked_edge_neighbors = edge_neighbors(induced_graph) - checked_edge_neighbors
#     return induced_graph


def get_tree_centroid(tree: nx.Graph, verbose=False):
    """
    Find a centroid of a tree, i.e., a node whose removal splits the tree into a forest with no more than
    half of the nodes in any component.
    """
    # tree = tree.to_nx()
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
        if verbose: print(f"Candidate nodes: {candidate_nodes}")


def fill_vstructures(g: nx.DiGraph, order=None):
    if order is None:
        order = nx.topological_sort(g)
    node2ix = {node: ix for ix, node in enumerate(order)}

    for node in reversed(order):
        parents = g.predecessors(node)
        for p1, p2 in itr.combinations(parents, 2):
            p1_, p2_ = (p1, p2) if node2ix[p1] < node2ix[p2] else (p2, p1)
            g.add_edge(p1_, p2_)

    return g


def direct_chordal_graph(chordal_graph: nx.Graph):
    chordal_graph_ = chordal_graph.copy()
    nodes = chordal_graph.nodes()
    weights = {node: 0 for node in nodes}
    order = []
    while len(order) < chordal_graph.number_of_nodes():
        max_weight = max(weights.items(), key=op.itemgetter(1))[1]
        max_weight_nodes = [node for node, weight in weights.items() if weight == max_weight]
        next_node = random.choice(max_weight_nodes)

        # add 1 to weight of all neighbors
        for nbr in chordal_graph_.neighbors(next_node):
            weights[nbr] += 1

        # delete node and add to order
        del weights[next_node]
        chordal_graph_.remove_node(next_node)
        order.append(next_node)

    d = nx.DiGraph()
    d.add_nodes_from(nodes)
    for node, nbr in itr.combinations(order, 2):
        if chordal_graph.has_edge(node, nbr):
            d.add_edge(node, nbr)
    return d
