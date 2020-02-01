import networkx as nx
import itertools as itr
import random
import operator as op


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


def get_directed_clique_graph(g, partial_order=True):
    cliques = nx.chordal_graph_cliques(g.to_undirected())
    dcg = nx.DiGraph()
    dcg.add_nodes_from(cliques)

    for c1, c2 in itr.combinations(cliques, 2):
        s = c1 & c2
        if s:
            c1_to_s = any(g.has_edge(i, j) for i, j in itr.product(c1 - s, s))
            c2_to_s = any(g.has_edge(i, j) for i, j in itr.product(c2 - s, s))

            if partial_order:
                if c1_to_s and not c2_to_s: dcg.add_edge(c1, c2)
                if c2_to_s and not c1_to_s: dcg.add_edge(c2, c1)
            else:
                if c1_to_s: dcg.add_edge(c1, c2)
                if c2_to_s: dcg.add_edge(c2, c1)
                if not c1_to_s and not c2_to_s: dcg.add_edge(c1, c2, bidirected=True)

    return dcg


def draw_directed_clique_graph(dcg):
    pos = nx.spectral_layout(dcg)
    nx.draw_networkx_nodes(dcg, pos)
    nx.draw_networkx_edges(dcg, pos)
    nx.draw_networkx_labels(dcg, pos, labels={nums: ','.join(map(str, nums)) for nums in dcg.nodes})

