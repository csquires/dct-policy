import networkx as nx
import itertools as itr
import random
import operator as op
from typing import Union, List
from causaldag import DAG


def random_directed_tree(nnodes: int):
    g = nx.random_tree(nnodes)
    root = random.randint(0, nnodes-1)
    d = nx.DiGraph()

    queue = [root]
    while queue:
        current_node = queue.pop()
        nbrs = list(g.neighbors(current_node))
        d.add_edges_from([(current_node, nbr) for nbr in nbrs])
        queue += nbrs
        g.remove_node(current_node)
    return d


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


def random_chordal_graph(nnodes, p=.1, ngraphs=1):
    if ngraphs == 1:
        g = nx.erdos_renyi_graph(nnodes, p)
        perm = random.sample(set(range(nnodes)), nnodes)
        for node in perm:
            nbrs = set(g.neighbors(node)) | {node}
            g.add_edges_from((i, j) for i, j in itr.combinations(nbrs, 2))

        d = nx.DiGraph()
        for node in perm:
            d.add_edges_from([(node, nbr) for nbr in g.neighbors(node)])
        return g
    else:
        return [random_chordal_graph(nnodes, p=p) for _ in range(ngraphs)]


def tree_plus(nnodes: int, e_min: int, e_max: int, ngraphs: int = 1):
    if ngraphs == 1:
        g = random_directed_tree(nnodes)
        order = list(nx.topological_sort(g))
        num_extra_edges = random.randint(e_min, e_max)
        extra_edges = random.sample(list(itr.combinations(order, 2)), num_extra_edges)
        g.add_edges_from(extra_edges)
        fill_vstructures(g, order)
        return g
    else:
        return [tree_plus(nnodes, e_min, e_max) for _ in range(ngraphs)]


def random_chordal_graph2(nnodes: int, k: int, ngraphs: int = 1, ensure_connected=True) -> Union[List[nx.DiGraph], nx.DiGraph]:
    if ngraphs == 1:
        for ix in itr.count():
            if ix > 100:
                raise ValueError("100 iterations without a connected graph, please change parameters")
            t = nx.random_tree(nnodes)

            subtrees = []
            for i in range(nnodes):
                x = random.randint(0, nnodes-1)
                t_i = nx.Graph()
                t_i.add_node(x)
                t_i_nbrs = {x: set(t.neighbors(x))}
                k_i = random.randint(1, 2*k-1)
                for j in range(k_i-1):
                    y = random.sample(t_i_nbrs.keys(), 1)[0]
                    z = random.sample(t_i_nbrs[y], 1)[0]
                    t_i.add_edge(y, z)
                    t_i_nbrs[y] -= {z}
                    t_i_nbrs[z] = set(t.neighbors(z)) - {y}
                    if not t_i_nbrs[y]:
                        del t_i_nbrs[y]
                    if not t_i_nbrs[z]:
                        del t_i_nbrs[z]
                subtrees.append(t_i)

            g = nx.Graph()
            g.add_nodes_from(range(nnodes))
            for (i, t_i), (j, t_j) in itr.combinations(enumerate(subtrees), 2):
                if t_i.nodes & t_j.nodes: g.add_edge(i, j)

            if not ensure_connected or nx.is_connected(g):
                return direct_chordal_graph(g)
    else:
        return [random_chordal_graph2(nnodes, k) for _ in range(ngraphs)]


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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from causaldag import DAG

    # g = nx.DiGraph()
    # g.add_edges_from([
    #     (5, 1),
    #     (5, 4),
    #     (2, 1),
    #     (2, 5),
    #     (2, 3),
    #     (3, 4),
    #     (3, 5)
    # ])
    # d = get_directed_clique_graph(g)
    #
    # ds = random_chordal_graph(10, ngraphs=100)
    # dcgs = [get_directed_clique_graph(d) for d in ds]
    # acyclic_list = [nx.is_directed_acyclic_graph(dcg) for dcg in dcgs]
    # print(all(acyclic_list))
    # plt.clf()
    # plt.ion()
    # draw_directed_clique_graph(d)

    # gs = random_chordal_graph2(10, 3, 50)
    # are_chordal = [nx.is_chordal(g) for g in gs]
    # print(all(are_chordal))
    # print([len(g.edges) for g in gs])

    # d = DAG.from_nx(random_chordal_graph2(50, 5))
    # dct = d.directed_clique_tree()
    # sdct = d.simplified_directed_clique_tree()
    # print(dct.number_of_nodes())
    # print(sdct.number_of_nodes())
    # print(dct.nodes())
    # print(sdct.nodes())
    #
    # from mixed_graph import LabelledMixedGraph
    #
    # dct_ = LabelledMixedGraph.from_nx(dct)
    # print({c for c in sdct if len(c) > 1})
    # print(dct_.bidirected)

    dags = tree_plus(20, 2, 5, 20)
