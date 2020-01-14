import networkx as nx
import itertools as itr
import random


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


def random_chordal_graph2(nnodes: int, k: int, ngraphs: int=1):
    if ngraphs == 1:
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

        return g
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

    gs = random_chordal_graph2(10, 3, 50)
    are_chordal = [nx.is_chordal(g) for g in gs]
    print(all(are_chordal))
    print([len(g.edges) for g in gs])


