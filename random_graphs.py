import networkx as nx
import itertools as itr
import random
from typing import Union, List
from directed_chordal_utils import direct_chordal_graph, fill_vstructures


def hairball(num_layers: int, degree: int):
    return nx.full_rary_tree(degree, sum((degree**i for i in range(num_layers))), create_using=nx.DiGraph)


def hairball_plus(num_layers: int, degree: int, e_min: int, e_max: int, ngraphs: int = 1):
    if ngraphs == 1:
        g = hairball(num_layers, degree)
        order = list(nx.topological_sort(g))
        num_extra_edges = random.randint(e_min, e_max)
        extra_edges = random.sample(list(itr.combinations(order, 2)), num_extra_edges)
        g.add_edges_from(extra_edges)
        fill_vstructures(g, order)
        return g
    else:
        return [tree_plus(num_layers, degree, e_min, e_max) for _ in range(ngraphs)]


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
