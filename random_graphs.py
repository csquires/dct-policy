import networkx as nx
import itertools as itr
import random
from typing import Union, List
import numpy as np
from graph_utils import fill_vstructures, direct_chordal_graph
from scipy.special import binom


def tree_of_cliques(
        degree: int,
        min_clique_size: int,
        max_clique_size: int,
        ngraphs: int = 1,
        nnodes: int = None
):
    if ngraphs == 1:
        counter = random.randint(min_clique_size, max_clique_size)
        source_clique = list(range(counter))
        previous_layer_cliques = [source_clique]
        current_layer_cliques = []
        arcs = set(itr.combinations(source_clique, 2))

        while counter < nnodes:
            for parent_clique in previous_layer_cliques:
                for d in range(degree):
                    if counter < nnodes:
                        clique_size = random.randint(min_clique_size, max_clique_size)
                        intersection_size = min(len(parent_clique)-1, random.randint(int(clique_size/2), clique_size-1))
                        num_new_nodes = clique_size - intersection_size
                        print(intersection_size)

                        indices = set(random.sample(parent_clique, intersection_size))
                        intersection = [
                            parent_clique[ix] for ix in range(len(parent_clique))
                            if ix in indices
                        ]
                        new_clique = intersection + list(range(counter, counter+num_new_nodes))
                        current_layer_cliques.append(new_clique)
                        arcs.update(set(itr.combinations(new_clique, 2)))
                        counter += num_new_nodes
            previous_layer_cliques = current_layer_cliques.copy()
        g = nx.DiGraph()
        g.add_edges_from(arcs)
        if not nx.is_connected(g.to_undirected()):
            raise RuntimeError
        return g
    else:
        return [tree_of_cliques(degree, min_clique_size, max_clique_size, nnodes=nnodes) for _ in range(ngraphs)]


def hairball(degree: int, num_layers: int = None, nnodes: int = None) -> nx.DiGraph:
    """
    Return a `degree`-ary tree with `nnodes` nodes or `num_layers` layers.
    """
    nnodes = sum((degree**i for i in range(num_layers))) if nnodes is None else nnodes
    return nx.full_rary_tree(degree, nnodes, create_using=nx.DiGraph)


def hairball_plus(
        degree: int,
        e_min: int = None,
        e_max: int = None,
        ngraphs: int = 1,
        num_layers: int = None,
        nnodes: int = None,
        edge_prob: float = None,
        nontree_factor: float = None
):
    """
    Generate a "hairball", then add a random number of edges between `e_min` and `e_max`, then
    triangulate the graph to make it chordal.
    """
    if ngraphs == 1:
        g = hairball(degree, num_layers=num_layers, nnodes=nnodes)
        order = list(nx.topological_sort(g))
        if e_min is not None:
            num_extra_edges = random.randint(e_min, e_max)
        elif edge_prob is not None:
            nnodes = g.number_of_nodes()
            num_missing_edges = binom(nnodes, 2) - (nnodes - 1)
            num_extra_edges = np.random.binomial(num_missing_edges, edge_prob)
        else:
            num_extra_edges = int(nontree_factor*(nnodes - 1))
        missing_edges = [(i, j) for i, j in itr.combinations(order, 2) if not g.has_edge(i, j)]
        extra_edges = random.sample(missing_edges, num_extra_edges)
        g.add_edges_from(extra_edges)
        fill_vstructures(g, order)
        return g
    else:
        return [hairball_plus(degree, e_min, e_max, num_layers=num_layers, nnodes=nnodes) for _ in range(ngraphs)]


def random_directed_tree(nnodes: int):
    """
    Generate a random undirected tree, then pick a random root to make it directed.
    """
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


def random_chordal_graph(nnodes, p=.1, ngraphs=1) -> Union[nx.DiGraph, List[nx.DiGraph]]:
    if ngraphs == 1:
        g = nx.erdos_renyi_graph(nnodes, p)
        perm = random.sample(set(range(nnodes)), nnodes)
        for node in perm:
            nbrs = set(g.neighbors(node)) | {node}
            g.add_edges_from((i, j) for i, j in itr.combinations(nbrs, 2))

        d = nx.DiGraph()
        for node in perm:
            d.add_edges_from([(node, nbr) for nbr in g.neighbors(node)])
        return d
    else:
        return [random_chordal_graph(nnodes, p=p) for _ in range(ngraphs)]


def shanmugam_random_chordal(nnodes, density):
    while True:
        d = nx.DiGraph()
        d.add_nodes_from(set(range(nnodes)))
        order = list(range(1, nnodes))
        for i in order:
            num_parents_i = max(1, np.random.binomial(i, density))
            parents_i = random.sample(list(range(i)), num_parents_i)
            d.add_edges_from({(p, i) for p in parents_i})
        for i in reversed(order):
            for j, k in itr.combinations(d.predecessors(i), 2):
                d.add_edge(min(j, k), max(j, k))

        perm = np.random.permutation(list(range(nnodes)))
        d = nx.relabel.relabel_nodes(d, dict(enumerate(perm)))

        return d


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


if __name__ == '__main__':
    import causaldag as cd

    d_nx = shanmugam_random_chordal(10, .1)
    print(d_nx.number_of_nodes())
    print(nx.chordal_graph_cliques(d_nx.to_undirected()))
    print(nx.is_connected(d_nx.to_undirected()))
    d = cd.DAG.from_nx(d_nx)
    print(d.vstructures())



