from chordal_utils import get_induced_chordal, get_clique_tree, get_tree_centroid, get_directed_clique_graph, get_clique_graph
from mixed_graph import LabelledMixedGraph


def incomparable(edge1, edge2, clique_graph):
    label1 = clique_graph.get_label(edge1)
    label2 = clique_graph.get_label(edge2)
    return not (label1 <= label2 or label2 < label1)


def add_edge_direction(clique_graph, clique_tree, c1, c2, dcg, verbose=False):
    if dcg.has_directed(c1, c2):
        clique_graph.to_directed(c1, c2)
        clique_tree.to_directed(c1, c2)
        if verbose: print(f"Clique intervention: directed {c1}->{c2}")
    elif dcg.has_directed(c2, c1):
        clique_graph.to_directed(c2, c1)
        clique_tree.to_directed(c2, c1)
        if verbose: print(f"Clique intervention: directed {c2}->{c1}")
    else:
        clique_graph.to_bidirected(c2, c1)
        clique_tree.to_bidirected(c2, c1)
        if verbose: print(f"Clique intervention: directed {c1}<->{c2}")


def apply_clique_intervention(
        clique_tree,
        induced_chordal,
        clique_graph,
        intervened_clique,
        dcg,
        verbose=False
):
    new_clique_tree = clique_tree.copy()
    new_clique_graph = clique_graph.copy()

    # === ADD DIRECTIONS TO CLIQUE GRAPH AND CLIQUE TREE
    for nbr_clique in clique_graph.neighbors_of(intervened_clique):
        add_edge_direction(new_clique_graph, new_clique_tree, nbr_clique, intervened_clique, dcg, verbose=verbose)

    current_unoriented_edges = new_clique_graph.undirected
    extra_nodes = set()

    # === ITERATIVELY ORIENT EDGES
    while True:
        if verbose: print('========')
        for (i, j), label in current_unoriented_edges.items():
            directed_with_same_label = new_clique_graph.directed_edges_with_label(label)
            onto_i = new_clique_graph.onto_edges(i)
            onto_j = new_clique_graph.onto_edges(j)

            if any(d[0] == i for d in directed_with_same_label):  # if C1 --S12--> C2 and C1 --S12-- C3, C1->C3
                new_clique_graph.to_directed(i, j)
                new_clique_tree.to_directed(i, j)
                if verbose: print(f"Directed {i}->{j} by equivalence")
            elif any(d[0] == j for d in directed_with_same_label):
                new_clique_tree.to_directed(j, i)
                new_clique_graph.to_directed(j, i)
                if verbose: print(f"Directed {j}->{i} by equivalence")
            elif any(incomparable(onto_edge, (i, j), clique_graph) for onto_edge in onto_i):
                new_clique_graph.to_directed(i, j)
                new_clique_tree.to_directed(i, j)
                if verbose: print(f"Directed {i}->{j} by propagation")
            elif any(incomparable(onto_edge, (i, j), clique_graph) for onto_edge in onto_j):
                new_clique_graph.to_directed(j, i)
                new_clique_tree.to_directed(i, j)
                if verbose: print(f"Directed {j}->{i} by propagation")
            else:
                if verbose: print(f"Could not direct {i}-{j}")
            # else:
            #     upstream_i = new_clique_graph.parents_of(i) | new_clique_graph.spouses_of(i)
            #     upstream_j = new_clique_graph.parents_of(j) | new_clique_graph.spouses_of(j)
            #
            #     if upstream_i & upstream_j:
            #         extra_nodes.update(i & j)
            #         add_edge_direction(new_clique_graph, new_clique_tree, i, j, dcg, verbose=verbose)

        new_unoriented_edges = new_clique_graph.undirected
        if current_unoriented_edges == new_unoriented_edges:
            break
        current_unoriented_edges = new_unoriented_edges

    return new_clique_tree, new_clique_graph, extra_nodes


def intervention_policy(ug, dag):
    clique_tree = get_clique_tree(ug)
    dcg = get_directed_clique_graph(dag)
    induced_chordal = get_clique_tree(clique_tree)
    clique_tree = LabelledMixedGraph.from_nx(clique_tree)
    clique_graph = get_clique_graph(ug)

    all_extra_nodes = set()
    while True:
        central_clique = get_tree_centroid(clique_tree)  # TODO: UPDATE CLIQUE TREE SO IT IS JUST OVER UNKNOWN EDGES
        new_clique_tree, new_clique_graph, extra_nodes = apply_clique_intervention(
            clique_tree,
            induced_chordal,
            clique_graph,
            central_clique,
            directed_clique_graph
        )
        all_extra_nodes.update(extra_nodes)


if __name__ == '__main__':
    import networkx as nx
    import causaldag as cd
    # g = nx.balanced_tree(2, 2)
    # d = cd.DAG(arcs=g.edges())
    #
    # clique_tree = get_clique_tree(g)
    # directed_clique_graph = get_directed_clique_graph(d)
    # induced_chordal = get_clique_tree(clique_tree)
    # clique_tree = LabelledMixedGraph.from_nx(clique_tree)
    # clique_graph = get_clique_graph(g)
    # print(clique_graph.num_undirected)
    #
    # nct, ncg, extra_nodes = apply_clique_intervention(
    #     clique_tree,
    #     induced_chordal,
    #     clique_graph,
    #     frozenset({0, 1}),
    #     directed_clique_graph,
    #     verbose=True
    # )
    #
    # print(nct.undirected)

    dct = LabelledMixedGraph()
    dct.add_directed(1, 3, {'d'})
    dct.add_directed(2, 3, {'a', 'd'})
    dct.add_directed(3, 4, {'a, c'})
    dct.add_directed(4, 5, {'a', 'b'})
    dct.add_directed(6, 5, {'a'})
    dct.add_directed(6, 7, {'e'})

    dcg = dct.copy()
    dcg.add_directed(1, 2, {'d'})
    dcg.add_directed(6, 4, {'a'})
    dcg.add_directed(6, 3, {'a'})
    dcg.add_directed(2, 6, {'a'})

    ct = dct.to_undirected()
    cg = dcg.to_undirected()

    iv_clique = 3
    nct, ncg, extra_edges = apply_clique_intervention(ct, None, cg, iv_clique, dcg, verbose=True)
    print(nct.undirected)




