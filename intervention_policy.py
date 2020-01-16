from chordal_utils import get_induced_chordal, get_clique_tree, get_tree_centroid, get_directed_clique_graph, get_clique_graph
from mixed_graph import LabelledMixedGraph
from causaldag import UndirectedGraph, DAG
import random


def incomparable(edge1, edge2, clique_graph):
    label1 = clique_graph.get_label(edge1)
    label2 = clique_graph.get_label(edge2)
    return not (label1 <= label2 or label2 < label1)


def add_edge_direction(
        clique_graph: LabelledMixedGraph,
        clique_tree: LabelledMixedGraph,
        c1,
        c2,
        dcg,
        verbose=False
):
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
        clique_tree: LabelledMixedGraph,
        induced_chordal,
        clique_graph: LabelledMixedGraph,
        target_clique,
        dcg: LabelledMixedGraph,
        verbose: bool = False,
        extra_interventions: bool = True
) -> (LabelledMixedGraph, LabelledMixedGraph, set):
    """
    Given a clique tree, "intervene" on the clique `target_clique`.

    Parameters
    ----------
    clique_tree
    induced_chordal
    clique_graph:

    target_clique:
        Clique on which the clique intervention is performed.
    dcg:
        The true directed clique graph. Used to determine edge directions that result from the clique intervention.
    verbose
    extra_interventions:
        If True, then perform extra interventions as needed to ensure that only one undirected subtree remains.

    Returns
    -------
    (new_clique_tree, new_clique_graph, extra_nodes)
        new_clique_tree: clique tree with edge directions added and propagated
        new_clique_graph: clique graph with edge directions added and propagated
        extra_nodes: nodes intervened on in order to get all clique tree edge directions
    """
    new_clique_tree = clique_tree.copy()
    new_clique_graph = clique_graph.copy()

    # === ADD DIRECTIONS TO CLIQUE GRAPH AND CLIQUE TREE
    for nbr_clique in clique_graph.neighbors_of(target_clique):
        add_edge_direction(new_clique_graph, new_clique_tree, nbr_clique, target_clique, dcg, verbose=verbose)

    current_unoriented_edges = new_clique_graph.undirected
    extra_nodes = set()

    # === ITERATIVELY ORIENT EDGES, CHECKING EACH EDGE UNTIL NO RULES CAN BE APPLIED
    while True:
        if verbose: print('========')
        for (c1, c2), label in current_unoriented_edges.items():
            directed_with_same_label = new_clique_graph.directed_edges_with_label(label)
            onto_c1 = new_clique_graph.onto_edges(c1)
            onto_c2 = new_clique_graph.onto_edges(c2)

            if any(d[0] == c1 for d in directed_with_same_label):  # if C1 --S13--> C3 and C1 --S13-- C2, C1->C2
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if verbose: print(f"Directed {c1}->{c2} by equivalence")
            elif any(d[0] == c2 for d in directed_with_same_label):  # if C2 --S12--> C3 and C1 --S13-- C2, C1<-C2
                new_clique_tree.to_directed(c2, c1)
                new_clique_graph.to_directed(c2, c1)
                if verbose: print(f"Directed {c2}->{c1} by equivalence")
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c1):  # propagate C1 -> C2
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if verbose: print(f"Directed {c1}->{c2} by propagation")
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c2):  # propagate C2 -> C1
                new_clique_graph.to_directed(c2, c1)
                new_clique_tree.to_directed(c1, c2)
                if verbose: print(f"Directed {c2}->{c1} by propagation")
            elif extra_interventions:
                upstream_c1 = new_clique_graph.parents_of(c1) | new_clique_graph.spouses_of(c1)
                upstream_c2 = new_clique_graph.parents_of(c2) | new_clique_graph.spouses_of(c2)

                if upstream_c1 & upstream_c2:  # if c1 and c2 have a common parent/spouse, then spend interventions
                    extra_nodes.update(c1 & c2)
                    add_edge_direction(new_clique_graph, new_clique_tree, c1, c2, dcg, verbose=verbose)
            else:
                if verbose: print(f"Could not direct {c1}-{c2}")

        new_unoriented_edges = new_clique_graph.undirected
        if current_unoriented_edges == new_unoriented_edges:
            break
        current_unoriented_edges = new_unoriented_edges

    return new_clique_tree, new_clique_graph, extra_nodes


def intervention_policy(ug: UndirectedGraph, dag: DAG):
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
            dcg
        )
        all_extra_nodes.update(extra_nodes)


def dct_policy(dag: DAG, verbose=False) -> set:
    ug = UndirectedGraph(nodes=dag.nodes, edges=dag.skeleton)
    full_clique_tree = get_clique_tree(ug)
    current_clique_subtree = full_clique_tree
    clique_graph = get_clique_graph(ug)
    dcg = get_directed_clique_graph(dag)

    intervened_nodes = set()
    if verbose: print("Phase I")
    while True:
        if len(current_clique_subtree.nodes) <= 1:
            break
        # INTERVENE ON THE CENTRAL CLIQUE
        central_clique = get_tree_centroid(current_clique_subtree, verbose=verbose)
        if verbose: print(f'Picked central clique: {central_clique}')

        full_clique_tree, clique_graph, extra_nodes = apply_clique_intervention(
            full_clique_tree,
            None,
            clique_graph,
            central_clique,
            dcg,
            verbose=verbose
        )

        # RECORD THE NODES THAT WERE INTERVENED ON
        intervened_nodes.update(central_clique)
        intervened_nodes.update(extra_nodes)

        # TAKE SUBTREE
        remaining_cliques = {
            clique for clique in full_clique_tree._nodes
            if full_clique_tree.neighbor_degree_of(clique) != 0
        }
        if verbose: print(f"Remaining cliques: {remaining_cliques}")
        current_clique_subtree = current_clique_subtree.induced_graph(remaining_cliques)

    if verbose: print("Phase II")
    while True:
        source_cliques = {clique for clique in clique_graph._nodes if clique_graph.indegree_of(clique) == 0}
        if len(source_cliques) == 0:
            break
        next_clique = random.choice(list(source_cliques))
        if verbose: print(f"Next clique: {next_clique}")
        clique_graph.remove_node(next_clique)

        # intervene on all nodes in this clique if it doesn't have a residual of size one
        if len(next_clique - intervened_nodes) > 1:
            intervened_nodes.update(next_clique)

    return intervened_nodes


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




