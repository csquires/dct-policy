from chordal_utils import get_induced_chordal, get_clique_tree, get_tree_centroid, get_directed_clique_graph, get_clique_graph
import operator as op
import numpy as np
import networkx as nx
from mixed_graph import LabelledMixedGraph
from causaldag import UndirectedGraph, DAG
import random
import itertools as itr
from collections import defaultdict
from networkx.utils import UnionFind


def dcg2dct(dcg: LabelledMixedGraph, verbose=True):
    clique_tree = nx.MultiDiGraph()
    clique_tree.add_nodes_from(dcg.nodes)
    print('============')

    subtrees = UnionFind()
    bidirected_components = UnionFind()
    for c1, c2 in sorted(dcg.directed_keys | dcg.bidirected_keys, key=lambda e: len(dcg.get_label(e)), reverse=True):
        if verbose: print(f"=== Considering edge {c1}-{c2}")
        if subtrees[c1] != subtrees[c2]:  # check adding edge won't make a cycle
            if dcg.has_bidirected((c1, c2)):  # if bidirected, add
                if verbose: print(f"Adding edge {c1}<->{c2}")
                clique_tree.add_edge(c1, c2)
                clique_tree.add_edge(c2, c1)
                bidirected_components.union(c1, c2)
                subtrees.union(c1, c2)
            else:  # if directed, check that c1 won't become a conflicting source
                c2_parent = bidirected_components[c2]
                bidirected_component = [
                    c for c, parent in bidirected_components.parents.items()
                    if parent == c2_parent
                ]
                has_source = any(
                    set(clique_tree.predecessors(c)) - set(clique_tree.successors(c))
                    for c in bidirected_component
                )
                if not has_source:
                    if verbose: print(f"Adding edge {c1}->{c2}")
                    clique_tree.add_edge(c1, c2)
                    subtrees.union(c1, c2)

    labels = {(c1, c2, 0): c1 & c2 for c1, c2 in clique_tree.edges()}
    nx.set_edge_attributes(clique_tree, labels, name='label')

    return LabelledMixedGraph.from_nx(clique_tree)


def cg2ct(cg: LabelledMixedGraph):
    ct = nx.Graph()
    subtrees = UnionFind()
    for c1, c2 in sorted(cg.undirected_keys, key=lambda e: len(cg.get_label(e)), reverse=True):
        if subtrees[c1] != subtrees[c2]:
            ct.add_edge(c1, c2)
            subtrees.union(c1, c2)

    return ct


def incomparable(edge1, edge2, clique_graph):
    """
    Check if `edge1` and `edge2` have incomparable labels in the given clique graph.

    Parameters
    ----------
    edge1
    edge2
    clique_graph

    Returns
    -------

    """
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


def apply_clique_intervention2(
        clique_tree: LabelledMixedGraph,
        induced_chordal,
        clique_graph: LabelledMixedGraph,
        target_clique,
        dcg: LabelledMixedGraph,
        verbose: bool = False,
        extra_interventions: bool = True,
        dag = None,
        check=True
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
        wrongly_inferred_bidirected = new_clique_graph.bidirected_keys - dcg.bidirected_keys
        wrongly_inferred_directed = new_clique_graph.directed_keys - dcg.directed_keys
        if clique_graph.num_edges != dcg.num_edges:
            raise ValueError("Screwed up the clique graph")
        if check and wrongly_inferred_directed:
            print(f"Wrongly inferred directed edges: {new_clique_graph.directed_keys - dcg.directed_keys}")
        if check and wrongly_inferred_bidirected:
            print(f"Wrongly inferred bidirected edges: {new_clique_graph.bidirected_keys - dcg.bidirected_keys}")

        forward_edges = {(i, j): label for (i, j), label in current_unoriented_edges.items()}
        backward_edges = {(j, i): label for (i, j), label in forward_edges.items()}
        for (c1, c2), label in itr.chain(forward_edges.items(), backward_edges.items()):
            onto_c1 = new_clique_graph.onto_edges(c1)

            if any((c3 & c2) == label for c3 in new_clique_graph.children_of(c2)):
                new_clique_graph.to_directed(c2, c1)
                if verbose: print(f"Directed {c2}->{c1} by label equivalence")
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c1):  # propagate C1 -> C2
                new_clique_graph.to_directed(c1, c2)
                if verbose: print(f"Directed {c1}->{c2} by propagation")
            elif any((c3 & c2) == label for c3 in new_clique_graph.onto_nodes(c2)):
                if not new_clique_graph.has_directed(c1, c2) and not new_clique_graph.has_bidirected((c1, c2)):
                    new_clique_graph.to_semidirected(c1, c2)
                    if verbose: print(f"Semi-directed {c1}o->{c2} by label equivalence")

        new_unoriented_edges = new_clique_graph.undirected
        if current_unoriented_edges == new_unoriented_edges:
            if not extra_interventions:
                break
            else:
                # === COMPLEX RULE: DEAL WITH CURRENT LABEL INCLUDING OTHER LABELS
                for (c1, c2), label in current_unoriented_edges.items():
                    c3 = next(
                        (c3 for c3 in new_clique_graph.onto_nodes(c1) & new_clique_graph.onto_nodes(c2) if (c3 & c1) == (c3 & c2))
                        , None
                    )

                    if c3 is not None and (c3 & c1) < label:  # if c1 and c2 have a common parent/spouse with a smaller intersection, then spend interventions
                        if verbose: print(f"Intervening on extra nodes {c1 & c2} to orient {c1}-{c2} with common_parent = {c3}")
                        extra_nodes.update(c1 & c2)
                        add_edge_direction(new_clique_graph, new_clique_tree, c1, c2, dcg, verbose=verbose)
                    elif c3 is not None and (c3 & c1) == label:
                        new_clique_graph.to_bidirected(c1, c2)
                        new_clique_tree.to_bidirected(c1, c2)
                        if verbose: print(f"Directed {c1}<->{c2} by equivalence")
                if current_unoriented_edges == new_unoriented_edges:
                    break
        current_unoriented_edges = new_unoriented_edges

        # new_clique_graph.remove_edges(removable_edges)
    return new_clique_tree, new_clique_graph, extra_nodes


def apply_clique_intervention(
        clique_tree: LabelledMixedGraph,
        induced_chordal,
        clique_graph: LabelledMixedGraph,
        target_clique,
        dcg: LabelledMixedGraph,
        verbose: bool = False,
        extra_interventions: bool = True,
        dag = None,
        check=True
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
    removable_edges = []

    # === ITERATIVELY ORIENT EDGES, CHECKING EACH EDGE UNTIL NO RULES CAN BE APPLIED
    while True:
        if verbose: print('========')
        wrongly_inferred_bidirected = new_clique_graph.bidirected_keys - dcg.bidirected_keys
        wrongly_inferred_directed = new_clique_graph.directed_keys - dcg.directed_keys
        if check and wrongly_inferred_directed:
            print(f"Wrongly inferred directed edges: {new_clique_graph.directed_keys - dcg.directed_keys}")
        if check and wrongly_inferred_bidirected:
            print(f"Wrongly inferred bidirected edges: {new_clique_graph.bidirected_keys - dcg.bidirected_keys}")

        forward_edges = {(i, j): label for (i, j), label in current_unoriented_edges.items()}
        backward_edges = {(j, i): label for (i, j), label in forward_edges.items()}
        for (c1, c2), label in {**forward_edges, **backward_edges}.items():
            directed_with_same_label = new_clique_graph.directed_edges_with_label(label) - frozenset({(c2, c1)})
            c1_parents = {p for p in new_clique_graph.parents_of(c1) if p & c1 >= label and p & c2 == label}
            c1_spouses = {s for s in new_clique_graph.spouses_of(c1) if s & c1 >= label and s & c2 == label}
            onto_c1 = new_clique_graph.onto_edges(c1)
            onto_c2 = new_clique_graph.onto_edges(c2)

            c3_ = next((c3 for c3 in c1_parents if new_clique_graph.has_directed(c3, c2)), None)
            # == SIMPLEST RULES: ALL OUTGOING EDGES FROM C1 WITH SAME LABEL HAVE THE SAME ORIENTATION
            if any(d[0] == c1 for d in directed_with_same_label):
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if verbose: print(f"Directed {c1}->{c2} by equivalence")
            # elif any(d[0] == c2 for d in directed_with_same_label):
            #     new_clique_graph.to_directed(c2, c1)
            #     new_clique_tree.to_directed(c2, c1)
            #     if verbose: print(f"Directed {c2}->{c1} by equivalence")
            # === COMPLEX RULES: DEAL WITH INCLUSION OF CURRENT LABEL IN A LARGER LABEL
            elif c3_:
                removable_edges.append((c1, c2))
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                # new_clique_graph.remove_edge(c1, c2)
                # new_clique_tree.remove_edge(c1, c2)
                # new_clique_tree.add_directed(c3_, c2, label)
                if verbose: print(f"Temporarily directed {c1}->{c2}")
            # elif any(d[1] == c1 for d in directed_with_same_label):
            #     new_clique_graph.remove_edge(c1, c2)
            #     new_clique_tree.remove_edge(c1, c2)
            #     c3 = next(c3 for c3, c1_ in directed_with_same_label if c1 == c1_)
            #     new_clique_tree.add_directed(c3, c2)
            #     if verbose: print(f"Removing {c1}-{c2} since it's redundant ({c3}->{c1})")
            elif any(new_clique_graph.has_bidirected((c3, c2)) for c3 in c1_parents):
                new_clique_graph.to_bidirected(c1, c2)
                new_clique_tree.to_bidirected(c1, c2)
                if check and not dcg.has_bidirected((c1, c2)):
                    raise RuntimeError("ORIENTATION RULES ARE WRONG")
                if verbose: print(f"Directed {c1}<->{c2} by rule (2)")
            elif any(new_clique_graph.has_directed(c3, c2) for c3 in c1_spouses):
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if check and not dcg.has_directed(c1, c2):
                    raise RuntimeError("ORIENTATION RULES ARE WRONG")
                if verbose: print(f"Directed {c1}->{c2} by rule (4)")
            elif any(new_clique_graph.has_bidirected((c3, c2)) for c3 in c1_spouses):
                new_clique_graph.to_bidirected(c1, c2)
                new_clique_tree.to_bidirected(c1, c2)
                if verbose: print(f"Directed {c1}<->{c2} by rule (5)")
            elif any(new_clique_graph.has_directed(c2, c3) for c3 in c1_parents | c1_spouses):
                new_clique_graph.to_directed(c2, c1)
                new_clique_tree.to_directed(c2, c1)
                if check and not dcg.has_directed(c2, c1):
                    raise RuntimeError("ORIENTATION RULES ARE WRONG")
                if verbose: print(f"Directed {c2}->{c1} by rule (3/6)")
            # === PROPAGATION RULES
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c1):  # propagate C1 -> C2
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if verbose: print(f"Directed {c1}->{c2} by propagation")
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c2):  # propagate C2 -> C1
                new_clique_graph.to_directed(c2, c1)
                new_clique_tree.to_directed(c2, c1)
                if verbose: print(f"Directed {c2}->{c1} by propagation")
            else:
                pass
                # if verbose: print(f"Could not direct {c1}-{c2}")

        new_unoriented_edges = new_clique_graph.undirected
        if current_unoriented_edges == new_unoriented_edges:
            if not extra_interventions:
                break
            else:
                # === COMPLEX RULE: DEAL WITH CURRENT LABEL INCLUDING OTHER LABELS
                for (c1, c2), label in current_unoriented_edges.items():
                    upstream_c1 = new_clique_graph.parents_of(c1) | new_clique_graph.spouses_of(c1)
                    upstream_c2 = new_clique_graph.parents_of(c2) | new_clique_graph.spouses_of(c2)
                    c3 = next((c3 for c3 in upstream_c1 & upstream_c2 if (c3 & c1) == (c3 & c2)), None)

                    if c3 is not None and (c3 & c1) < label:  # if c1 and c2 have a common parent/spouse with a smaller intersection, then spend interventions
                        if verbose: print(f"Intervening on extra nodes {c1 & c2} to orient {c1}-{c2} with common_parent = {c3}")
                        extra_nodes.update(c1 & c2)
                        add_edge_direction(new_clique_graph, new_clique_tree, c1, c2, dcg, verbose=verbose)
                    elif c3 is not None and (c3 & c1) == label:
                        new_clique_graph.to_bidirected(c1, c2)
                        new_clique_tree.to_bidirected(c1, c2)
                        if verbose: print(f"Directed {c1}<->{c2} by equivalence")
                if current_unoriented_edges == new_unoriented_edges:
                    break
        current_unoriented_edges = new_unoriented_edges

        # new_clique_graph.remove_edges(removable_edges)
    return new_clique_tree, new_clique_graph, extra_nodes


def dct_policy(dag: DAG, verbose=False, check=False) -> set:
    """
    Use the DCT policy to fully orient the given DAG, as if it was an undirected graph.

    Parameters
    ----------
    dag
    verbose

    Returns
    -------

    """
    random.seed(897986274)
    if check:
        optimal_ivs = len(dag.optimal_fully_orienting_interventions())
        nx_graph = dag.to_nx()
        cliques = nx.chordal_graph_cliques(nx_graph.to_undirected())
        log_num_cliques = np.ceil(np.log2(len(cliques)))
        # clique_size = max((len(c) for c in cliques))
        # true_dct = LabelledMixedGraph.from_nx(dag.directed_clique_tree())

    ug = UndirectedGraph(nodes=dag.nodes, edges=dag.skeleton)
    cliques = nx.chordal_graph_cliques(ug.to_nx())  # costly
    full_clique_tree = get_clique_tree(ug, cliques=cliques)
    current_clique_subtree = full_clique_tree.undirected_copy().to_nx()
    clique_graph = get_clique_graph(ug, cliques=cliques)  # costly
    dcg = get_directed_clique_graph(dag, clique_graph=clique_graph)

    intervened_nodes = set()
    regular_phase1 = set()
    cliques_phase1 = []
    all_extra_nodes = set()
    if verbose: print("Phase I")
    while True:
        if len(current_clique_subtree.nodes) == 1:
            intervened_nodes.update(list(current_clique_subtree.nodes)[0])
            break
        if len(current_clique_subtree.nodes) == 0:
            break

        # INTERVENE ON THE CENTRAL CLIQUE
        central_clique = get_tree_centroid(current_clique_subtree, verbose=False)
        if verbose: print(f'Picked central clique: {central_clique}')

        full_clique_tree, clique_graph, extra_nodes = apply_clique_intervention2(
            full_clique_tree,
            None,
            clique_graph,
            central_clique,
            dcg,
            verbose=verbose,
            dag=dag
        )

        # RECORD THE NODES THAT WERE INTERVENED ON
        intervened_nodes.update(central_clique)
        intervened_nodes.update(extra_nodes)
        regular_phase1.update(central_clique)
        all_extra_nodes.update(extra_nodes)
        cliques_phase1.append(central_clique)

        # TAKE SUBTREE
        remaining_cliques = {
            clique for clique in clique_graph._nodes
            if clique_graph.neighbor_degree_of(clique) != 0
        }
        # print(clique_graph.undirected_keys)
        if verbose: print(f"Remaining cliques: {remaining_cliques}")
        # current_clique_subtree = current_clique_subtree.induced_graph(remaining_cliques)
        # print(current_clique_subtree.undirected_keys)
        if not remaining_cliques:
            break
        current_clique_subtree = cg2ct(clique_graph.induced_graph(remaining_cliques))

    new_dct = dcg2dct(clique_graph)
    cg = clique_graph.copy()
    cg.remove_all_undirected()
    cg.all_to_undirected()
    cg_nx = cg.to_nx()
    # print('connected', nx.is_connected(cg_nx))
    if verbose: print(f"*** {len(regular_phase1)} regular intervened in Phase I")
    if check and len(cliques_phase1) > log_num_cliques:
        print(f"Phase I bound: {log_num_cliques}, used={len(cliques_phase1)}")
        print('------ BROKEN ------')
        print(cliques_phase1)
        raise RuntimeError

    if verbose: print("Phase II")
    phase2_nodes = set()
    resolved_cliques = set()
    nx_clique_tree = new_dct.to_nx()
    clique2ancestors = {c: nx.ancestors(nx_clique_tree, c) for c in new_dct.nodes}
    sorted_cliques = sorted(clique2ancestors.items(), key=lambda x: len(x[1]))

    for next_clique, _ in sorted_cliques:
        # intervene on all nodes in this clique if it doesn't have a residual of size one
        if len(next_clique - resolved_cliques) > 1:
            if verbose: print(f"intervened on {next_clique}")
            intervened_nodes.update(next_clique - resolved_cliques)
            phase2_nodes.update(next_clique - resolved_cliques)

        resolved_cliques.update(next_clique)
    if verbose: print(f"*** {len(phase2_nodes)} intervened in Phase II")
    # if verbose: print(f"extra nodes not intervened in Phase II: {all_extra_nodes - phase2_nodes}")
    if check and 3*optimal_ivs < len(phase2_nodes):
        print(f"Phase II bound: {3*optimal_ivs}, used={len(phase2_nodes)}")
        print("------------ BROKEN -------------")
        raise RuntimeError
    if check and not all_extra_nodes <= phase2_nodes:
        print(f"extra nodes not intervened in Phase II: {all_extra_nodes - phase2_nodes}")
        print(f"---- BROKEN ----")
        raise RuntimeError

    return intervened_nodes


if __name__ == '__main__':
    from directed_chordal_utils import tree_plus
    from line_profiler import LineProfiler
    import causaldag as cd

    d = cd.DAG.from_nx(tree_plus(100, 2, 5))

    def run_dct_policy():
        iv_nodes = dct_policy(d)
        print(iv_nodes)

    lp = LineProfiler()
    lp.add_function(dct_policy)
    lp.runcall(run_dct_policy)
    lp.print_stats()




