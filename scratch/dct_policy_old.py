from graph_utils import get_clique_tree, get_tree_centroid, get_directed_clique_graph, get_clique_graph
import numpy as np
from mixed_graph import LabelledMixedGraph
from causaldag import UndirectedGraph, DAG
from collections import defaultdict


def dcg2dct(dcg: LabelledMixedGraph):
    # === get max cliques
    max_cliques = dcg.nodes
    overlap_to_edges = defaultdict(set)
    for edge, label in dcg.directed.items():
        overlap_to_edges[len(label)].add(edge)
    for edge, label in dcg.bidirected.items():
        overlap_to_edges[len(label)].add(edge)

    clique_tree = nx.MultiDiGraph()
    # greedily choose edges to create a directed clique tree
    current_threshold = max(overlap_to_edges)
    for _ in range(len(max_cliques)-1):
        while current_threshold > 0:
            candidate_edges = list(overlap_to_edges[current_threshold])
            candidate_trees = [clique_tree.copy() for _ in candidate_edges]
            for t, edge in zip(candidate_trees, candidate_edges):
                c1, c2 = edge
                if isinstance(edge, frozenset):
                    t.add_edge(c1, c2)
                    t.add_edge(c2, c1)
                else:
                    t.add_edge(c1, c2)

            # only keep forests
            candidate_trees = [t for t in candidate_trees if nx.is_forest(t.to_undirected())]

            if not candidate_trees:
                current_threshold -= 1
                continue

            # only keep if single source in each component
            candidate_trees_no_collider = []
            for t in candidate_trees:
                components = [nx.subgraph(t, component) for component in nx.strongly_connected_components(t)]
                components2parents = [set.union(*(set(t.predecessors(node)) - c.nodes() for node in c)) for c in components]
                if all(len(parents) <= 1 for parents in components2parents):
                    candidate_trees_no_collider.append(t)
            if not candidate_trees_no_collider:
                raise Exception("Can't find a collider-free tree")

            # choose clique tree without new -><->, if possible
            preferred_candidate_trees = []
            for t in candidate_trees_no_collider:
                new_edge = set(t.edges()) - set(clique_tree.edges())
                c1, c2 = list(new_edge)[0]
                if not t.has_edge(c2, c1):
                    preferred_candidate_trees.append(t)
                else:
                    if not set(t.predecessors(c1)) - {c2} and not set(t.predecessors(c2)) - {c1}:
                        preferred_candidate_trees.append(t)

            clique_tree = preferred_candidate_trees[0] if preferred_candidate_trees else candidate_trees_no_collider[0]
            break

    labels = {(c1, c2, 0): c1 & c2 for c1, c2 in clique_tree.edges()}
    nx.set_edge_attributes(clique_tree, labels, name='label')

    return LabelledMixedGraph.from_nx(clique_tree)


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


def apply_clique_intervention(
        clique_tree: LabelledMixedGraph,
        induced_chordal,
        clique_graph: LabelledMixedGraph,
        target_clique,
        dcg: LabelledMixedGraph,
        verbose: bool = False,
        extra_interventions: bool = True,
        dag = None
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
        if wrongly_inferred_directed:
            print(f"Wrongly inferred directed edges: {new_clique_graph.directed_keys - dcg.directed_keys}")
        if wrongly_inferred_bidirected:
            print(f"Wrongly inferred bidirected edges: {new_clique_graph.bidirected_keys - dcg.bidirected_keys}")
        for (c1, c2), label in current_unoriented_edges.items():
            directed_with_same_label = new_clique_graph.directed_edges_with_label(label)
            c1_parents = {p for p in new_clique_graph.parents_of(c1) if p & c1 >= label and p & c2 == label}
            c1_spouses = {s for s in new_clique_graph.spouses_of(c1) if s & c1 >= label and s & c2 == label}
            onto_c1 = new_clique_graph.onto_edges(c1)
            onto_c2 = new_clique_graph.onto_edges(c2)

            c3_ = next((c3 for c3 in c1_parents if new_clique_graph.has_directed(c3, c2)), None)
            if any(d[0] == c1 for d in directed_with_same_label):
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
            elif any(d[0] == c2 for d in directed_with_same_label):
                new_clique_graph.to_directed(c2, c1)
                new_clique_tree.to_directed(c2, c1)
            elif c3_:
                new_clique_graph.remove_edge(c1, c2)
                new_clique_tree.remove_edge(c1, c2)
                new_clique_tree.add_directed(c3_, c2, label)
            elif any(new_clique_graph.has_bidirected((c3, c2)) for c3 in c1_parents):
                new_clique_graph.to_bidirected(c1, c2)
                new_clique_tree.to_bidirected(c1, c2)
                if not dcg.has_bidirected((c1, c2)):
                    print(dag)
                    print(new_clique_graph.bidirected_keys)
                    print(new_clique_graph.directed_keys)
                    print(c1_parents & new_clique_graph.spouses_of(c2))
            elif any(new_clique_graph.has_directed(c3, c2) for c3 in c1_spouses):
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if not dcg.has_directed(c1, c2):
                    print(new_clique_graph.bidirected_keys)
                    print(new_clique_graph.directed_keys)
                    print(c1_spouses & new_clique_graph.parents_of(c2))
            elif any(new_clique_graph.has_bidirected((c3, c2)) for c3 in c1_spouses):
                new_clique_graph.to_bidirected(c1, c2)
                new_clique_tree.to_bidirected(c1, c2)
            elif any(new_clique_graph.has_directed(c2, c3) for c3 in c1_parents | c1_spouses):
                new_clique_graph.to_directed(c2, c1)
                new_clique_tree.to_directed(c2, c1)
                if not dcg.has_directed(c2, c1):
                    print('------------')
                    print(new_clique_graph.bidirected_keys)
                    print(new_clique_graph.directed_keys)
                    print((c1_parents | c1_spouses) & new_clique_graph.children_of(c2))

            # if any(d[0] == c1 for d in directed_with_same_label):  # if C1 --S13--> C3 and C1 --S13-- C2, C1->C2
            #     new_clique_graph.to_directed(c1, c2)
            #     new_clique_tree.to_directed(c1, c2)
            #     if verbose: print(f"Directed {c1}->{c2} by equivalence")
            # elif any(d[0] == c2 for d in directed_with_same_label):  # if C2 --S12--> C3 and C1 --S13-- C2, C1<-C2
            #     new_clique_tree.to_directed(c2, c1)
            #     new_clique_graph.to_directed(c2, c1)
            #     if verbose: print(f"Directed {c2}->{c1} by equivalence")
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c1):  # propagate C1 -> C2
                new_clique_graph.to_directed(c1, c2)
                new_clique_tree.to_directed(c1, c2)
                if verbose: print(f"Directed {c1}->{c2} by propagation")
            elif any(incomparable(onto_edge, (c1, c2), clique_graph) for onto_edge in onto_c2):  # propagate C2 -> C1
                new_clique_graph.to_directed(c2, c1)
                new_clique_tree.to_directed(c2, c1)
                if verbose: print(f"Directed {c2}->{c1} by propagation")
            else:
                if verbose:
                    print(f"Could not direct {c1}-{c2}")
                    print(directed_with_same_label)

        new_unoriented_edges = new_clique_graph.undirected
        if current_unoriented_edges == new_unoriented_edges:
            if not extra_interventions:
                break
            else:
                for (c1, c2), label in current_unoriented_edges.items():
                    upstream_c1 = new_clique_graph.parents_of(c1) | new_clique_graph.spouses_of(c1)
                    upstream_c2 = new_clique_graph.parents_of(c2) | new_clique_graph.spouses_of(c2)
                    c3 = next((c3 for c3 in upstream_c1 & upstream_c2 if label < (c1 & c2)), None)

                    if c3 is not None:  # if c1 and c2 have a common parent/spouse with a smaller intersection, then spend interventions
                        print(f"Intervening on extra nodes {c1 & c2} to orient {c1}-{c2} with common_parent = {c3}")
                        extra_nodes.update(c1 & c2)
                        add_edge_direction(new_clique_graph, new_clique_tree, c1, c2, dcg, verbose=verbose)
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


def dct_policy(dag: DAG, verbose=False, check=True) -> set:
    """
    Use the DCT policy to fully orient the given DAG, as if it was an undirected graph.

    Parameters
    ----------
    dag
    verbose

    Returns
    -------

    """
    if check:
        optimal_ivs = len(dag.optimal_fully_orienting_interventions())
        nx_graph = dag.to_nx()
        cliques = nx.chordal_graph_cliques(nx_graph.to_undirected())
        log_num_cliques = np.ceil(np.log2(len(cliques)))
        clique_size = max((len(c) for c in cliques))
        true_dct = LabelledMixedGraph.from_nx(dag.directed_clique_tree())

    ug = UndirectedGraph(nodes=dag.nodes, edges=dag.skeleton)
    full_clique_tree = get_clique_tree(ug)
    # print(full_clique_tree.num_edges)
    current_clique_subtree = full_clique_tree
    clique_graph = get_clique_graph(ug)
    dcg = get_directed_clique_graph(dag)

    intervened_nodes = set()
    regular_phase1 = set()
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

        full_clique_tree, clique_graph, extra_nodes = apply_clique_intervention(
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

        # TAKE SUBTREE
        remaining_cliques = {
            clique for clique in full_clique_tree._nodes
            if full_clique_tree.neighbor_degree_of(clique) != 0
        }
        if verbose: print(f"Remaining cliques: {remaining_cliques}")
        current_clique_subtree = current_clique_subtree.induced_graph(remaining_cliques)

    new_dct = dcg2dct(clique_graph)
    cg = clique_graph.copy()
    print(f'number undirected in cg: {cg.num_undirected}')
    cg.remove_all_undirected()
    cg.all_to_undirected()
    cg_nx = cg.to_nx()
    # print('connected', nx.is_connected(cg_nx))
    if verbose: print(f"*** {len(regular_phase1)} regular intervened in Phase I")
    if check:
        if log_num_cliques*clique_size < len(regular_phase1):
            print(f"Phase I bound: {log_num_cliques*clique_size}, used={len(regular_phase1)}")
            print('------ BROKEN ------')

    if verbose: print("Phase II")
    phase2_nodes = set()
    resolved_cliques = set()
    # print(f"Clique graph: directed={clique_graph.directed}, bidirected = {clique_graph.bidirected}")
    # print(f"Clique tree: directed={full_clique_tree.directed}, bidirected = {full_clique_tree.bidirected}")
    # if check:
    #     print(f"False negatives: directed={set(true_dct.directed.keys()) - set(full_clique_tree.directed.keys())}, "
    #           f"bidirected={set(true_dct.bidirected.keys()) - set(full_clique_tree.bidirected.keys())}")
    #     print(f"False positives: directed={set(full_clique_tree.directed.keys()) - set(true_dct.directed.keys())}, "
    #           f"bidirected={set(full_clique_tree.bidirected.keys()) - set(true_dct.bidirected.keys())}")
    #     print(full_clique_tree.num_edges)
    #     print(true_dct.num_edges)
    #     print(full_clique_tree.to_undirected().undirected_keys - true_dct.to_undirected().undirected_keys)
    #     print(true_dct.to_undirected().undirected_keys - full_clique_tree.to_undirected().undirected_keys)
    #     print(full_clique_tree.num_undirected)
    #     print('==== graph =====')
    #     print(set(dcg.directed.keys()))
    #     print(set(clique_graph.directed.keys()))
    #     print(f"False negatives: directed={set(dcg.directed.keys()) - set(clique_graph.directed.keys())}, "
    #           f"bidirected={set(dcg.bidirected.keys()) - set(clique_graph.bidirected.keys())}")
    #     print(f"False positives: directed={set(clique_graph.directed.keys()) - set(dcg.directed.keys())}, "
    #           f"bidirected={set(clique_graph.bidirected.keys()) - set(dcg.bidirected.keys())}")
    #     print(clique_graph.num_edges)
    #     print(dcg.num_edges)
    #     print(clique_graph.num_undirected)

    # print(full_clique_tree.directed_keys)
    nx_clique_tree = new_dct.to_nx()
    clique2ancestors = {c: nx.ancestors(nx_clique_tree, c) for c in new_dct.nodes}
    sorted_cliques = sorted(clique2ancestors.items(), key=lambda x: len(x[1]))
    # print(sorted_cliques)

    for next_clique, _ in sorted_cliques:
        # source_cliques = {clique for clique in clique_graph._nodes if clique_graph.indegree_of(clique) == 0}
        # if verbose: print(f"source cliques={source_cliques}")
        # if len(source_cliques) == 0:
        #     break
        # next_clique = random.choice(list(source_cliques))
        # clique_graph.remove_node(next_clique)

        # intervene on all nodes in this clique if it doesn't have a residual of size one
        if len(next_clique - resolved_cliques) > 1:
            if verbose: print(f"intervened on {next_clique}")
            intervened_nodes.update(next_clique - resolved_cliques)
            phase2_nodes.update(next_clique - resolved_cliques)

        resolved_cliques.update(next_clique)
    if verbose: print(f"*** {len(phase2_nodes)} intervened in Phase II")
    if verbose: print(f"extra nodes not intervened in Phase II: {all_extra_nodes - phase2_nodes}")
    if check:
        if 3*optimal_ivs < len(phase2_nodes):
            print(f"Phase II bound: {3*optimal_ivs}, used={len(phase2_nodes)}")
            print("------------ BROKEN -------------")

    return intervened_nodes


if __name__ == '__main__':
    import networkx as nx

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
    nct, ncg, extra_edges = apply_clique_intervention(ct, None, cg, iv_clique, dcg, verbose=False)
    print(nct.undirected)




