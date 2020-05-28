import causaldag as cd
import utils
import itertools as itr
from scipy.special import factorial
import numpy as np
import networkx as nx
from fractions import Fraction
from directed_chordal_utils import random_chordal_graph2
from functools import lru_cache


def make_adversarial2(p):
    arcs = set(itr.combinations(range(p), 2))
    for j in range(0, p-2):
        arcs.update({(k, p+j) for k in range(j+2)})
        arcs.add((p+j, j+2))
    i = 2*p-2
    arcs.add((0, i))
    arcs.update({(i, p+j) for j in range(p-2)})
    return cd.DAG(arcs=arcs)


def make_clique_star(clique_size, star_size):
    edges = set(itr.combinations(range(clique_size), 2))
    edges.update({(clique_size-1, j) for j in range(clique_size, clique_size+star_size)})
    return cd.PDAG(edges=edges)


def make_adversarial(clique_size, directed=False):
    p = int(np.ceil(np.log2(clique_size)))
    nodes = set(range(clique_size + p))
    edges = set(itr.combinations(range(clique_size), 2))

    # attach auxiliary node i to main nodes with a 1 in their binary string after position i
    for i in range(p):
        edges.update({(clique_size + i, j) for j in range(clique_size - 2**i)})

    # make a clique between auxiliary nodes
    edges.update({(clique_size + i, clique_size + j) for i, j in itr.combinations(range(p), 2)})

    if directed:
        return cd.DAG(nodes=nodes, arcs=edges)
    return cd.PDAG(nodes=nodes, edges=edges)


def make_fuzzy_adversarial(p, fuzz):
    edges = set(itr.combinations(range(2**p), 2))
    for i in range(p):
        node = 2**p + i*(fuzz+1)
        edges.update({(node, j) for j in range(2**p) if bin(j).zfill(p)[-i-1] == '1'})
        edges.update({(node, node + 1 + j) for j in range(fuzz)})
    return cd.PDAG(edges=edges)


def largest_clique(graph):
    return max(len(c) for c in nx.chordal_graph_cliques(graph))


def to_nx(g):
    return nx.Graph(list(g.edges))


def to_essgraph(nx_graph):
    return cd.PDAG(nodes=set(nx_graph.nodes), edges=set(nx_graph.edges))


def dominated_nodes(essgraph):
    node2nbrs = essgraph.undirected_neighbors
    dom_nodes = {
        node
        for node, nbrs in node2nbrs.items()
        if (len(nbrs) == 1 and len(node2nbrs[nbrs.pop()]) != 1) or len(nbrs) == 0
    }

    # dom_nodes = set()
    # for node, nbrs in node2nbrs.items():
    #     if any(nbrs < node2nbrs[nbr] for nbr in nbrs) or not nbrs:
    #         dom_nodes.add(node)

    return dom_nodes


def is_clique(essgraph, s):
    return all(essgraph.has_edge(i, j) for i, j in itr.combinations(s, 2))


def hash_essgraph(essgraph):
    return hash(frozenset(essgraph._arcs)), hash(frozenset(essgraph._edges))


def optimal_policy(essgraph, essgraph2opt, use_clique=True, rename=True):
    """Returns (v*, expected_time, num_dags)"""
    if len(essgraph.edges) == 0:
        return None, 0, 1
    if len(essgraph.edges) == 1:
        return None, 1, 2

    chain_components = essgraph.chain_components(rename=rename)
    if len(chain_components) > 1:
        results = [optimal_policy(component, essgraph2opt, use_clique=use_clique, rename=rename) for component in chain_components]
        interventions, times, ndags_list = zip(*results)
        return None, sum(times), np.prod(ndags_list)

    if use_clique:
        adjacent_to_undirected = {v for v, nbrs in essgraph.undirected_neighbors.items() if nbrs}
        if is_clique(essgraph, adjacent_to_undirected):
            # print('clique')
            return None, comp_tp_closed(len(adjacent_to_undirected)), factorial(len(adjacent_to_undirected))

    if hash_essgraph(essgraph) in essgraph2opt:
        return essgraph2opt[hash_essgraph(essgraph)]

    v_opt = None
    time_opt = float('inf')

    # print(len(essgraph.nodes) - len(dominated_nodes(essgraph)), '/', len(essgraph.nodes))
    for v in essgraph.nodes - dominated_nodes(essgraph):
        total_time = 0
        ndags = 0
        for c in utils.powerset_predicate(essgraph.undirected_neighbors[v], lambda s: is_clique(essgraph, s)):
            # === NEW ESSGRAPH
            essgraph_iv = essgraph.copy()
            # print(essgraph_iv.arcs, essgraph_iv.edges)
            # print(v, c)
            essgraph_iv.assign_parents(v, c)

            # === UPDATE
            v_next, exp_time_next, ndags_next = optimal_policy(essgraph_iv, essgraph2opt, use_clique=use_clique, rename=rename)
            ndags += ndags_next
            total_time += exp_time_next * ndags_next

        exp_time = total_time/ndags + 1
        v_opt, time_opt = (v, exp_time) if exp_time < time_opt else (v_opt, time_opt)

    essgraph2opt[hash_essgraph(essgraph)] = v_opt, time_opt, ndags
    # print(len(essgraph2opt))

    return v_opt, time_opt, ndags


def comp_tp(p):
    if p == 0 or p == 1: return 0
    return 1 + 1/p*sum(comp_tp(k) + comp_tp(p-1-k) for k in range(p))


def comp_tp_closed(p):
    return 2/3*p - 1/3


@lru_cache(maxsize=30)
def comp_exp_downstream(p):
    if p == 1: return 0
    return 1 + Fraction(1, p)*sum(comp_exp_downstream(k) for k in range(1, p))


if __name__ == '__main__':
    import time
    import random
    import line_profiler
    profiler = line_profiler.LineProfiler()

    profiler.add_function(optimal_policy)

    random.seed(1729)

    EPSILON = 1e-6
    start = time.time()

    nnodes = 12
    ngraphs = 20
    graphs_nx = random_chordal_graph2(nnodes, 2, ngraphs)
    # graphs_nx = random_chordal_graph2(nnodes, 2, ngraphs) + random_chordal_graph2(nnodes, 3, ngraphs)
    largest_cliques = np.array([largest_clique(g) for g in graphs_nx])
    cpdags = [to_essgraph(g) for g in graphs_nx]

    # dag_in_degrees = np.array([[dag.indegree(node) for node in dag.nodes] for dag in dags])
    # dag_out_degrees = np.array([[dag.outdegree(node) for node in dag.nodes] for dag in dags])

    # print(dag_in_degrees.mean())
    # print(dag_in_degrees.max())
    # print([dag.sparsity for dag in dags])

    # dags = cd.rand.directed_erdos(nnodes, .9, ngraphs)
    # cpdags = [d.cpdag() for d in dags]
    #
    # print('Computing graph times')
    # ts = []
    # num_distinct = []
    #
    # def run_optimal():
    #     for cpdag in tqdm(cpdags):
    #         essgraph2opt = dict()
    #         _, t, ndags = optimal_policy(cpdag, essgraph2opt, use_clique=True, rename=True)
    #         num_distinct.append(len(essgraph2opt))
    #         ts.append(t)
    #
    # profiler.add_function(run_optimal)
    # p = profiler(run_optimal)
    # p()
    # profiler.print_stats()
    #
    # num_distinct = np.array(num_distinct)
    # ts = np.array(ts) + EPSILON
    #
    # lc_times = largest_cliques*2/3 - 1/3
    # print(lc_times <= ts)
    # print(np.where(lc_times > ts))
    # print("Total wall clock time:", time.time() - start)
    # print("Average number of interventions:", ts.mean())
    # print("std number of interventions:", ts.std())
    #
    # print("Average number of essgraphs:", num_distinct.mean())


    # print([comp_tp(p) for p in range(10)])

    # g = make_adversarial(9)
    # print(nx.is_chordal(to_nx(g)))
    # _, t, ndags = optimal_policy(g, dict(), use_clique=True)
    # print(t)
    # print(largest_clique(to_nx(g))*2/3 - 1/3)

    # print('Computing subgraph times')
    # subgraphs = [g.induced_subgraph(set(random.sample(g.nodes, random.randint(3, nnodes-1)))) for g in cpdags]
    #
    # subgraph_times = []
    # for subgraph in tqdm(subgraphs):
    #     subgraph_times.append(optimal_policy(subgraph, dict())[1])
    # subgraph_times = np.array(subgraph_times)
    #
    # print('All less?', all(subgraph_times <= ts))
    #
    # print([len(cpdag.nodes) for cpdag in cpdags])
    # print([len(cpdag.edges) for cpdag in cpdags])
    # print(np.where(subgraph_times > ts))
    #
    # g = cd.PDAG(edges={(0, 1), (0, 4), (2, 4), (3, 4)})
    # print(optimal_policy(g, {})[1])
    # sub_g = cd.PDAG(edges={(0, 1), (0, 4), (2, 4)})
    # print(optimal_policy(sub_g, {})[1])

    # clique_size = 4
    # clique_star = make_clique_star(clique_size, 20)
    # print(optimal_policy(clique_star, {})[1])
    # print(clique_size*2/3 - 1/3)
    #
    # p = 2
    # fuzz = 10
    # clique_size = 2**p
    # g = make_fuzzy_adversarial(p, fuzz)
    # v, t, ndags = optimal_policy(g, {})
    # print(t)
    # print(clique_size*2/3 - 1/3)

    # arcs = set(itr.combinations(range(4), 2))
    # arcs.update({(0, 4), (4, 1)})
    # arcs.update({(1, 5), (5, 2)})
    # arcs.update({(2, 6), (6, 3)})
    # arcs.update({(0, 6)})
    # arcs.update({(0, 5)})
    # arcs.update({(1, 6)})
    # d = cd.DAG(arcs=arcs)
    # g = d.cpdag()
    # print(nx.is_chordal(to_nx(g)))

    p = 4
    # d = make_adversarial(p, directed=True)
    d = make_adversarial2(p)
    print('v-structures:', d.vstructures())
    c = d.cpdag()
    # ivs = [{p+i} for i in range(int(np.ceil(np.log2(p))))]
    # print('interventions:', ivs)
    # g = d.interventional_cpdag(ivs, cpdag=c)
    # print('edges:', g.edges)

    # arcs = set(itr.combinations(range(4), 2))
    # arcs.update({(0, 4), (0, 5), (0, 6)})
    # arcs.update({(1, 6), (5, 1), (6, 2), (4, 5), (4, 6), (4, 1)})
    # d = cd.DAG(arcs=arcs)
    # g = d.interventional_cpdag([frozenset({4})], d.cpdag())
    #
    # import matplotlib.pyplot as plt
    # ps = np.arange(3, 25)
    # ts = np.array([comp_exp_downstream(i) for i in ps])
    # plt.clf()
    # plt.plot(ps, ts, label='interventions')
    # plt.plot(ps, np.log(ps), label='log')
    # plt.plot(ps, ts - np.log(ps), label='diff')
    # plt.legend()
    # plt.show()

    arcs = set(itr.combinations(range(5), 2))
    arcs.update(set(itr.combinations({0, 3, 4, 5}, 2)))
    arcs.update(set(itr.combinations({0, 3, 5, 6}, 2)))
    arcs.update(set(itr.combinations({3, 4, 5, 7}, 2)))
    d = cd.DAG(arcs=arcs)
    g = d.cpdag()
    gn = to_nx(g)
    cs = nx.chordal_graph_cliques(gn)


