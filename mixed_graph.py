from collections import defaultdict
import networkx as nx


class LabelledMixedGraph:
    def __init__(self, nodes=set(), directed=dict(), undirected=dict(), bidirected=dict()):
        self._nodes = set(nodes)
        self._directed = {(i, j): label for (i, j), label in directed.items()}
        self._bidirected = {frozenset({i, j}): label for (i, j), label in bidirected.items()}
        self._undirected = {frozenset({i, j}): label for (i, j), label in undirected.items()}

        self._neighbors = defaultdict(set)
        self._spouses = defaultdict(set)
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._adjacent = defaultdict(set)

        for i, j in self._directed.keys():
            self._children[i].add(j)
            self._parents[j].add(i)
            self._nodes.add(i)
            self._nodes.add(j)
        for i, j in self._bidirected.keys():
            self._spouses[j].add(i)
            self._spouses[i].add(j)
            self._nodes.add(i)
            self._nodes.add(j)
        for i, j in self._undirected.keys():
            self._neighbors[i].add(j)
            self._neighbors[j].add(i)
            self._nodes.add(i)
            self._nodes.add(j)
        for node in self._nodes:
            self._adjacent[node] = self._children[node] | self._parents[node] | self._spouses[node] | self._neighbors[node]

    # === BUILT-INS
    @property
    def __str__(self):
        s = ""
        s += f"Directed edges: {self._directed}\n"
        s += f"Undirected edges: {self._undirected}\n"
        s += f"Bidirected edges: {self._bidirected}\n"
        return s

    @property
    def __repr__(self):
        return str(self)

    def copy(self):
        return LabelledMixedGraph(
            nodes=self._nodes,
            directed=self._directed,
            bidirected=self._bidirected,
            undirected=self._undirected
        )

    # === PROPERTIES
    @property
    def nodes(self):
        return set(self._nodes)

    @property
    def directed(self):
        return dict(self._directed)

    @property
    def undirected(self):
        return dict(self._undirected)

    @property
    def bidirected(self):
        return dict(self._bidirected)

    @property
    def num_directed(self):
        return len(self._directed)

    @property
    def num_undirected(self):
        return len(self._undirected)

    @property
    def num_bidirected(self):
        return len(self._bidirected)

    # === CONVERTERS
    @classmethod
    def from_nx(cls, nx_graph):
        if isinstance(nx_graph, nx.MultiDiGraph):
            directed = {(i, j): nx_graph.get_edge_data(i, j)[0]['label'] for i, j in nx_graph.edges()}
            bidirected_keys = set(directed.keys()) & {(j, i) for i, j in directed}
            bidirected = {(i, j): directed[(i, j)] for i, j in bidirected_keys}
            directed = {
                (i, j): val for (i, j), val in directed.items()
                if (i, j) not in bidirected_keys and (j, i) not in bidirected_keys
            }
            return LabelledMixedGraph(nodes=nx_graph.nodes(), directed=directed, bidirected=bidirected)
        if isinstance(nx_graph, nx.Graph):
            undirected = {frozenset({i, j}): nx_graph.get_edge_data(i, j)['label'] for i, j in nx_graph.edges()}
            return LabelledMixedGraph(nodes=nx_graph.nodes(), undirected=undirected)

    def to_nx(self) -> nx.Graph:
        if self._bidirected or self._directed:
            raise ValueError("Can only convert if the graph has only undirected edges")
        nx_graph = nx.Graph()
        nx_graph.add_nodes_from(self._nodes)
        nx_graph.add_edges_from(self._undirected.keys())
        nx.set_edge_attributes(nx_graph, self._undirected, 'label')
        return nx_graph

    def induced_graph(self, nodes):
        return LabelledMixedGraph(
            nodes,
            directed={(i, j): val for (i, j), val in self._directed.items() if i in nodes and j in nodes},
            bidirected={(i, j): val for (i, j), val in self._bidirected.items() if i in nodes and j in nodes},
            undirected={(i, j): val for (i, j), val in self._undirected.items() if i in nodes and j in nodes},
        )

    def to_undirected(self):
        edges = {
            **self._undirected,
            **self._bidirected,
            **{frozenset({i, j}): label for (i, j), label in self._directed.items()}
        }
        return LabelledMixedGraph(nodes=self._nodes, undirected=edges)

    # === PREDICATES
    def has_directed(self, i, j):
        return (i, j) in self._directed

    def has_bidirected(self, edge):
        return frozenset(*edge) in self._bidirected

    def has_undirected(self, edge):
        return frozenset(*edge) in self._undirected

    # === NODE-WISE SETS
    def indegree_of(self, node):
        return len(self._parents[node])

    def outdegree_of(self, node):
        return len(self._children[node])

    def spouse_degree_of(self, node):
        return len(self._spouses[node])

    def neighbor_degree_of(self, node):
        return len(self._neighbors[node])

    def neighbors_of(self, node):
        return set(self._neighbors[node])

    def spouses_of(self, node):
        return set(self._spouses[node])

    def children_of(self, node):
        return set(self._children[node])

    def parents_of(self, node):
        return set(self._parents[node])

    def adjacent_to(self, node):
        return set(self._adjacent[node])

    def onto_edges(self, node):
        directed_onto = {(p, node): self._directed[(p, node)] for p in self._parents[node]}
        bidirected_onto = {(s, node): self._bidirected[frozenset({s, node})] for s in self._spouses[node]}
        return {**directed_onto, **bidirected_onto}

    # === EDGE FUNCTIONALS
    def get_label(self, edge, ignore_error=True):
        try:
            label = self._directed.get(edge)
            if label: return label
            label = self._directed.get(tuple(reversed(edge)))
            if label: return label
            label = self._undirected.get(frozenset({*edge}))
            if label: return label
            label = self._bidirected[frozenset({*edge})]
            if label: return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def directed_edges_with_label(self, label):
        return {edge for edge, l in self._directed.items() if l == label}

    def undirected_edges_with_label(self, label):
        return {edge for edge, l in self._undirected.items() if l == label}

    def bidirected_edges_with_label(self, label):
        return {edge for edge, l in self._bidirected.items() if l == label}

    # === ADDERS
    def add_directed(self, i, j, label):
        self._directed[(i, j)] = label
        self._parents[j].add(i)
        self._children[i].add(j)

    def add_bidirected(self, i, j, label):
        self._bidirected[frozenset({i, j})] = label
        self._spouses[j].add(i)
        self._spouses[i].add(j)

    # === REMOVERS
    def remove_node(self, i):
        self._nodes.remove(i)

        for parent in self._parents[i]:
            self._children[parent].remove(i)
        del self._parents[i]

        for child in self._children[i]:
            self._parents[child].remove(i)
        del self._children[i]

        for spouse in self._spouses[i]:
            self._spouses[spouse].remove(i)
        del self._spouses[i]

        for nbr in self._neighbors[i]:
            self._neighbors[nbr].remove(i)
        del self._neighbors[i]

        for adj in self._adjacent[i]:
            self._adjacent[adj].remove(i)
        del self._adjacent[i]

        self._directed = {(j, k): val for (j, k), val in self._directed.items() if i != j and i != k}
        self._bidirected = {(j, k): val for (j, k), val in self._bidirected.items() if i != j and i != k}
        self._undirected = {(j, k): val for (j, k), val in self._undirected.items() if i != j and i != k}

    def remove_directed(self, i, j, ignore_error=True):
        try:
            label = self._directed.pop((i, j))
            self._parents[j].remove(i)
            self._children[i].remove(j)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_bidirected(self, i, j, ignore_error=True):
        try:
            label = self._bidirected.pop(frozenset({(i, j)}))
            self._spouses[i].remove(j)
            self._spouses[j].remove(i)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_undirected(self, i, j, ignore_error=True):
        try:
            label = self._undirected.pop(frozenset({i, j}))
            self._neighbors[i].remove(j)
            self._neighbors[j].remove(i)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    # === MUTATORS
    def to_directed(self, i, j):
        label = self.remove_bidirected(i, j)
        label = self.remove_undirected(i, j) if label is None else label
        self.add_directed(i, j, label)

    def to_bidirected(self, i, j):
        label = self.remove_undirected(i, j)
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        self.add_bidirected(i, j, label)





