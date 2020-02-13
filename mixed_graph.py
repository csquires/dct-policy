from collections import defaultdict
import networkx as nx
from typing import Union


class LabelledMixedGraph:
    def __init__(self, nodes=set(), directed=dict(), undirected=dict(), bidirected=dict(), semidirected=dict()):
        self._nodes = set(nodes)
        self._directed = {(i, j): label for (i, j), label in directed.items()}
        self._bidirected = {frozenset({i, j}): label for (i, j), label in bidirected.items()}
        self._undirected = {frozenset({i, j}): label for (i, j), label in undirected.items()}
        self._semidirected = {(i, j): label for (i, j), label in semidirected.items()}

        self._neighbors = defaultdict(set)
        self._spouses = defaultdict(set)
        self._parents = defaultdict(set)
        self._children = defaultdict(set)
        self._semiparents = defaultdict(set)
        self._semichildren = defaultdict(set)

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
        for i, j in self._semidirected.keys():
            self._semichildren[i].add(j)
            self._semiparents[j].add(i)
            self._nodes.add(i)
            self._nodes.add(j)

    # === BUILT-INS
    def __str__(self):
        s = ""
        s += f"Directed edges: {self._directed}\n"
        s += f"Undirected edges: {self._undirected}\n"
        s += f"Bidirected edges: {self._bidirected}\n"
        s += f"Semidirected edges: {self._semidirected}\n"
        return s

    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        same_dir = self._directed == other._directed
        same_bidir = self._bidirected == other._bidirected
        same_undir = self._undirected == other._undirected
        same_semidir = self._semidirected == other._semidirected
        return same_dir and same_bidir and same_undir and same_semidir

    def copy(self):
        return LabelledMixedGraph(
            nodes=self._nodes,
            directed=self._directed,
            bidirected=self._bidirected,
            undirected=self._undirected,
            semidirected=self._semidirected
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
    def semidirected(self):
        return dict(self._semidirected)

    @property
    def directed_keys(self):
        return set(self._directed.keys())

    @property
    def bidirected_keys(self):
        return set(self._bidirected.keys())

    @property
    def undirected_keys(self):
        return set(self._undirected.keys())

    @property
    def semidirected_keys(self):
        return set(self._semidirected.keys())

    @property
    def nnodes(self):
        return len(self._nodes)

    @property
    def num_directed(self):
        return len(self._directed)

    @property
    def num_undirected(self):
        return len(self._undirected)

    @property
    def num_bidirected(self):
        return len(self._bidirected)

    @property
    def num_semidirected(self):
        return len(self._semidirected)

    @property
    def num_edges(self):
        return self.num_bidirected + self.num_directed + self.num_undirected + self.num_semidirected

    # === CONVERTERS
    @classmethod
    def from_nx(cls, nx_graph):
        """
        Create a LabelledMixedGraph from a networkx graph with labelled edges.
        """
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

    def to_nx(self) -> Union[nx.Graph, nx.DiGraph]:
        """
        Return a networkx graph. If the current graph has no undirected edges, return a DiGraph.
        If it has no directed or undirected edges, return an undirected Graph.
        """
        if not self._undirected:
            nx_graph = nx.DiGraph()
            nx_graph.add_nodes_from(self._nodes)
            nx_graph.add_edges_from(self._directed.keys())
            bidirected = {(i, j) for i, j in self._bidirected.keys()}
            nx_graph.add_edges_from(bidirected | {(j, i) for i, j in bidirected})
            nx.set_edge_attributes(nx_graph, self._directed, name='label')
            nx.set_edge_attributes(nx_graph, self._bidirected, name='label')
            nx.set_edge_attributes(nx_graph, {(j, i): l for (i, j), l in self.bidirected.items()}, name='label')
            return nx_graph
        if not self._directed and not self._bidirected:
            nx_graph = nx.Graph()
            nx_graph.add_nodes_from(self._nodes)
            nx_graph.add_edges_from(self._undirected.keys())
            nx.set_edge_attributes(nx_graph, self._undirected, 'label')
            return nx_graph
        else:
            raise ValueError("Can only convert if the graph has only undirected edges or no undirected edges")

    def induced_graph(self, nodes):
        """Return the induced subgraph of this graph over `nodes`."""
        return LabelledMixedGraph(
            nodes,
            directed={(i, j): val for (i, j), val in self._directed.items() if i in nodes and j in nodes},
            bidirected={(i, j): val for (i, j), val in self._bidirected.items() if i in nodes and j in nodes},
            undirected={(i, j): val for (i, j), val in self._undirected.items() if i in nodes and j in nodes},
            semidirected={(i, j): val for (i, j), val in self._semidirected.items() if i in nodes and j in nodes}
        )

    def undirected_copy(self):
        """Return a copy of this graph with all edges undirected."""
        edges = {
            **self._undirected,
            **self._bidirected,
            **{frozenset({i, j}): label for (i, j), label in self._directed.items()},
            **{frozenset({i, j}): label for (i, j), label in self._semidirected.items()}
        }
        return LabelledMixedGraph(nodes=self._nodes, undirected=edges)

    # === PREDICATES
    def has_directed(self, i, j):
        """Check if this graph has the directed edge i->j."""
        return (i, j) in self._directed

    def has_bidirected(self, edge):
        """Check if this graph has the bidirected edge `edge`."""
        return frozenset({*edge}) in self._bidirected

    def has_undirected(self, edge):
        """Check if this graph has the undirected edge `edge`."""
        return frozenset({*edge}) in self._undirected

    def has_semidirected(self, i, j):
        """Check if this graph has the semidirected edge i*->j."""
        return (i, j) in self._semidirected

    def has_any_edge(self, edge):
        """Check if this graph has any edge `edge`."""
        i, j = edge
        return self.has_directed(i, j) or self.has_directed(j, i) or self.has_bidirected(edge) \
            or self.has_undirected(edge) or self.has_semidirected(i, j) or self.has_semidirected(j, i)

    # === NODE-WISE SETS
    def indegree_of(self, node):
        """Return the number of parents of a node"""
        return len(self._parents[node])

    def outdegree_of(self, node):
        """Return the number of children of a node"""
        return len(self._children[node])

    def spouse_degree_of(self, node):
        """Return the number of spouses of a node"""
        return len(self._spouses[node])

    def neighbor_degree_of(self, node):
        """Return the number of neighbors of a node"""
        return len(self._neighbors[node])

    def semi_indegree_of(self, node):
        """Return the number of semi-parents of a node"""
        return len(self._parents[node])

    def semi_outdegree_of(self, node):
        """Return the number of semi-children of a node"""
        return len(self._children[node])

    def neighbors_of(self, node):
        return set(self._neighbors[node])

    def spouses_of(self, node):
        return set(self._spouses[node])

    def children_of(self, node):
        return set(self._children[node])

    def parents_of(self, node):
        return set(self._parents[node])

    def semichildren_of(self, node):
        return set(self._semichildren[node])

    def semiparents_of(self, node):
        return set(self._semiparents[node])

    def adjacent_to(self, node):
        """Return all nodes adjacent to `node`."""
        return self.parents_of(node) | self.children_of(node) | self.spouses_of(node) | self.neighbors_of(node) \
            | self.semiparents_of(node) | self.semichildren_of(node)

    def onto_edges(self, node):
        """
        Return all edges with an arrowhead at `node`.
        """
        directed_onto = {(p, node): self._directed[(p, node)] for p in self._parents[node]}
        bidirected_onto = {(s, node): self._bidirected[frozenset({s, node})] for s in self._spouses[node]}
        semi_directed_onto = {(p, node): self._semidirected[(p, node)] for p in self._semiparents[node]}
        return {**directed_onto, **bidirected_onto, **semi_directed_onto}

    def onto_nodes(self, node):
        """Return all parents, spouses, and semiparents of `node`."""
        return {*self._parents[node], *self._spouses[node], *self._semiparents[node]}

    # === EDGE FUNCTIONALS
    def get_label(self, edge, ignore_error=True):
        i, j = edge
        label = self._directed.get((i, j))
        if label: return label
        label = self._directed.get((j, i))
        if label: return label
        label = self._undirected.get(frozenset({*edge}))
        if label: return label
        label = self._bidirected.get(frozenset({*edge}))
        if label: return label
        label = self._semidirected.get((i, j))
        if label: return label
        label = self._semidirected.get((j, i))
        if label: return label

        if not ignore_error:
            raise KeyError(f"No edge {edge}")

    def directed_edges_with_label(self, label):
        return {edge for edge, l in self._directed.items() if l == label}

    def undirected_edges_with_label(self, label):
        return {edge for edge, l in self._undirected.items() if l == label}

    def bidirected_edges_with_label(self, label):
        return {edge for edge, l in self._bidirected.items() if l == label}

    def semidirected_edges_with_label(self, label):
        return {edge for edge, l in self._semidirected.items() if l == label}

    # === ADDERS
    def add_directed(self, i, j, label):
        """Add i->j with label `label` to this graph."""
        self._directed[(i, j)] = label
        self._parents[j].add(i)
        self._children[i].add(j)

    def add_bidirected(self, i, j, label):
        """Add i<->j with label `label` to this graph."""
        self._bidirected[frozenset({i, j})] = label
        self._spouses[j].add(i)
        self._spouses[i].add(j)

    def add_undirected(self, i, j, label):
        """Add i-j with label `label` to this graph."""
        self._undirected[frozenset({i, j})] = label
        self._neighbors[j].add(i)
        self._neighbors[i].add(j)

    def add_semidirected(self, i, j, label):
        """Add i*->j with label `label` to this graph."""
        self._semidirected[(i, j)] = label
        self._semiparents[j].add(i)
        self._semichildren[i].add(j)

    # === REMOVERS
    def remove_node(self, i):
        """
        Remove the node `i`, and all incident edges, from this graph.
        """
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

        for sp in self._semiparents[i]:
            self._semichildren[sp].remove(i)
        del self._semiparents[i]

        for sc in self._semichildren[i]:
            self._semiparents[sc].remove(i)
        del self._semichildren[i]

        self._directed = {(j, k): val for (j, k), val in self._directed.items() if i != j and i != k}
        self._bidirected = {frozenset({j, k}): val for (j, k), val in self._bidirected.items() if i != j and i != k}
        self._undirected = {frozenset({j, k}): val for (j, k), val in self._undirected.items() if i != j and i != k}
        self._semidirected = {(j, k): val for (j, k), val in self._semidirected.items() if i != j and i != k}

    def remove_directed(self, i, j, ignore_error=True):
        """Remove a directed edge i->j from this graph."""
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
        """Remove a bidirected edge i<->j from this graph."""
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
        """Remove an undirected edge i-j from this graph."""
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

    def remove_semidirected(self, i, j, ignore_error=True):
        """Remove a semidirected edge i*->j from this graph."""
        try:
            label = self._semidirected.pop((i, j))
            self._semiparents[j].remove(i)
            self._semichildren[i].remove(j)
            return label
        except KeyError as e:
            if ignore_error:
                pass
            else:
                raise e

    def remove_edge(self, i, j, ignore_error=True):
        """
        Remove the edge between i and j in this graph, if any exists.
        """
        label = self.remove_directed(i, j)
        if label: return label
        label = self.remove_directed(j, i)
        if label: return label
        label = self.remove_bidirected(i, j)
        if label: return label
        label = self.remove_undirected(i, j)
        if label: return label
        label = self.remove_semidirected(i, j)
        if label: return label
        label = self.remove_semidirected(j, i)
        if label: return label

        if not label and not ignore_error:
            raise KeyError("i-j is not an edge in this graph")

    def remove_edges(self, edges, ignore_error=True):
        """
        Remove all edges in `edges` from this graph.
        """
        for edge in edges:
            self.remove_edge(*edge, ignore_error=ignore_error)

    def remove_all_directed(self):
        """Remove all directed edges from this graph."""
        for i, j in self._directed:
            self._parents[j].remove(i)
            self._children[i].remove(j)
        self._directed = defaultdict(set)

    def remove_all_bidirected(self):
        """Remove all bidirected edges from this graph."""
        for i, j in self._bidirected:
            self._spouses[i].remove(j)
            self._spouses[j].remove(i)
        self._bidirected = defaultdict(set)

    def remove_all_undirected(self):
        """Remove all undirected edges from this graph."""
        for i, j in self._undirected:
            self._neighbors[i].remove(j)
            self._neighbors[j].remove(i)
        self._undirected = defaultdict(set)

    def remove_all_semidirected(self):
        """Remove all semidirected edges from this graph."""
        for i, j in self._semidirected:
            self._semiparents[j].remove(i)
            self._semichildren[i].remove(j)
        self._semidirected = defaultdict(set)

    # === MUTATORS
    def to_directed(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i->j"""
        label = self.remove_bidirected(i, j)
        label = self.remove_undirected(i, j) if label is None else label
        label = self.remove_semidirected(i, j) if label is None else label
        label = self.remove_semidirected(j, i) if label is None else label
        if label or not check_exists:
            self.add_directed(i, j, label)

    def to_bidirected(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i<->j"""
        label = self.remove_undirected(i, j)
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        label = self.remove_semidirected(i, j) if label is None else label
        label = self.remove_semidirected(j, i) if label is None else label
        if label or not check_exists:
            self.add_bidirected(i, j, label)

    def to_undirected(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i-j"""
        label = self.remove_bidirected(i, j)
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        label = self.remove_semidirected(i, j) if label is None else label
        label = self.remove_semidirected(j, i) if label is None else label
        if label or not check_exists:
            self.add_undirected(i, j, label)

    def to_semidirected(self, i, j, check_exists=True):
        """Replace the edge between i and j, if any exists, with i o-> j"""
        label = self.remove_undirected(i, j)
        label = self.remove_bidirected(i, j) if label is None else label
        label = self.remove_directed(i, j) if label is None else label
        label = self.remove_directed(j, i) if label is None else label
        if label or not check_exists:
            self.add_semidirected(i, j, label)

    def all_to_undirected(self):
        """
        Change all edges in this graph into undirected edges.
        """
        self._undirected.update({frozenset({i, j}): label for (i, j), label in self._directed.items()})
        self._undirected.update({frozenset({i, j}): label for (i, j), label in self._bidirected.items()})
        self._undirected.update({frozenset({i, j}): label for (i, j), label in self._semidirected.items()})
        self.remove_all_directed()
        self.remove_all_bidirected()
        self.remove_all_semidirected()
