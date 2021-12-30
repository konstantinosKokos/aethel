from typing import Generic, TypeVar, Any, Iterator, Callable, overload
from functools import reduce

Node = TypeVar('Node')
Nodes = set[Node]
NodeIterator = Iterator[Node]


class Edge(Generic[Node]):
    __match_args__ = ('source', 'target', 'label')

    def __init__(self, source: Node, target: Node, label: str) -> None:
        self.source = source
        self.target = target
        self.label = label

    def __repr__(self):
        return f'{self.source}:{self.label}:{self.target}'

    def __iter__(self):
        yield self.source
        yield self.target
        yield self.label

    def __eq__(self, other: Any) -> bool:
        return tuple(self) == tuple(other)

    def __hash__(self) -> int:
        return hash(tuple(self))


Edges = set[Edge]
Path = tuple[Edge, ...]
Paths = set[Path]


class DAG(Generic[Node]):
    def __init__(self, nodes: Nodes, edges: Edges, attribs: dict[Node, dict[str, str]], meta: Any):
        self.nodes = nodes
        self.edges = edges
        self.attribs = attribs
        self.meta = meta

    def __len__(self) -> int:
        return len(self.nodes)

    @overload
    def get(self, node: Node, attr: str) -> str | None: ...
    @overload
    def get(self, node: Node) -> dict[str, str]: ...

    def get(self, node, attr=None):
        if attr is None:
            return self.attribs[node]
        return self.attribs[node][attr] if attr in self.attribs[node] else None

    @overload
    def set(self, node: Node, attr: dict[str, str]) -> None: ...
    @overload
    def set(self, node: Node, attr: str, value: str) -> None: ...

    def set(self, node, attr, value=None):
        if value is None:
            self.attribs[node].update(attr)
        else:
            self.attribs[node][attr] = value

    def is_empty(self):
        return len(self.nodes) == 0

    def occurring_nodes(self) -> Nodes:
        return {node for edge in self.edges for node in (edge.source, edge.target)}

    def incoming_edges(self, node: Node) -> Edges:
        return {edge for edge in self.edges if edge.target == node}

    def outgoing_edges(self, node: Node) -> Edges:
        return {edge for edge in self.edges if edge.source == node}

    def get_roots(self) -> Nodes:
        return {node for node in self.nodes if not self.incoming_edges(node)}

    def get_leaves(self) -> Nodes:
        return {node for node in self.nodes if self.is_leaf(node)}

    def is_leaf(self, node: Node) -> bool:
        return not self.outgoing_edges(node)

    def parents(self, node: Node) -> Nodes:
        return {edge.source for edge in self.incoming_edges(node)}

    def children(self, node: Node) -> Nodes:
        return {edge.target for edge in self.outgoing_edges(node)}

    def successors(self, node: Node) -> NodeIterator:
        yield from (children := self.children(node))
        for child in children:
            yield from self.successors(child)

    def predecessors(self, node: Node) -> NodeIterator:
        yield from (parents := self.parents(node))
        for parent in parents:
            yield from self.predecessors(parent)

    def is_reachable(self, node: Node, target: Node) -> bool:
        return target == node or target in self.successors(node)

    def first_common_predecessor(self, *nodes: Node) -> Node | None:
        match nodes:
            case []:
                return None
            case [n]:
                return n
            case [n1, n2]:
                return next((n for n in (n1, *self.predecessors(n1)) if self.is_reachable(n, n2)), None)
            case [n1, n2, *ns]:
                return self.first_common_predecessor(self.first_common_predecessor(n1, n2), *ns)

    def remove_node(self, node: Node) -> 'DAG[Node]':
        self.nodes.remove(node)
        self.edges = {edge for edge in self.edges if edge.source != node and edge.target != node}
        self.attribs.pop(node)
        return self

    def remove_edge(self, edge: Edge) -> 'DAG[Node]':
        self.edges.remove(edge)
        self.nodes = self.occurring_nodes()
        self.attribs = {node: attrib for node, attrib in self.attribs.items() if node in self.nodes}
        return self

    @overload
    def remove_nodes(self, to_remove: Nodes) -> 'DAG[Node]': ...
    @overload
    def remove_nodes(self, to_remove: Callable[[Node], bool]) -> 'DAG[Node]': ...

    def remove_nodes(self, to_remove):
        if callable(to_remove):
            to_remove = {node for node in self.nodes if to_remove(node)}
        self.nodes -= to_remove
        self.edges = {edge for edge in self.edges if edge.source in self.nodes and edge.target in self.nodes}
        self.attribs = {node: attrib for node, attrib in self.attribs.items() if node in self.nodes}
        return self

    @overload
    def remove_edges(self, to_remove: Edges) -> 'DAG[Node]': ...
    @overload
    def remove_edges(self, to_remove: Callable[[Edge], bool]) -> 'DAG[Node]': ...

    def remove_edges(self, to_remove):
        if callable(to_remove):
            to_remove = {edge for edge in self.edges if to_remove(edge)}
        self.edges -= to_remove
        self.nodes = self.occurring_nodes()
        self.attribs = {node: attrib for node, attrib in self.attribs.items() if node in self.nodes}
        return self

    def oneway(self, node: Node) -> bool:
        return len(self.outgoing_edges(node)) == 1 and len(self.incoming_edges(node)) == 1

    def remove_oneway(self, node: Node) -> 'DAG[Node]':
        incoming = next(iter(self.incoming_edges(node)))
        outgoing = next(iter(self.outgoing_edges(node)))
        self.edges.add(Edge(incoming.source, outgoing.target, incoming.label))
        self.remove_node(node)
        return self

    def remove_oneways(self) -> 'DAG[Node]':
        return reduce(lambda dag, node: dag.remove_oneway(node), filter(self.oneway, self.nodes), self)

    def get_rooted_subgraph(self, root: Node) -> 'DAG[Node]':
        return DAG(reachable := {node for node in self.nodes if self.is_reachable(root, node)},
                   {edge for edge in self.edges if edge.source in reachable and edge.target in reachable},
                   {node: attrib for node, attrib in self.attribs.items() if node in reachable},
                   self.meta.copy())

    def get_rooted_subgraphs(self) -> list['DAG[Node]']:
        return [self.get_rooted_subgraph(root) for root in self.get_roots()]

    def distinct_paths(self, source: Node, target: Node) -> Paths:
        if source == target:
            return {()}

        def expand_path(edge: Edge, path: Path) -> Paths:
            path = path + (edge,)
            if edge.target == target:
                return {path}
            return {new_path for edge in self.outgoing_edges(edge.target) for new_path in expand_path(edge, path)}
        return set().union(*[expand_path(edge, ()) for edge in self.outgoing_edges(source)])

    def shortest_path(self, source: Node, target: Node) -> Path:
        return min(self.distinct_paths(source, target), key=len)

    def __add__(self, other: 'DAG[Node]') -> 'DAG[Node]':
        return DAG(self.nodes | other.nodes, self.edges | other.edges, self.attribs | other.attribs, self.meta)

    def is_subgraph_of(self, other: 'DAG[Node]') -> bool:
        return self.nodes < other.nodes and self.edges < other.edges

    def __lt__(self, other) -> bool: return self.is_subgraph_of(other)
    def __le__(self, other) -> bool: return self.is_subgraph_of(other) or self == other
    def __eq__(self, other) -> bool: return self.nodes == other.nodes and self.edges == other.edges
    def __ne__(self, other) -> bool: return not self == other
    def __gt__(self, other) -> bool:return not self <= other
