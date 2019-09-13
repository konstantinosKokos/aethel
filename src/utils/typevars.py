from typing import *
from functools import reduce

fst = lambda x: x[0]
snd = lambda x: x[1]

T1 = TypeVar('T1')
T2 = TypeVar('T2')


def unfoldr(f: Callable[[T1], Optional[Tuple[T1, T2]]], x: T1) -> List[T2]:
    while True:
        u = f(x)
        if u is None:
            break
        x = u[1]
        yield u[0]


Node = TypeVar('Node')
Nodes = Set[Node]
Dep = TypeVar('Dep')


class Edge(NamedTuple):
    source: Node
    target: Node
    dep: Dep

    def adjacent(self) -> Nodes:
        return {self.source, self.target}


Edges = Set[Edge]


def occuring_nodes(edges: Edges) -> Nodes:
    return reduce(set.union, map(lambda edge: edge.adjacent(), edges), set())


class DAG(NamedTuple):
    nodes: Nodes
    edges: Edges
    node_attribs: Dict[Node, Dict]

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def get_root(self) -> Node:
        return list(filter(lambda node: len(self.incoming(node)) == 0, self.nodes))[0]

    def occuring_nodes(self):
        return reduce(set.union, map(lambda edge: edge.adjacent(), self.edges))

    def points_to(self, node: Node) -> Nodes:
        ret = set()
        fringe_nodes = {node}

        while True:
            fringe_edges = reduce(set.union, list(map(self.outgoing, fringe_nodes)), set())
            if not len(fringe_edges):
                break
            fringe_nodes = {edge.target for edge in fringe_edges}
            ret = set.union(ret, fringe_nodes)
        return ret

    def pointed_by(self, node: Node) -> Nodes:
        ret = set()
        fringe_nodes = {node}

        while True:
            fringe_edges = reduce(set.union, list(map(self.incoming, fringe_nodes)), set())
            if not len(fringe_edges):
                break
            fringe_nodes = {edge.source for edge in fringe_edges}
            ret = set.union(ret, fringe_nodes)
        return ret

    def is_underneath(self, node0: Node, node1: Node) -> bool:
        return node1 in self.points_to(node0)

    def remove_nodes(self, condition: Callable[[Node], bool]) -> 'DAG':
        nodes = set(filter(condition, self.nodes))
        edges = set(filter(lambda edge: edge.source in nodes and edge.target in nodes, self.edges))
        node_attribs = {n: a for n, a in self.node_attribs.items() if n in nodes}
        return DAG(nodes=nodes, edges=edges, node_attribs=node_attribs)

    def remove_edges(self, condition: Callable[[Edge], bool]) -> 'DAG':
        edges = set(filter(condition, self.edges))
        nodes = occuring_nodes(edges)
        node_attribs = {n: a for n, a in self.node_attribs.items() if n in nodes}
        return DAG(nodes=nodes, edges=edges, node_attribs=node_attribs)

    def incoming(self, node: Node) -> Edges:
        return set(filter(lambda edge: edge.target == node, self.edges))

    def outgoing(self, node: Node) -> Edges:
        return set(filter(lambda edge: edge.source == node, self.edges))

    def incoming_many(self, nodes: Nodes) -> Edges:
        return reduce(set.union, list(map(self.incoming, nodes)), set())

    def outgoing_many(self, nodes: Nodes) -> Edges:
        return reduce(set.union, list(map(self.outgoing, nodes)), set())

    def oneway(self, node: Node) -> bool:
        return True if len(self.incoming(node)) == 1 and len(self.outgoing(node)) == 1 else False

    def remove_oneway(self, path: Node) -> 'DAG':
        incoming = list(self.incoming(path))[0]
        outgoing = list(self.outgoing(path))[0]
        edge = Edge(incoming.source, outgoing.target, outgoing.dep)
        newdag = self.remove_nodes(lambda node: node == path)
        newdag.edges.add(edge)
        return newdag

    def remove_oneways(self):
        changed = True
        while changed:
            changed = False
            for node in list(filter(self.oneway, self.edges)):
                self.remove_oneway(node)
                changed = True

    def get_subgraphs(self, start: Callable[[Nodes], Node] = lambda n: sorted(n)[0]) -> List['DAG']:
        return list(unfoldr(lambda dag: dag.bfs_split(start), self))

    def bfs_split(self, start: Callable[[Nodes], Node]) -> Optional[Tuple['DAG', 'DAG']]:
        if self.is_empty():
            return

        fringe_nodes = flooded_nodes = {start(self.nodes)}
        flooded_edges = set()

        while True:
            new_fringe_edges = set.union(self.incoming_many(fringe_nodes), self.outgoing_many(fringe_nodes), set())
            new_fringe_edges = set.difference(new_fringe_edges, flooded_edges)
            new_fringe_nodes = reduce(set.union, list(map(lambda edge: edge.adjacent(), new_fringe_edges)), set())
            new_fringe_nodes = set.difference(new_fringe_nodes, flooded_nodes)
            fringe_nodes = new_fringe_nodes
            fringe_edges = new_fringe_edges

            if not len(fringe_nodes):
                break
            else:
                flooded_nodes = set.union(flooded_nodes, fringe_nodes)
                flooded_edges = set.union(flooded_edges, fringe_edges)

        flooded_attribs = {node: attrib for node, attrib in self.node_attribs.items() if node in flooded_nodes}

        rem_nodes = set.difference(self.nodes, flooded_nodes)
        rem_edges = set.difference(self.edges, flooded_edges)
        rem_attribs = {node: attrib for node, attrib in self.node_attribs.items() if node in rem_nodes}

        return DAG(nodes=flooded_nodes, edges=flooded_edges, node_attribs=flooded_attribs), \
               DAG(nodes=rem_nodes, edges=rem_edges, node_attribs=rem_attribs)