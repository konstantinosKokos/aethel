from typing import *


def fst(x: Sequence) -> Any:
    return x[0]


def snd(x: Sequence) -> Any:
    return x[1]


def last(x: Sequence) -> Any:
    return x[-1]


_T1 = TypeVar('_T1')
_T2 = TypeVar('_T2')


def unfoldr(f: Callable[[_T1], Optional[Tuple[_T2, _T1]]], x: _T1) -> Iterable[_T2]:
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
Path = Sequence[Edge]
Paths = Set[Path]


def occuring_nodes(edges: Edges) -> Nodes:
    return set.union(*list(map(lambda edge: edge.adjacent(), edges))) if edges else set()


class DAG(NamedTuple):
    nodes: Nodes
    edges: Edges
    attribs: Dict[Node, Dict]
    meta: Any = None

    def is_empty(self) -> bool:
        return len(self.nodes) == 0

    def get_roots(self) -> Nodes:
        return set(filter(lambda node: not len(self.incoming(node)), self.nodes))

    def is_leaf(self, node: Node) -> bool:
        return not len(self.outgoing(node))

    def get_edges(self, dep: Dep) -> Iterable[Edge]:
        return filter(lambda edge: edge.dep == dep, self.edges)

    def occuring_nodes(self):
        return set.union(*list(map(lambda edge: edge.adjacent(), self.edges)))

    def incoming(self, node: Node) -> Edges:
        return set(filter(lambda edge: edge.target == node, self.edges))

    def predecessors(self, node: Node) -> Nodes:
        incoming = self.incoming(node)
        return set.union(*list(map(lambda edge: {edge.source}, incoming))) if incoming else set()

    def first_common_predecessor(self, nodes: Nodes) -> Optional[Node]:
        predecessors = list(map(self.pointed_by, nodes))
        common_predecessors = set.intersection(*predecessors) if predecessors else set()
        upsets = list(map(lambda pred: (pred, self.pointed_by(pred)), common_predecessors))
        upsets = list(filter(lambda upset:
                             all(list(map(lambda comp:
                                          comp.difference(snd(upset)) == set(),
                                          list(map(snd, upsets))))),
                             upsets))
        return fst(fst(upsets)) if upsets else None

    def outgoing(self, node: Node) -> Edges:
        return set(filter(lambda edge: edge.source == node, self.edges))

    def successors(self, node: Node) -> Nodes:
        outgoing = self.outgoing(node)
        return set.union(*list(map(lambda edge: {edge.target}, outgoing))) if outgoing else set()

    def incoming_many(self, nodes: Nodes) -> Edges:
        return set.union(*list(map(self.incoming, nodes)))

    def outgoing_many(self, nodes: Nodes) -> Edges:
        return set.union(*list(map(self.outgoing, nodes)))

    def points_to(self, node: Node) -> Nodes:
        ret = set()
        fringe_nodes = {node}

        while True:
            fringe_edges = set.union(*list(map(self.outgoing, fringe_nodes)))
            if not len(fringe_edges):
                break
            fringe_nodes = {edge.target for edge in fringe_edges}
            ret = set.union(ret, fringe_nodes)
        return ret

    def pointed_by(self, node: Node) -> Nodes:
        ret = set()
        fringe_nodes = {node}

        while True:
            fringe_edges = set.union(*list(map(self.incoming, fringe_nodes)))
            if not len(fringe_edges):
                break
            fringe_nodes = {edge.source for edge in fringe_edges}
            ret = set.union(ret, fringe_nodes)
        return ret

    def exists_path(self, node0: Node, node1: Node) -> bool:
        return node1 in self.points_to(node0)

    def remove_nodes(self, condition: Callable[[Node], bool], normalize: bool = True) -> 'DAG':
        nodes = set(filter(condition, self.nodes))
        node_attribs = {n: a for n, a in self.attribs.items() if n in nodes}
        if normalize:
            edges = set(filter(lambda edge: edge.source in nodes and edge.target in nodes, self.edges))
            return DAG(nodes=nodes, edges=edges, attribs=node_attribs, meta=self.meta)
        else:
            return DAG(nodes=nodes, edges=self.edges, attribs=node_attribs, meta=self.meta)

    def remove_edges(self, condition: Callable[[Edge], bool], normalize: bool = True) -> 'DAG':
        edges = set(filter(condition, self.edges))
        if normalize:
            nodes = occuring_nodes(edges)
            node_attribs = {n: a for n, a in self.attribs.items() if n in nodes}
            return DAG(nodes=nodes, edges=edges, attribs=node_attribs, meta=self.meta)
        else:
            return DAG(nodes=self.nodes, edges=edges, attribs=self.attribs, meta=self.meta)

    def oneway(self, node: Node) -> bool:
        return True if len(self.incoming(node)) == 1 and len(self.outgoing(node)) == 1 else False

    def remove_oneway(self, path: Node) -> 'DAG':
        incoming = list(self.incoming(path))[0]
        outgoing = list(self.outgoing(path))[0]
        edge = Edge(incoming.source, outgoing.target, outgoing.dep)
        newdag = self.remove_nodes(lambda node: node == path)
        newdag.edges.add(edge)
        return newdag

    def remove_oneways(self) -> 'DAG':
        newdag = self
        while True:
            oneways = list(filter(self.oneway, self.edges))
            if not len(oneways):
                return newdag
            for node in list(filter(self.oneway, self.edges)):
                newdag = newdag.remove_oneway(node)

    def get_rooted_subgraphs(self, erasing: bool = False) -> List['DAG']:
        roots = self.get_roots()
        if len(roots) == 1:
            return [self]
        subnodes = list(map(lambda root: self.points_to(root).union({root}), roots))

        if erasing:
            subnodes = list(map(lambda i:
                                set(subnodes[i]).difference(set.union(*map(lambda j: set(subnodes[j]),
                                                                           list(filter(lambda k: i != k,
                                                                                       range(len(subnodes))))))),
                                range(len(subnodes))))
        else:
            subnodes = list(map(set, subnodes))

        subedges = list(map(lambda subgraph: set(filter(lambda edge: edge.adjacent().issubset(subgraph), self.edges)),
                            subnodes))

        return list(filter(lambda subgraph: not subgraph.is_empty(),
                           list(map(lambda idx: DAG(nodes=subnodes[idx], edges=subedges[idx],
                                                    attribs={k: v for k, v in self.attribs.items()
                                                             if k in subnodes[idx]},
                                                    meta=self.meta),
                                    range(len(subnodes))))))

    def get_subgraphs(self,
                      start: Callable[[Nodes], Node] = lambda n: fst(sorted(n, key=lambda x: int(x)))) -> List['DAG']:
        return list(unfoldr(lambda dag: dag.bfs_split(start), self))

    def bfs_split(self, start: Callable[[Nodes], Node]) -> Optional[Tuple['DAG', 'DAG']]:
        if self.is_empty():
            return

        fringe_nodes = flooded_nodes = {start(self.nodes)}
        flooded_edges = set()

        while True:
            new_fringe_edges = set.union(self.incoming_many(fringe_nodes), self.outgoing_many(fringe_nodes), set())
            new_fringe_edges = set.difference(new_fringe_edges, flooded_edges)
            new_fringe_nodes = set.union(*list(map(lambda e: e.adjacent(), new_fringe_edges))) \
                if new_fringe_edges else set()
            new_fringe_nodes = set.difference(new_fringe_nodes, flooded_nodes)
            fringe_nodes = new_fringe_nodes
            fringe_edges = new_fringe_edges

            if not len(fringe_nodes):
                break
            else:
                flooded_nodes = set.union(flooded_nodes, fringe_nodes)
                flooded_edges = set.union(flooded_edges, fringe_edges)

        flooded_attribs = {node: attrib for node, attrib in self.attribs.items() if node in flooded_nodes}

        rem_nodes = set.difference(self.nodes, flooded_nodes)
        rem_edges = set.difference(self.edges, flooded_edges)
        rem_attribs = {node: attrib for node, attrib in self.attribs.items() if node in rem_nodes}

        return (DAG(nodes=flooded_nodes, edges=flooded_edges, attribs=flooded_attribs, meta=self.meta),
                DAG(nodes=rem_nodes, edges=rem_edges, attribs=rem_attribs, meta=self.meta))

    def distinct_paths_to(self, source: Node, target: Node) -> Paths:
        def expand_path(edge_: Edge, path_: Path) -> Paths:
            path_ = tuple(path_) + (edge_,)
            if edge_.target == target:
                return {path_}
            expansions = list(map(lambda out: expand_path(out, path_), self.outgoing(edge_.target)))
            return set.union(*expansions) if expansions else set()

        if target not in self.points_to(source) or not self.outgoing(source):
            return set()
        else:
            return set.union(*[expand_path(edge, tuple()) for edge in self.outgoing(source)])

