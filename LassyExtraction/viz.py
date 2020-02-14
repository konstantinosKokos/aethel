import graphviz as gv  # type: ignore

from LassyExtraction.graphutils import *


class ToGraphViz(object):
    def __init__(self, properties: Iterable[str] = ('id', 'word', 'pos', 'cat', 'index', 'type', 'pt')) -> None:
        self.properties = properties

    def make_node_label(self, attribs: Dict) -> str:
        return '\n'.join([str(attribs[k]) for k in self.properties if k in attribs.keys()])

    def make_edge_label(self, edge: Any) -> str:
        return edge

    def dag_to_gv(self, dag: DAG) -> gv.Digraph:
        graph = gv.Digraph()
        for n in dag.nodes:
            graph.node(n, label=self.make_node_label(dag.attribs[n]))
        for edge in dag.edges:
            graph.edge(edge.source, edge.target, self.make_edge_label(edge.dep))
        return graph

    def tree_to_gv(self, tree: Any) -> gv.Digraph:
        nodes = set(tree.iter('node'))
        edges = [Edge(source.attrib['id'], target.attrib['id'], target.attrib['rel'])
                 for source in nodes for target in source.findall('node')]
        attribs = {node.attrib['id']: node.attrib for node in nodes}
        return self.dag_to_gv(DAG(set(attribs.keys()), set(edges), attribs))

    def __call__(self, parse: Any, view: bool = True, **kwargs):
        if isinstance(parse, DAG):
            graph = self.dag_to_gv(parse)
        else:
            graph = self.tree_to_gv(parse)
        if view:
            graph.render(view=True, **kwargs)
