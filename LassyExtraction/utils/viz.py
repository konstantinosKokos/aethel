"""
    A helper class for visualizing extracted/transformed graphs.
"""


import graphviz as gv
from .graph import DAG


def render(dag: DAG[str], **kwargs) -> None:
    Renderer.render(dag, **kwargs)


class Renderer:
    properties = ('id', 'word', 'cat', 'index', 'type', 'pt', 'proof')

    @staticmethod
    def make_node_label(node: dict) -> str:
        return '\n'.join(f'{k}: {node[k] if k != "proof" else node[k].type}'
                         for k in Renderer.properties if k in node.keys())

    @staticmethod
    def make_html_label(node: dict) -> str:
        return '<' + '<br/>'.join(f'<b>{k if k != "proof" else "type"}</b>: <i>{node[k] if k != "proof" else node[k].type}</i>'
                                  for k in Renderer.properties if k in node.keys()) + '>'

    @staticmethod
    def render(dag: DAG, **kwargs) -> None:
        graph = gv.Digraph(graph_attr={'ordering': 'in'})
        for node in sorted(dag.nodes, key=int):
            graph.node(node, label=Renderer.make_html_label(dag.attribs[node]),
                       _attributes={'shape': 'none'})
        for edge in dag.edges:
            graph.edge(edge.source, edge.target, label=edge.label,
                       _attributes={'arrowhead': 'none'})
        graph.render(view=True, **kwargs)
