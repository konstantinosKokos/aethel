from .lassy import Lassy
from .transformations import tree_to_dag, collapse_mwu, order_nodes
from .graphutils import DAG
from typing import List, Tuple, Set
from itertools import groupby
from .viz import ToGraphViz

lassy = Lassy(transform=lambda x: collapse_mwu(tree_to_dag(x[-1])))
tgv = ToGraphViz()


def is_np(dag: DAG, node: str):
    d = dag.attribs[node]
    return ('cat' in d.keys() and d['cat'] == 'np') or \
           ('pos' in d.keys() and d['pos'] in {'noun', 'pron'}
            and any(map(lambda p: not is_np(dag, p), dag.predecessors(node))))


def leaves_below(dag: DAG, node: str) -> List[str]:
    if dag.is_leaf(node):
        return [node]
    return sorted(filter(lambda suc: dag.is_leaf(suc), dag.points_to(node)), key=int)


def find_labels(dag: DAG):
    def find_vp(xs):
        return [leaves_below(dag, edge.target) for edge in xs if edge.dep == 'hd'][0]

    def find_np(xs):
        return [leaves_below(dag, edge.target)
                for edge in xs if is_np(dag, edge.target) and edge.dep == 'su'][0]

    edges = sorted(dag.edges, key=lambda edge: edge.source)
    edges = [list(vs) for k, vs in groupby(edges, key=lambda edge: edge.source)]

    return [(find_vp(vs), find_np(vs)) for vs in edges
            if any(map(lambda edge: edge.dep == 'su' and is_np(dag, edge.target), vs))
            and any(map(lambda edge: edge.dep == 'hd', vs))]


def find_nps(dag: DAG):
    noun_nodes = [k for k in dag.attribs if is_np(dag, k)]
    return [leaves_below(dag, np) for np in sorted(noun_nodes, key=int)]


def get_sentence(dag: DAG) -> List[Tuple[str, List[str]]]:
    leaves = order_nodes(dag, list(dag.get_leaves()))
    return list(map(lambda leaf: (leaf, dag.attribs[leaf]['word'].split()), leaves))


def get_sample(dag: DAG) -> Tuple[List[str], List[List[int]], List[List[int]], List[int]]:
    pairs = find_labels(dag)
    nouns = find_nps(dag)
    sent = get_sentence(dag)

    def make_span(n_ids: Set[str]):
        return sum([[1 if idx in n_ids else 0] * len(w) for idx, w in sent], [])
    n_spans = [make_span(set(noun)) for noun in nouns]
    v_spans = [make_span(set(verb)) for verb, _ in pairs]
    labels = [nouns.index(pair) for _, pair in pairs]
    sent = sum([w for _, w in sent], [])
    return sent, n_spans, v_spans, labels
