import pdb
from .graphutils2 import DAG, Edge
from xml.etree.cElementTree import ElementTree
from .viz2 import render


def etree_to_dag(etree: ElementTree, name: str | None = None) -> list[DAG[str]]:
    def f(_dag: DAG[str]) -> DAG[str]:
        _dag = relabel_crd_pairs(_dag)
        _dag = normalize_ghost_positions(_dag)
        _dag = remove_understood_argument(_dag)
        _dag = refine_body(_dag)
        _dag = collapse_mwu(_dag)
        _dag = relabel_determiners(_dag)
        _dag = swap_np_heads(_dag)
        _dag = collapse_mwu(_dag)
        _dag = raise_nouns(_dag)
        return _dag
    nodes = set(etree.iter('node'))
    edges = {Edge(s.attrib['id'], t.attrib['id'], t.attrib['rel']) for s in nodes for t in s.findall('node')}
    attribs = {n.attrib['id']: {k: v for k, v in n.attrib.items() if k != 'rel'} for n in nodes}
    initial = ad_hoc_fixes(DAG(set(attribs.keys()), edges, attribs, {'name': name}))
    return [f(dag) for dag in salvage_headless(initial)]


def is_indexed(dag: DAG[str], node: str) -> bool:
    return dag.get(node, 'index') is not None


def is_ghost(dag: DAG[str], node: str) -> bool:
    return dag.attribs[node].keys() & {'pos', 'cat'} == set()
    

def find_coindex(dag: DAG[str], index: str) -> str | None:
    return next((node for node in dag.nodes if not is_ghost(dag, node) and dag.get(node, 'index') == index), None)


def get_material(dag: DAG[str], node: str) -> str:
    return node if not is_indexed(dag, node) else find_coindex(dag, dag.get(node, 'index'))


def distance_to(dag: DAG[str], node: str, target: str) -> int:
    return len(dag.shortest_path(node, target))


def normalize_ghost_positions(dag: DAG[str]) -> DAG[str]:
    _indexed_nodes = {n for n in dag.nodes if is_indexed(dag, n)}
    indexed_nodes = {index: {n for n in _indexed_nodes if dag.get(n, 'index') == index}
                     for index in set(dag.get(n, 'index') for n in _indexed_nodes)}
    root = next(iter(dag.get_roots()))
    for index, nodes in indexed_nodes.items():
        min_dist = min((root_distances := {n: distance_to(dag, root, n) for n in nodes}).values())
        highest_nodes = {n for n, d in root_distances.items() if d == min_dist}
        if (material := find_coindex(dag, index)) not in highest_nodes:
            dag = reroot_subgraphs(dag, material, next(iter(sort_nodes(dag, highest_nodes))))
    return dag


def reroot_subgraphs(dag: DAG[str], node_a: str, node_b: str) -> DAG[str]:
    to_remove = (tr_a := {edge for edge in dag.edges if edge.source == node_a}) | \
                (tr_b := {edge for edge in dag.edges if edge.source == node_b})
    to_add = {Edge(node_b, tgt, label) for _, tgt, label in tr_a} | {Edge(node_a, tgt, label) for _, tgt, label in tr_b}
    attrs_a, attrs_b = ({k: v for k, v in dag.get(node_a).items() if k != 'id'} | {'id': node_b},
                        {k: v for k, v in dag.get(node_b).items() if k != 'id'} | {'id': node_a})
    dag.edges -= to_remove
    dag.edges |= to_add
    dag.set(node_a, attrs_b)
    dag.set(node_b, attrs_a)
    return dag


def add_fresh_node(dag: DAG[str]) -> str:
    dag.nodes.add(node := str(max((int(dag.get(n, 'id')) for n in dag.nodes)) + 1))
    dag.attribs[node] = {'id': node}
    return node


def add_fresh_nodes(dag: DAG[str], count: int) -> tuple[str, ...]:
    return tuple(add_fresh_node(dag) for _ in range(count))


def add_ghost_of(dag: DAG[str], node: str) -> str:
    fresh_node = add_fresh_node(dag)
    dag.set(fresh_node, {'index': dag.get(node, 'index'),
                         'begin': dag.get(node, 'begin'),
                         'end': dag.get(node, 'end')})
    return fresh_node


def print_dag(dag: DAG[str], root: str | None = None) -> str:
    nodes = sort_nodes(dag) if root is None else sort_nodes(dag, set(dag.successors(root)) | {root})
    return ' '.join([word for n in nodes if (word := dag.get(n, 'word')) is not None])


def sorting_key(dag: DAG[str], node: str) -> tuple[int, int, int]:
    return int(dag.get(node, 'begin')), int(dag.get(node, 'end')), int(dag.get(node, 'id'))


def sort_nodes(dag: DAG[str], nodes: set[str] | None = None) -> list[str]:
    return sorted(nodes or dag.nodes, key=lambda n: sorting_key(dag, n))


def remove_understood_argument(dag: DAG[str]) -> DAG[str]:
    def has_sentential_parent(node: str) -> bool:
        def is_sentential(_node: str) -> bool:
            return (cat := dag.get(_node, 'cat')) in {'sv1', 'smain', 'ssub', 'inf', 'ti'} or \
                   (cat == 'conj' and has_sentential_parent(_node))
        return any(map(is_sentential, dag.parents(node)))

    def top_rel_coindex(_node: str) -> bool:
        nodes = {node for node in dag.nodes if dag.get(node, 'index') == dag.get(_node, 'index')}
        if not {edge.label for node in nodes for edge in dag.incoming_edges(node)} & {'rhd', 'whd'}:
            return False
        common_ancestor = dag.first_common_predecessor(*nodes)
        distances = {node: len(dag.shortest_path(common_ancestor, node)) for node in nodes if is_ghost(dag, node)}
        return distances[_node] == min(distances.values())

    def candidate(edge: Edge[str]) -> bool: return edge.label in {'su', 'obj1', 'obj2', 'sup', 'pobj'}
    def infinitival(node: str) -> bool: return dag.get(node, 'cat') in {'inf', 'ppart'}

    def cond(e: Edge[str]) -> bool:
        return is_ghost(dag, e.target) and candidate(e) and infinitival(e.source) and \
               has_sentential_parent(e.source) and not top_rel_coindex(e.target)
    return dag.remove_edges(cond)


def relabel_determiners(dag: DAG[str]) -> DAG[str]:
    # todo: fix labels
    possessives = {'mij', 'mijn', 'mijn', 'je', 'jouw', 'uw', 'zijn', 'zijne', 'haar', 'ons', 'onze', 'hun', 'wier',
                   'wien', 'wiens',  'hum', 'z\'n', 'z´n', 'm\'n', 'm´n', 'zin', 'huin', 'onst', 'welks'}
    determiners = {'welke', 'die', 'deze', 'dit', 'dat', 'zo\'n', 'zo´n', 'wat', 'wélke', 'zo`n', 'díe', 'dát', 'déze'}
    edges = {edge for edge in dag.edges
             if edge.label == 'det' or (edge.label == 'mod' and dag.get(edge.source, 'cat') == 'np')}
    for edge in edges:
        attrs = dag.get(get_material(dag, edge.target))
        if 'cat' in attrs:
            edge.label = 'mod'
        else:
            word, pos, pt = attrs['word'], attrs['pos'], attrs['pt']
            if pt == 'lid' or word in determiners:
                edge.label = 'det'
            elif word in possessives:
                edge.label = 'det'
            elif (word == 'u' or word.endswith('\'') or word.endswith('\'s')) and edge.label == 'det':
                edge.label = 'det'
            else:
                edge.label = 'mod'
    return dag


def raise_nouns(dag: DAG[str]) -> DAG[str]:
    # http: // web.stanford.edu / group / cslipublications / cslipublications / HPSG / 3 / van.pdf
    nouns = {node: dag.parents(node) for node in dag.nodes if dag.get(node, 'pt') == 'n'}
    for noun, parents in nouns.items():
        if not any(dag.get(parent, 'pos') == 'np' for parent in parents):
            dag.set(noun, {'pos': 'np'})
    return dag


def majority_vote(dag: DAG[str], nodes: list[str], attr: str) -> str:
    return max(xs := [dag.get(n, attr) for n in nodes], key=xs.count)


def collapse_mwu(dag: DAG[str]) -> DAG[str]:
    def propagate_mwu_info(nodes: list[str]) -> dict[str, str]:
        return {'begin': min(dag.get(n, 'begin') for n in nodes),
                'end': max(dag.get(n, 'end') for n in nodes),
                'word': ' '.join(word for n in nodes if (word := dag.get(n, 'word')) is not None),
                'pos': majority_vote(dag, nodes, 'pos'),
                'pt': majority_vote(dag, nodes, 'pt')}

    successors = {n: sort_nodes(dag, set(dag.successors(n))) for n in dag.nodes if dag.get(n, 'cat') == 'mwu'}
    for mwu in successors.keys():
        dag.set(mwu, propagate_mwu_info(successors[mwu]))
    return dag.remove_nodes(set().union(*map(set, successors.values())))


def refine_body(dag: DAG[str]) -> DAG[str]:
    clauses = {n: (body, next(label for edge in out if (label := edge.label) in {'cmp', 'rhd', 'whd'}))
               for n in dag.nodes
               if (body := next(filter(lambda e: e.label == 'body', (out := dag.outgoing_edges(n))), None)) is not None}
    for src, (body, head) in clauses.items():
        match head:
            case 'cmp': label = 'cmpbody'
            case 'rhd': label = 'relcl'
            case 'whd': label = 'whbody'
            case _: raise ValueError(f'Unexpected label {head}')
        dag.edges.remove(body)
        dag.edges.add(Edge(body.source, body.target, label))
    return dag


def relabel_crd_pairs(dag: DAG[str]) -> DAG[str]:
    second_crds = {next(iter(sorted(pair, key=lambda e: sorting_key(dag, e.target), reverse=True)))
                   for n in dag.nodes if dag.get(n, 'cat') == 'conj' and
                   len(pair := (set(filter(lambda out: out.label == 'crd', dag.outgoing_edges(n))))) == 2}
    for edge in second_crds:
        dag.edges.remove(edge)
        # todo: how should I label the new edge?
        dag.edges.add(Edge(edge.source, edge.target, 'cor'))
    return dag


def swap_np_heads(dag: DAG[str]) -> DAG[str]:
    head_nouns = {n: (det, set(filter(lambda edge: edge.label in {'hd', 'crd'}, out - det))) for n in dag.nodes
                  if (det := set(filter(lambda edge: edge.label == 'det', out := dag.outgoing_edges(n))))}
    for src, (dets, heads) in head_nouns.items():
        if not len(dets) == len(heads) == 1:
            render(dag)
            raise AssertionError
        det, head = next(iter(dets)), next(iter(heads))
        if head.label == 'crd':
            continue
        dag.edges.add(Edge(head.source, head.target, 'np_head'))
        dag.edges.remove(head)
    return dag


def salvage_headless(dag: DAG[str]) -> list[DAG[str]]:
    def is_headless(edge: Edge[str]) -> bool: return edge.label in {'dp', 'sat', 'nucl', 'tag', 'du', '--'}

    def replace_ghost(_subgraph: DAG[str], root: str) -> DAG[str]:
        floating_nodes = {n for n in _subgraph.nodes
                          if is_ghost(dag, n) and find_coindex(_subgraph, _subgraph.get(n, 'index')) is None}
        floating_nodes = {index: min({n for n in floating_nodes if dag.get(n, 'index') == index},
                                     key=lambda n: (distance_to(_subgraph, root, n), sorting_key(dag, n)))
                          for index in {dag.get(n, 'index') for n in floating_nodes}}
        for index, highest_float in floating_nodes.items():
            _subgraph += (rooted := dag.get_rooted_subgraph(material_root := find_coindex(dag, index)))
            to_remove = {e for e in _subgraph.edges if e.target == highest_float}
            to_add = {Edge(e.source, material_root, e.label) for e in to_remove}
            _subgraph.edges |= to_add
            _subgraph.remove_edges(to_remove)
            _subgraph.attribs |= rooted.attribs
        return _subgraph

    def insert_punct(_subgraph: DAG[str], root: str) -> DAG[str]:
        # todo: adjacent punctuation
        begin, end = int(_subgraph.get(root, 'begin')), int(_subgraph.get(root, 'end'))
        puncts = {n for n in dag.nodes
                  if dag.get(n, 'pos') == 'punct' and
                  begin < int(dag.get(n, 'begin')) - 1 < end and
                  n not in _subgraph.nodes}
        _subgraph.edges |= {Edge(root, n, 'punct') for n in puncts}
        _subgraph.nodes |= puncts
        _subgraph.attribs |= {n: dag.get(n) for n in puncts}
        _subgraph.set(root, 'end', max((dag.get(p, 'end') for p in puncts), default=_subgraph.get(root, 'end')))
        return insert_punct(_subgraph, root) if len(puncts) > 0 else _subgraph

    def rename(_subgraph: DAG[str], root: str) -> DAG[str]:
        _subgraph.meta['name'] += f'_{root}'
        return _subgraph

    def f(_subgraph: DAG[str], root: str) -> DAG[str]:
        return rename(insert_punct(replace_ghost(_subgraph, root), root), root)

    def maximal(_subgraphs: list[DAG[str]]) -> list[DAG[str]]:
        return [graph for graph in _subgraphs if not(any(map(lambda g: graph < g, _subgraphs)))]

    bad_edges = {e for e in dag.edges if is_headless(e)}
    # todo: do not filter single node graphs unless they are punctuation?
    return maximal([f(subgraph, root) for edge in bad_edges
                    if not set(filter(is_headless, (subgraph := dag.get_rooted_subgraph(root := edge.target)).edges))
                    and len(subgraph) > 1])


def factor_distributed_subgraphs(dag: DAG[str]) -> DAG[str]:
    def group_by_label(_edges: set[Edge[str]]) -> \
            tuple[list[tuple[str, list[Edge[str]]]],
                  list[tuple[str, list[Edge[str]]]],
                  list[tuple[str, list[Edge[str]]]]]:
        def f(_set: set[str]) -> list[tuple[str, list[Edge[str]]]]:
            return [(_label, [e for e in _edges if e.label == _label]) for _label in _set]
        return (f(heady := (occurring := {e.label for e in _edges}) & {'hd', 'rhd', 'whd', 'cmp'}),
                f(moddy := occurring & {'mod', 'app', 'predm'}),
                f(occurring - heady - moddy))

    distributed = [(index, edge) for edge in dag.edges if (index := dag.get(edge.target, 'index')) is not None]
    by_index = {index: group_by_label({e for i, e in distributed if i == index})
                for index in {i for i, _ in distributed}}

    for index, groups in by_index.items():
        match groups:
            case ([], [(label, edges)], []) | ([], [], [(label, edges)]) | ([(label, edges)], [], []):
                # distributed edges of a common label
                dag = _factor_group(dag, index, label, edges)
            case ([('rhd' | 'whd', [_])], [(label, [edge])], []) | ([('rhd' | 'whd', [_])], [], [(label, [edge])]):
                # simple relative clause
                continue
            case ([('rhd' | 'whd', [_])], [(label, edges)], []) | ([('rhd' | 'whd', [_])], [], [(label, edges)]):
                # simple relative clause over a conjunction
                continue
            case _:
                raise NotImplementedError
    return dag


def _factor_group(dag: DAG[str], index: str, label: str, edges: list[Edge[str]]) -> DAG[str]:
    if len(edges) == 1:
        return dag
    material = find_coindex(dag, index)
    fresh_node = add_ghost_of(dag, material)
    common_ancestor = dag.first_common_predecessor(*{edge.target for edge in edges})
    material_src = next((edge.source for edge in edges if edge.target == material))
    dag.edges.add(Edge(material_src, fresh_node, label))
    dag.edges.add(Edge(common_ancestor, material, label))
    dag.edges.remove(Edge(material_src, material, label))
    return dag


def ad_hoc_fixes(dag: DAG[str]) -> DAG[str]:
    if (name := dag.meta['name']) == 'WS-U-E-A-0000000211.p.25.s.1.xml':
        ppart = add_fresh_node(dag)
        dag.edges -= {(Edge('0', '1', '--')), Edge('10', '18', 'hd'), Edge('10', '17', 'obj1')}
        dag.edges |= {Edge('10', '1', 'hd'), Edge('10', ppart, 'vc'),
                      Edge(ppart, '18', 'hd'), Edge(ppart, '17', 'obj1')}
        dag.set(ppart, {'cat': 'ppart', 'begin': min(dag.get('18', 'begin'), dag.get('17', 'begin')),
                        'end': max(dag.get('18', 'end'), dag.get('17', 'end'))})
        dag.set('10', 'cat', 'inf')
    elif name == 'WS-U-E-A-0000000211.p.16.s.2.xml':
        dag.edges.remove(Edge('23', '24', 'vc'))
    elif name == 'WR-P-E-C-0000000021.p.27.s.2.xml':
        dag = dag.get_rooted_subgraph('28')
    elif name == 'wiki-5318.p.26.s.2.xml':
        detp1, detp2 = add_fresh_nodes(dag, 2)
        dag.edges -= {Edge('12', '13', 'mod'), Edge('12', '14', 'det'),
                      Edge('32', '33', 'mod'), Edge('32', '34', 'det')}
        dag.edges |= {Edge('12', detp1, 'det'), Edge(detp1, '13', 'mod'), Edge(detp1, '14', 'hd'),
                      Edge('32', detp2, 'det'), Edge(detp2, '33', 'mod'), Edge(detp2, '34', 'hd')}
        dag.set(detp1, {'cat': 'detp', 'begin': dag.get('13', 'begin'), 'end': dag.get('14', 'end')})
        dag.set(detp2, {'cat': 'detp', 'begin': dag.get('33', 'begin'), 'end': dag.get('34', 'end')})
    return dag
