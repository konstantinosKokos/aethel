"""
    Pipeline for making data graphs compatible with the proof extraction algorithm.
"""

import pdb
from .utils.graph import DAG, Edge
from xml.etree.cElementTree import ElementTree
from typing import Iterator


class TransformationError(Exception):
    pass


def prepare_many(lassy: Iterator[tuple[ElementTree, str | None]]) -> list[DAG[str]]:
    dags = []
    for etree in lassy:
        try:
            dags.extend(prepare_for_extraction(*etree))
        except TransformationError:
            continue
    return dags


def prepare_for_extraction(etree: ElementTree, name: str | None = None,) -> list[DAG[str]]:
    def f(_dag: DAG[str]) -> DAG[str]:
        _dag = punct_to_crd(_dag)
        _dag = relabel_extra_crds(_dag)
        _dag = normalize_ghost_positions(_dag)
        _dag = remove_understood_argument(_dag)
        _dag = refine_body(_dag)
        _dag = collapse_mwu(_dag)
        _dag = relabel_determiners(_dag)
        _dag = swap_np_heads(_dag)
        _dag = relocate_nominal_modifiers(_dag)
        _dag = raise_nouns(_dag)
        _dag = factor_distributed_subgraphs(_dag)
        _dag = coerce_conjunctions(_dag)
        assertions(_dag)
        return _dag
    return sorted([f(dag) for dag in salvage_headless(ad_hoc_fixes(etree_to_dag(etree, name)))],
                  key=lambda dag: int(dag.meta['name'].split('(')[1].rstrip(')')))


def etree_to_dag(etree: ElementTree, name: str | None) -> DAG[str]:
    nodes = set(etree.iter('node'))
    edges = {Edge(s.attrib['id'], t.attrib['id'], t.attrib['rel']) for s in nodes for t in s.findall('node')}
    attribs = {n.attrib['id']: {k: v for k, v in n.attrib.items() if k != 'rel'} for n in nodes}
    return DAG(set(attribs.keys()), edges, attribs, {'name': name} if name is not None else {})


def is_indexed(dag: DAG[str], node: str) -> bool:
    return dag.get(node, 'index') is not None


def is_ghost(dag: DAG[str], node: str) -> bool:
    return dag.attribs[node].keys() & {'pos', 'cat'} == set()


def find_coindexed(dag: DAG[str], index: str) -> set[str]:
    return {n for n in dag.nodes if dag.get(n, 'index') == index}


def find_coindex(dag: DAG[str], index: str) -> str | None:
    return next((node for node in dag.nodes if not is_ghost(dag, node) and dag.get(node, 'index') == index), None)


def get_material(dag: DAG[str], node: str) -> str:
    return node if not is_indexed(dag, node) else find_coindex(dag, dag.get(node, 'index'))


def distance_to(dag: DAG[str], node: str, target: str) -> int:
    return len(dag.shortest_path(node, target))


def group_by_index(dag: DAG[str]) -> dict[str, set[str]]:
    return {index: {n for n in dag.nodes if dag.get(n, 'index') == index}
            for index in set(dag.get(n, 'index') for n in dag.nodes)}


def punct_to_crd(dag: DAG[str]) -> DAG[str]:
    def crdless(edges: set[Edge[str]]) -> bool: return not any(map(lambda e: e.label == 'crd', edges))
    puncts = {(n, inc.source, node_to_key(dag, n))
              for n in dag.nodes
              if (inc := next(iter(dag.incoming_edges(n)), None)) is not None and inc.label == 'punct'}
    conjunctions = {n: [edge.target for edge in out if edge.label == 'cnj'] for n in dag.nodes
                    if dag.get(n, 'cat') == 'conj' and crdless(out := dag.outgoing_edges(n))}
    for c, conjuncts in conjunctions.items():
        starts, ends, _ = zip(*[node_to_key(dag, cnj) for cnj in conjuncts])
        start, end = min(starts), max(ends)
        ps = [(p, source) for p, source, (s, e, _) in puncts if start < s < end]
        if len(ps) == 1:
            par, source = next(iter(ps))
            dag.edges -= {Edge(source, par, 'punct')}
            dag.edges |= {Edge(c, par, 'crd')}
    return dag


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
    dag.attribs[node_a] = attrs_b
    dag.attribs[node_b] = attrs_a
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


def get_lex_nodes(dag: DAG[str], root: str | None = None) -> list[str]:
    nodes = sort_nodes(dag) if root is None else sort_nodes(dag, set(dag.successors(root)) | {root})
    return [node for node in nodes if
            all(edge.label == 'mwp' for edge in dag.outgoing_edges(node))
            and not any(edge.label == 'mwp' for edge in dag.incoming_edges(node))]


def get_sentence(dag: DAG[str], root: str | None = None) -> str:
    return ' '.join([dag.get(node, 'word') for node in get_lex_nodes(dag, root)])


def node_to_key(dag: DAG[str], node: str) -> tuple[int, int, int]:
    return int(dag.get(node, 'begin')), int(dag.get(node, 'end')), int(dag.get(node, 'id'))


def sort_nodes(dag: DAG[str], nodes: set[str] | None = None) -> list[str]:
    return sorted(nodes if nodes is not None else dag.nodes, key=lambda n: node_to_key(dag, n))


def relabel_extra_crds(dag: DAG[str]) -> DAG[str]:
    extra_crds = {crds for n in dag.nodes if dag.get(n, 'cat') == 'conj'
                  and len(crds := tuple(filter(lambda edge: edge.label == 'crd', dag.outgoing_edges(n)))) > 1}
    for crds in extra_crds:
        _, *rest = sorted(crds, key=lambda edge: node_to_key(dag, edge.target))
        dag.edges -= set(rest)
        dag.edges |= {Edge(edge.source, edge.target, 'cor') for edge in rest}
    return dag


def remove_understood_argument(dag: DAG[str]) -> DAG[str]:
    def infinitival(_node: str) -> bool:
        return ((cat := dag.get(_node, 'cat')) in {'inf', 'ti', 'ppart', 'ssub', } or
                cat == 'conj' and any(infinitival(p) for p in dag.parents(_node)))

    def sentential(_node: str) -> bool:
        return ((cat := dag.get(_node, 'cat')) in {'sv1', 'smain', 'ssub', 'inf', 'ti', 'ahi', 'ppart'} or
                (cat == 'conj' and any(map(sentential, dag.parents(_node)))))

    def candidate(label: str) -> bool:
        # todo: should I actually be removing obj edges
        return label in {'su', 'obj1', 'obj2', 'sup', 'pobj1'}

    def is_understood(edge: Edge[str]) -> bool:
        if is_ghost(dag, edge.target) and candidate(edge.label):
            index = dag.get(edge.target, 'index')
            return any(sentential(p) and any(dag.get(c, 'index') == index for c in dag.children(p))
                       for p in dag.predecessors(edge.source))
        return False
    return dag.remove_edges(is_understood)


def relabel_determiners(dag: DAG[str]) -> DAG[str]:
    possessives = {'mij', 'mijn', 'mijn', 'je', 'jouw', 'uw', 'zijn', 'zijne', 'haar', 'ons', 'onze', 'hun', 'wier',
                   'wien', 'wiens',  'hum', 'z\'n', 'z´n', 'm\'n', 'm´n', 'zin', 'huin', 'onst', 'welks'}
    determiners = {'welke', 'die', 'deze', 'dit', 'dat', 'zo\'n', 'zo´n', 'wat', 'wélke', 'zo`n', 'díe', 'dát', 'déze'}
    edges = {edge for edge in dag.edges
             if edge.label == 'det' or (edge.label == 'mod' and dag.get(edge.source, 'cat') == 'np')}
    for edge in edges:
        attrs = dag.get(get_material(dag, edge.target))
        dag.edges.remove(edge)
        if 'cat' in attrs:
            dag.edges.add(Edge(edge.source, edge.target, 'mod'))
        else:
            word, pos, pt = attrs['word'], attrs['pos'], attrs['pt']
            if pt == 'lid' or (word.lower() in determiners and not word == word.upper()) or word.lower() in possessives:
                dag.edges.add(Edge(edge.source, edge.target, 'det'))
            else:
                dag.edges.add(Edge(edge.source, edge.target, 'mod'))
    return dag


def relocate_nominal_modifiers(dag: DAG[str]) -> DAG[str]:
    nominal_mods = {node: (det, mods, hd)
                    for node in dag.nodes if dag.get(node, 'cat') == 'np' and
                    (mods := set(filter(lambda edge: edge.label in {'mod', 'app', 'predm'},
                                        (edges := dag.outgoing_edges(node))))) and
                    (det := next((edge for edge in edges if edge.label == 'det'), None)) is not None and
                    (hd := next((edge for edge in edges if edge.label == 'np_head'), None)) is not None}
    for node, (det, mods, hd) in nominal_mods.items():
        premods = {mod for mod in mods
                   if int(dag.get(det.target, 'end')) <= int(dag.get(mod.target, 'begin'))
                   and int(dag.get(mod.target, 'end')) <= int(dag.get(hd.target, 'begin'))}
        if premods:
            intermediate = add_fresh_node(dag)
            dag.edges |= {Edge(node, intermediate, hd.label), Edge(intermediate, hd.target, hd.label)}
            dag.edges |= {Edge(intermediate, mod.target, mod.label) for mod in premods}
            dag.edges -= (premods | {Edge(node, hd.target, hd.label)})
            dag.set(intermediate, {'cat': dag.get(get_material(dag, hd.target), 'pt'),
                                   'begin': str(min(int(dag.get(n, 'begin')) for n in dag.successors(intermediate))),
                                   'end': str(max(int(dag.get(n, 'end')) for n in dag.successors(intermediate)))})
    return dag


def raise_nouns(dag: DAG[str]) -> DAG[str]:
    # http://web.stanford.edu/group/cslipublications/cslipublications/HPSG/3/van.pdf
    nouns = {node: dag.parents(node) for node in dag.nodes
             if dag.is_leaf(node) and dag.get(node, 'pt') in {'n', 'spec'}}
    for noun, parents in nouns.items():
        if not any(dag.get(parent, 'cat') in {'np', 'n', 'spec'} for parent in parents):
            dag.set(noun, {'pt': 'np'})
    return dag


def majority_vote(xs: list[str]) -> str | None:
    return 'np' if 'np' in xs else 'n' if ('n' in xs or 'spec' in xs) else max(xs, key=xs.count, default=None)


def collapse_mwu(dag: DAG[str]) -> DAG[str]:
    def propagate_mwu_info(nodes: list[str]) -> dict[str, str | list[str]]:
        return {'begin': str(min(int(dag.get(n, 'begin')) for n in nodes)),
                'end': str(max(int(dag.get(n, 'end')) for n in nodes)),
                'word': ' '.join(word for n in nodes if (word := dag.get(n, 'word')) is not None),
                'pos': majority_vote([pos for node in nodes if (pos := dag.get(node, 'pos')) is not None]),
                'pt': majority_vote([pt for node in nodes if (pt := dag.get(node, 'pt')) is not None]),
                'lemma': ' '.join(lemma for node in nodes if (lemma := dag.get(node, 'lemma')) is not None)}

    successors = {n: sort_nodes(dag, set(dag.successors(n))) for n in dag.nodes if dag.get(n, 'cat') == 'mwu'}
    for mwu in successors.keys():
        dag.set(mwu, propagate_mwu_info(successors[mwu]))
    return dag
    # return dag.remove_nodes(set().union(*map(set, successors.values())))


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


def cnj_to_mod(dag: DAG[str]) -> DAG[str]:
    # todo
    def modding(nodes: list[str]) -> bool:
        return {dag.get(n, 'pt') for n in nodes}.issuperset({'vnw', 'pt'})

    conjunctions = {node: cnjs for node in dag.nodes if dag.get(node, 'cat') == 'conj'
                    if modding(cnjs := [edge.target for edge in dag.outgoing_edges(node) if edge.label == 'cnj'])}
    if conjunctions:
        pdb.set_trace()
    return dag


def swap_np_heads(dag: DAG[str]) -> DAG[str]:
    head_nouns = {n: (det, set(filter(lambda edge: edge.label in {'hd', 'crd'}, out - det))) for n in dag.nodes
                  if (det := set(filter(lambda edge: edge.label == 'det', out := dag.outgoing_edges(n))))}
    for src, (dets, heads) in head_nouns.items():
        if not len(dets) == len(heads) == 1:
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
                                     key=lambda n: (distance_to(_subgraph, root, n), node_to_key(dag, n)))
                          for index in {dag.get(n, 'index') for n in floating_nodes}}
        for index, highest_float in floating_nodes.items():
            _subgraph += (rooted := dag.get_rooted_subgraph(material_root := find_coindex(dag, index)))
            to_remove = {e for e in _subgraph.edges if e.target == highest_float}
            to_add = {Edge(e.source, material_root, e.label) for e in to_remove}
            _subgraph.edges |= to_add
            _subgraph.remove_edges(to_remove)
            _subgraph.attribs |= rooted.attribs
        return _subgraph

    def add_boundaries(_dag: DAG[str], puncts: set[str], left: int, right: int) -> set[str]:
        included = {p for p in puncts if left - 1 < int(dag.get(p, 'begin')) < right}
        puncts -= included
        left_boundary = next((p for p in puncts if int(dag.get(p, 'begin')) == left - 1), None)
        right_boundary = next((p for p in puncts if int(dag.get(p, 'begin')) == right), None)

        illegal = {'\\', '/', '-', ':', ';', ',', '~', '@', '#', '$', '%', '^', '&', '*'}
        illegal_lefts = {'.', '?', '!', ')', '>>', '»', '/'}.union(illegal)
        illegal_rights = {'(', '<<', '«'}.union(illegal)

        if left_boundary is not None and dag.get(left_boundary, 'word') not in illegal_lefts:
            included.add(left_boundary)
        if right_boundary is not None and dag.get(right_boundary, 'word') not in illegal_rights:
            included.add(right_boundary)
        return included

    def insert_punct(_subgraph: DAG[str], root: str) -> DAG[str]:
        begin, end = int(_subgraph.get(root, 'begin')), int(_subgraph.get(root, 'end'))
        puncts = {n for n in dag.nodes if dag.get(n, 'pos') == 'punct' and n not in _subgraph.nodes}
        puncts = add_boundaries(dag, puncts, begin, end)
        _subgraph.edges |= {Edge(root, n, 'punct') for n in puncts}
        _subgraph.nodes |= puncts
        _subgraph.attribs |= {n: dag.get(n) for n in puncts}
        _subgraph.set(root, 'begin', str(min((int(dag.get(n, 'begin')) for n in puncts), default=str(begin))))
        _subgraph.set(root, 'end', str(max((int(dag.get(p, 'end')) for p in puncts), default=str(end))))
        return insert_punct(_subgraph, root) if len(puncts) > 0 else _subgraph

    def rename(_subgraph: DAG[str], root: str) -> DAG[str]:
        _subgraph.meta['name'] += f'({root})'
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
            case ([('rhd' | 'whd', [_])], [(_, [_])], []) | ([('rhd' | 'whd', [_])], [], [(_, [_])]):
                # simple relative clause
                continue
            case ([('rhd' | 'whd', [_])], [(_, [_, *_])], []) | ([('rhd' | 'whd', [_])], [], [(_, [_, *_])]):
                # simple relative clause over a conjunction
                continue
            case _:
                # ???
                continue
    return dag


def _factor_group(dag: DAG[str], index: str, label: str, edges: list[Edge[str]]) -> DAG[str]:
    if len(edges) == 1:
        return dag
    common_ancestor = dag.first_common_predecessor(*{edge.target for edge in edges})
    if dag.get(common_ancestor, 'cat') != 'conj':
        # todo: deal with some cases of non-conjunction node sharing
        raise TransformationError(f'Cannot redistribute edges over a non-conjunction ancestor')
    material = find_coindex(dag, index)
    fresh_node = add_ghost_of(dag, material)
    common_ancestor = dag.first_common_predecessor(*{edge.target for edge in edges})

    material_src = next((edge.source for edge in edges if edge.target == material))
    dag.edges |= {Edge(material_src, fresh_node, label), Edge(common_ancestor, material, label)}
    dag.edges.remove(Edge(material_src, material, label))
    return dag


def coerce_conjunctions(dag: DAG[str]) -> DAG[str]:
    def bottom_first(_node) -> int: return sum(map(lambda n: dag.is_reachable(_node, n), conjunctions.keys()))

    conjunctions = {node: [edge.target for edge in dag.outgoing_edges(node) if edge.label == 'cnj']
                    for node in dag.nodes if dag.get(node, 'cat') == 'conj'}
    for node, cnjs in sorted(conjunctions.items(), key=lambda t: bottom_first(t[0])):
        dag.set(node, 'cat', majority_vote([pt if (pt := dag.get(cnj, 'pt')) is not None
                                            else dag.get(cnj, 'cat') for cnj in cnjs]))
    return dag


def ad_hoc_fixes(dag: DAG[str]) -> DAG[str]:
    match dag.meta['name']:
        case 'WS-U-E-A-0000000211.p.25.s.1.xml':
            ppart = add_fresh_node(dag)
            dag.edges -= {(Edge('0', '1', '--')), Edge('10', '18', 'hd'), Edge('10', '17', 'obj1')}
            dag.edges |= {Edge('10', '1', 'hd'), Edge('10', ppart, 'vc'),
                          Edge(ppart, '18', 'hd'), Edge(ppart, '17', 'obj1')}
            dag.set(ppart, {'cat': 'ppart', 'begin': min(dag.get('18', 'begin'), dag.get('17', 'begin')),
                            'end': max(dag.get('18', 'end'), dag.get('17', 'end'))})
            dag.set('10', 'cat', 'inf')
        case 'WS-U-E-A-0000000211.p.16.s.2.xml':
            dag.edges.remove(Edge('23', '24', 'vc'))
        case 'WR-P-E-C-0000000021.p.27.s.2.xml':
            dag = dag.get_rooted_subgraph('28')
        case 'wiki-5318.p.26.s.2.xml':
            detp1, detp2 = add_fresh_nodes(dag, 2)
            dag.edges -= {Edge('12', '13', 'mod'), Edge('12', '14', 'det'),
                          Edge('32', '33', 'mod'), Edge('32', '34', 'det')}
            dag.edges |= {Edge('12', detp1, 'det'), Edge(detp1, '13', 'mod'), Edge(detp1, '14', 'hd'),
                          Edge('32', detp2, 'det'), Edge(detp2, '33', 'mod'), Edge(detp2, '34', 'hd')}
            dag.set(detp1, {'cat': 'detp', 'begin': dag.get('13', 'begin'), 'end': dag.get('14', 'end')})
            dag.set(detp2, {'cat': 'detp', 'begin': dag.get('33', 'begin'), 'end': dag.get('34', 'end')})
        case 'dpc-qty-000931-nl-sen.p.25.s.1.xml':
            copula = add_fresh_node(dag)
            predcs = add_fresh_node(dag)
            dag.edges -= {Edge('10', '15', 'predc'), Edge('10', '11', 'su'), Edge('10', '14', 'hd'),
                          Edge('15', '16', 'cnj'), Edge('15', '28', 'cnj'), Edge('15', '23', 'cnj')}
            dag.edges |= {Edge(copula, '14', 'hd'), Edge(copula, '11', 'su'), Edge(copula, predcs, 'predc'),
                          Edge(predcs, '16', 'cnj'), Edge(predcs, '28', 'cnj'), Edge(predcs, '23', 'cnj'),
                          Edge('15', copula, 'cnj'), Edge('0', '15', '--')}
            dag.remove_nodes({'9', '10'})
            dag.set(copula, {'cat': 'smain', 'begin': dag.get('11', 'begin'), 'end': dag.get('28', 'end')})
            dag.set(predcs, {'cat': 'conj', 'begin': dag.get('16', 'begin'), 'end': dag.get('28', 'end')})
            # convert last comma to crd to avoid losing the sample
            dag.edges -= {Edge('0', '6', '--')}
            dag.edges |= {Edge(predcs, '6', 'crd')}
            dag.set('6', {'word': 'en', 'pos': 'vg', 'pt': 'vg', })
        case 'dpc-ibm-001314-nl-sen.p.57.s.1.xml':
            dag.edges |= {Edge('15', '48', 'cnj')}
            dag.remove_nodes({'45', '46', '47'})
        case 'WR-P-E-I-0000050381.p.1.s.338.xml':
            # (het ene _ na het andere _) X case
            dag.remove_nodes({'18', '21'})
            dag.edges -= {Edge('18', '27', 'hd'), Edge('18', '19', 'det'), Edge('18', '20', 'mod'),
                          Edge('18', '21', 'mod'), Edge('23', '26', 'hd')}
            left, conj = add_fresh_nodes(dag, 2)
            dag.edges |= {Edge('17', conj, 'su'), Edge(conj, '22', 'crd'), Edge(conj, left, 'cnj'),
                          Edge(conj, '23', 'cnj'), Edge(left, '27', 'hd'), Edge(left, '19', 'det'),
                          Edge(left, '20', 'mod'), Edge('23', '26', 'hd'), Edge('23', '24', 'det'),
                          Edge('23', '25', 'mod')}
            dag.set(conj, {'begin': dag.get('19', 'begin'), 'end': dag.get('26', 'end'), 'cat': 'conj'})
            dag.set(left, {'begin': dag.get('19', 'begin'), 'end': dag.get('20', 'end'), 'cat': 'np'})
    return dag


def assertions(dag: DAG[str]):
    # assert single root
    if len(rs := dag.get_roots()) != 1:
        raise ValueError(f'Too many roots {dag.meta["name"]}')
    # assert tree structure
    assert all((len(dag.incoming_edges(n)) == 1 for n in dag.nodes if n not in rs))
