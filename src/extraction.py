from collections import defaultdict
from itertools import chain

from src.graphutils import *
from src.milltypes import AtomicType, WordType, ColoredType, WordTypes, strings, binarize, invariance_check
from src.transformations import majority_vote, _cats_of_type, order_nodes

# # # Extraction variables # # #
# Mapping from phrasal categories and POS tags to Atomic Types
_cat_dict = {'advp': 'ADV', 'ahi': 'AHI', 'ap': 'AP', 'cp': 'CP', 'detp': 'DETP', 'inf': 'INF', 'np': 'NP',
             'oti': 'OTI', 'pp': 'PP', 'ppart': 'PPART', 'ppres': 'PPRES', 'rel': 'REL', 'smain': 'SMAIN',
             'ssub': 'SSUB', 'sv1': 'SV1', 'svan': 'SVAN', 'ti': 'TI', 'whq': 'WHQ', 'whrel': 'WHREL',
             'whsub': 'WHSUB'}
_pos_dict = {'adj': 'ADJ', 'adv': 'ADV', 'comp': 'COMP', 'comparative': 'COMPARATIVE', 'det': 'DET',
             'fixed': 'FIXED', 'name': 'NAME', 'noun': 'N', 'num': 'NUM', 'part': 'PART',
             'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'PRON', 'punct': 'PUNCT', 'tag': 'TAG',
             'verb': 'VERB', 'vg': 'VG'}
_pt_dict = {'adj': 'ADJ', 'bw': 'BW', 'let': 'LET', 'lid': 'LID', 'n': 'N', 'spec': 'SPEC', 'tsw': 'TSW',
            'tw': 'TW', 'vg': 'VG', 'vnw': 'VNW', 'vz': 'VZ', 'ww': 'WW'}
_cat_dict = {k: AtomicType(v) for k, v in _cat_dict.items()}
_pos_dict = {k: AtomicType(v) for k, v in _pos_dict.items()}
_pt_dict = {k: AtomicType(v) for k, v in _pt_dict.items()}
# Head and modifier dependencies
_head_deps = {'hd', 'rhd', 'whd', 'cmp', 'crd'}
_mod_deps = {'mod', 'predm', 'app'}
# Obliqueness Hierarchy
_obliqueness_order = (
    ('mod', 'app', 'predm'),  # modifiers
    ('body', 'rhd_body', 'whd_body'),  # clause bodies
    ('svp',),  # phrasal verb part
    ('ld', 'me', 'vc'),  # verb complements
    ('predc', 'obj2', 'se', 'pc', 'hdf'),  # verb secondary arguments
    ('obj1',),  # primary object
    ('pobj',),  # preliminary object
    ('su',),  # primary subject
    ('sup',),  # preliminary subject
    ('invdet',),  # NP head
)


# Callable version
class ObliquenessSort(object):
    def __init__(self, order: Iterable[Iterable[str]]):
        order = {k: i for i, k in enumerate(reversed(list(chain.from_iterable(order))))}
        self.order = defaultdict(lambda: -1, {**order, **{'cnj': -2}})

    def __call__(self, argcolors: Iterable[Tuple[WordType, str]]) -> List[Tuple[WordType, str]]:
        return sorted(argcolors, key=lambda x: (self.order[snd(x)], str(fst(x))), reverse=True)


_obliqueness_sort = ObliquenessSort(_obliqueness_order)


class ExtractionError(AssertionError):
    def __init__(self, message: str, meta: Any = None):
        super().__init__(message)
        self.meta = meta


def get_type_plain(dag: DAG, node: Node, type_dict: Dict[str, AtomicType], pos_set: str) -> AtomicType:
    if pos_set in dag.attribs[node].keys():
        return type_dict[dag.attribs[node][pos_set]]
    else:
        cat = dag.attribs[node]['cat']
        if cat == 'conj':
            return type_dict[majority_vote(dag, dag.successors(node), pos_set)]
        else:
            return type_dict[dag.attribs[node]['cat']]


def modifier_of(modified: WordType, dep: str) -> ColoredType:
    return ColoredType(argument=modified, result=modified, color=dep)


def rebinarize(sorting_fn: Callable[[List[Tuple[WordType, str]]], List[Tuple[WordType, str]]], arguments: WordTypes,
               colors: strings, result: WordType, mod_deps: Set[str]) -> ColoredType:
    x = result
    while isinstance(x, ColoredType) and x.color not in mod_deps:
        arguments += (x.argument,)
        colors += (x.color,)
        x = x.result
    return binarize(sorting_fn, arguments, colors, x)


def type_top(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str):
    root = fst(list(dag.get_roots()))
    root_type = get_type_plain(dag, root, type_dict, pos_set)
    dag.attribs[root]['type'] = root_type


def type_bot(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: Set[str], mod_deps: Set[str]) -> bool:
    heads = set(map(lambda edge: edge.target, list(filter(lambda edge: edge.dep in hd_deps, dag.edges))))
    heads = heads.difference(set(filter(lambda node: is_gap(dag, node, hd_deps), dag.nodes)))
    typed = set(filter(lambda node: 'type' in dag.attribs[node].keys(), dag.nodes))
    fringe = heads.union(typed)
    temp = type_bot_step(dag, type_dict, pos_set, hd_deps, mod_deps, fringe)
    changed = False
    while temp is not None:
        changed = True
        fringe, attribs = temp
        dag.attribs.update(attribs)
        temp = type_bot_step(dag, type_dict, pos_set, hd_deps, mod_deps, fringe)
    return changed


def type_bot_step(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: Set[str], mod_deps: Set[str],
                  fringe: Nodes) -> Optional[Tuple[Nodes, Dict[Node, Dict]]]:
    def is_fringe(node: Node) -> bool:
        return ((not len(set(filter(lambda edge: edge.dep in mod_deps.union({'cnj'}), dag.incoming(node)))))
                and (dag.is_leaf(node)
                     or all(list(map(lambda out: out.dep in mod_deps or out.target in fringe or
                                                 dag.attribs[out.source]['cat'] == 'conj',
                                     dag.outgoing(node)))))
                and node not in fringe)

    def is_cnj_fringe(node: Node) -> bool:
        return (not is_gap(dag, node, hd_deps)
               and 'type' not in dag.attribs[node]
               and len(set(dag.incoming(node))) == 1 and fst(list(dag.incoming(node))).dep == 'cnj'
               and 'type' in dag.attribs[fst(list(dag.incoming(node))).source])

    new_fringe = set(filter(is_fringe, dag.nodes))
    new_cnj_fringe = set(filter(is_cnj_fringe, dag.nodes))
    if new_fringe.intersection(new_cnj_fringe):
        raise ExtractionError('Fringes overlap.', meta=dag.meta)
    if not new_fringe and not new_cnj_fringe:
        return
    return (new_fringe.union(fringe).union(new_cnj_fringe),
            {**{node: {**dag.attribs[node], **{'type': get_type_plain(dag, node, type_dict, pos_set)}} for node
             in new_fringe},
             **{node: {**dag.attribs[node], **{'type': dag.attribs[fst(list(dag.incoming(node))).source]['type']}}
                for node in new_cnj_fringe}})


def type_mods_step(dag: DAG, mod_deps: Set[str]) -> Optional[Dict[Node, Dict]]:
    typed_nodes = set(filter(lambda node: 'type' in dag.attribs[node].keys(), dag.nodes))
    modding_edges = list(filter(lambda edge: edge.source in typed_nodes
                                             and edge.dep in mod_deps
                                             and edge.target not in typed_nodes, dag.edges))
    if not modding_edges:
        return
    return {edge.target: {**dag.attribs[edge.target], **{'type': modifier_of(dag.attribs[edge.source]['type'],
                                                                              edge.dep)}}
            for edge in modding_edges}


def type_mods(dag: DAG, mod_deps: Set[str]) -> bool:
    temp = type_mods_step(dag, mod_deps)
    changed = False
    while temp is not None:
        changed = True
        attribs = temp
        dag.attribs.update(attribs)
        temp = type_mods_step(dag, mod_deps)
    return changed


def type_heads_step(dag: DAG, head_deps: Set[str], mod_deps: Set[str]) -> Optional[Dict[Node, Dict]]:
    def make_functor(res: WordType, argcolors: Tuple[WordTypes, strings]) -> ColoredType:
        return rebinarize(_obliqueness_sort, fst(argcolors), snd(argcolors), res, mod_deps)

    heading_edges = list(filter(lambda edge: edge.dep in head_deps.difference({'crd'})
                                             and 'type' in dag.attribs[edge.source].keys()
                                             and 'type' not in dag.attribs[edge.target].keys()
                                             and not is_gap(dag, edge.target, head_deps),
                                dag.edges))
    heading_edges = list(map(lambda edge: (edge, dag.outgoing(edge.source)), heading_edges))
    heading_edges = list(map(lambda pair: (fst(pair), set(map(lambda edge: edge.target,
                                                              filter(lambda edge: edge.dep in head_deps, snd(pair))))),
                             heading_edges))
    heading_edges = list(map(lambda pair: (fst(pair), order_nodes(dag, snd(pair))), heading_edges))
    double_heads = list(map(fst, filter(lambda pair: fst(pair).target != fst(snd(pair)), heading_edges)))
    double_heads = list(map(lambda edge: edge.target, double_heads))

    single_heads = list(map(fst, filter(lambda pair: fst(pair) not in double_heads, heading_edges)))
    heading_edges = list(map(lambda edge: (edge, dag.outgoing(edge.source)), single_heads))

    heading_edges = list(map(lambda pair: (fst(pair),
                                           list(filter(lambda edge:
                                                       edge.dep not in mod_deps and edge != fst(pair) and
                                                       edge.target not in double_heads,
                                                       snd(pair)))),
                             heading_edges))

    heading_edges = list(filter(lambda pair: all(list(map(lambda out: 'type' in dag.attribs[out.target].keys(),
                                                          snd(pair)))),
                                heading_edges))
    if not heading_edges and not double_heads:
        return
    heading_argcs = list(map(lambda pair:
                             (fst(pair).target,
                              dag.attribs[fst(pair).source]['type'],
                              list(zip(*map(lambda out: (dag.attribs[out.target]['type'], out.dep), snd(pair))))),
                             heading_edges))
    head_types = {**{node: {**dag.attribs[node], **{'type': make_functor(res, argcolors) if argcolors else res}}
                  for (node, res, argcolors) in heading_argcs},
                  **{node: {**dag.attribs[node], **{'type': AtomicType('_')}} for node in double_heads}}
    return head_types


def type_heads(dag: DAG, head_deps: Set[str], mod_deps: Set[str]) -> bool:
    temp = type_heads_step(dag, head_deps, mod_deps)
    changed = False
    while temp is not None:
        changed = True
        attribs = temp
        dag.attribs.update(attribs)
        temp = type_heads_step(dag, head_deps, mod_deps)
    return changed


def type_core(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, head_deps: Set[str], mod_deps: Set[str]):
    changed = True
    while changed:
        bot = type_bot(dag, type_dict, pos_set, head_deps, mod_deps)
        mod = type_mods(dag, mod_deps)
        head = type_heads(dag, head_deps, mod_deps)
        changed = mod or head or bot


def is_gap(dag: DAG, node: Node, head_deps: Set[str]) -> bool:
    incoming = set(map(lambda edge: edge.dep, dag.incoming(node)))
    return len(incoming) > 1 and len(incoming.intersection(head_deps)) > 0


def type_gaps(dag: DAG, head_deps: Set[str], mod_deps: Set[str]):
    def make_gap_functor(emb_type: WordType, interm: Tuple[WordType, str], top: Tuple[WordType, str]) -> ColoredType:
        if snd(interm) in mod_deps.union(head_deps):
            argument = ColoredType(argument=emb_type, result=fst(interm), color='embedded')
        else:
            argument = ColoredType(argument=emb_type, result=fst(interm), color=snd(interm))
        return ColoredType(argument=argument, result=fst(top), color=snd(top)+'_body')

    def get_interm_top(gap: Node) -> Tuple[Tuple[WordType, str], Tuple[WordType, str]]:
        incoming = dag.incoming(gap)

        top = list(filter(lambda inc: inc.dep in head_deps, incoming))
        if len(set(map(lambda edge: dag.attribs[edge.source]['type'], top))) != 1:
            raise ExtractionError('Multiple top types.')
        top = fst(top)
        top_type, top_dep = dag.attribs[top.source]['type'], top.dep

        interm = list(filter(lambda node: gap in dag.points_to(node), dag.successors(top.source)))
        if len(interm) != 1:
            raise ExtractionError('Multiple intermediate nodes.')
        interm_type = dag.attribs[fst(interm)]['type']
        interm_color = list(map(lambda edge: edge.dep, filter(lambda edge: edge.dep not in head_deps, incoming)))
        if len(set(interm_color)) > 1:
            raise ExtractionError('Multiple intermediate colors.')
        interm_color = fst(interm_color)
        return (interm_type, interm_color), (top_type, top_dep)

    gap_nodes = list(filter(lambda node: is_gap(dag, node, head_deps), dag.nodes))
    if not gap_nodes:
        return
    if any(list(map(lambda node: 'type' not in dag.attribs[node].keys(), gap_nodes))):
        raise ExtractionError('Untyped gap.')
    emb_types = list(map(lambda node: dag.attribs[node]['type'], gap_nodes))
    interms, tops = list(zip(*map(get_interm_top, gap_nodes)))
    gap_types = list(map(make_gap_functor, emb_types, interms, tops))
    gap_types = {node: {**dag.attribs[node], **{'type': gap_type}} for node, gap_type in zip(gap_nodes, gap_types)}
    dag.attribs.update(gap_types)

    derived_gaps = list(filter(lambda node: not dag.is_leaf(node), gap_nodes))
    descendants = list(chain.from_iterable(map(dag.points_to, derived_gaps)))
    descendant_attribs = {node: {k: v for k, v in dag.attribs[node].items() if k != 'type'} for node in descendants}
    dag.attribs.update(descendant_attribs)


def is_copy(dag: DAG, node: Node) -> bool:
    incoming = list(map(lambda edge: edge.dep, dag.incoming(node)))
    return len(incoming) > 1 and any(map(lambda inc: len(list(filter(lambda other: other == inc, incoming))) > 1,
                                         incoming))


def type_copies(dag: DAG, head_deps: Set[str], mod_deps: Set[str]):
    def daughterhood_conditions(daughter: Edge) -> bool:
        return daughter.dep not in head_deps.union(mod_deps)

    def normalize_gap_copies(typecolors: Sequence[Tuple[WordType, Set[str]]]) -> Sequence[Tuple[WordType, str]]:
        def normalize_gap_copy(tc: Tuple[WordType, Set[str]]) -> Tuple[WordType, str]:
            if len(snd(tc)) == 1:
                return fst(tc), fst(list(snd(tc)))
            elif len(snd(tc)) == 2:
                color = fst(list(filter(lambda c: c not in head_deps, snd(tc))))
                return fst(tc).argument.argument, color if color not in mod_deps else 'embedded'
            else:
                raise ExtractionError('Multi-colored copy.', meta=dag.meta)
        return list(map(normalize_gap_copy, typecolors))

    def make_polymorphic_x(initial: WordType, missing: Sequence[Tuple[WordType, str]]) -> ColoredType:
        missing = list(map(lambda pair: (fst(pair), snd(pair) if snd(pair) not in mod_deps else 'embedded'),
                           missing))
        return binarize(_obliqueness_sort, list(map(fst, missing)), list(map(snd, missing)), initial)

    def make_crd_type(poly_x: WordType, repeats: int) -> ColoredType:
        ret = poly_x
        while repeats:
            ret = ColoredType(argument=poly_x, result=ret, color='cnj')
            repeats -= 1
        return ret

    conjuncts = list(_cats_of_type(dag, 'conj'))

    gap_conjuncts = list(filter(lambda node: is_gap(dag, node, head_deps), conjuncts))
    if gap_conjuncts:
        raise ExtractionError('Gap conjunction.')

    conj_groups = list(map(dag.outgoing, conjuncts))
    crds = list(map(lambda conj_group: list(filter(lambda edge: edge.dep == 'crd', conj_group)), conj_groups))
    if any(list(map(lambda conj_group: len(conj_group) == 0, crds))):
        raise ExtractionError('Headless conjunction.', meta={'dag': dag.meta})
    conj_groups = list(map(lambda conj_group: list(filter(daughterhood_conditions, conj_group)), conj_groups))
    conj_groups = list(map(lambda conj_group: list(map(lambda edge: edge.target, conj_group)), conj_groups))

    initial_types = list(map(lambda conj_group: set(map(lambda daughter: dag.attribs[daughter]['type'], conj_group)),
                             conj_groups))
    if any(list(map(lambda conj_group: len(conj_group) != 1, initial_types))):
        raise ExtractionError('Non-polymorphic conjunction.', meta={'dag': dag.meta})

    initial_types = list(map(lambda conj_group: fst(list(conj_group)), initial_types))
    # todo: assert all missing args are the same
    downsets = list(map(lambda conj_group: list(map(lambda daughter: dag.points_to(daughter).union({daughter}),
                                                    conj_group)),
                        conj_groups))
    common_downsets = list(map(lambda downset: set.intersection(*downset), downsets))
    minimal_downsets = list(map(lambda downset:
                                set(filter(lambda node: len(dag.pointed_by(node).intersection(downset)) == 0,
                                           downset)),
                                common_downsets))

    accounted_copies = set.union(*minimal_downsets) if common_downsets else set()
    all_copies = set(filter(lambda node: is_copy(dag, node), dag.nodes))
    if accounted_copies != all_copies:
        raise ExtractionError('Unaccounted copies.', meta=dag.meta)
    if any(list(map(lambda acc: 'type' not in dag.attribs[acc].keys(), accounted_copies))):
        raise ExtractionError('Untyped copies.', meta=dag.meta)

    copy_typecolors = list(map(lambda downset: list(map(lambda node: (dag.attribs[node]['type'],
                                                                      set(map(lambda edge: edge.dep,
                                                                              dag.incoming(node)))),
                                                        downset)),
                               minimal_downsets))
    copy_typecolors = list(map(normalize_gap_copies, copy_typecolors))
    polymorphic_xs = list(map(make_polymorphic_x, initial_types, copy_typecolors))
    crd_types = list(map(make_crd_type, polymorphic_xs, list(map(len, conj_groups))))
    secondary_crds = list(chain.from_iterable(crd[1::] for crd in crds))
    secondary_crds = list(map(lambda crd: crd.target, secondary_crds))
    crds = list(map(fst, crds))
    crds = list(map(lambda crd: crd.target, crds))
    copy_types = {crd: {**dag.attribs[crd], **{'type': crd_type}} for crd, crd_type in zip(crds, crd_types)}
    dag.attribs.update(copy_types)
    secondary_types = {crd: {**dag.attribs[crd], **{'type': AtomicType('_')}} for crd in secondary_crds}
    dag.attribs.update(secondary_types)


def type_dag(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: Set[str], mod_deps: Set[str],
             check: bool = True) -> DAG:
    type_top(dag, type_dict, pos_set)
    type_core(dag, type_dict, pos_set, hd_deps, mod_deps)
    type_gaps(dag, hd_deps, mod_deps)
    type_core(dag, type_dict, pos_set, hd_deps, mod_deps)
    type_copies(dag, hd_deps, mod_deps)
    type_core(dag, type_dict, pos_set, hd_deps, mod_deps)

    if check:
        try:
            premises = list(map(lambda node: dag.attribs[node]['type'], filter(lambda node: dag.is_leaf(node),
                                                                               dag.nodes)))
        except KeyError:
            raise ExtractionError('Untyped leaves.')
        goal = dag.attribs[fst(list(dag.get_roots()))]['type']
        if not invariance_check(premises, goal):
            raise ExtractionError('Invariance check failed.', meta=dag.meta)
    return dag


class Extraction(object):
    def __init__(self, cat_dict: Dict[str, AtomicType], pos_dict: Dict[str, AtomicType], pos_set: str,
                 head_deps: Set[Dep], mod_deps: Set[Dep],
                 ):
        self.type_dict = {**cat_dict, **pos_dict}
        self.pos_set = pos_set
        self.head_deps = head_deps
        self.mod_deps = mod_deps

    def __call__(self, dag: DAG, raise_errors: bool = False) -> Optional[DAG]:
        try:
            return type_dag(dag, self.type_dict, self.pos_set, self.head_deps, self.mod_deps)
        except ExtractionError as e:
            if raise_errors:
                raise e
            else:
                return


typer = Extraction(_cat_dict, _pt_dict, 'pt', _head_deps, _mod_deps)
