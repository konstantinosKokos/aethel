from collections import defaultdict
from itertools import chain

from LassyExtraction.graphutils import *
from LassyExtraction.milltypes import (AtomicType, WordType, FunctorType, WordTypes, DiamondType, BoxType, strings,
                                       invariance_check, reduce)
from LassyExtraction.transformations import majority_vote, _cats_of_type, order_nodes

# # # Extraction variables # # #
# Mapping from phrasal categories and POS tags to Atomic Types
_CatDict = {'advp': 'ADV', 'ahi': 'AHI', 'ap': 'AP', 'cp': 'CP', 'detp': 'DETP', 'inf': 'INF', 'np': 'NP',
            'oti': 'OTI', 'pp': 'PP', 'ppart': 'PPART', 'ppres': 'PPRES', 'rel': 'REL', 'smain': 'SMAIN',
            'ssub': 'SSUB', 'sv1': 'SV1', 'svan': 'SVAN', 'ti': 'TI', 'whq': 'WHQ', 'whrel': 'WHREL',
            'whsub': 'WHSUB'}
_PosDict = {'adj': 'ADJ', 'adv': 'ADV', 'comp': 'COMP', 'comparative': 'COMPARATIVE', 'det': 'DET',
            'fixed': 'FIXED', 'name': 'NAME', 'noun': 'N', 'num': 'NUM', 'part': 'PART',
            'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'PRON', 'punct': 'PUNCT', 'tag': 'TAG',
            'verb': 'VERB', 'vg': 'VG'}
_PtDict = {'adj': 'ADJ', 'bw': 'BW', 'let': 'LET', 'lid': 'LID', 'n': 'N', 'spec': 'SPEC', 'tsw': 'TSW',
           'tw': 'TW', 'vg': 'VG', 'vnw': 'VNW', 'vz': 'VZ', 'ww': 'WW'}

CatDict = {k: AtomicType(v) for k, v in _CatDict.items()}
PosDict = {k: AtomicType(v) for k, v in _PosDict.items()}
PtDict = {k: AtomicType(v) for k, v in _PtDict.items()}

# Head and modifier dependencies
HeadDeps = frozenset(['hd', 'rhd', 'whd', 'cmp', 'crd', 'det'])
ModDeps = frozenset(['mod', 'predm', 'app'])

# Obliqueness Hierarchy
ObliquenessOrder = (
    ('mod', 'app', 'predm'),  # modifiers
    ('body', 'rhd_body', 'whd_body'),  # clause bodies
    ('svp',),  # phrasal verb part
    ('ld', 'me', 'vc'),  # verb complements
    ('predc', 'obj2', 'se', 'pc', 'hdf'),  # verb secondary arguments
    ('obj1',),  # primary object
    ('pobj',),  # preliminary object
    ('su',),  # primary subject
    ('sup',),  # preliminary subject
    ('det',),  # NP head
)


ArgSeq = List[Tuple[WordType, Optional[str]]]


# Callable version
class ObliquenessSort(object):
    def __init__(self, order: Iterable[Iterable[str]]):
        order = {k: i for i, k in enumerate(reversed(list(chain.from_iterable(order))))}
        self.order = defaultdict(lambda: -1, {**order, **{'cnj': -2}})

    def __call__(self, argcolors: ArgSeq) -> ArgSeq:
        return sorted(argcolors, key=lambda x: (self.order[snd(x)], str(fst(x))), reverse=True)


_obliqueness_sort = ObliquenessSort(ObliquenessOrder)


class ExtractionError(AssertionError):
    def __init__(self, message: str, meta: Any = None):
        super().__init__(message)
        self.meta = meta


def is_gap(dag: DAG, node: Node, head_deps: FrozenSet[str] = HeadDeps) -> bool:
    incoming = set(map(lambda edge: edge.dep, dag.incoming(node)))
    return len(incoming) > 1 and len(incoming.intersection(head_deps)) > 0


def is_copy(dag: DAG, node: Node) -> bool:
    incoming = list(map(lambda edge: edge.dep, dag.incoming(node)))
    return len(incoming) > 1 and any(map(lambda inc: len(list(filter(lambda other: other == inc, incoming))) > 1,
                                         incoming))


def get_type_plain(dag: DAG, node: Node, type_dict: Dict[str, AtomicType], pos_set: str) -> AtomicType:
    if pos_set in dag.attribs[node]:
        return type_dict[dag.attribs[node][pos_set]]
    else:
        cat = dag.attribs[node]['cat']
        if cat == 'conj':
            return type_dict[majority_vote(dag, dag.successors(node), pos_set)]
        else:
            return type_dict[cat]


def make_functor(argument: WordType, result: WordType, dep: Optional[str]) -> FunctorType:
    if dep is None:
        return FunctorType(argument=argument, result=result)
    else:
        if dep in ModDeps:
            return BoxType(argument, result, dep)
        if dep == 'np_hd':
            return BoxType(argument, result, 'det')
        else:
            return DiamondType(argument, result, dep)


def make_ho_functor(argument: WordType, result: WordType, dep: Optional[str]) -> FunctorType:
    if dep is None or dep in HeadDeps or dep == 'np_hd':
        return FunctorType(argument, result)
    else:
        return make_functor(argument, result, dep)


def modifier_of(modified: WordType, dep: str) -> BoxType:
    return BoxType(argument=modified, result=modified, box=dep)


def binarize(argcolors: List[Tuple[WordType, Optional[str]]], result: WordType,
             sorting_fn: Callable[[ArgSeq], ArgSeq] = _obliqueness_sort) -> WordType:
    argcolors = sorting_fn(argcolors)
    return reduce(lambda x, y:
                  make_functor(argument=y[0], result=x, dep=y[1]), argcolors, result)


def binarize_hots(argcolors: List[Tuple[WordType, Optional[str]]], result: WordType,
                  sorting_fn: Callable[[ArgSeq], ArgSeq] = _obliqueness_sort) -> WordType:
    argcolors = sorting_fn(argcolors)
    return reduce(lambda x, y:
                  make_ho_functor(argument=y[0], result=x, dep=y[1]), argcolors, result)


def rebinarize(argcolors: List[Tuple[WordType, Optional[str]]], result: WordType,
               sorting_fn: Callable[[ArgSeq], ArgSeq] = _obliqueness_sort) -> WordType:
    if not argcolors:
        return result

    arguments, colors = list(zip(*argcolors))

    x = result
    while isinstance(x, DiamondType):
        arguments += (x.argument,)
        colors += (x.diamond,)
        x = x.result
    return binarize(list(zip(arguments, colors)), result, sorting_fn)


def type_top(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str) -> None:
    root = fst(list(dag.get_roots()))
    root_type = get_type_plain(dag, root, type_dict, pos_set)
    dag.attribs[root]['type'] = root_type


def type_bot(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: FrozenSet[str] = HeadDeps,
             mod_deps: FrozenSet[str] = ModDeps) -> bool:
    heads = set(map(lambda edge: edge.target,
                    list(filter(lambda edge: edge.dep in hd_deps, dag.edges))))
    heads = heads.difference(set(filter(lambda node: is_gap(dag, node, hd_deps), dag.nodes)))
    typed = set(filter(lambda node: 'type' in dag.attribs[node].keys(), dag.nodes))
    fringe = heads.union(typed)

    changed = False

    while True:
        fringe, attribs = type_bot_step(dag, fringe, type_dict, pos_set, hd_deps, mod_deps)
        if not attribs:
            break
        dag.attribs.update(attribs)
        changed = True

    return changed


def type_bot_step(dag: DAG[Node, str], fringe: Nodes, type_dict: Dict[str, AtomicType], pos_set: str,
                  hd_deps: FrozenSet[str] = HeadDeps, mod_deps: FrozenSet[str] = ModDeps) \
        -> Tuple[Nodes, Dict[Node, Dict]]:
    def is_fringe(_node: Node) -> bool:
        is_fn = len(set(filter(lambda edge: edge.dep in mod_deps.union({'cnj'}), dag.incoming(_node))))
        in_fringe = _node in fringe
        is_leaf = dag.is_leaf(_node)
        has_no_daughters = all(list(map(lambda out: out.dep in mod_deps
                                                    or out.target in fringe
                                                    or dag.attribs[out.source]['cat'] == 'conj',
                                        dag.outgoing(_node))))
        return not is_fn and (is_leaf or has_no_daughters) and not in_fringe

    def is_cnj_fringe(_node: Node) -> bool:
        gap = is_gap(dag, _node, hd_deps)
        typed = 'type' in dag.attribs[_node].keys()
        single_inc = len(set(dag.incoming(_node))) == 1
        is_cnj = single_inc and fst(list(dag.incoming(_node))).dep == 'cnj'
        typed_parent = single_inc and 'type' in dag.attribs[fst(list(dag.incoming(_node))).source]
        return not gap and not typed and is_cnj and typed_parent

    new_fringe = set(filter(lambda n: is_fringe(n), dag.nodes))
    new_cnj_fringe = set(filter(lambda n: is_cnj_fringe(n), dag.nodes))

    if new_cnj_fringe.intersection(new_fringe):
        raise ExtractionError('Fringes overlap', meta=dag.meta)

    fringe_types = [(node, get_type_plain(dag, node, type_dict, pos_set)) for node in new_fringe] + \
                   [(node, dag.attribs[fst(list(dag.incoming(node))).source]['type']) for node in new_cnj_fringe]

    return (new_cnj_fringe.union(new_fringe).union(fringe),
            {**{node: {**dag.attribs[node], **{'type': _type}} for node, _type in fringe_types}})


def type_mods(dag: DAG, mod_deps: FrozenSet[str] = ModDeps) -> bool:
    changed = False

    while True:
        temp = type_mods_step(dag, mod_deps)
        if not temp:
            break
        changed = True
        dag.attribs.update(temp)
    return changed


def type_mods_step(dag: DAG[Node, str], mod_deps: FrozenSet[str]) -> Dict[Node, Dict]:
    typed_nodes = set(filter(lambda node: 'type' in dag.attribs[node].keys(), dag.nodes))
    modding_edges = list(filter(lambda edge: edge.source in typed_nodes
                                             and edge.dep in mod_deps
                                             and edge.target not in typed_nodes,
                                dag.edges))
    mod_types = [(edge.target, modifier_of(dag.attribs[edge.source]['type'], edge.dep)) for edge in modding_edges]
    return {node: {**dag.attribs[node], **{'type': _type}} for node, _type in mod_types}


def type_heads_step(dag: DAG, head_deps: FrozenSet[str], mod_deps: FrozenSet[str]) -> Optional[Dict[str, Dict]]:
    def make_hd_functor(result: WordType, argcs: Tuple[WordTypes, strings]) -> WordType:
        return rebinarize(list(zip(*argcs)), result) if argcs else result

    heads_nodes: List[Tuple[Edge, List[str]]] \
        = list(map(lambda edge: (edge,
                                 order_nodes(dag, list(set(map(lambda edge: edge.target,
                                                               filter(lambda edge: edge.dep in head_deps,
                                                                      dag.outgoing(edge.source))))))),
                   filter(lambda edge: edge.dep in head_deps.difference({'crd'})
                                       and 'type' in dag.attribs[edge.source].keys()
                                       and 'type' not in dag.attribs[edge.target].keys()
                                       and not is_gap(dag, edge.target, head_deps),
                          dag.edges)))

    double_heads = list(map(lambda edge: edge.target,
                            map(fst, filter(lambda pair: fst(pair).target != fst(snd(pair)), heads_nodes))))
    single_heads = list(map(fst, filter(lambda pair: fst(pair) not in double_heads, heads_nodes)))

    heading_edges: List[Tuple[Edge, List[Edge]]] \
        = list(filter(lambda pair: all(list(map(lambda out: 'type' in dag.attribs[out.target].keys(),
                                                snd(pair)))),
                      map(lambda pair: (fst(pair),
                                        list(filter(lambda edge: edge.dep not in mod_deps
                                                                 and edge != fst(pair)
                                                                 and edge.target not in double_heads,
                                                    snd(pair)))),
                          map(lambda edge: (edge, dag.outgoing(edge.source)), single_heads))))

    targets: strings = list(map(lambda pair: fst(pair).target, heading_edges))
    types: WordTypes = list(map(lambda pair: dag.attribs[fst(pair).source]['type'], heading_edges))

    def extract_argcs(edges: List[Edge]) -> Tuple[WordTypes, strings]:
        args = list(map(lambda edge: dag.attribs[edge.target]['type'], edges))
        cs = list(map(lambda edge: edge.dep, edges))
        return args, cs

    argcolors: List[Tuple[WordTypes, strings]] = list(map(lambda pair: extract_argcs(snd(pair)),
                                                          heading_edges))

    head_types = [(node, make_hd_functor(res, argcs)) for node, res, argcs in zip(targets, types, argcolors)] + \
                 [(node, AtomicType('_')) for node in double_heads]

    return {**{node: {**dag.attribs[node], **{'type': _type}} for node, _type in head_types}}


def type_heads(dag: DAG, head_deps: FrozenSet[str] = HeadDeps, mod_deps: FrozenSet[str] = ModDeps) -> bool:
    changed = False

    while True:
        attribs = type_heads_step(dag, head_deps, mod_deps)
        if not attribs:
            break
        changed = True
        dag.attribs.update(attribs)
    return changed


def type_core(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, head_deps: FrozenSet[str],
              mod_deps: FrozenSet[str]):
    changed = True

    while changed:
        bot = type_bot(dag, type_dict, pos_set, head_deps, mod_deps)
        mod = type_mods(dag, mod_deps)

        head = type_heads(dag, head_deps, mod_deps)
        changed = mod or head or bot


def type_gaps(dag: DAG, head_deps: FrozenSet[str] = HeadDeps):
    def make_gap_functor(emb_type: WordType, interm: Tuple[WordType, str], top: Tuple[WordType, str]) -> FunctorType:
        # if isinstance(emb_type, FunctorType):
        #     # argument = FunctorType(argument=emb_type, result=fst(interm))
        # else:
        argument = DiamondType(argument=emb_type, result=fst(interm), diamond=snd(interm))
        return DiamondType(argument=argument, result=fst(top), diamond=snd(top)+'_body')

    def get_interm_top(gap: Node) -> Tuple[Tuple[WordType, str], Tuple[WordType, str]]:
        incoming = dag.incoming(gap)

        top_ = list(filter(lambda inc: inc.dep in head_deps, incoming))
        if len(set(map(lambda edge: dag.attribs[edge.source]['type'], top_))) != 1:
            raise ExtractionError('Multiple top types.')
        top = fst(top_)
        top_type: WordType = dag.attribs[top.source]['type']
        top_dep: str = top.dep

        interm = list(filter(lambda node: gap in dag.points_to(node), dag.successors(top.source)))
        if len(interm) != 1:
            raise ExtractionError('Multiple intermediate nodes.')
        interm_type: WordType = dag.attribs[fst(interm)]['type']
        interm_color_ = list(map(lambda edge: edge.dep, filter(lambda edge: edge.dep not in head_deps, incoming)))
        if len(set(interm_color_)) > 1:
            raise ExtractionError('Multiple intermediate colors.')
        interm_color = fst(interm_color_)
        return (interm_type, interm_color), (top_type, top_dep)

    gap_nodes = list(filter(lambda node: is_gap(dag, node, head_deps), dag.nodes))
    gap_nodes = list(filter(lambda node: '_gap_typed' not in dag.attribs[node].keys(), gap_nodes))

    if not gap_nodes:
        return None
    if any(list(map(lambda node: 'type' not in dag.attribs[node].keys(), gap_nodes))):
        raise ExtractionError('Untyped gap.')
    emb_types = list(map(lambda node: dag.attribs[node]['type'], gap_nodes))
    interms, tops = list(zip(*map(get_interm_top, gap_nodes)))
    gap_types_ = list(map(make_gap_functor, emb_types, interms, tops))
    gap_types = {node: {**dag.attribs[node], **{'type': gap_type, '_gap_typed': 1}}
                 for node, gap_type in zip(gap_nodes, gap_types_)}
    dag.attribs.update(gap_types)

    non_term_gaps = list(filter(lambda node: not dag.is_leaf(node), gap_nodes))
    descendants = list(chain.from_iterable(map(dag.points_to, non_term_gaps)))
    # clear type information from non-terminal gap descendants
    descendant_attribs = {node: {k: v for k, v in dag.attribs[node].items()
                                 if k != 'type' and k != '_gap_typed'} for node in descendants}
    dag.attribs.update(descendant_attribs)


def type_copies(dag: DAG[Node, str], head_deps: FrozenSet[str] = HeadDeps, mod_deps: FrozenSet[str] = ModDeps):
    def daughterhood_conditions(daughter: Edge[Node, str]) -> bool:
        return daughter.dep not in head_deps.union(mod_deps)

    def normalize_gap_copies(typecolors: List[Tuple[WordType, Set[str]]]) -> ArgSeq:
        def normalize_gap_copy(tc: Tuple[WordType, Set[str]]) -> Tuple[WordType, str]:
            if len(snd(tc)) == 1:
                return fst(tc), fst(list(snd(tc)))
            elif len(snd(tc)) == 2:
                color = fst(list(filter(lambda c: c not in head_deps, snd(tc))))
                return fst(tc).argument.argument, color if color not in mod_deps else None
            else:
                raise ExtractionError('Multi-colored copy.', meta=dag.meta)
        return list(map(normalize_gap_copy, typecolors))

    def make_polymorphic_x(initial: WordType, missing: ArgSeq) -> WordType:
        # missing = list(map(lambda pair: (fst(pair), snd(pair) if snd(pair) not in mod_deps else 'embedded'),
        #                    missing))
        return binarize_hots(missing, initial)

    def make_crd_type(poly_x: WordType, repeats: int) -> WordType:
        ret = poly_x
        while repeats:
            ret = DiamondType(argument=poly_x, result=ret, diamond='cnj')
            repeats -= 1
        return ret

    conjuncts = list(_cats_of_type(dag, 'conj'))

    gap_conjuncts = list(filter(lambda node: is_gap(dag, node, head_deps), conjuncts))
    if gap_conjuncts:
        raise ExtractionError('Gap conjunction.')

    # the edges coming out of each conjunct
    conj_outgoing_edges: List[Edges] = list(map(lambda c: dag.outgoing(c), conjuncts))

    # the list of coordinator edges coming out of each conjunct
    crds = list(map(lambda cg:
                    list(filter(lambda edge: edge.dep == 'crd', cg)),
                    conj_outgoing_edges))

    if any(list(map(lambda conj_group: len(conj_group) == 0, crds))):
        raise ExtractionError('Headless conjunction.', meta={'dag': dag.meta})

    # the list of non-excluded edges coming out of each conjunct
    conj_outgoing_edges = list(map(lambda cg:
                                   set(filter(daughterhood_conditions, cg)),
                                   conj_outgoing_edges))

    # the list of non-excluded nodes pointed by each conjunct
    conj_targets: List[Nodes] = list(map(lambda cg:
                                         set(map(lambda edge: edge.target, cg)),
                                         conj_outgoing_edges))

    # the list including only typed branches
    conj_targets = list(filter(lambda cg:
                               all(list(map(lambda daughter: 'type' in dag.attribs[daughter].keys(), cg))),
                               conj_targets))

    initial_typegroups: List[Set[WordType]] \
        = list(map(lambda conj_group: set(map(lambda daughter: dag.attribs[daughter]['type'], conj_group)),
                   conj_targets))

    if any(list(map(lambda conj_group: len(conj_group) != 1, initial_typegroups))):
        raise ExtractionError('Non-polymorphic conjunction.', meta={'dag': dag.meta})

    initial_types: WordTypes = list(map(lambda conj_group: fst(list(conj_group)), initial_typegroups))
    downsets: List[List[Nodes]] \
        = list(map(lambda group_targets:
                   list(map(lambda daughter: dag.points_to(daughter).union({daughter}),
                            group_targets)),
                   conj_targets))
    common_downsets: List[Nodes] = list(map(lambda downset: set.intersection(*downset), downsets))
    minimal_downsets: List[Nodes] = list(map(lambda downset:
                                             set(filter(lambda node:
                                                        len(dag.pointed_by(node).intersection(downset)) == 0,
                                                        downset)),
                                             common_downsets))

    accounted_copies = set.union(*minimal_downsets) if common_downsets else set()
    all_copies = set(filter(lambda node: is_copy(dag, node), dag.nodes))

    if accounted_copies != all_copies:
        raise ExtractionError('Unaccounted copies.', meta=dag.meta)
    if any(list(map(lambda acc: 'type' not in dag.attribs[acc].keys(), accounted_copies))):
        raise ExtractionError('Untyped copies.', meta=dag.meta)

    copy_colorsets = list(map(lambda downset: list(map(lambda node: (dag.attribs[node]['type'],
                                                                     set(map(lambda edge: edge.dep,
                                                                             dag.incoming(node)))),
                                                       downset)),
                              minimal_downsets))

    copy_types_and_colors = list(map(normalize_gap_copies, copy_colorsets))

    polymorphic_xs = list(map(make_polymorphic_x, initial_types, copy_types_and_colors))
    crd_types = list(map(make_crd_type, polymorphic_xs, list(map(len, conj_targets))))
    secondary_crds = list(map(lambda crd: crd.target,
                              chain.from_iterable(crd[1::] for crd in crds)))
    primary_crds = list(map(lambda crd: crd.target, map(fst, crds)))
    copy_types = {crd: {**dag.attribs[crd], **{'type': crd_type}} for crd, crd_type in zip(primary_crds, crd_types)}
    dag.attribs.update(copy_types)
    secondary_types = {crd: {**dag.attribs[crd], **{'type': AtomicType('_')}} for crd in secondary_crds}
    dag.attribs.update(secondary_types)


def type_dag(dag: DAG[str, str], type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: FrozenSet[str],
             mod_deps: FrozenSet[str], check: bool = True) -> DAG:
    def fully_typed(dag_: DAG) -> bool:
        return all(list(map(lambda node: 'type' in dag.attribs[node].keys(), dag_.nodes)))

    type_top(dag, type_dict, pos_set)
    while not fully_typed(dag):
        type_core(dag, type_dict, pos_set, hd_deps, mod_deps)
        type_gaps(dag, hd_deps)
        type_copies(dag, hd_deps, mod_deps)

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


def untype_dag(dag: DAG) -> None:
    for k in dag.attribs.keys():
        if 'type' in dag.attribs[k].keys():
            del dag.attribs[k]['type']
        if '_gap_typed' in dag.attribs[k].keys():
            del dag.attribs[k]['_gap_typed']


class Extraction(object):
    def __init__(self, cat_dict: Dict[str, AtomicType], pos_dict: Dict[str, AtomicType], pos_set: str,
                 head_deps: FrozenSet[str], mod_deps: FrozenSet[str],
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
                return None


extractor = Extraction(CatDict, PtDict, 'pt', HeadDeps, ModDeps)
