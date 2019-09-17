from src.graphutils import *
from src.milltypes import AtomicType, WordType, ColoredType, WordTypes, strings, binarize
from src.transformations import majority_vote
from collections import defaultdict
from itertools import chain


# # # Extraction variables # # #

# Mapping from phrasal categories and POS tags to Atomic Types
cat_dict = {'advp': 'ADV', 'ahi': 'AHI', 'ap': 'AP', 'cp': 'CP', 'detp': 'DETP', 'inf': 'INF', 'np': 'NP',
            'oti': 'OTI', 'pp': 'PP', 'ppart': 'PPART', 'ppres': 'PPRES', 'rel': 'REL', 'smain': 'SMAIN',
            'ssub': 'SSUB', 'sv1': 'SV1', 'svan': 'SVAN', 'ti': 'TI', 'whq': 'WHQ', 'whrel': 'WHREL',
            'whsub': 'WHSUB'}
pos_dict = {'adj': 'ADJ', 'adv': 'ADV', 'comp': 'COMP', 'comparative': 'COMPARATIVE', 'det': 'DET',
            'fixed': 'FIXED', 'name': 'NAME', 'noun': 'N', 'num': 'NUM', 'part': 'PART',
            'prefix': 'PREFIX', 'prep': 'PREP', 'pron': 'PRON', 'punct': 'PUNCT', 'tag': 'TAG',
            'verb': 'VERB', 'vg': 'VG'}
pt_dict = {'adj': 'ADJ', 'bw': 'BW', 'let': 'LET', 'lid': 'LID', 'n': 'N', 'spec': 'SPEC', 'tsw': 'TSW',
           'tw': 'TW', 'vg': 'VG', 'vnw': 'VNW', 'vz': 'VZ', 'ww': 'WW'}
cat_dict = {k: AtomicType(v) for k, v in cat_dict.items()}
pos_dict = {k: AtomicType(v) for k, v in pos_dict.items()}
pt_dict = {k: AtomicType(v) for k, v in pt_dict.items()}

# Head and modifier dependencies
head_deps = {'hd', 'rhd', 'whd', 'cmp', 'crd'}
mod_deps = {'mod', 'predm', 'app'}

# Obliqueness Hierarchy
obliqueness_order = (
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


obliqueness_sort = ObliquenessSort(obliqueness_order)


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


def type_bot(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: Set[str], mod_deps: Set[str]):
    heads = set(map(lambda edge: edge.target, list(filter(lambda edge: edge.dep in hd_deps, dag.edges))))
    heads = heads.difference(set(filter(lambda node: is_gap(dag, node, hd_deps), dag.nodes)))
    temp = type_bot_step(dag, type_dict, pos_set, mod_deps, heads)
    while temp is not None:
        fringe, attribs = temp
        dag.attribs.update(attribs)
        temp = type_bot_step(dag, type_dict, pos_set, mod_deps, fringe)


def type_bot_step(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, mod_deps: Set[str],
                  fringe: Nodes) -> Optional[Tuple[Nodes, Dict[Node, Dict]]]:
    def is_fringe(node: Node) -> bool:
        return (not len(set(filter(lambda edge: edge.dep in mod_deps and edge.target == node, dag.edges)))) \
               and (dag.is_leaf(node) or all(list(map(lambda out: out.dep in mod_deps or out.target in fringe,
                                                      dag.outgoing(node))))) \
               and node not in fringe

    new_fringe = set(filter(is_fringe, dag.nodes))
    if not new_fringe:
        return
    return (new_fringe.union(fringe),
            {node: {**dag.attribs[node], **{'type': get_type_plain(dag, node, type_dict, pos_set)}} for node
             in new_fringe})


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
        return rebinarize(obliqueness_sort, fst(argcolors), snd(argcolors), res, mod_deps)

    heading_edges = list(filter(lambda edge: edge.dep in head_deps.difference({'crd'})
                                             and 'type' in dag.attribs[edge.source].keys()
                                             and 'type' not in dag.attribs[edge.target].keys()
                                             and not is_gap(dag, edge.target, head_deps),
                                dag.edges))

    heading_edges = list(map(lambda edge: (edge, dag.outgoing(edge.source)), heading_edges))
    heading_edges = list(map(lambda pair: (fst(pair),
                                           list(filter(lambda edge:
                                                       edge.dep not in mod_deps and edge != fst(pair),
                                                       snd(pair)))),
                             heading_edges))

    heading_edges = list(filter(lambda pair: all(list(map(lambda out: 'type' in dag.attribs[out.target].keys(),
                                                          snd(pair)))),
                                heading_edges))
    if not heading_edges:
        return
    heading_argcs = list(map(lambda pair:
                             (fst(pair).target,
                              dag.attribs[fst(pair).source]['type'],
                              list(zip(*map(lambda out: (dag.attribs[out.target]['type'], out.dep), snd(pair))))),
                             heading_edges))
    head_types = {node: {**dag.attribs[node], **{'type': make_functor(res, argcolors) if argcolors else res}}
                  for (node, res, argcolors) in heading_argcs}
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


def type_head_mods(dag: DAG, head_deps: Set[str], mod_deps: Set[str]):
    changed = True
    while changed:
        mod = type_mods(dag, mod_deps)
        head = type_heads(dag, head_deps, mod_deps)
        changed = mod or head


def type_gaps(dag: DAG, head_deps: Set[str], mod_deps: Set[str]):
    def make_gap_functor(emb_type: WordType, interm: Tuple[WordType, str], top: Tuple[WordType, str]) -> ColoredType:
        if snd(interm) in mod_deps:
            argument = ColoredType(argument=emb_type, result=fst(interm), color='embedded')
        else:
            argument = ColoredType(argument=emb_type, result=fst(interm), color=snd(interm))
        return ColoredType(argument=argument, result=fst(top), color=snd(top))

    def get_interm_top(gap: Node) -> Tuple[Tuple[WordType, str], Tuple[WordType, str]]:
        incoming = dag.incoming(gap)
        interm = list(filter(lambda inc: inc.dep not in head_deps, incoming))
        top = list(filter(lambda inc: inc.dep in head_deps, incoming))
        assert len(interm) == len(top) == 1
        interm = fst(interm)
        interm = dag.attribs[interm.source]['type'], interm.dep
        top = fst(top)
        top = dag.attribs[top.source]['type'], top.dep
        return interm, top

    gap_nodes = list(filter(lambda node: is_gap(dag, node, head_deps), dag.nodes))
    if not gap_nodes:
        return
    emb_types = list(map(lambda node: dag.attribs[node]['type'], gap_nodes))
    interms, tops = list(zip(*map(get_interm_top, gap_nodes)))
    gap_types = list(map(make_gap_functor, emb_types, interms, tops))
    gap_types = {node: {**dag.attribs[node], **{'type': gap_type}} for node, gap_type in zip(gap_nodes, gap_types)}
    dag.attribs.update(gap_types)


def is_gap(dag: DAG, node: Node, head_deps: Set[str]) -> bool:
    incoming = set(map(lambda edge: edge.dep, dag.incoming(node)))
    return len(incoming) > 1 and len(incoming.intersection(head_deps)) > 0


def placeholder(dag: DAG, type_dict: Dict[str, AtomicType], pos_set: str, hd_deps: Set[str], mod_deps: Set[str]) -> DAG:
    type_top(dag, type_dict, pos_set)
    type_bot(dag, type_dict, pos_set, hd_deps, mod_deps)
    type_head_mods(dag, head_deps, mod_deps)
    type_gaps(dag, head_deps, mod_deps)
    return dag


class Extraction(object):
    def __init__(self, cat_dict: Dict[str, AtomicType], pos_dict: Dict[str, AtomicType], pos_set: str,
                 head_deps: Set[Dep], mod_deps: Set[Dep],
                 ):
        self.type_dict = {**cat_dict, **pos_dict}
        self.pos_set = pos_set
        self.head_deps = head_deps
        self.mod_deps = mod_deps

    def __call__(self, dag: DAG) -> DAG:
        return placeholder(dag, self.type_dict, self.pos_set, self.head_deps, self.mod_deps)


typer = Extraction(cat_dict, pt_dict, 'pt', head_deps, mod_deps)