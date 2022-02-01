"""
    Pipeline for assigning a proof to a transformed tree and its subtrees.
"""


import pdb
from .mill.types import Type, Atom, Functor, Diamond, Box, Proof, T
from .transformations import DAG, is_ghost, node_to_key, get_material, find_coindexed
from functools import reduce
from typing import Iterable, Callable


class ExtractionError(Exception):
    pass


Atoms = {'adj':     (ADJ    := Atom('ADJ')),
         'bw':      (BW     := Atom('BW')),
         'punct':   (PUNCT  := Atom('PUNCT')),
         'lid':     (LID    := Atom('LID')),
         'n':       (N      := Atom('N')),
         'spec':    N,
         'vnw':     (VNW    := Atom('VNW', (N,))),
         'np':      (NP     := Atom('NP', (N,))),
         'tsw':     (TSW    := Atom('TSW')),
         'tw':      (TW     := Atom('TW')),
         'vg':      (VG     := Atom('VG')),
         'vz':      (VZ     := Atom('VZ')),
         'ww':      (INF     := Atom('INF')),
         'advp':    (ADV    := Atom('ADV')),
         'ahi':     (AHI    := Atom('AHI')),
         'ap':      (AP     := Atom('AdjP')),
         'cp':      (CP     := Atom('CP')),
         'inf':     INF,
         'detp':    (DETP   := Atom('DETP')),
         'oti':     (OTI    := Atom('OTI')),
         'ti':      (TI     := Atom('TI')),
         'pp':      (PP     := Atom('PP')),
         'ppart':   (PPART  := Atom('PPART')),
         'ppres':   (PPRES  := Atom('PPRES')),
         'rel':     (REL    := Atom('REL')),
         # sentential types
         's':       (S      := Atom('S')),
         'smain':   (Smain  := Atom('SMAIN', (S,))),
         'ssub':    (Ssub   := Atom('SSUB', (S,))),
         'sv1':     (Sv1    := Atom('SV1', (S,))),
         'svan':    (Svan   := Atom('SVAN', (S,))),
         'whq':     (WHq    := Atom('WHQ', (S,))),
         'whrel':   (WHrel  := Atom('WHREL', (S,))),
         'whsub':   (WHsub  := Atom('WHSUB', (S,)))}


head_labels = {'hd', 'rhd', 'whd', 'cmp', 'np_head', 'crd'}
adjunct_labels = {'mod', 'app', 'predm'}


ObliquenessOrder = (
    ('crd', 'hd', 'np_head', 'whd', 'rhd', 'cmp'),  # heads
    ('mod', 'app', 'predm'),                        # modifiers
    ('body', 'relcl', 'whbody', 'cmpbody'),         # clause bodies
    ('sup',),                                       # preliminary subject
    ('su',),                                        # primary subject
    ('pobj',),                                      # preliminary object
    ('obj1',),                                      # primary object
    ('predc', 'obj2', 'se', 'pc', 'hdf'),           # verb secondary arguments
    ('ld', 'me', 'vc'),                             # verb complements
    ('obcomp',),                                    # comparison complement
    ('svp',),                                       # separable verb part
    ('det',))                                       # determiner head


def dep_to_key(dep: str) -> int:
    return next((i for i, d in enumerate(sum(ObliquenessOrder, ())) if d == dep), -1)


def assign_proof(
        fn: [[], T]) -> Callable[[DAG[str], str, str | None, T | None], T]:
    def f(dag: DAG[str], root: str | None, label: str | None = None, hint: T | None = None) -> T:
        dag.attribs[root]['proof'] = (proof := fn(dag, root, label, hint))
        return proof
    return f


def make_functor(result: T, arguments: list[Type]) -> T:
    return reduce(lambda x, y: Functor(y, x), arguments, result)


def unbox_and_apply(original: T, boxes: list[T]) -> T:
    return reduce(lambda x, y: Proof.apply(Proof.unbox(y), x), reversed(boxes), original)


def apply(functor: T, arguments: list[T]) -> T:
    return reduce(Proof.apply, reversed(arguments), functor)


def endo_of(_type: T) -> T:
    return Functor(_type, _type)


def abstract(term: T, abstraction_condition: Callable[[T], bool]) -> T:
    if term.rule == Proof.Rule.Axiom:
        return term
    abstractions = [f for f in term.free() if abstraction_condition(f)]
    for var in reversed(abstractions):
        term = Proof.abstract(var, term)
    return term


@assign_proof
def _prove(dag: DAG, root: str, label: str | None, hint: T, ) -> T:
    # utilities
    def coindexed_with(nodes: set[str]) -> Callable[[T], bool]:
        indices = {index for node in nodes if (index := dag.get(node, 'index')) is not None}
        return lambda proof: dag.get(str(proof.variable), 'index') in indices

    def make_adj(_adjuncts: list[tuple[str, str]], _top_type: T) -> list[T]:
        return [_prove(dag, _child, _label, Box(_label, endo_of(_top_type))) for _label, _child in _adjuncts]

    def make_args(
            _dependents: list[tuple[str, str]],
            abstract_if: Callable[[T], bool] = lambda _: False,
            forced_type: T | None = None) -> list[T]:
        return [Proof.diamond(_label, abstract(_term, abstract_if))
                if _term.rule != Proof.Rule.Axiom else abstract(_term, abstract_if)
                for _label, _term in
                [(_label, _prove(dag, _child, _label, forced_type)) for _label, _child in _dependents]
                if _label != 'punct']

    def shared_and_distributed(_roots: list[tuple[str, str]],
                               siblings: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        indices = {index for node in set().union(*[dag.successors(_root) for _, _root in _roots])
                   if (index := dag.get(node, 'index')) is not None}
        return ((shared := [(dep, node) for dep, node in siblings if dag.get(node, 'index') in indices]),
                [sib for sib in siblings if sib not in shared])

    def get_type_of(_node_id: str, _conjuncts: list[str]) -> T:
        coindexed = find_coindexed(dag, dag.get(_node_id, 'index'))
        per_conjunct = [coindexed & set(dag.successors(_conjunct)) for _conjunct in _conjuncts]
        # if len(set(map(len, per_conjunct))) != 1:
        #     pdb.set_trace()
        abstraction_types = [type(dag.get(var, 'proof')) for c in per_conjunct for var in c]
        if not abstraction_types:
            raise ExtractionError('No type found for {}'.format(_node_id))
            # todo: see e.g. WR-P-E-I-0000027216.p.1.s.23.xml
        # todo: assertion about polymorphism, or choice of most general abstraction type
        # if len(set(abstraction_types)) != 1:
        #     print([(print_dag(dag, next(iter(pc))), abst) for pc, abst in zip(per_conjunct, abstraction_types)])

        return next(iter(abstraction_types))

    # terminal case
    if dag.is_leaf(root):
        material, node_id = get_material(dag, root), int(dag.get(root, 'id'))
        if hint is not None:
            return hint.var(node_id) if is_ghost(dag, root) else hint.con(node_id)
        match is_ghost(dag, root), dag.get(material, 'cat'), dag.get(material, 'pt'):
            case False, ('mwu' | None), pt:
                return Atoms[pt].con(node_id)
            case False, cat, None:
                return Atoms[cat].con(node_id)
            case True, cat, pt:
                # variables come with their modalities preassigned
                node_type = Atoms[pt] if pt is not None else Atoms[cat]
                if label in adjunct_labels:
                    node_type = Box(label, node_type)
                elif label not in head_labels:
                    node_type = Diamond(label, node_type)
                return node_type.var(node_id)
            case _: raise ValueError

    # non-terminal case
    top_type = Atoms[dag.get(root, 'cat')] if hint is None else hint
    match split_children(dag, root):
        case [], dets, [('np_head', head)], adjuncts, arguments, []:
            # print(f'{root} case 1 ')
            # noun phrase [possibly with determiner, adjuncts and arguments]
            # todo: when do I actually have arguments in a determiner phrase?
            np_term = _prove(dag, head, None, None)
            arg_terms = make_args(arguments)
            det_terms = [
                _prove(dag, det, None, Box('det', make_functor(top_type, [type(a) for a in arg_terms + [np_term]])))
                for _, det in dets]
            adj_terms = make_adj(adjuncts, top_type)
            assert len(det_terms) <= 1
            return unbox_and_apply(apply(unbox_and_apply(np_term, det_terms), arg_terms), adj_terms)
        case [], [], [(_, head)], adjuncts, arguments, []:
            # print(f'{root} case 2 ')
            # simple or relative clause
            arg_terms = make_args(arguments, coindexed_with({head}))
            adj_terms = make_adj(adjuncts, top_type)
            head_term = _prove(dag, head, None, make_functor(top_type, [type(a) for a in arg_terms]))
            return unbox_and_apply(apply(head_term, arg_terms), adj_terms)
        # todo: consider complex arguments that share material deeper in the tree, both in case 3 and 4
        case conjuncts, [], [('crd', crd), *heads], adjuncts, arguments, correlatives:
            # print(f'{root} case 3 ')
            # conjunction of clauses possibly sharing adjuncts, arguments and/or heads
            shared_adjuncts, distributed_adjuncts = shared_and_distributed(conjuncts, adjuncts)
            shared_nodes = {node for _, node in heads + shared_adjuncts + arguments}
            cnj_terms = make_args(conjuncts, coindexed_with(shared_nodes), top_type)
            hd_terms = [_prove(dag, head, None, get_type_of(head, [cnj for _, cnj in conjuncts])) for _, head in heads]
            shared_adj_terms = make_adj(shared_adjuncts, top_type)
            dist_adj_terms = make_adj(distributed_adjuncts, top_type)
            cor_terms, arg_terms = make_args(correlatives),  make_args(arguments)
            all_args = shared_adj_terms + arg_terms + hd_terms + cnj_terms + cor_terms
            crd_type = make_functor(top_type, [type(a) for a in all_args])
            crd_term = _prove(dag, crd, None, crd_type)
            return unbox_and_apply(apply(crd_term, all_args), dist_adj_terms)
        case conjuncts, [('det', det)], [('crd', crd)], adjuncts, arguments, correlatives:
            # print(f'{root} case 4 ')
            # conjunction of noun phrases sharing their determiner (and possibly adjuncts)
            shared_adjuncts, distributed_adjuncts = shared_and_distributed(conjuncts, adjuncts)
            shared_nodes = {node for _, node in shared_adjuncts + arguments} | {det}
            cnj_terms = make_args(conjuncts, coindexed_with(shared_nodes), top_type)
            det_term = _prove(dag, det, None, get_type_of(det, [cnj for _, cnj in conjuncts]))
            shared_adj_terms = make_adj(shared_adjuncts, top_type)
            dist_adj_terms = make_adj(distributed_adjuncts, top_type)
            cor_terms, arg_terms = make_args(correlatives),  make_args(arguments)
            all_args = shared_adj_terms + arg_terms + [det_term] + cnj_terms + cor_terms
            crd_type = make_functor(top_type, [type(a) for a in all_args])
            crd_term = _prove(dag, crd, None, crd_type)
            return unbox_and_apply(apply(crd_term, all_args), dist_adj_terms)
        case [*cs], _, [*heads], _, _, _:
            raise ExtractionError('Headless conjunction')
        case _:
            pdb.set_trace()
            raise ExtractionError('Unhandled case')


def split_children(dag: DAG[str], root: str) -> tuple[list[tuple[str, str]], ...]:
    def depsort(children: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(children, key=lambda lc: dep_to_key(lc[0]))

    def nodesort(children: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(children, key=lambda lc: node_to_key(dag, lc[1]))

    outgoing = {(edge.label, edge.target) for edge in dag.outgoing_edges(root)}
    conjuncts = {(label, child) for label, child in outgoing if label == 'cnj'}
    adjuncts = {(label, child) for label, child in outgoing if label in adjunct_labels}
    heads = {(label, child) for label, child in outgoing if label in head_labels}
    dets = {(label, child) for label, child in outgoing if label == 'det'}
    cors = {(label, child) for label, child in outgoing if label == 'cor'}
    dependents = outgoing - conjuncts - adjuncts - heads - dets - cors
    return nodesort(conjuncts), nodesort(dets), depsort(heads), nodesort(adjuncts), depsort(dependents), nodesort(cors)


def prove(dag: DAG[str]) -> T:
    if len(roots := dag.get_roots()) > 1:
        raise ValueError('Multiple roots')
    proof = _prove(dag, next(iter(roots)), None, None)
    if proof.free():
        raise ExtractionError('Free variables unaccounted for')
    return proof
