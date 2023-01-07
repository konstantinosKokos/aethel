from .mill.proofs import Proof, variable, constant, Logical, Structural, make_extractable, deep_extract
from .mill.types import Atom, Type, Functor, Diamond, Box
from .mill.terms import Variable
from .transformations import DAG, is_ghost, node_to_key, get_material, find_coindexed, is_bottom
from typing import Callable, Iterable
from functools import reduce


class ExtractionError(Exception):
    pass


Atoms = {'adj':     (ADJ    := Atom('ADJ')),
         'bw':      (BW     := Atom('BW')),
         'let':     (PUNCT  := Atom('PUNCT')),
         'lid':     (LID    := Atom('LID')),
         'n':       (N      := Atom('N')),
         'np':      (NP     := Atom('NP')),
         'spec':    NP,
         'vnw':     (VNW    := Atom('VNW')),
         'tsw':     (TSW    := Atom('TSW')),
         'tw':      (TW     := Atom('TW')),
         'vg':      (VG     := Atom('VG')),
         'vz':      (VZ     := Atom('VZ')),
         'ww':      (WW    := Atom('WW')),
         'advp':    (ADV    := Atom('ADV')),
         'ahi':     (AHI    := Atom('AHI')),
         'ap':      (AP     := Atom('AdjP')),
         'cp':      (CP     := Atom('CP')),
         'inf':     (INF    := Atom('INF')),
         'detp':    (DETP   := Atom('DETP')),
         'oti':     (OTI    := Atom('OTI')),
         'ti':      (TI     := Atom('TI')),
         'pp':      (PP     := Atom('PP')),
         'ppart':   (PPART  := Atom('PPART')),
         'ppres':   (PPRES  := Atom('PPRES')),
         'rel':     (REL    := Atom('REL')),
         # sentential types
         'smain':   (Smain  := Atom('SMAIN')),
         'ssub':    (Ssub   := Atom('SSUB')),
         'sv1':     (Sv1    := Atom('SV1')),
         'svan':    (Svan   := Atom('SVAN')),
         'whq':     (WHq    := Atom('WHQ')),
         'whrel':   (WHrel  := Atom('WHREL')),
         'whsub':   (WHsub  := Atom('WHSUB'))}


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


def make_functor(result: Type, arguments: list[Type]) -> Type:
    return reduce(lambda x, y: Functor(y, x), arguments, result)  # type: ignore


def unbox_and_apply(original: Proof, boxes: list[Proof]) -> Proof:
    def go(f: Proof) -> Proof:
        unboxed = f.unbox()
        if isinstance(f.term, Variable):
            return unboxed.undiamond(where=f.term,
                                     becomes=variable(Diamond(unboxed.term.decoration, f.term.type),
                                                      f.term.index))
        return unboxed
    return reduce(lambda x, y: go(y) @ x, reversed(boxes), original)


def apply(function: Proof, arguments: list[Proof]) -> Proof: return reduce(Proof.apply, reversed(arguments), function)
def endo(of: Type) -> Type: return Functor(of, of)


def abstract(proof: Proof, condition: Callable[[Variable], bool]) -> Proof:
    def var_to_key(_var: Variable) -> int:
        return dep_to_key(_var.type.decoration) if isinstance(_var.type, (Diamond, Box)) else -1

    if proof.rule == Logical.Variable:
        return proof
    variables = sorted([(ctx, v) for ctx, v in proof.vars() if condition(v)],
                       key=lambda pair: var_to_key(pair[1]))

    for ctx, var in variables:
        if ctx:
            proof, var = make_extractable(proof, var)
            proof, var = deep_extract(proof, var)
        proof = proof.abstract(var)
    return proof


def prove_dag(dag: DAG) -> Proof:
    return rename_duplicates(prove(dag, next(iter(dag.get_roots())), None, None, set()))


def prove(dag: DAG, root: str, label: str | None, hint: Type | None, coindex_ctx: set[str]) -> Proof:
    dag.attribs[root]['proof'] = (ret := _prove(dag, root, label, hint, coindex_ctx))
    return ret


def _prove(dag: DAG, root: str, label: str | None, hint: Type | None, coindex_ctx: set[str]) -> Proof:
    def nodes_to_indices(nodes: Iterable[str]) -> set[str]:
        return {index for node in nodes if (index := dag.get(node, 'index')) is not None}

    def shared_content(branch_roots: Iterable[str]) -> set[str]:
        return set.intersection(
            *({index for node in dag.successors(branch_root) if (index := dag.get(node, 'index')) is not None}
              for branch_root in branch_roots))

    def coindexed_with(indices: set[str], ids: set[str]) -> Callable[[Variable], bool]:
        def go(_var: Variable) -> bool:
            node = str(_var.index)
            return dag.get(node, 'index') in indices and node not in ids
        return go

    def make_adj(_adjuncts: list[tuple[str, str]], _top_type: Type) -> list[Proof]:
        return [prove(dag, _child, _label, Box(_label, endo(_top_type)), set()) for _label, _child in _adjuncts]

    def make_args(_dependents: list[tuple[str, str]],
                  _coindex_ctx: set[str],
                  abstract_if: Callable[[Variable], bool] = lambda _: False,
                  forced_type: Type | None = None) -> list[Proof]:
        return [_proof.diamond(_label) if _proof.rule != Logical.Variable else _proof for _label, _proof in
                [(_label, abstract(prove(dag, _child, _label, forced_type, _coindex_ctx), abstract_if))
                 for _label, _child in _dependents]
                if _label != 'punct']

    def shared_and_distributed(_roots: list[tuple[str, str]],
                               siblings: list[tuple[str, str]]) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        indices = {index for node in set().union(*[dag.successors(_root) for _, _root in _roots])
                   if (index := dag.get(node, 'index')) is not None}
        return ((shared := [(dep, node) for dep, node in siblings if dag.get(node, 'index') in indices]),
                [sib for sib in siblings if sib not in shared])

    def get_type_of(_node_id: str, _conjuncts: list[str]) -> Type:
        coindexed = find_coindexed(dag, dag.get(_node_id, 'index'))
        per_conjunct = [coindexed & set(dag.successors(_conjunct)) for _conjunct in _conjuncts]
        abstraction_types = [dag.get(var, 'proof').type for c in per_conjunct for var in c]
        if not abstraction_types:
            raise ExtractionError(f'Missing type information.')
        # todo: polymorphism assertion or type coercion
        return next(iter(abstraction_types))

    # terminal case
    if is_bottom(dag, root):
        material, node_id = get_material(dag, root), int(dag.get(root, 'id'))
        if hint is not None:
            return (variable if is_ghost(dag, root) else constant)(hint, node_id)
        match is_ghost(dag, root), dag.get(material, 'cat'), dag.get(material, 'pt'):
            case False, ('mwu' | None), pt: return constant(Atoms[pt], node_id)
            case False, cat, None: return constant(Atoms[cat], node_id)
            case True, cat, pt:
                node_type = Atoms[pt] if pt is not None else Atoms[cat]
                if label in adjunct_labels:
                    node_type = Box(label, node_type)
                elif label not in head_labels:
                    node_type = Diamond(label, node_type)
                return variable(node_type, node_id)
            case _: raise ExtractionError(f'No pattern for node {dag.get(root)}')

    # non-terminal case
    top_type = Atoms[dag.get(root, 'cat')] if hint is None else hint
    match split_children(dag, root):
        case [], dets, [('np_head', head)], adjuncts, arguments, []:
            np_proof = prove(dag, head, 'np_head', None, coindex_ctx)
            arg_proofs = make_args(arguments, coindex_ctx)
            det_proofs = [
                prove(dag=dag,
                      root=det,
                      label='det',
                      hint=Box('det', make_functor(top_type, [a.type for a in arg_proofs + [np_proof]])),
                      coindex_ctx=coindex_ctx)
                for _, det in dets]
            adj_proofs = make_adj(adjuncts, top_type)
            assert len(det_proofs) <= 1
            return unbox_and_apply(apply(unbox_and_apply(np_proof, det_proofs), arg_proofs), adj_proofs)
        case [], [], [(_, head)], adjuncts, arguments, []:
            coindex_ctx = {*coindex_ctx, *(a for d, a in arguments if d == 'su')}
            coindex_against = {head, *coindex_ctx}
            arg_proofs = make_args(_dependents=arguments,
                                   _coindex_ctx=coindex_ctx,
                                   abstract_if=coindexed_with(nodes_to_indices(coindex_against), coindex_against))
            adj_proofs = make_adj(adjuncts, top_type)
            head_term = prove(dag, head, 'hd', make_functor(top_type, [a.type for a in arg_proofs]), coindex_ctx)
            return unbox_and_apply(apply(head_term, arg_proofs), adj_proofs)
        case conjuncts, [], [('crd', crd), *heads], adjuncts, arguments, correlatives:
            shared_adjuncts, distributed_adjuncts = shared_and_distributed(conjuncts, adjuncts)
            shared_indices = nodes_to_indices(node for _, node in heads + shared_adjuncts + arguments)
            shared_indices |= shared_content(c for _, c in conjuncts)

            cnj_proofs = make_args(_dependents=conjuncts,
                                   _coindex_ctx=coindex_ctx,
                                   abstract_if=coindexed_with(shared_indices, set()),
                                   forced_type=top_type)
            hd_proofs = [prove(dag=dag,
                               root=head,
                               label='hd',
                               hint=get_type_of(head, [cnj for _, cnj in conjuncts]),
                               coindex_ctx=coindex_ctx)
                         for _, head in heads]
            shared_adj_proofs = make_adj(shared_adjuncts, top_type)
            dist_adj_proofs = make_adj(distributed_adjuncts, top_type)
            cor_proofs = make_args(_dependents=correlatives, _coindex_ctx=set())
            arg_proofs = make_args(_dependents=arguments, _coindex_ctx=coindex_ctx)
            all_args = shared_adj_proofs + arg_proofs + hd_proofs + cnj_proofs + cor_proofs
            crd_type = make_functor(top_type, [a.type for a in all_args])
            crd_proof = prove(dag, crd, 'crd', crd_type, coindex_ctx)
            return unbox_and_apply(apply(crd_proof, all_args), dist_adj_proofs)
        case conjuncts, [('det', det)], [('crd', crd)], adjuncts, arguments, correlatives:
            shared_adjuncts, distributed_adjuncts = shared_and_distributed(conjuncts, adjuncts)
            shared_nodes = {node for _, node in shared_adjuncts + arguments} | {det}
            cnj_proofs = make_args(_dependents=conjuncts,
                                   _coindex_ctx=coindex_ctx,
                                   abstract_if=coindexed_with(nodes_to_indices(shared_nodes), set()),
                                   forced_type=top_type)
            det_term = prove(dag=dag,
                             root=det,
                             label='det',
                             hint=get_type_of(det, [cnj for _, cnj in conjuncts]),
                             coindex_ctx=coindex_ctx)
            shared_adj_proofs = make_adj(shared_adjuncts, top_type)
            dist_adj_proofs = make_adj(distributed_adjuncts, top_type)
            cor_proofs = make_args(_dependents=correlatives, _coindex_ctx=set())
            arg_proofs = make_args(_dependents=arguments, _coindex_ctx=coindex_ctx)
            all_args = shared_adj_proofs + arg_proofs + [det_term] + cnj_proofs + cor_proofs
            crd_type = make_functor(top_type, [a.type for a in all_args])
            crd_proof = prove(dag, crd, 'crd', crd_type, set())
            return unbox_and_apply(apply(crd_proof, all_args), dist_adj_proofs)
        case [_, *_], _, _, _, _, _:
            raise ExtractionError('Headless conjunction')
        case _:
            raise ExtractionError('Unhandled case')


def split_children(dag: DAG[str], root: str) -> tuple[list[tuple[str, str]], ...]:
    def depsort(children: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(children, key=lambda lc: dep_to_key(lc[0]))

    def nodesort(children: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        return sorted(children, key=lambda lc: node_to_key(dag, lc[1]))

    outgoing = {(edge.label, edge.target) for edge in dag.outgoing_edges(root) if edge.label != 'mwp'}
    conjuncts = {(label, child) for label, child in outgoing if label == 'cnj'}
    adjuncts = {(label, child) for label, child in outgoing if label in adjunct_labels}
    heads = {(label, child) for label, child in outgoing if label in head_labels}
    dets = {(label, child) for label, child in outgoing if label == 'det'}
    cors = {(label, child) for label, child in outgoing if label == 'cor'}
    dependents = outgoing - conjuncts - adjuncts - heads - dets - cors
    return nodesort(conjuncts), nodesort(dets), depsort(heads), nodesort(adjuncts), depsort(dependents), nodesort(cors)


def rename_duplicates(proof: Proof) -> Proof:
    def go(_proof: Proof, counter: int, trans: dict[int, int]) -> tuple[Proof, int, dict[int, int]]:
        match _proof.rule:
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                focus = _proof.focus
                trans[focus.index] = (counter := counter + 1)
                body, counter, trans = go(body, counter, trans)
                return body.abstract(Variable(focus.type, trans[focus.index])), counter, trans
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                focus = _proof.focus
                ye_ole_trans = trans[focus.index]
                becomes, counter, trans = go(becomes, counter, trans)
                trans[focus.index] = (counter := counter + 1)
                original, counter, trans = go(original, counter, trans)
                ye_new_trans = trans[focus.index]
                trans[focus.index] = ye_ole_trans
                return original.undiamond(Variable(focus.type, ye_new_trans), becomes), counter, trans
            case Logical.Variable:
                index = _proof.term.index
                return variable(_proof.type, trans[index]), counter, trans
            case Logical.Constant:
                return _proof, counter, trans
            case Logical.ArrowElimination:
                (fn, arg) = _proof.premises
                fn, counter, trans = go(fn, counter, trans)
                arg, counter, trans = go(arg, counter, trans)
                return fn @ arg, counter, trans
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                body, counter, trans = go(body, counter, trans)
                return body.diamond(struct.brackets), counter, trans
            case Logical.BoxElimination:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                body, counter, trans = go(body, counter, trans)
                return body.unbox(struct.brackets), counter, trans
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                (struct,) = body.structure
                body, counter, trans = go(body, counter, trans)
                return body.box(struct.brackets), counter, trans
            case Structural.Extract:
                (body,) = _proof.premises
                focus = _proof.focus
                body, counter, trans = go(body, counter, trans)
                return body.extract(Variable(focus.type, trans[focus.index])), counter, trans
            case _:
                raise ValueError
    try:
        _proof, _, _ = go(proof, -1, {})
    except KeyError:
        raise ExtractionError('Proof is not linear')
    if not _proof.is_linear():
        raise ExtractionError('Proof is not linear')
    return _proof
