from functools import reduce

from src.extraction import order_nodes, is_gap, is_copy, _head_deps, _mod_deps
from src.graphutils import *
from src.milltypes import polarize_and_index_many, WordType, PolarizedIndexedType, ColoredType
from src.transformations import _cats_of_type

ProofNet = Set[Tuple[int, int]]


def match(proofnet: ProofNet, positive: WordType, negative: WordType) -> ProofNet:
    if positive != negative:
        raise ProofError('Formulas are not equal.\t{}\t{}'.format(positive, negative))
    if any(map(lambda x: not is_indexed(x), [positive, negative])):
        raise ProofError('Input formulas are not fully indexed.\t{}\t{}'.format(positive, negative))
    if isinstance(positive, PolarizedIndexedType):
        if not positive.polarity:
            raise ProofError('Positive formula has negative index.\t{}\t{}'.format(positive, negative))
        if negative.polarity:
            raise ProofError('Negative formula has positive index.\t{}\t{}'.format(positive, negative))

        if positive.index in set(map(fst, proofnet)):
            raise ProofError('Positive formula already assigned.\t{}\t{}\t{}'.format(positive, negative,
                                                                                     proofnet))
        if negative.index in set(map(snd, proofnet)):
            raise ProofError('Negative formula already assigned.\t{}\t{}'.format(positive, negative))
        proofnet = proofnet.union({(positive.index, negative.index)})
    else:
        proofnet = match(proofnet, negative.argument, positive.argument)
        proofnet = match(proofnet, positive.result, negative.result)
    return proofnet


def merge_proof(core: ProofNet, local: ProofNet) -> ProofNet:
    for k, v in local:
        if k in set(map(fst, core)):
            raise ProofError('Positive formula already assigned in core proof.\t{}'.format(k))
        if v in set(map(snd, core)):
            raise ProofError('Negative formula already assigned in core proof.')
        core = core.union({(k, v)})
    return core


def merge_proofs(core: ProofNet, locals_: Sequence[ProofNet]) -> ProofNet:
    return reduce(merge_proof, locals_, core)


class ProofError(AssertionError):
    def __init__(self, message: str):
        super().__init__(message)


def is_indexed(wordtype: WordType) -> bool:
    return all(list(map(lambda subtype: isinstance(subtype, PolarizedIndexedType), wordtype.get_atomic())))


def get_annotated_nodes(dag: DAG) -> Set[Node]:
    return set(filter(lambda node: is_indexed(dag.attribs[node]['type']), dag.nodes))


def get_functor_result(functor: WordType) -> WordType:
    result = functor
    while isinstance(result, ColoredType):
        if result.color in _mod_deps:
            break
        result = result.result
    return result


def get_functor_body(functor: WordType) -> WordType:
    if functor.color in _mod_deps:
        return functor.argument.argument
    return functor.argument


def find_first_conjunction_above(dag: DAG, node: Node) -> Optional[Node]:
    conjunctions = _cats_of_type(dag, 'conj')
    node_ancestors = dag.pointed_by(node)
    node_ancestors = node_ancestors.intersection(conjunctions)
    first_conj_above = list(filter(lambda ancestor: not any(list(map(lambda other: dag.exists_path(ancestor, other),
                                                                     node_ancestors))), node_ancestors))
    if len(first_conj_above) == 1:
        return fst(first_conj_above)
    return


def isolate_xs(crd_type: WordType) -> List[WordType]:
    def step_cnj(crd_type_: WordType) -> Optional[Tuple[WordType, WordType]]:
        if crd_type_.color == 'cnj':
            return crd_type_.argument, crd_type_.result
        return
    return list(unfoldr(step_cnj, crd_type))


def last_instance_of(crd_type: WordType) -> WordType:
    def step_cnj(crd_type_: WordType) -> Optional[Tuple[WordType, WordType]]:
        if crd_type_.color == 'cnj':
            return crd_type_.result, crd_type_.result
        return
    return last(list(unfoldr(step_cnj, crd_type)))


def identify_missing(polymorphic_x: WordType, missing: WordType, dep: str) -> WordType:
    while polymorphic_x.color != dep and polymorphic_x.argument != missing:
        polymorphic_x = polymorphic_x.result
    return polymorphic_x.argument


def update_types(dag: DAG, nodes: List[Node], wordtypes: List[WordType]) -> None:
    leaftypes = {node: {**dag.attribs[node], **{'type': wordtype}} for (node, wordtype) in zip(nodes, wordtypes)}
    dag.attribs.update(leaftypes)


def get_simple_argument(dag: DAG, node: Node) -> WordType:
    type_ = dag.attribs[node]['type']
    if is_gap(dag, node, _head_deps):
        return type_.argument.argument
    else:
        return type_


def get_simple_functor(dag: DAG, node: Node) -> WordType:
    type_ = dag.attribs[node]['type']
    if is_gap(dag, node, _head_deps):
        return ColoredType(argument=type_.argument.result, color=type_.color, result=type_.result)
    else:
        return type_


def align_mods(mod_input: WordType, mods: Sequence[WordType], proof: ProofNet) -> Tuple[ProofNet, WordType]:
    def match_modchain(proof_: ProofNet, modpair: Tuple[WordType, WordType]) -> ProofNet:
        prev = fst(modpair)
        curr = snd(modpair)
        proof_ = match(proof_, prev.result, curr.argument)
        return proof_

    print(mod_input)
    print(mods)
    mod_output = last(mods).result
    proof = match(proof, mod_input, fst(mods).argument)
    mods = list(zip(mods, mods[1:]))
    proof = reduce(match_modchain, mods, proof)
    return proof, mod_output


def align_args(functor: WordType, argtypes: Sequence[WordType], deps: Sequence[str], proof: ProofNet) \
        -> Tuple[ProofNet, WordType]:
    def color_fold(functor_: WordType) -> Iterable[Tuple[WordType, str]]:
        def step(x: WordType) -> Optional[Tuple[Tuple[WordType, str], WordType]]:
            return ((x.color, x.argument), x.result) if isinstance(x, ColoredType) else None
        return unfoldr(step, functor_)

    def match_args(proof_: ProofNet, pair: Tuple[WordType, WordType]) -> ProofNet:
        pos = fst(pair)
        neg = snd(pair)
        return match(proof_, pos, neg)

    functor_argcolors = color_fold(functor)
    functor_argcolors = list(filter(lambda ac: fst(ac) not in _mod_deps, functor_argcolors))
    argdeps = list(zip(deps, argtypes))
    argdeps = sorted(argdeps, key=lambda x: functor_argcolors.index(x))

    pairs = list(zip(list(map(snd, argdeps)), list(map(snd, functor_argcolors))))
    proof = reduce(match_args, pairs, proof)
    return proof, get_functor_result(functor)


def annotate_leaves(dag: DAG) -> int:
    leaves = set(filter(dag.is_leaf, dag.nodes))
    leaves = order_nodes(dag, leaves)
    leaftypes = list(map(lambda leaf: dag.attribs[leaf]['type'], leaves))
    idx, leaftypes = polarize_and_index_many(leaftypes, 0)
    update_types(dag, leaves, leaftypes)
    return idx


def annotate_simple_branch(dag: DAG, parent: Node) -> Tuple[ProofNet, WordType]:
    branch_proof = set()

    outgoing = dag.outgoing(parent)

    head = list(filter(lambda out: out.dep in _head_deps, outgoing))
    head = list(filter(lambda out: fst(str(dag.attribs[out.target]['type'])) != '_', head))
    assert len(head) == 1
    head = fst(head)
    outgoing = list(filter(lambda out: out != head, outgoing))
    mods = list(filter(lambda out: out.dep in _mod_deps, outgoing))
    mods = order_nodes(dag, set(map(lambda out: out.target, mods)))
    args = list(filter(lambda out: out.dep not in _mod_deps, outgoing))

    branch_output = get_simple_functor(dag, head.target)

    if args:
        branch_proof, branch_output = align_args(branch_output,
                                                 list(map(lambda out: get_simple_argument(dag, out.target), args)),
                                                 list(map(lambda out: out.dep, args)),
                                                 branch_proof)
    if mods:
        branch_proof, branch_output = align_mods(branch_output,
                                                 list(map(lambda node: get_simple_argument(dag, node), mods)),
                                                 branch_proof)
    return branch_proof, branch_output


def get_simple_fringe(dag: DAG) -> List[Node]:
    def is_simple(parent: Node) -> bool:
        downset = dag.points_to(parent)
        copies = set(filter(lambda down: is_copy(dag, down), downset))
        first_conj_above = list(map(lambda copy: find_first_conjunction_above(dag, copy), copies))
        first_conj_above = list(filter(lambda x: x is not None, first_conj_above))
        return all(list(map(lambda fca: is_indexed(dag.attribs[fca]['type']), first_conj_above)))

    annotated = get_annotated_nodes(dag)

    parents = set(filter(lambda node: not dag.is_leaf(node), dag.nodes))
    parents = parents.difference(annotated)
    parents = list(filter(lambda parent: all(list(map(lambda child: child in annotated,
                                                      dag.successors(parent)))), parents))
    simple = list(map(is_simple, parents))
    return list(map(fst, filter(snd, zip(parents, simple))))


def annotate_simple_fringe(dag: DAG, proof: ProofNet) -> Optional[ProofNet]:
    parents = list(get_simple_fringe(dag))
    if parents:
        temp = list(map(lambda branch: annotate_simple_branch(dag, branch), parents))
        branch_proofs, parent_types = list(zip(*temp))
        update_types(dag, parents, parent_types)
        return merge_proofs(proof, branch_proofs)
    return


def iterate_simple_fringe(dag: DAG, proof: ProofNet) -> Optional[ProofNet]:
    cond = True
    changed = False
    while cond:
        cond = False
        temp = annotate_simple_fringe(dag, proof)
        if temp is not None:
            proof = temp
            cond = changed = True
    return proof if changed else None


def get_complex_fringe(dag: DAG) -> List[Node]:
    def doable(parent: Node) -> bool:
        downset = dag.points_to(parent)
        downset_square = list(map(lambda down: dag.points_to(down), downset))
        return all(list(map(lambda ds: all(list(map(lambda down: is_indexed(dag.attribs[down]['type']), ds))),
                            downset_square)))

    conjunctions = _cats_of_type(dag, 'conj')
    return list(filter(doable, conjunctions))


def align_copied_arguments(crd_type: WordType, copy_type: WordType, copy_color: str, sharing_functors: List[WordType]) \
        -> Tuple[ProofNet, List[WordType], WordType]:
    def match_args(proof_: ProofNet, pair: Tuple[WordType, WordType]) -> ProofNet:
        pos = fst(pair)
        neg = snd(pair)
        return match(proof_, pos, neg)

    local_proof = set()
    xs, result = isolate_xs(crd_type), last_instance_of(crd_type)
    missing_in_result = identify_missing(result, copy_type, copy_color)
    missing_in_crd_args = list(map(identify_missing,
                               xs,
                               [copy_type for _ in range(len(sharing_functors))],
                               [copy_color for _ in range(len(sharing_functors))]))
    missing_in_functors = list(map(identify_missing,
                                   sharing_functors,
                                   [copy_type for _ in range(len(sharing_functors))],
                                   [copy_color for _ in range(len(sharing_functors))]))
    parent_types = list(map(get_functor_result, sharing_functors))
    conjunction_type = get_functor_result(result)

    local_proof = match(local_proof, copy_type, missing_in_result)
    local_proof = reduce(match_args, zip(missing_in_crd_args, missing_in_functors), local_proof)

    return local_proof, parent_types, conjunction_type


def align_copied_functor(crd_type: WordType, copy_type: WordType, sharing_args: List[List[Tuple[str, WordType]]]):
    local_proof = set()
    xs, result = isolate_xs(crd_type), last_instance_of(crd_type)
    local_proof = match(local_proof, copy_type, result.argument)
    parent_types = list(map(get_functor_result, xs))
    functor_bodies = list(map(get_functor_body, xs))

    sharing_types = list(map(lambda sharing: list(map(fst, sharing)), sharing_args))
    sharing_deps = list(map(lambda sharing: list(map(snd, sharing)), sharing_args))
    functor_results = list(map(align_args, functor_bodies, sharing_types, sharing_deps, local_proof))

    import pdb
    pdb.set_trace()

    # todo: functor case


def annotate_conjunction_branch(dag: DAG, parent: Node) -> Tuple[ProofNet, WordType]:
    def get_copy_parents(copy_) -> List[Node]:
        incoming = dag.incoming(copy_)
        if len(set(map(lambda edge_: edge_.dep, incoming))) != 1:
            import pdb
            pdb.set_trace()
            from src.viz import ToGraphViz
            ToGraphViz()(dag)
        return order_nodes(dag, set(map(lambda edge_: edge_.source, incoming)))

    def get_successor_arguments(copy_parent_: Node) -> List[Optional[Tuple[WordType, str]]]:
        outgoing_ = dag.outgoing(copy_parent_)
        outgoing_ = list(filter(lambda out_: out_.dep not in _mod_deps.union(_head_deps), outgoing_))
        return list(map(lambda out_:
                        None if is_copy(dag, out_.target) else (dag.attribs[out_.target]['type'], out_.dep),
                        outgoing_))

    def get_successor_functors(copy_parent_: Node) -> Optional[WordType]:
        outgoing_ = dag.outgoing(copy_parent_)
        outgoing_ = list(filter(lambda out_: out_.dep in _head_deps, outgoing_))
        assert len(outgoing_) == 1
        outgoing_ = fst(outgoing_).target
        return dag.attribs[outgoing_]['type'] if not is_copy(dag, outgoing_) else None

    def get_copy_color(copy_: Node) -> str:
        incoming_ = set(map(lambda inc_: inc_.dep, dag.incoming(copy_)))
        if len(incoming_) > 1:
            # todo: gap copy case
            import pdb
            from src.viz import ToGraphViz
            pdb.set_trace()
            ToGraphViz()(dag)
        return fst(list(incoming_))

    conjunction_proof = set()

    outgoing = dag.outgoing(parent)

    crd = list(filter(lambda out: out.dep == 'crd', outgoing))
    crd = list(filter(lambda out: fst(str(dag.attribs[out.target]['type'])) != '_', crd))
    assert len(crd) == 1
    crd = fst(crd).target
    crd_type = dag.attribs[crd]['type']

    copies = list(filter(lambda node: is_copy(dag, node), dag.points_to(parent)))
    copies = list(filter(lambda copy: not any(list(map(lambda other: dag.exists_path(copy, other), copies))), copies))
    copies_colors = list(map(get_copy_color, copies))

    copied_functors = list(filter(lambda copy: snd(copy) in _head_deps, zip(copies, copies_colors)))
    copied_args = list(filter(lambda copy: copy not in copied_functors, zip(copies, copies_colors)))

    copied_functors_parents = list(map(get_copy_parents, map(fst, copied_functors)))
    copied_args_parents = list(map(get_copy_parents, map(fst, copied_args)))

    sharing_args = list(map(lambda parents: list(map(get_successor_arguments, parents)), copied_functors_parents))
    sharing_functors = list(map(lambda parents: list(map(get_successor_functors, parents)), copied_args_parents))

    # todo: mixed case

    if sharing_args:
        # todo: functor case
        raise NotImplementedError

    temp = list(map(lambda type_, color_, functor_:
                    align_copied_arguments(crd_type, type_, color_, functor_),
                    list(map(lambda copy:
                             dag.attribs[fst(copy)]['type'],
                             copied_args)),
                    list(map(snd, copied_args)),
                    sharing_functors))
    arg_matchings, parent_types, conjunction_type = list(zip(*temp))

    conjunction_proof = merge_proofs(conjunction_proof, arg_matchings)
    for cap, pt in zip(copied_args_parents, parent_types):
        update_types(dag, cap, pt)

    temp = list(map(lambda type_, args_: align_copied_functor(crd_type, type_, args_),
                    list(map(lambda copy: dag.attribs[fst(copy)]['type'], copied_functors)),
                    sharing_args))

    return conjunction_proof, conjunction_type


def annotate_complex_fringe(dag: DAG, proof: ProofNet) -> Optional[ProofNet]:
    parents = list(get_complex_fringe(dag))
    if parents:
        temp = list(map(lambda branch: annotate_conjunction_branch(dag, branch), parents))
        conjunction_proofs, new_parent_types = list(zip(*temp))
        update_types(dag, parents, new_parent_types)
        return merge_proofs(proof, conjunction_proofs)
    return


def iterate_complex_fringe(dag: DAG, proof: ProofNet) -> Optional[ProofNet]:
    cond = True
    changed = False
    while cond:
        cond = False
        temp = annotate_complex_fringe(dag, proof)
        if temp is not None:
            proof = temp
            cond = changed = True
    return proof if changed else None


def annotate_dag(dag: DAG) -> DAG:
    proof = set()
    idx = annotate_leaves(dag)

    cond = True
    while cond:
        simple = iterate_simple_fringe(dag, proof)
        if simple is not None:
            proof = simple
        complex_ = iterate_complex_fringe(dag, proof)
        if complex_ is not None:
            proof = complex_
        cond = simple is not None or complex_ is not None
    # root = fst(list(dag.get_roots()))
    # root_type = dag.attribs[root]['type']
    # _, conclusion = polarize_and_index(root_type, False, idx)
    # match(proof, root_type, conclusion)
    return dag


class Prove(object):
    def __init__(self):
        pass

    def __call__(self, dag: DAG, raise_errors: bool = True) -> Optional[DAG]:
        try:
            return annotate_dag(dag)
        except ProofError as e:
            if raise_errors:
                raise e
            else:
                return

