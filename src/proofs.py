from src.milltypes import polarize_and_index, polarize_and_index_many, WordType, PolarizedIndexedType, ColoredType
from src.graphutils import *
from src.extraction import order_nodes, is_gap, is_copy, _head_deps, _mod_deps
from src.transformations import _cats_of_type

from functools import reduce

from src.viz import ToGraphViz


ProofNet = Dict[int, int]


def match(proofnet: ProofNet, positive: WordType, negative: WordType):
    if positive != negative:
        raise ProofError('Formulas are not equal.\t{}\t{}'.format(positive, negative))
    if any(map(lambda x: not is_indexed(x), [positive, negative])):
        raise ProofError('Input formulas are not fully indexed.\t{}\t{}'.format(positive, negative))
    if isinstance(positive, PolarizedIndexedType):
        if not positive.polarity:
            raise ProofError('Positive formula has negative index.\t{}\t{}'.format(positive, negative))
        if negative.polarity:
            raise ProofError('Negative formula has positive index.\t{}\t{}'.format(positive, negative))
        if positive.index in proofnet.keys():
            raise ProofError('Positive formula already assigned.\t{}\t{}\t{}'.format(positive, negative,
                                                                                     proofnet))
        if negative.index in proofnet.values():
            raise ProofError('Negative formula already assigned.\t{}\t{}'.format(positive, negative))
        proofnet[positive.index] = negative.index
    else:
        match(proofnet, negative.argument, positive.argument)
        match(proofnet, positive.result, negative.result)


class ProofError(AssertionError):
    def __init__(self, message: str):
        super().__init__(message)


def is_indexed(wordtype: WordType) -> bool:
    return all(list(map(lambda subtype: isinstance(subtype, PolarizedIndexedType), wordtype.get_atomic())))


def find_first_conjunction_above(dag: DAG, node: Node) -> Optional[Node]:
    conjunctions = _cats_of_type(dag, 'conj')
    node_ancestors = dag.pointed_by(node)
    node_ancestors = node_ancestors.intersection(conjunctions)
    first_conj_above = list(filter(lambda ancestor: not any(list(map(lambda other: dag.exists_path(ancestor, other),
                                                                     node_ancestors))), node_ancestors))
    if len(first_conj_above) == 1:
        return fst(first_conj_above)
    return


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


def align_mods(mod_input: WordType, mods: Sequence[WordType], proof: ProofNet) -> WordType:
    def match_modchain(prev: WordType, curr: WordType) -> WordType:
        match(proof, prev.result, curr.argument)
        return curr
    match(proof, mod_input, fst(mods).argument)
    mod_output = reduce(match_modchain, mods).result
    return mod_output


def align_args(functor: WordType, argtypes: Sequence[WordType], deps: Sequence[str], proof: ProofNet) -> WordType:
    def color_fold(functor_: WordType) -> Iterable[Tuple[WordType, str]]:
        def step(x: WordType) -> Optional[Tuple[Tuple[WordType, str], WordType]]:
            return ((x.color, x.argument), x.result) if isinstance(x, ColoredType) else None
        return unfoldr(step, functor_)

    def match_args(functor_: WordType, arg: WordType) -> WordType:
        match(proof, arg, functor_.argument)
        return functor_.result

    functor_argcolors = color_fold(functor)
    functor_argcolors = list(filter(lambda ac: fst(ac) not in _mod_deps, functor_argcolors))
    argdeps = list(zip(deps, argtypes))
    argdeps = sorted(argdeps, key=lambda x: functor_argcolors.index(x))
    functor = reduce(match_args, list(map(snd, argdeps)), functor)
    return functor


def annotate_leaves(dag: DAG) -> int:
    leaves = set(filter(dag.is_leaf, dag.nodes))
    leaves = order_nodes(dag, leaves)
    leaftypes = list(map(lambda leaf: dag.attribs[leaf]['type'], leaves))
    idx, leaftypes = polarize_and_index_many(leaftypes, 0)
    update_types(dag, leaves, leaftypes)
    return idx


def annotate_simple_branch(dag: DAG, parent: Node, proof: ProofNet) -> WordType:
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
        branch_output = align_args(branch_output,
                                   list(map(lambda out: get_simple_argument(dag, out.target), args)),
                                   list(map(lambda out: out.dep, args)),
                                   proof)
    if mods:
        branch_output = align_mods(branch_output,
                                   list(map(lambda node: get_simple_argument(dag, node), mods)),
                                   proof)
    return branch_output


def annotate_simple_fringe(dag: DAG, proof: ProofNet) -> bool:
    parents = list(get_annotated_fringe(dag))
    parents = list(map(fst, filter(lambda x: snd(x), parents)))
    if parents:
        new_parent_types = list(map(lambda branch: annotate_simple_branch(dag, branch, proof), parents))
        update_types(dag, parents, new_parent_types)
        return True
    return False


def iterate_simple_fringe(dag: DAG, proof: ProofNet) -> None:
    temp = True
    while temp:
        temp = annotate_simple_fringe(dag, proof)


def annotate_dag(dag: DAG) -> DAG:
    proof = dict()
    # proof = dict() if 'proof' not in dag.meta.keys() else dag.meta['proof']
    idx = annotate_leaves(dag)
    iterate_simple_fringe(dag, proof)

    # root = fst(list(dag.get_roots()))
    # root_type = dag.attribs[root]['type']
    # _, conclusion = polarize_and_index(root_type, False, idx)
    # match(proof, root_type, conclusion)
    return dag


def get_annotated_nodes(dag: DAG) -> Set[Node]:
    return set(filter(lambda node: is_indexed(dag.attribs[node]['type']), dag.nodes))


def get_annotated_fringe(dag: DAG) -> List[Tuple[Node, bool]]:
    def is_simple(parent: Node) -> bool:
        downset = dag.points_to(parent)
        conjunctions = _cats_of_type(dag, 'conj')
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
    return list(zip(parents, simple))


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

