from src.milltypes import polarize_and_index, polarize_and_index_many, WordType, PolarizedIndexedType, ColoredType
from src.graphutils import *
from src.extraction import order_nodes, is_gap, is_copy, _head_deps, _mod_deps

from functools import reduce

from src.viz import ToGraphViz


ProofNet = Dict[int, int]


def match(proofnet: ProofNet, positive: WordType, negative: WordType):
    if positive != negative:
        raise ProofError('Formulas are not equal.')
    if any(map(lambda x: not is_indexed(x), [positive, negative])):
        raise ProofError('Input formulas are not fully indexed.')
    if isinstance(positive, PolarizedIndexedType):
        if not positive.polarity:
            raise ProofError('Positive formula has negative index.')
        if negative.polarity:
            raise ProofError('Negative formula has positive index.')
        if positive.index in proofnet.keys():
            raise ProofError('Positive formula already assigned')
        if negative.index in proofnet.values():
            raise ProofError('Negative formula already assigned.')
        proofnet[positive.index] = negative.index
    else:
        match(proofnet, negative.argument, positive.argument)
        match(proofnet, positive.result, negative.result)


class ProofError(AssertionError):
    def __init__(self, message: str):
        super().__init__(message)


def is_indexed(wordtype: WordType) -> bool:
    return all(list(map(lambda subtype: isinstance(subtype, PolarizedIndexedType), wordtype.get_atomic())))


def update_types(dag: DAG, nodes: List[Node], wordtypes: List[WordType]) -> None:
    leaftypes = {node: {**dag.attribs[node], **{'type': wordtype}} for (node, wordtype) in zip(nodes, wordtypes)}
    dag.attribs.update(leaftypes)


def align_mods(mod_input: WordType, mods: Sequence[WordType], proof: ProofNet) -> WordType:
    def match_modchain(prev: WordType, curr: WordType) -> WordType:
        match(proof, prev.result, curr.argument)
        return curr
    match(proof, mod_input, fst(mods).argument)
    mod_output = reduce(match_modchain, mods).result
    return mod_output


def align_args(functor: WordType, argtypes: Sequence[WordType], deps: Sequence[str], proof: ProofNet) -> WordType:
    def color_fold(functor_: WordType) -> Any:
        def step(x: WordType) -> Optional[Tuple[WordType, str]]:
            return (x.color, x.result) if isinstance(x, ColoredType) else None
        return unfoldr(step, functor_)
    func_colors = color_fold(functor)
    argdeps = zip(argtypes, deps)
    argdeps = sorted(argdeps, key=lambda x: )



def annotate_perimeter(dag: DAG):
    leaves = set(filter(dag.is_leaf, dag.nodes))
    leaves = order_nodes(dag, leaves)
    leaftypes = list(map(lambda leaf: dag.attribs[leaf]['type'], leaves))
    idx, leaftypes = polarize_and_index_many(leaftypes, 0)
    root = fst(list(dag.get_roots()))
    _, root_type = polarize_and_index(dag.attribs[root]['type'], False, idx)
    update_types(dag, leaves + [root], leaftypes + [root_type])


def annotate_branch_simple(dag: DAG, parent: Node, proof: ProofNet):
    outgoing = dag.outgoing(parent)
    if any(list(map(lambda su: is_copy(dag, su) or is_gap(dag, su, _head_deps),
                    list(map(lambda out: out.target, outgoing))))):
        ToGraphViz()(dag)
        import pdb
        pdb.set_trace()
    head = list(filter(lambda out: out.dep in _head_deps, outgoing))
    head = list(filter(lambda out: fst(str(dag.attribs[out.target]['type'])) != '_', head))
    assert len(head) == 1
    head = fst(head)
    mods = list(filter(lambda out: out.dep in _mod_deps, outgoing))
    mods = order_nodes(dag, set(map(lambda out: out.target, mods)))
    args = list(filter(lambda out: out not in mods, outgoing))

    branch_output = dag.attribs[head.target]['type']

    if args:
        branch_output = align_args(branch_output,
                                   list(map(lambda out: dag.attribs[out.target]['type'], args)),
                                   list(map(lambda out: out.dep, args)),
                                   proof)
    if mods:
        branch_output = align_mods(branch_output,
                                   list(map(lambda node: dag.attribs[node]['type'], mods)),
                                   proof)
    update_types(dag, [parent], [branch_output])



def annotate_fringe(dag: DAG, proof: ProofNet):
    branches = annotated_fringe(dag)
    for branch in branches:
        annotate_branch_simple(dag, branch, proof)


def annotate_dag(dag: DAG) -> DAG:
    annotate_perimeter(dag)
    return dag


def annotated_nodes(dag: DAG) -> Set[Node]:
    return set(filter(lambda node: is_indexed(dag.attribs[node]['type']), dag.nodes))


def annotated_fringe(dag: DAG) -> Set[Node]:
    annotated = annotated_nodes(dag)
    parents = set(filter(lambda node: not dag.is_leaf(node), dag.nodes))
    parents = parents.difference(annotated)
    return set(filter(lambda parent: all(list(map(lambda child: child in annotated,
                                              dag.successors(parent)))), parents))


class Proof(object):
    def __init__(self):
        pass

    def __call__(self, dag: DAG, raise_errors: bool = False) -> Optional[DAG]:
        try:
            return annotate_dag(dag)
        except ProofError as e:
            if raise_errors:
                raise e
            else:
                return
