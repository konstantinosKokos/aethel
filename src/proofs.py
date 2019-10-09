from functools import reduce

from src.extraction import order_nodes, is_gap, is_copy, _head_deps, _mod_deps
from src.graphutils import *
from src.milltypes import polarize_and_index_many, WordType, PolarizedIndexedType, ColoredType
from src.transformations import _cats_of_type

from src.viz import ToGraphViz

ProofNet = Set[Tuple[int, int]]

Placeholder = None


class ProofError(AssertionError):
    def __init__(self, message: str):
        super().__init__(message)


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
        if k in set(map(fst, core)) and (k, v) not in core:
            raise ProofError('Positive formula already assigned in core proof.\t{}\t{}\t{}'.format(k, v, core))
        if v in set(map(snd, core)) and (k, v) not in core:
            raise ProofError('Negative formula already assigned in core proof.\t{}\t{}\t{}'.format(k, v, core))
        core = core.union({(k, v)})
    return core


def merge_proofs(core: ProofNet, locals_: Sequence[ProofNet]) -> ProofNet:
    return reduce(merge_proof, locals_, core)


def is_indexed(wordtype: WordType) -> bool:
    return all(list(map(lambda subtype: isinstance(subtype, PolarizedIndexedType), wordtype.get_atomic())))


def get_annotated_nodes(dag: DAG) -> Set[Node]:
    return set(filter(lambda node: is_indexed(dag.attribs[node]['type']), dag.nodes))


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


def split_functor(functor: WordType) -> Tuple[Optional[WordType], WordType]:
    result = functor
    body = None
    while isinstance(result, ColoredType):
        if result.color in _mod_deps:
            break
        body = result.argument
        result = result.result
    return body, result


def get_functor_result(functor: WordType) -> WordType:
    return snd(split_functor(functor))


def get_functor_body(functor: WordType) -> Optional[WordType]:
    return fst(split_functor(functor))


def get_functor_from_poly_x(functor: WordType) -> WordType:
    while functor.color not in _head_deps:
        functor = functor.argument
    return functor.argument


def isolate_xs(crd_type: WordType) -> List[WordType]:
    def step_cnj(crd_type_: WordType) -> Optional[Tuple[WordType, WordType]]:
        if isinstance(crd_type_, ColoredType) and crd_type_.color == 'cnj':
            return crd_type_.argument, crd_type_.result
        return
    return list(unfoldr(step_cnj, crd_type))


def last_instance_of(crd_type: WordType) -> WordType:
    def step_cnj(crd_type_: WordType) -> Optional[Tuple[WordType, WordType]]:
        if isinstance(crd_type_, ColoredType) and crd_type_.color == 'cnj':
            return crd_type_.result, crd_type_.result
        return
    return last(list(unfoldr(step_cnj, crd_type)))


def identify_missing(polymorphic_x: WordType, missing: WordType, dep: str) -> WordType:
    while polymorphic_x.color != dep or polymorphic_x.argument != missing:
        polymorphic_x = polymorphic_x.result
    return polymorphic_x.argument


def update_types(dag: DAG, nodes: List[Node], wordtypes: List[WordType]) -> DAG:
    new_types = {node: {**dag.attribs[node], **{'type': wordtype}} for (node, wordtype) in zip(nodes, wordtypes)}
    return DAG(nodes=dag.nodes, edges=dag.edges, attribs={**dag.attribs, **new_types})


def add_ghost_nodes(dag: DAG) -> DAG:
    edges = dag.edges
    copies = set(filter(lambda edge: is_copy(dag, edge.target), edges))
    gaps = set(filter(lambda edge: is_gap(dag, edge.target, _head_deps), edges))
    copy_gaps = copies.intersection(gaps)

    copies = copies - copy_gaps
    copy_types = list(map(lambda copy: get_copy_type(dag, copy), copies))
    dag = reduce(lambda dag_, pair_: add_edge(dag_, fst(pair_), snd(pair_)), zip(copies, copy_types), dag)

    if copy_gaps:
        ToGraphViz()(dag)
        import pdb
        pdb.set_trace()

    return dag


def delete_ghost_nodes(dag: DAG) -> DAG:
    ghost = set(filter(lambda node: set(dag.attribs[node].keys()) == {'index', 'type', 'id'}, dag.nodes))
    return dag.remove_nodes(lambda node: node not in ghost)


def add_edge(dag: DAG, edge: Edge, type_: Optional[WordType] = Placeholder) -> DAG:
    def get_fresh_node(nodes_: Nodes) -> Node:
        node = -1
        while str(node) in nodes_:
            node -= 1
        return str(node)
    fresh_node = get_fresh_node(dag.nodes)
    fresh_edge = Edge(source=edge.source, target=fresh_node, dep=edge.dep)

    fresh_attrib = {fresh_node: {'index': dag.attribs[edge.target]['index'], 'type': type_}}

    return DAG(nodes=dag.nodes.union({fresh_node}),
               edges=dag.edges.union({fresh_edge}),
               attribs={**dag.attribs, **fresh_attrib})


def get_copy_type(dag: DAG, edge: Edge) -> WordType:
    copy = edge.target
    conjunction = find_first_conjunction_above(dag, copy)
    missing_type = dag.attribs[edge.target]['type']

    crd = list(filter(lambda out: out.dep == 'crd' and dag.attribs[out.target]['type'] != '_CRD',
                      dag.outgoing(conjunction)))
    assert len(crd) == 1
    crd = fst(crd).target
    crd_type = dag.attribs[crd]['type']

    conjunction_daughters = list(filter(lambda out: out.dep not in _head_deps.union(_mod_deps),
                                        dag.outgoing(conjunction)))
    conjunction_daughters = set(map(lambda out: out.target, conjunction_daughters))
    conjunction_daughters = order_nodes(dag, conjunction_daughters)
    conjunction_daughters = list(map(lambda daughter: dag.exists_path(daughter, edge.source) or daughter == edge.source,
                                     conjunction_daughters))
    daughter_index = conjunction_daughters.index(True)

    xs, result = isolate_xs(crd_type), last_instance_of(crd_type)

    return identify_missing(xs[daughter_index], missing_type, edge.dep)


def find_first_conjunction_above(dag: DAG, node: Node) -> Optional[Node]:
    conjunctions = _cats_of_type(dag, 'conj')
    node_ancestors = dag.pointed_by(node)
    node_ancestors = node_ancestors.intersection(conjunctions)
    first_conj_above = list(filter(lambda ancestor: not any(list(map(lambda other: dag.exists_path(ancestor, other),
                                                                     node_ancestors))), node_ancestors))
    if len(first_conj_above) == 1:
        return fst(first_conj_above)
    return


def iterate_simple_fringe(dag: DAG) -> Optional[Tuple[ProofNet, DAG]]:
    unfolded = list(unfoldr(annotate_simple_branches, dag))
    return merge_proofs(set(), list(map(fst, unfolded))), snd(last(unfolded)) if unfolded else None


def annotate_simple_branches(dag: DAG) -> Optional[Tuple[Tuple[ProofNet, DAG], DAG]]:
    parents = list(get_simple_fringe(dag))
    if parents:
        temp = list(map(lambda parent: annotate_simple_branch(dag, parent), parents))
        branch_proofs, parent_types = list(zip(*temp))
        dag = update_types(dag, parents, parent_types)
        return (merge_proofs(set(), branch_proofs), dag), dag
    return


def get_simple_fringe(dag: DAG) -> List[Node]:
    annotated = get_annotated_nodes(dag)

    parents = set(filter(lambda node: not dag.is_leaf(node), dag.nodes))
    parents = parents.difference(annotated)
    parents = list(filter(lambda parent: all(list(map(lambda child: child in annotated,
                                                      dag.successors(parent)))), parents))
    return parents


def annotate_simple_branch(dag: DAG, parent: Node) -> Tuple[ProofNet, WordType]:
    def simplify_crd(crd_type: WordType) -> WordType:
        xs, result = isolate_xs(crd_type), last_instance_of(crd_type)
        xs = list(map(get_functor_result, xs))
        result = get_functor_result(result)
        return reduce(lambda res_, arg_: ColoredType(arg_, res_, 'cnj'), xs, result)

    branch_proof = set()

    outgoing = dag.outgoing(parent)
    outgoing = set(filter(lambda edge: not is_copy(dag, edge.target), outgoing))

    head = list(filter(lambda out: out.dep in _head_deps, outgoing))
    assert len(head) == 1
    head = fst(head)
    outgoing = list(filter(lambda out: out != head, outgoing))
    mods = list(filter(lambda out: out.dep in _mod_deps, outgoing))
    mods = order_nodes(dag, set(map(lambda out: out.target, mods)))
    args = list(filter(lambda out: out.dep not in _mod_deps, outgoing))

    if dag.attribs[parent]['cat'] == 'conj':
        branch_output = simplify_crd(dag.attribs[head.target]['type'])
    else:
        branch_output = get_simple_functor(dag, head.target)

    arg_proof, branch_output = align_args(branch_output,
                                          list(map(lambda out: get_simple_argument(dag, out.target), args)),
                                          list(map(lambda out: out.dep, args)))
    mod_proof, branch_output = align_mods(branch_output, list(map(lambda node: get_simple_argument(dag, node), mods)))
    branch_proof = merge_proofs(branch_proof, (arg_proof, mod_proof))

    return branch_proof, branch_output


def align_args(functor: WordType, argtypes: Sequence[WordType], deps: Sequence[str]) -> Tuple[ProofNet, WordType]:
    def color_fold(functor_: WordType) -> Iterable[Tuple[WordType, str]]:
        def step(x: WordType) -> Optional[Tuple[Tuple[WordType, str], WordType]]:
            return ((x.color, x.argument), x.result) if isinstance(x, ColoredType) else None
        return unfoldr(step, functor_)

    def match_args(proof_: ProofNet, pair: Tuple[WordType, WordType]) -> ProofNet:
        pos = fst(pair)
        neg = snd(pair)
        return match(proof_, pos, neg)

    proof = set()
    if argtypes:
        functor_argcolors = color_fold(functor)
        functor_argcolors = list(filter(lambda ac: fst(ac) not in _mod_deps, functor_argcolors))
        argdeps = list(zip(deps, argtypes))
        argdeps = sorted(argdeps, key=lambda x: functor_argcolors.index(x))

        pairs = list(zip(list(map(snd, argdeps)), list(map(snd, functor_argcolors))))
        proof = reduce(match_args, pairs, proof)
        return proof, get_functor_result(functor)
    return proof, functor


def align_mods(mod_input: WordType, mods: Sequence[WordType]) \
        -> Tuple[ProofNet, WordType]:
    def match_modchain(proof_: ProofNet, modpair: Tuple[WordType, WordType]) -> ProofNet:
        prev = fst(modpair)
        curr = snd(modpair)
        proof_ = match(proof_, prev.result, curr.argument)
        return proof_

    proof = set()
    if mods:
        mod_output = last(mods).result
        proof = match(proof, mod_input, fst(mods).argument)
        mods = list(zip(mods, mods[1:]))
        proof = reduce(match_modchain, mods, proof)
        return proof, mod_output
    return proof, mod_input


def annotate_dag(dag: DAG) -> Tuple[ProofNet, DAG]:
    proof = set()
    new_dag, idx = annotate_leaves(dag)
    new_dag = add_ghost_nodes(new_dag)

    temp = iterate_simple_fringe(new_dag)
    if temp is not None:
        proof, new_dag = temp

    if set(map(fst, proof)).union(set(map(snd, proof))) != set(range(idx)):
        ToGraphViz()(new_dag)
        import pdb
        pdb.set_trace()

    return proof, new_dag


def annotate_leaves(dag: DAG) -> Tuple[DAG, int]:
    leaves = set(filter(dag.is_leaf, dag.nodes))
    leaves = order_nodes(dag, leaves)
    leaftypes = list(map(lambda leaf: dag.attribs[leaf]['type'], leaves))
    idx, leaftypes = polarize_and_index_many(leaftypes, 0)
    dag = update_types(dag, leaves, leaftypes)
    return dag, idx


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

