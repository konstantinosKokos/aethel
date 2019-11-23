from functools import reduce
from itertools import chain

from LassyExtraction.extraction import order_nodes, is_gap, is_copy, _head_deps, _mod_deps
from LassyExtraction.graphutils import *
from LassyExtraction.milltypes import polarize_and_index_many, polarize_and_index, WordType, \
    PolarizedIndexedType, ColoredType, AtomicType, depolarize
from LassyExtraction.transformations import _cats_of_type

from LassyExtraction.viz import ToGraphViz

ProofNet = Set[Tuple[int, int]]

placeholder = AtomicType('_')


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


def remove_functor_arguments(functor: WordType, arguments: List[WordType]) -> WordType:
    ret = functor
    while arguments:
        body, res = split_functor(ret)
        if body != fst(arguments):
            return ret
        ret = res
        arguments = arguments[1::]


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
    return DAG(nodes=dag.nodes, edges=dag.edges, attribs={**dag.attribs, **new_types}, meta=dag.meta)


def add_ghost_nodes(dag: DAG) -> DAG:
    edges = dag.edges
    copies = set(filter(lambda edge: is_copy(dag, edge.target), edges))
    gaps = set(filter(lambda edge: is_gap(dag, edge.target, _head_deps), edges))
    copy_gaps = copies.intersection(gaps)
    copies = list(copies - copy_gaps)

    # todo. assertions

    copy_gaps = set(filter(lambda edge: edge.dep not in _head_deps, copy_gaps))
    copy_gaps = list(copy_gaps)

    copy_types = list(map(lambda copy: get_copy_annotation(dag, copy), copies))
    gap_types = list(map(lambda copy: get_copy_annotation(dag, copy), copy_gaps))

    dag = reduce(lambda dag_, pair_: add_edge(dag_, fst(pair_), snd(pair_)), zip(copies, copy_types), dag)
    dag = reduce(lambda dag_, pair_: add_edge(dag_, fst(pair_), snd(pair_)), zip(copy_gaps, gap_types), dag)

    return dag


def delete_ghost_nodes(dag: DAG) -> DAG:
    return dag.remove_nodes(lambda node: int(dag.attribs[node]['id']) >= 0)


def add_edge(dag: DAG, edge: Edge, type_: Optional[WordType] = placeholder) -> DAG:
    def get_fresh_node(nodes_: Nodes) -> Node:
        node = -1
        while str(node) in nodes_:
            node -= 1
        return str(node)
    fresh_node = get_fresh_node(dag.nodes)
    fresh_edge = Edge(source=edge.source, target=fresh_node, dep=edge.dep)

    fresh_attrib = {fresh_node: {'id': fresh_node,
                                 'index': dag.attribs[edge.target]['index'],
                                 'type': type_,
                                 'begin': dag.attribs[edge.target]['begin'],
                                 'end': dag.attribs[edge.target]['end']}}

    return DAG(nodes=dag.nodes.union({fresh_node}),
               edges=dag.edges.union({fresh_edge}),
               attribs={**dag.attribs, **fresh_attrib},
               meta=dag.meta)


def get_crd_type(dag: DAG, conjunction: Node) -> WordType:
    crd = list(filter(lambda out: out.dep == 'crd' and dag.attribs[out.target]['type'] != placeholder,
                      dag.outgoing(conjunction)))
    assert len(crd) == 1
    crd = fst(crd).target
    return dag.attribs[crd]['type']


def get_conjunction_daughters(dag: DAG, conjunction: Node) -> List[Node]:
    conjunction_daughters = list(filter(lambda out: out.dep not in _head_deps.union(_mod_deps),
                                        dag.outgoing(conjunction)))
    conjunction_daughters = set(map(lambda out: out.target, conjunction_daughters))
    return order_nodes(dag, conjunction_daughters)


def get_copy_annotation(dag: DAG, edge: Edge) -> WordType:

    source = edge.source

    while True:

        conjunction = find_first_conjunction_above(dag, source)
        if conjunction is None:
            ToGraphViz()(dag)
            import pdb
            pdb.set_trace()
        missing_type = get_simple_argument(dag, edge.target)
        crd_type = get_crd_type(dag, conjunction)

        conjunction_daughters = get_conjunction_daughters(dag, conjunction)
        conjunction_daughters = list(map(lambda daughter: dag.exists_path(daughter, edge.source) or daughter == edge.source,
                                         conjunction_daughters))
        if any(conjunction_daughters):
            break
        source = conjunction

    daughter_index = conjunction_daughters.index(True)

    xs, result = isolate_xs(crd_type), last_instance_of(crd_type)

    return identify_missing(xs[daughter_index], missing_type, edge.dep)


def intra_crd_match(dag: DAG, hierarchy: List[Tuple[Node, List[Optional[Node]]]],
                    copy_type_: WordType, copy_dep_: str) -> ProofNet:

    hierarchy = list(map(lambda pair:
                         (isolate_xs(get_crd_type(dag, fst(pair))), snd(pair)), hierarchy))

    hierarchy = list(map(lambda h: list(zip(*h)), hierarchy))

    hierarchy = list(filter(lambda pair: snd(pair) is not None, list(chain.from_iterable(hierarchy))))

    hierarchy = list(map(lambda pair: (fst(pair), last_instance_of(get_crd_type(dag, snd(pair)))),
                         hierarchy))

    try:
        matches = list(map(lambda pair: (identify_missing(fst(pair), copy_type_, copy_dep_),
                                     identify_missing(snd(pair), copy_type_, copy_dep_)),
                       hierarchy))
    except AttributeError:
        ToGraphViz()(dag)
        print(hierarchy)
        import pdb
        pdb.set_trace()

    return reduce(lambda proof_, pair: match(proof_, fst(pair), snd(pair)), matches, set())


def match_copies_with_crds(dag: DAG) -> ProofNet:
    def get_copy_color(copy_: Node) -> str:
        incoming_ = set(map(lambda inc_: inc_.dep, dag.incoming(copy_)))
        assert len(incoming_) == 1
        return fst(list(incoming_))

    proof = set()

    copies = list(filter(lambda node: is_copy(dag, node) and not is_gap(dag, node, _head_deps),
                         dag.nodes))
    copy_colors = list(map(get_copy_color, copies))
    copy_types = list(map(lambda copy: dag.attribs[copy]['type'], copies))

    conjunction_hierarchies: List[List[Tuple[Node, List[Optional[Node]]]]] = \
        list(map(lambda node: participating_conjunctions(dag, node), copies))

    proof = merge_proofs(proof, list(map(lambda ch, ct, cc:
                                         intra_crd_match(dag, ch, ct, cc),
                                         conjunction_hierarchies,
                                         copy_types,
                                         copy_colors)))

    crd_types = list(map(lambda ch: get_crd_type(dag, fst(fst(ch))), conjunction_hierarchies))
    results = list(map(last_instance_of, crd_types))
    matches = list(map(identify_missing,
                       results,
                       copy_types,
                       copy_colors))
    return reduce(lambda proof_, pair: match(proof_, fst(pair), snd(pair)), zip(copy_types, matches), proof)


def match_copy_gaps_with_crds(dag: DAG) -> ProofNet:
    proof = set()

    gaps = list(filter(lambda node: is_copy(dag, node) and is_gap(dag, node, _head_deps), dag.nodes))
    gap_types = list(map(lambda node: dag.attribs[node]['type'], gaps))
    gap_colors = list(map(lambda type_: type_.argument.color, gap_types))
    gap_types = list(map(lambda type_: type_.argument.argument, gap_types))
    conjunction_hierarchies = list(map(lambda node: participating_conjunctions(dag, node, exclude_heads=True),
                                       gaps))

    proof = merge_proofs(proof, list(map(lambda ch, gt, gc:
                                         intra_crd_match(dag, ch, gt, gc),
                                         conjunction_hierarchies,
                                         gap_types,
                                         gap_colors)))

    crd_types = list(map(lambda ch: get_crd_type(dag, fst(fst(ch))), conjunction_hierarchies))
    results = list(map(last_instance_of, crd_types))
    matches = list(map(identify_missing, results, gap_types, gap_colors))
    return reduce(lambda proof_, pair: match(proof_, fst(pair), snd(pair)), zip(gap_types, matches), proof)


def find_first_conjunction_above(dag: DAG, node: Node) -> Optional[Node]:
    conjunctions = _cats_of_type(dag, 'conj')
    node_ancestors = dag.pointed_by(node)
    node_ancestors = node_ancestors.intersection(conjunctions)
    first_conj_above = list(filter(lambda ancestor: not any(list(map(lambda other: dag.exists_path(ancestor, other),
                                                                     node_ancestors))), node_ancestors))
    if len(first_conj_above) == 1:
        return fst(first_conj_above)
    return


def participating_conjunctions(dag: DAG, node: Node, exclude_heads: bool = False) \
        -> List[Tuple[Node, List[Optional[Node]]]]:

    def impose_order(conjunctions_: Nodes) -> List[Tuple[Node, List[Optional[Node]]]]:
        daughters = list(conjunctions_)
        daughters = list(zip(daughters, list(map(lambda conj: get_conjunction_daughters(dag, conj), daughters))))

        daughters_connections = list(map(lambda pair:
                                         # the conjunction
                                         (
                                             # the conjunction
                                             fst(pair),
                                             # the subset of conjunctions contained by each daughter
                                             list(map(lambda daughter:
                                                      set(filter(lambda other: dag.exists_path(daughter, other) or
                                                                               daughter == other, conjunctions_)),
                                                      snd(pair)))
                                         ),
                                         daughters))

        daughters_connections = list(map(lambda pair:
                                         (fst(pair),
                                          # the subset of connections that have the most paths to node
                                          list(map(lambda daughter_connections:
                                                   set(filter(lambda connection:
                                                              not any(map(lambda other:
                                                                          len(dag.distinct_paths_to(other, node)) >
                                                                          len(dag.distinct_paths_to(connection, node)),
                                                                          daughter_connections)),
                                                              daughter_connections)),
                                                   snd(pair)))),
                                         daughters_connections))

        daughters_connections = list(map(lambda pair:
                                         (fst(pair),
                                          # the singular connection that lies lowest (not over any conjunction?)
                                          list(map(lambda daughter_connections:
                                                   set(filter(lambda connection:
                                                              not any(map(lambda other:
                                                                          other in dag.points_to(connection),
                                                                          daughter_connections)),
                                                              daughter_connections)),
                                                   snd(pair)))
                                          ),
                                         daughters_connections))

        if any(map(lambda conj: any(map(lambda conns: len(list(filter(lambda x: x, conns))) > 1, snd(conj))),
                   daughters_connections)):
            ToGraphViz()(dag)
            raise ProofError('wtf')

        daughters_connections = list(map(lambda pair: (fst(pair),
                                                       list(map(lambda daughter:
                                                                fst(list(daughter)) if len(daughter) else None,
                                                                snd(pair)))),
                                         daughters_connections))

        return daughters_connections

    incoming = list(filter(lambda edge: edge.dep not in _head_deps or not exclude_heads, dag.incoming(node)))
    parents = set(map(lambda edge: edge.source, incoming))
    top = dag.first_common_predecessor(parents)
    if top is None or dag.attribs[top]['cat'] != 'conj':
        ToGraphViz()(dag)
        raise ProofError('Top is not a conj or no top.')
    conjunctions = _cats_of_type(dag, 'conj', dag.points_to(top).intersection(dag.pointed_by(node)).union({top}))
    conjunctions = set(filter(lambda conj: len(dag.distinct_paths_to(conj, node)) > 1, conjunctions))
    conjunctions = sorted(impose_order(conjunctions), key=lambda pair: fst(pair) == top, reverse=True)
    return conjunctions


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
    def simplify_crd(crd_type: WordType, arg_types_: List[WordType]) -> WordType:
        xs, result = isolate_xs(crd_type), last_instance_of(crd_type)

        if len(set(map(depolarize, arg_types))) > 1:
            raise ProofError('Non polymorphic conjunction.')

        if arg_types_ == xs:
            return crd_type
        else:
            xs, result = list(map(lambda x: x.result, xs)), result.result
            crd_type = reduce(lambda res_, arg_: ColoredType(arg_, res_, 'cnj'), reversed(xs), result)
            return simplify_crd(crd_type, arg_types_)
            # xs, result = list(map(get_functor_result, xs)), get_functor_result(result)
        # return reduce(lambda res_, arg_: ColoredType(arg_, res_, 'cnj'), reversed(xs), result)

    def is_gap_copy_parent(edge_: Edge) -> bool:
        return edge_.dep in _head_deps and is_copy(dag, edge_.target) and is_gap(dag, edge_.target, _head_deps)

    branch_proof = set()

    outgoing = dag.outgoing(parent)
    outgoing = set(filter(lambda edge: not is_copy(dag, edge.target) or is_gap_copy_parent(edge), outgoing))

    head = list(filter(lambda out: out.dep in _head_deps and dag.attribs[out.target]['type'] != placeholder, outgoing))
    if len(head) != 1:
        ToGraphViz()(dag)
        import pdb
        pdb.set_trace()
    head = fst(head)
    outgoing = list(filter(lambda out: out.dep not in _head_deps, outgoing))
    mods = list(filter(lambda out: out.dep in _mod_deps, outgoing))
    mods = order_nodes(dag, set(map(lambda out: out.target, mods)))
    args = list(filter(lambda out: out.dep not in _mod_deps, outgoing))
    sorted_args = order_nodes(dag, set(map(lambda out: out.target, args)))
    args = sorted(args, key=lambda x: sorted_args.index(x.target))
    arg_types = list(map(lambda out: get_simple_argument(dag, out.target), args))
    arg_deps = list(map(lambda out: out.dep, args))

    if dag.attribs[parent]['cat'] == 'conj':
        branch_output = simplify_crd(dag.attribs[head.target]['type'], arg_types)
    else:
        branch_output = get_simple_functor(dag, head.target)

    arg_proof, branch_output = align_args(branch_output,
                                          arg_types,
                                          arg_deps)
    mod_proof, branch_output = align_mods(branch_output, list(map(lambda node: get_simple_argument(dag, node), mods)))
    branch_proof = merge_proofs(branch_proof, (arg_proof, mod_proof))
    return branch_proof, branch_output


def align_args(functor: WordType, argtypes: Sequence[WordType], deps: Sequence[str]) -> Tuple[ProofNet, WordType]:
    def color_fold(functor_: WordType) -> Iterable[Tuple[str, WordType]]:
        def step(x: WordType) -> Optional[Tuple[Tuple[str, WordType], WordType]]:
            return ((x.color, x.argument), x.result) if isinstance(x, ColoredType) \
                                                        and x.color not in _mod_deps else None
        return unfoldr(step, functor_)

    def match_args(proof_: ProofNet, pair: Tuple[WordType, WordType]) -> ProofNet:
        pos = fst(pair)
        neg = snd(pair)
        return match(proof_, pos, neg)

    def make_pairs(argdeps_: List[Tuple[str, WordType]], functor_argdeps_: List[Tuple[str, WordType]]) \
            -> Tuple[List[Tuple[WordType, WordType]], List[Tuple[str, WordType]]]:
        ret = []
        rem = []
        for neg in functor_argdeps_:
            if neg in argdeps_:
                pos = argdeps_.pop(argdeps_.index(neg))
                ret.append((snd(pos), snd(neg)))
            else:
                rem.append(neg)
        return ret, rem

    proof = set()

    if argtypes:
        functor_argcolors = list(color_fold(functor))
        functor_argcolors = list(filter(lambda ac: fst(ac) not in _mod_deps, functor_argcolors))
        argdeps = list(zip(deps, argtypes))
        pairs, rem = make_pairs(argdeps, functor_argcolors)
        proof = reduce(match_args, pairs, proof)
        return proof, reduce(lambda x, y: ColoredType(result=x, argument=y[1], color=y[0]), rem,
                             get_functor_result(functor))
    return proof, functor


def align_mods(mod_input: WordType, mods: Sequence[WordType]) -> Tuple[ProofNet, WordType]:
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

    try:
        if dag.edges:

            temp = iterate_simple_fringe(new_dag)
            if temp is not None:
                proof, new_dag = temp

            new_dag = delete_ghost_nodes(new_dag)
            copy_proof = match_copies_with_crds(new_dag)
            copy_gap_proof = match_copy_gaps_with_crds(new_dag)
            proof = merge_proofs(proof, [copy_proof, copy_gap_proof])
    except ProofError as e:
        raise e

    root_type = new_dag.attribs[fst(list(new_dag.get_roots()))]['type']

    if not isinstance(root_type, AtomicType):
        raise ProofError('Derived a complex type.')

    idx, conclusion = polarize_and_index(root_type.depolarize(), False, idx)

    proof = match(proof, root_type, conclusion)

    positives = set(map(fst, proof))
    negatives = set(map(snd, proof))
    immaterial = set(map(lambda type_: type_.index, filter(lambda type_: type_ == placeholder,
                                                           map(lambda leaf: new_dag.attribs[leaf]['type'],
                                                               filter(new_dag.is_leaf, dag.nodes)))))

    if set.union(positives, negatives, immaterial) != set(range(idx)):
        ToGraphViz()(new_dag)
        print(set(range(idx)).difference(set(map(fst, proof)).union(set(map(snd, proof)))))
        raise ProofError('Unmatched types.')

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

    def __call__(self, dag: DAG, raise_errors: bool = False) -> Optional[Tuple[ProofNet, DAG]]:
        try:
            return annotate_dag(dag)
        except ProofError as e:
            if raise_errors:
                raise e
            else:
                return

