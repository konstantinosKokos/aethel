from functools import reduce
from itertools import chain

from LassyExtraction.extraction import order_nodes, is_gap, is_copy, HeadDeps, ModDeps, make_functor
from LassyExtraction.graphutils import *
from LassyExtraction.milltypes import (polarize_and_index_many, polarize_and_index, WordType, AtomicType, DiamondType,
                                       PolarizedType, BoxType, WordTypes, FunctorType, depolarize)
from LassyExtraction.transformations import _cats_of_type

ProofNet = Set[Tuple[int, int]]

placeholder = AtomicType('_')


def match(proofnet: ProofNet, positive: WordType, negative: WordType) -> ProofNet:
    if positive != negative:
        raise ProofError(f'Formulas are not equal.\t{positive}\t{negative}.')
    if any(map(lambda x: not is_indexed(x), [positive, negative])):
        raise ProofError(f'Input formulas are not fully indexed.\t{positive}\t{negative}.')
    if isinstance(positive, PolarizedType) and isinstance(negative, PolarizedType):
        if not positive.polarity:
            raise ProofError(f'Positive formula has negative index.\t{positive}\t{negative}.')
        if negative.polarity:
            raise ProofError(f'Negative formula has positive index.\t{positive}\t{negative}.')

        if positive.index in set(map(fst, proofnet)):
            raise ProofError(f'Positive formula already assigned.\n{positive}\n{negative}\n{proofnet}.')
        if negative.index in set(map(snd, proofnet)):
            raise ProofError(f'Negative formula already assigned.\t{positive}\t{negative}.')
        proofnet = proofnet.union({(positive.index, negative.index)})
    elif isinstance(negative, FunctorType) and isinstance(positive, FunctorType):
        proofnet = match(proofnet, negative.argument, positive.argument)
        proofnet = match(proofnet, positive.result, negative.result)
    else:
        raise ProofError(f'Unexpected types.')
    return proofnet


def merge_proof(core: ProofNet, local: ProofNet) -> ProofNet:
    for k, v in local:
        if k in set(map(fst, core)) and (k, v) not in core:
            raise ProofError('Positive formula already assigned in core proof.\t{}\t{}\t{}'.format(k, v, core))
        if v in set(map(snd, core)) and (k, v) not in core:
            raise ProofError('Negative formula already assigned in core proof.\t{}\t{}\t{}'.format(k, v, core))
        core = core.union({(k, v)})
    return core


def merge_proofs(locals_: List[ProofNet], core: Optional[ProofNet] = None) -> ProofNet:
    if core is None:
        return reduce(merge_proof, locals_) if locals_ else set()
    return reduce(merge_proof, locals_, core)


def make_proofnet(dag: DAG[str, str]) -> ProofNet:
    if not dag.edges:
        return set()

    deannotate_dag(dag)
    idx = annotate_leaves(dag)
    add_ghost_nodes(dag)
    core_proof = iterate_simple_fringe(dag)
    delete_ghost_nodes(dag)
    crd_proof = match_copies_with_crds(dag)
    gap_proof = match_copied_gaps_with_crds(dag)
    root_type = dag.attribs[fst(list(dag.get_roots()))]['type']
    idx, conclusion = polarize_and_index(root_type.depolarize(), False, idx)
    core_proof = match(core_proof, root_type, conclusion)
    proof = merge_proofs([core_proof, crd_proof, gap_proof])
    correctness_check(proof, dag, idx)
    return proof


def correctness_check(proof: ProofNet, dag: DAG, idx: int) -> None:
    positives = set(map(fst, proof))
    negatives = set(map(snd, proof))
    immaterial = set(map(lambda type_:
                         type_.index,
                         filter(lambda type_:
                                type_ == placeholder,
                                map(lambda leaf:
                                    dag.attribs[leaf]['type'],
                                    filter(lambda node:
                                           dag.is_leaf(node),
                                           dag.nodes)))))
    if set.union(positives, negatives, immaterial) != set(range(idx)):
        raise ProofError('Unmatched types.')


def iterate_simple_fringe(dag: DAG[str, str]) -> ProofNet:
    proof: ProofNet = set()
    while True:
        temp = annotate_simple_branches(dag)
        if temp is None:
            break
        proof = merge_proof(proof, temp)
    return proof


def annotate_simple_branches(dag: DAG[str, str]) -> Optional[ProofNet]:
    parents = get_simple_branches(dag)

    if not parents:
        return None

    temp = list(map(lambda parent: annotate_simple_branch(dag, parent), parents))
    branch_proofs, branch_outputs = list(zip(*temp))
    update_types(dag, parents, branch_outputs)
    return merge_proofs(branch_proofs)


def annotate_simple_branch(dag: DAG[str, str], parent: str) -> Tuple[ProofNet, WordType]:
    def simplify_crd(crd_type: WordType, arg_types_: WordTypes) -> WordType:
        xs = isolate_xs(crd_type)
        result = last_instance_of(crd_type)

        if len(set(map(depolarize, arg_types_))) > 1:
            raise ProofError('Non polymorphic conjunction.')

        if arg_types == xs:
            return crd_type
        else:
            xs = list(map(lambda x: x.result, xs))
            result = result.result

            crd_type = reduce(lambda res_, arg_:
                              DiamondType(argument=arg_, result=res_, diamond='cnj'),
                              reversed(xs),
                              result)
            return simplify_crd(crd_type, arg_types_)

    def is_gap_copy_parent(edge_: Edge[str, str]) -> bool:
        return edge_.dep in HeadDeps and is_copy(dag, edge_.target) and is_gap(dag, edge_.target, HeadDeps)

    outgoing = dag.outgoing(parent)
    outgoing = set(filter(lambda edge:
                          not is_copy(dag, edge.target) or is_gap_copy_parent(edge),
                          outgoing))

    heads = list(filter(lambda edge:
                        edge.dep in HeadDeps and dag.attribs[edge.target]['type'] != placeholder,
                        outgoing))

    if len(heads) != 1:
        raise ProofError(f'Too many heads: {heads}.')

    head = fst(heads)

    outgoing = set(filter(lambda edge:
                          edge.dep not in HeadDeps,
                          outgoing))
    modding_edges = list(filter(lambda edge:
                                edge.dep in ModDeps,
                                outgoing))
    arg_edges = list(filter(lambda edge:
                            edge.dep not in ModDeps,
                            outgoing))
    mods = order_nodes(dag, list(map(lambda edge: edge.target, modding_edges)))
    args = order_nodes(dag, list(map(lambda edge: edge.target, arg_edges)))
    arg_edges = sorted(arg_edges, key=lambda x: args.index(x.target))
    arg_types = list(map(lambda edge:
                         get_simple_argument(dag, edge.target),
                         arg_edges))
    arg_deps = list(map(lambda edge:
                        edge.dep,
                        arg_edges))

    if dag.attribs[parent]['cat'] == 'conj':
        branch_output = simplify_crd(dag.attribs[head.target]['type'], arg_types)
    else:
        branch_output = get_simple_functor(dag, head.target)

    arg_proof, branch_output = align_args(branch_output,
                                          arg_types,
                                          arg_deps)
    mod_proof, branch_output = align_mods(branch_output, list(map(lambda node: get_simple_argument(dag, node), mods)))
    return merge_proof(arg_proof, mod_proof), branch_output


def align_args(functor: WordType, argtypes: WordTypes, deps: List[str]) -> Tuple[ProofNet, WordType]:
    def color_fold(functor_: WordType) -> List[Tuple[str, WordType]]:
        def step(x: WordType) -> Optional[Tuple[Tuple[str, WordType], WordType]]:
            if isinstance(x, FunctorType):
                if isinstance(x, DiamondType):
                    return (x.diamond, x.argument), x.result
                if isinstance(x, BoxType) and x.box == 'det':
                    return ('np_hd', x.argument), x.result
            return None
        return list(unfoldr(step, functor_))

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

    def match_args(proof_: ProofNet, pair: Tuple[WordType, WordType]) -> ProofNet:
        return match(proof_, fst(pair), snd(pair))

    proof: Set[Tuple[int, int]] = set()

    if argtypes:
        functor_argcolors = color_fold(functor)
        functor_argcolors = list(filter(lambda ac: fst(ac) not in ModDeps, functor_argcolors))
        material_argcolors = list(zip(deps, argtypes))
        pairs, rem = make_pairs(material_argcolors, functor_argcolors)
        proof = reduce(match_args, pairs, proof)
        return proof, reduce(lambda x, y:
                             make_functor(argument=y[1], dep=y[0], result=x),
                             rem,
                             get_functor_result(functor))

    return proof, functor


def align_mods(mod_input: WordType, mods: WordTypes) -> Tuple[ProofNet, WordType]:
    def match_modchain(proof_: ProofNet, modpair: Tuple[WordType, WordType]) -> ProofNet:
        prev = fst(modpair)
        curr = snd(modpair)
        if isinstance(prev, FunctorType) and isinstance(curr, FunctorType):
            return match(proof_, prev.result, curr.argument)
        else:
            raise ProofError(f'Modifiers {prev}, {curr} are not FunctorTypes.')

    proof: Set[Tuple[int, int]] = set()

    if mods:
        mod_output = last(mods).result
        proof = match(proof, mod_input, fst(mods).argument)
        zipped_mods = list(zip(mods, mods[1:]))
        proof = reduce(match_modchain, zipped_mods, proof)
        return proof, mod_output
    return proof, mod_input


def match_copies_with_crds(dag: DAG[Node, Any]) -> ProofNet:
    def get_copy_color(copy_: Node) -> str:
        incoming_ = set(map(lambda inc_: inc_.dep, dag.incoming(copy_)))
        assert len(incoming_) == 1
        return fst(list(incoming_))

    copies = list(filter(lambda node:
                         is_copy(dag, node) and not is_gap(dag, node, HeadDeps),
                         dag.nodes))

    copy_colors = list(map(lambda copy: get_copy_color(copy), copies))
    copy_types: WordTypes = list(map(lambda copy: dag.attribs[copy]['type'], copies))

    conjunction_hierarchies: List[List[Tuple[Node, List[Optional[Node]]]]]
    conjunction_hierarchies = list(map(lambda node:
                                       participating_conjunctions(dag, node),
                                       copies))

    proofs = list(map(lambda ch, ct, cc:
                      intra_crd_match(dag, ch, ct, cc),
                      conjunction_hierarchies,
                      copy_types,
                      copy_colors))
    proof = merge_proofs(proofs)

    crds = list(map(lambda ch: get_crd_type(dag, fst(fst(ch))), conjunction_hierarchies))
    results = list(map(lambda crd: last_instance_of(crd), crds))
    matches = list(map(lambda res, ct, cc:
                       identify_missing(res, ct, cc),
                       results,
                       copy_types,
                       copy_colors))
    return reduce(lambda proof_, pair:
                  match(proof_, fst(pair), snd(pair)),
                  zip(copy_types, matches),
                  proof)


def participating_conjunctions(dag: DAG[Node, str], node: Node, exclude_heads: bool = False) -> \
        List[Tuple[Node, List[Optional[Node]]]]:

    def impose_order(conjunctions_: List[Node]) -> List[Tuple[Node, List[Optional[Node]]]]:

        def link_daughter_with_conjs(daughter: Node, conjs: List[Node]) -> Nodes:
            return set(filter(lambda conj_:
                              dag.exists_path(daughter, conj_) or daughter == conj_,
                              conjs))

        def filter_maximally_linked(candidates: Nodes) -> Nodes:
            return set(filter(lambda candidate:
                              not any(map(lambda contestant:
                                          len(dag.distinct_paths_to(contestant, node)) >
                                          len(dag.distinct_paths_to(candidate, node)),
                                          candidates)),
                              candidates))

        def select_lowest(candidates: Nodes) -> Optional[Node]:
            filtered = set(filter(lambda candidate:
                                  not any(map(lambda contestant:
                                              contestant in dag.points_to(candidate),
                                              candidates)),
                                  candidates))
            return fst(list(filtered)) if filtered else None

        # each conj paired with its daughters
        daughters: List[Tuple[Node, List[Node]]]
        daughters = list(map(lambda conj:
                             (conj, get_conjunction_daughters(dag, conj)),
                             conjunctions))
        # each conj daughter also paired with the set of conjunctions it covers
        paired: List[Tuple[Node, List[Nodes]]]
        paired = list(map(lambda pair:
                          (fst(pair), list(map(lambda daughter:
                                               link_daughter_with_conjs(daughter, conjunctions_),
                                               snd(pair)))),
                          daughters))

        maximal: List[Tuple[Node, List[Nodes]]]
        maximal = list(map(lambda pair:
                           (fst(pair), list(map(lambda daughter_group:
                                                filter_maximally_linked(daughter_group),
                                                snd(pair)))),
                           paired))
        lowest = list(map(lambda pair:
                          (fst(pair), list(map(lambda daughter_group:
                                               select_lowest(daughter_group),
                                               snd(pair)))),
                          maximal))
        return lowest

    incoming = list(filter(lambda edge:
                           edge.dep not in HeadDeps or not exclude_heads, dag.incoming(node)))
    parents = set(map(lambda edge: edge.source, incoming))
    top = dag.first_common_predecessor(parents)
    if top is None or dag.attribs[top]['cat'] != 'conj':
        raise ProofError('Top is not a conj or no top.')
    conjunctions = _cats_of_type(dag, 'conj', dag.points_to(top).intersection(dag.pointed_by(node)).union({top}))
    conjunctions = set(filter(lambda conj: len(dag.distinct_paths_to(conj, node)) > 1, conjunctions))
    return sorted(impose_order(list(conjunctions)), key=lambda pair: fst(pair) == top, reverse=True)


def intra_crd_match(dag: DAG[Node, str], hierarchy: List[Tuple[Node, List[Optional[Node]]]],
                    copy_type: WordType, copy_color: str) -> ProofNet:

    isolated: List[Tuple[WordTypes, List[Optional[Node]]]]
    isolated = list(map(lambda pair:
                        (isolate_xs(get_crd_type(dag, fst(pair))), snd(pair)),
                        hierarchy))

    zipped: List[List[Tuple[WordType, Optional[Node]]]]
    zipped = [list(zip(*x)) for x in isolated]

    filtered: List[Tuple[WordType, Optional[Node]]]
    filtered = list(filter(lambda pair: snd(pair) is not None, chain.from_iterable(zipped)))

    paired: List[Tuple[WordType, WordType]]
    paired = list(map(lambda pair:
                      (fst(pair),
                       last_instance_of(get_crd_type(dag, snd(pair)))),
                      filtered))

    matches = list(map(lambda pair:
                       (identify_missing(fst(pair), copy_type, copy_color),
                        identify_missing(snd(pair), copy_type, copy_color)),
                       paired))

    return reduce(lambda proof_, pair:
                  match(proof_, fst(pair), snd(pair)),
                  matches, set())


def match_copied_gaps_with_crds(dag: DAG[str, str]) -> ProofNet:
    def extract_color(type_: WordType) -> str:
        if isinstance(type_, FunctorType):
            if isinstance(type_.argument, DiamondType):
                return type_.argument.diamond
            if isinstance(type_.argument, BoxType):
                return type_.argument.box
        raise ProofError(f'Expected {type_} to be a higher order functor.')

    def extract_arg(type_: WordType) -> WordType:
        if isinstance(type_, FunctorType):
            if isinstance(type_.argument, DiamondType):
                return type_.argument.argument
            if isinstance(type_.argument, BoxType):
                return type_.argument.argument
        raise ProofError(f'Expected {type_} to be a higher order functor.')

    gaps = list(filter(lambda node:
                       is_copy(dag, node) and is_gap(dag, node, HeadDeps),
                       dag.nodes))
    gap_types: WordTypes = list(map(lambda node: dag.attribs[node]['type'], gaps))

    gap_colors = list(map(lambda gaptype: extract_color(gaptype), gap_types))
    gap_args = list(map(lambda gaptype: extract_arg(gaptype), gap_types))

    conj_hierarchies = list(map(lambda node:
                                participating_conjunctions(dag, node, exclude_heads=True),
                                gaps))

    proofs = list(map(lambda ch, ga, gc:
                      intra_crd_match(dag, ch, ga, gc),
                      conj_hierarchies,
                      gap_args,
                      gap_colors))

    crds = list(map(lambda ch:
                    get_crd_type(dag, fst(fst(ch))),
                    conj_hierarchies))

    results = list(map(lambda crd: last_instance_of(crd), crds))

    matches = list(map(lambda res, ga, gc:
                       identify_missing(res, ga, gc),
                       results,
                       gap_args,
                       gap_colors))

    return reduce(lambda proof_, pair:
                  match(proof_, fst(pair), snd(pair)),
                  zip(gap_args, matches),
                  merge_proofs(proofs))


def get_functor_result(functor: WordType) -> WordType:
    return snd(split_functor(functor))


def split_functor(functor: WordType) -> Tuple[Optional[WordType], WordType]:
    result = functor
    body = None
    while isinstance(result, FunctorType):
        if isinstance(result, BoxType) and result.box in ModDeps:
            break
        body = result.argument
        result = result.result
    return body, result


def get_simple_branches(dag: DAG[Node, Any]) -> List[Node]:
    annotated = get_annotated_nodes(dag)

    parents = set(filter(lambda node: not dag.is_leaf(node), dag.nodes))
    parents = parents.difference(annotated)
    return list(filter(lambda parent:
                       all(list(map(lambda child: child in annotated, dag.successors(parent)))),
                       parents))


def get_annotated_nodes(dag: DAG[Node, Any]) -> Nodes:
    return set(filter(lambda node:
                      is_indexed(dag.attribs[node]['type']),
                      dag.nodes))


def is_indexed(type_: WordType) -> bool:
    return all(list(map(lambda subtype: isinstance(subtype, PolarizedType), type_.get_atomic())))


def add_edge(dag: DAG[str, Any], edge: Edge[str, Any], type_: Optional[WordType]) -> None:
    def get_fresh_node(nodes_: Set[str]) -> str:
        node = -1
        while str(node) in nodes_:
            node -= 1
        return str(node)

    fresh_node = get_fresh_node(dag.nodes)
    fresh_edge = Edge(source=edge.source, target=fresh_node, dep=edge.dep)

    fresh_attrib = {'id': fresh_node,
                    'index': dag.attribs[edge.target]['index'],
                    'type': type_,
                    'begin': dag.attribs[edge.target]['begin'],
                    'end': dag.attribs[edge.target]['end']}

    dag.nodes.add(fresh_node)
    dag.edges.add(fresh_edge)
    dag.attribs[fresh_node] = fresh_attrib


def find_first_conjunction_above(dag: DAG[Node, Any], source: Node) -> Optional[Node]:
    conjunctions = _cats_of_type(dag, 'conj')
    node_ancestors = dag.pointed_by(source)

    conjunction_ancestors = node_ancestors.intersection(conjunctions)

    for ca in conjunction_ancestors:
        if not any(list(map(lambda other: dag.exists_path(ca, other), conjunction_ancestors))):
            return ca
    return None


def get_simple_argument(dag: DAG[Node, Any], node: Node) -> WordType:
    type_: WordType = dag.attribs[node]['type']
    if is_gap(dag, node, HeadDeps):
        if isinstance(type_, DiamondType):
            if isinstance(type_.argument, FunctorType):
                return type_.argument.argument
            else:
                raise ProofError(f'Gap argument {type_.argument} is not a FunctorType')
        else:
            raise ProofError(f'Gap type {type_} is not a DiamondType')
    return type_


def get_simple_functor(dag: DAG[Node, Any], node: Node) -> WordType:
    type_: WordType = dag.attribs[node]['type']
    if is_gap(dag, node, HeadDeps):
        if isinstance(type_, DiamondType):
            return DiamondType(argument=type_.argument.result, diamond=type_.diamond, result=type_.result)
        else:
            raise ProofError(f'Gap type {type_} is not a DiamondType')
    return type_


def get_crd_type(dag: DAG[Node, Any], node: Node) -> WordType:
    crds = list(filter(lambda out:
                       out.dep == 'crd' and dag.attribs[out.target]['type'] != placeholder,
                       dag.outgoing(node)))
    if len(crds) != 1:
        raise ProofError('Too many coordinators.')
    crd = fst(crds).target
    return dag.attribs[crd]['type']


def get_conjunction_daughters(dag: DAG[Node, Any], conjunction: Node) -> List[Node]:
    daughter_edges = list(filter(lambda out:
                                 out.dep not in HeadDeps.union(ModDeps),
                                 dag.outgoing(conjunction)))
    daughter_nodes = list(map(lambda out: out.target, daughter_edges))
    return order_nodes(dag, daughter_nodes)


def isolate_xs(coordinator: WordType) -> WordTypes:
    def step_cnj(coordinator_: WordType) -> Optional[Tuple[WordType, WordType]]:
        if isinstance(coordinator_, DiamondType) and coordinator_.diamond == 'cnj':
            return coordinator_.argument, coordinator_.result
        return None
    return list(unfoldr(step_cnj, coordinator))


def last_instance_of(coordinator: WordType) -> WordType:
    def step_cnj(coordinator_: WordType) -> Optional[Tuple[WordType, WordType]]:
        if isinstance(coordinator_, DiamondType) and coordinator_.diamond == 'cnj':
            return coordinator_.result, coordinator_.result
        return None
    return last(list(unfoldr(step_cnj, coordinator)))


def identify_missing(poly_x: WordType, missing: WordType, dep: str) -> WordType:
    while True:
        if isinstance(poly_x, DiamondType):
            if poly_x.diamond == dep and poly_x.argument == missing:
                break
        elif isinstance(poly_x, BoxType):
            if poly_x.box == dep and poly_x.argument == missing:
                break
        elif isinstance(poly_x, FunctorType) and poly_x.argument == missing and dep in HeadDeps.union({'np_hd'}):
            break
        poly_x = poly_x.result
    return poly_x.argument


def get_copy_annotation(dag: DAG, edge: Edge) -> WordType:
    source = edge.source

    while True:
        mother_node = find_first_conjunction_above(dag, source)

        if mother_node is None:
            raise ProofError('Conjunctionless copy.')

        missing_type = get_simple_argument(dag, edge.target)
        crd_type = get_crd_type(dag, mother_node)

        conjunction_daughters = get_conjunction_daughters(dag, mother_node)
        paths = list(map(lambda daughter:
                         dag.exists_path(daughter, edge.source) or daughter == edge.source,
                         conjunction_daughters))

        if any(paths):
            break

        source = mother_node

    daughter_index = paths.index(True)

    xs, res = isolate_xs(crd_type), last_instance_of(crd_type)
    return identify_missing(xs[daughter_index], missing_type, edge.dep)


def add_ghost_nodes(dag: DAG[str, Any]) -> None:
    edges = dag.edges
    copy_edges = set(filter(lambda edge: is_copy(dag, edge.target), edges))
    gap_edges = set(filter(lambda edge: is_gap(dag, edge.target, HeadDeps), edges))
    copy_gap_edges = copy_edges.intersection(gap_edges)
    copy_edges = copy_edges - copy_gap_edges

    copy_gap_edges = set(filter(lambda edge: edge.dep not in HeadDeps, copy_gap_edges))

    copies_and_types = list(map(lambda copy: (copy, get_copy_annotation(dag, copy)), copy_edges.union(copy_gap_edges)))
    for c, t in copies_and_types:
        add_edge(dag, c, t)


def delete_ghost_nodes(dag: DAG[str, Any]) -> None:
    dag.nodes = set(filter(lambda node: int(dag.attribs[node]['id']) > 0, dag.nodes))
    dag.attribs = {n: a for n, a in dag.attribs.items() if n in dag.nodes}
    dag.edges = set(filter(lambda edge: edge.source in dag.nodes and edge.target in dag.nodes, dag.edges))


def update_types(dag: DAG[Node, str], leaves: List[Node], types: WordTypes) -> None:
    for leaf, _type in zip(leaves, types):
        dag.attribs[leaf]['type'] = _type


def annotate_leaves(dag: DAG[str, str]) -> int:
    leaf_set = list(dag.get_leaves())
    leaves_sorted = order_nodes(dag, leaf_set)
    leaf_types: WordTypes = list(map(lambda leaf: dag.attribs[leaf]['type'], leaves_sorted))
    idx, leaf_types = polarize_and_index_many(leaf_types)
    update_types(dag, leaves_sorted, leaf_types)
    return idx


def deannotate_dag(dag: DAG) -> None:
    for node in dag.attribs.keys():
        if 'type' in dag.attribs[node].keys():
            dag.attribs[node]['type'] = depolarize(dag.attribs[node]['type'])


class ProofError(AssertionError):
    def __init__(self, message: str):
        super().__init__(message)


class Prove(object):
    def __init__(self):
        pass

    def __call__(self, dag: DAG[str, str], raise_errors: bool = False) -> Optional[Tuple[DAG, ProofNet]]:
        try:
            pn = make_proofnet(dag)
            return dag, pn
        except ProofError as e:
            if raise_errors:
                raise e
            else:
                return None


prover = Prove()
