from src.Extraction import *

Proof = Dict[int, int]


class Proofify(object):
    def __init__(self, decomposer: Decompose):
        self.decomposer = decomposer

    def __call__(self, grouped: Grouped, type_dict: Dict[str, WordType], type_sequence: WordTypes,
                 top_type: PolarizedIndexedType):
        atomic_lexicon = reduce(lambda x, y: x.union(y.get_atomic()), type_sequence, {top_type})
        proof = iterate_top_down(grouped, type_dict, self.decomposer)
        result = type_dict[fst(list(self.decomposer.get_disconnected(grouped))).attrib['id']]
        arg_match(top_type, result, proof)
        check_proof(atomic_lexicon, proof)
        return proof


def iterate_top_down(grouped: Grouped, type_dict: Dict[str, WordType], decompose: Decompose):

    buffer = dict()

    node_dict = {node.attrib['id']: node for node in
                 set(grouped.keys()).union(set([v[0] for v in chain.from_iterable(grouped.values())]))}
    proof = dict()

    left = list(grouped.keys())
    done = []

    fringe = decompose.fringe_heads_bottom_up(grouped, left, done)
    while fringe:
        for f in fringe:
            match_branch(f, grouped, type_dict, decompose, proof, buffer)
            left.remove(f)
            done += [f]
        fringe = decompose.fringe_heads_bottom_up(grouped, left, done)
    decompose.annotate_nodes(type_dict, node_dict)
    ToGraphViz()(grouped)
    return proof


def check_proof(atomic_lexicon: Set[PolarizedIndexedType], proof: Proof):
    negatives = set(map(lambda x: x.index, filter(lambda x: not x.polarity, atomic_lexicon)))
    positives = set(map(lambda x: x.index, filter(lambda x: x.polarity, atomic_lexicon)))
    keys = set(proof.keys())
    values = set(proof.values())
    if not keys == negatives:
        raise AssertionError('Disagreement between keys and negatives\n'
                             'N-K : {} \n'
                             'K-N : {} \n'.format(negatives.difference(keys), keys.difference(negatives)))
    if not values == positives:
        raise AssertionError('Disagreement between keys and negatives\n'
                             'P-V : {} \n'
                             'V-P : {} \n'.format(positives.difference(values), values.difference(positives)))


def match_branch(parent: Optional[ET.Element], grouped: Grouped, type_dict: Dict[str, WordType],
                 decompose: Decompose, proof: Proof, buffer: Dict[int, int]):
    child_rels = grouped[parent]
    # distinguish between args and mods
    args = list(filter(lambda x: decompose.get_rel(snd(x)) not in decompose.mod_candidates, child_rels))
    # mods are ambiguous, therefore need to be sorted
    mods = decompose.order_siblings(list(filter(lambda x: x not in args, child_rels)))
    # todo: when does an embedded moddifier apply itself ?
    mod_gaps = list(map(lambda x: decompose.is_gap(fst(x)), mods))

    # settle arg/head first
    head = fst(decompose.choose_head(args))
    head_type = type_dict[head.attrib['id']]

    args = list(filter(lambda x: fst(x) != head, args))

    if parent is not None and 'cat' in parent.attrib.keys() and parent.attrib['cat'] == 'conj':
        args = decompose.order_siblings(args)
        top_type = match_branch_conj(head_type, args, type_dict, proof, buffer, decompose)
    else:
        top_type = match_branch_args(head_type, args, type_dict, decompose, proof, decompose.is_gap(head),
                                     decompose.is_copy(head), buffer)

    # now do modifiers
    if top_type.get_arity() == 0 or top_type.color in decompose.mod_candidates + ('invdet',):
        # easy case; modifier of atom or modifier of modifier
        top_type = match_branch_mods(top_type, mods, type_dict, proof, mod_gaps)
    elif top_type.get_arity() == 1:
        # ok case; modifier of gap
        owed = []
        while top_type.get_arity() > 0 and top_type.color not in decompose.mod_candidates:
            owed.append((top_type.argument, top_type.color))
            top_type = top_type.result
        top_type = match_branch_mods(top_type, mods, type_dict, proof, mod_gaps)
        args, colors = list(zip(*owed))
        top_type = ColoredType(arguments=args, colors=colors, result=top_type)
    elif mods or mod_gaps:
        # must be head copy?
        print(top_type)
        print(top_type.get_arity())
        raise NotImplementedError('seriously')

    if parent is not None:
        type_dict[parent.attrib['id']] = top_type


def match_branch_args(head_type: WordType, args: ChildRels, type_dict: Dict[str, WordType],
                      decompose: Decompose, proof: Proof, gap: bool, copy: bool, buffer: Any):

    owed = []

    if gap:
        head_type = ColoredType(arguments=(head_type.argument.result,), colors=(head_type.color,),
                                result=head_type.result)

    if copy:
        pre = []
        post = head_type.result

    while isinstance(head_type, ComplexType):
        color = head_type.color
        if color in decompose.mod_candidates:
            break

        a = [fst(a) for a in args if decompose.get_rel(snd(a)) == color]
        if len(a) > 1:
            raise AssertionError('Too many ({}) arguments with rel: {}'.format(len(a), color))
        elif not len(a):
            return head_type
        a = fst(a)

        is_copy = decompose.is_copy(a)
        is_gap = decompose.is_gap(a)

        if not any([is_copy, is_gap]):
            # standard case
            a_type = type_dict[a.attrib['id']]
        elif all([is_copy, is_gap]):
            # two subcases; we are at the copy part or we are at the gap part (could we be at both?)
            if Counter(list(map(lambda x: decompose.get_rel(x), a.attrib['rel'].values())))[color] > 1:
                owed.append((head_type.argument, color))
                buffer[head_type.argument.index] = type_dict[a.attrib['id']].argument.argument
                head_type = head_type.result
                continue
            else:
                raise NotImplementedError('Neither exactly copy nor gap.')
        elif is_gap:
            # gap case -- simply take embedded argument
            a_type = type_dict[a.attrib['id']].argument.argument
        elif is_copy:
            # propagate negative arguments upwards
            owed.append((head_type.argument, color))
            # associate negative argument with its positive counterpart
            buffer[head_type.argument.index] = type_dict[a.attrib['id']]
            head_type = head_type.result
            continue

        # if I reach this point, I know what the argument type is
        if not copy:
            arg_match(head_type.argument, a_type, proof)  # placeholder
        else:
            buffer[a_type.index] = head_type.argument
            pre.append((a_type, color))
        head_type = head_type.result

    if owed:
        arguments, colors = list(zip(*owed))
        head_type = ColoredType(arguments=arguments, colors=colors, result=head_type)
    if copy:
        if not pre:
            raise NotImplementedError('Copied head with no common arguments?')
        arguments, colors = list(zip(*pre))
        head_type = ColoredType(arguments=arguments, colors=colors, result=post)
        head_type = ColoredType(arguments=(head_type,), colors=('embedded',), result=post)
    return head_type


def arg_match(negative_arg: WordType, positive_arg: WordType, proof: Proof):
    if not isinstance(negative_arg, PolarizedIndexedType):
        raise AssertionError('Received non-atomic negative arg {} with match {}'.format(negative_arg, positive_arg))
    if not isinstance(positive_arg, PolarizedIndexedType):
        raise AssertionError('Received non-atomic positive arg {} with match {}'.format(positive_arg, negative_arg))
    if negative_arg.polarity:
        raise AssertionError('Received negative polarity arg {} with match {}'.format(negative_arg, positive_arg))
    assert positive_arg.polarity
    assert negative_arg.result == positive_arg.result
    assert negative_arg.index not in proof.keys()
    if positive_arg.index in proof.values():
        raise AssertionError('Received positive arg {} as match to {}, '
                             'already linked to {}'.format(positive_arg, negative_arg, [k for k in proof.keys() if
                                                                          proof[k] == positive_arg.index][0]))
    proof[negative_arg.index] = positive_arg.index


def match_branch_mods(top_type: WordType, mods: ChildRels, type_dict: Dict[str, WordType], proof: Proof,
                      mod_gaps: Sequence[bool]):
    for mod, gap in zip(mods[::-1], mod_gaps[::-1]):
        node, rel = mod
        top_type = mod_match(top_type,
                             type_dict[node.attrib['id']].argument.argument if gap else type_dict[node.attrib['id']],
                             proof)
    return top_type


def mod_match(result: WordType, mod: WordType, proof: Proof):
    mod_atoms = sorted(list(map(lambda x: (x.index, x.polarity), mod.argument.get_atomic())))
    result_atoms = sorted(list(map(lambda x: (x.index, x.polarity), result.get_atomic())))
    if len(mod_atoms) != len(result_atoms):
        raise AssertionError('Non-matching number of mod and result atoms: {} -- {}'.format(mod_atoms, result_atoms))
    for m, r in zip(mod_atoms, result_atoms):
        if m[1] is False:
            assert m[0] not in proof.keys()
            proof[m[0]] = r[0]
        else:
            assert r[0] not in proof.keys()
            proof[r[0]] = m[0]
    return mod.result


def match_branch_conj(head_type: WordType, args: ChildRels, type_dict: Dict[str, WordType], proof: Proof, buffer: Any,
                      decompose: Decompose) -> WordType:
    arity = head_type.get_arity()

    if arity > 2:
        for a, _ in args[:-1]:
            # each argument is a second order functor
            current_conj = type_dict[a.attrib['id']]   # (A->T)-> T // + - -  (non-indexed)
            current_conj = current_conj.argument   # A->T + -, partially indexed
            crd_type = head_type.argument       # (A_>T)->T  // - + +  (indexed)
            head_type = head_type.result        # X->X..->X

            crd_type_result = crd_type.result   # T (indexed, +)
            crd_type = crd_type.argument        # A-> T  (indexed, -)

            while isinstance(crd_type, ComplexType):
                subarg_arg = current_conj.argument
                current_conj = current_conj.result
                subarg_crd = crd_type.argument
                crd_type = crd_type.result
                arg_match(subarg_crd, subarg_arg, proof)
                del buffer[subarg_arg.index]
            arg_match(crd_type_result, crd_type, proof)

        # last argument
        a, _ = args[-1]
        current_conj = type_dict[a.attrib['id']]  # (A->T)-> T // + - -  (non-indexed)
        current_conj = current_conj.argument  # A->T + -, partially indexed
        crd_type = head_type.argument  # (A_>T)->T  // - + +  (indexed)
        head_type = head_type.result  #   (A->T) -> T // indexed, + - -, needs matching
        head_type_main = head_type.argument
        head_type_result = head_type.result

        crd_type_result = crd_type.result  # T (indexed, +)
        crd_type = crd_type.argument  # A-> T  (indexed, -)

        while isinstance(crd_type, ComplexType):
            subarg_arg = current_conj.argument
            current_conj = current_conj.result
            crd_type_arg = crd_type.argument
            crd_type = crd_type.result
            head_type_arg = head_type_main.argument
            head_type_main = head_type_main.result
            arg_match(crd_type_arg, subarg_arg, proof)
            arg_match(buffer[subarg_arg.index], head_type_arg, proof)

        arg_match(crd_type_result, crd_type, proof)
        # todo; what if the copy here was modified?
        # if current_conj.index in proof.values():
        #     mod_link = [k for k in proof.keys() if proof[k] == current_conj.index][0]
        #     print(proof)
        #     import pdb
        #     pdb.set_trace()
        arg_match(head_type_main, current_conj, proof)
        return head_type_result

    elif arity == 1:
        # simple conjunction
        for a, _ in args:
            arg_match(head_type.argument, type_dict[a.attrib['id']], proof)
            head_type = head_type.result
        return head_type
    elif arity == 2:
        # missing args
        for a, _ in args[:-1]:
            # identify conjunct within coordinator
            current_conj = head_type.argument

            # move coordinator to the right
            head_type = head_type.result

            # identify conjunct argument
            arg_type = type_dict[a.attrib['id']]

            while isinstance(arg_type, ComplexType):
                # subargument within conjunct (negative)
                subarg_arg = arg_type.argument
                arg_type = arg_type.result
                # subargument within coordinator (positive)
                subarg_conj = current_conj.argument
                current_conj = current_conj.result
                arg_match(subarg_arg, subarg_conj, proof)
            arg_match(current_conj, arg_type, proof)

        current_conj = head_type.argument
        head_type = head_type.result
        a, _ = args[-1]
        arg_type = type_dict[a.attrib['id']]

        temp = head_type
        while isinstance(arg_type, ComplexType):
            subarg_arg = arg_type.argument  # negative argument
            arg_type = arg_type.result

            subarg_conj = current_conj.argument  # positive
            current_conj = current_conj.result
            arg_match(subarg_arg, subarg_conj, proof)

            subarg_res = temp.argument  # negative
            temp = head_type.result
            if subarg_arg.index in buffer.keys():
                arg_match(subarg_res, buffer[subarg_arg.index], proof)
        arg_match(current_conj, arg_type, proof)

        if head_type.color in decompose.mod_candidates:
            return head_type
        else:
            return temp
