from src.Extraction import *

Proof = Dict[int, int]


class Proofify(object):
    def __init__(self, decomposer: Decompose):
        self.decomposer = decomposer

    def __call__(self, grouped: Grouped, type_dict: Dict[str, WordType], type_sequence: WordTypes,
                 top_type: PolarizedIndexedType):
        atomic_lexicon = reduce(lambda x, y: x.union(y.get_atomic()), type_sequence, {top_type})
        proof = iterate_top_down(grouped, type_dict, self.decomposer)
        check_proof(atomic_lexicon, proof)
        return proof


def iterate_top_down(grouped: Grouped, type_dict: Dict[str, WordType], decompose: Decompose):
    node_dict = {node.attrib['id']: node for node in
                 set(grouped.keys()).union(set([v[0] for v in chain.from_iterable(grouped.values())]))}
    proof = dict()

    left = list(grouped.keys())
    done = []

    fringe = decompose.fringe_heads_bottom_up(grouped, left, done)
    while fringe:
        for f in fringe:
            try:
                match_branch(f, grouped, type_dict, decompose, proof)
            except AttributeError:
                decompose.annotate_nodes(type_dict, node_dict)
                ToGraphViz()(grouped)
                raise AttributeError
            left.remove(f)
            done += [f]
        fringe = decompose.fringe_heads_bottom_up(grouped, left, done)
    decompose.annotate_nodes(type_dict, node_dict)
    ToGraphViz()(grouped)

    return proof


def check_proof(atomic_lexicon: Set[PolarizedIndexedType], proof: Proof):
    atomic_lexicon = {a.index: (AtomicType(a.result), a.polarity) for a in atomic_lexicon}
    for k, v in proof.items():
        atom_neg = atomic_lexicon[k]
        atom_pos = atomic_lexicon[v]
        assert atom_pos[0] == atom_neg[0]
        assert atom_neg[1] is False
        assert atom_pos[1] is True


def match_branch(parent: Optional[ET.Element], grouped: Grouped, type_dict: Dict[str, WordType],
                 decompose: Decompose, proof: Proof):
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
        top_type = match_branch_conj(head_type, args, type_dict, proof)
    else:
        top_type = match_branch_args(head_type, args, type_dict, decompose, proof, decompose.is_gap(head))

    # now do modifiers
    top_type = match_branch_mods(top_type, mods, type_dict, proof, mod_gaps)

    if parent is not None:
        type_dict[parent.attrib['id']] = top_type


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
    assert len(mod_atoms) == len(result_atoms)
    for m, r in zip(mod_atoms, result_atoms):
        if m[1] is False:
            assert m[0] not in proof.keys()
            proof[m[0]] = r[0]
        else:
            assert r[0] not in proof.keys()
            proof[r[0]] = m[0]
    return mod.result


def match_branch_args(head_type: WordType, args: ChildRels, type_dict: Dict[str, WordType],
                           decompose: Decompose, proof: Proof, gap: bool):
    if gap:
        head_type = ColoredType(arguments=(head_type.argument.result,), result=head_type.result,
                                colors=(head_type.color,),)
    owed = []
    # todo: track owed id also and then propagate when necessary
    while isinstance(head_type, ComplexType):
        color = head_type.color
        if color in decompose.mod_candidates:
            break
        a = [fst(a) for a in args if decompose.get_rel(snd(a)) == color]

        if len(a) > 1:
            raise AssertionError('Too many ({}) arguments with rel: {}'.format(len(a), color))
        else:
            a = fst(a)
            if decompose.is_copy(a):
                if decompose.is_gap(a):
                    if Counter(list(map(lambda x: decompose.get_rel(x), a.attrib['rel'].values())))[color] > 1:
                        # we are inspecting a copied edge of a gap node, use embedded argument
                        print(type_dict[a.attrib['id']])
                    else:
                        # we are inspecting a non-copied edge of a gap node, use result of argument
                        # and associate the embedded argument with the head
                        print(1)
                    head_type = head_type.result
                else:
                    print(2)
                #
                #
                #
                # # if the argument is a copy and we are at a copied dependency
                # if Co:
                #     print(head_type)
                #     owed.append((DoubleIndexedType(result=head_type.argument.result,
                #                                    polarity=head_type.argument.polarity,
                #                                    memory=type_dict[a.attrib['id']].argument.argument.index,
                #                                    index=head_type.argument.polarity), color))
                #     print(owed)
                # else:
                #     owed.append((DoubleIndexedType(result=head_type.result,
                #                                    polarity=head_type.polarity,
                #                                    memory=type_dict[a.attrib['id']].index,
                #                                    index=head_type.polarity), color))
                # head_type = head_type.result
                # continue
            else:
                a_type = type_dict[a.attrib['id']]
                arg_match(head_type.argument, a_type, proof)
                head_type = head_type.result
    if owed:
        arguments, colors = list(zip(*owed))
        return ColoredType(arguments=arguments, colors=colors, result=head_type)
    else:
        return head_type


def arg_match(head_arg: WordType, arg: WordType, proof: Proof):
    assert head_arg.index not in proof.keys()
    try:
        proof[head_arg.index] = arg.index
    except AttributeError:
        print(head_arg, arg)
        raise AttributeError


def match_branch_conj(head_type: WordType, args: ChildRels, type_dict: Dict[str, WordType], proof: Proof) -> WordType:
    arity = head_type.get_arity()

    if arity > 2:
        raise NotImplementedError('Too hard ({})'.format(arity))
    if arity == 1:
        # simple conjunction
        for a, _ in args:
            arg_match(head_type.argument, type_dict[a.attrib['id']], proof)
            head_type = head_type.result
        return head_type
    elif arity == 2:
        # missing args
        for a, _ in args:
            # identify conjunct within coordinator
            current_conj = head_type.argument
            # move coordinator to the right
            head_type = head_type.result

            # identify conjunct argument
            arg_type = type_dict[a.attrib['id']]

            while isinstance(arg_type, ComplexType):
                print(arg_type)
                print(current_conj)
                # subargument within conjunct
                subarg_arg = arg_type.argument
                arg_type = arg_type.result
                # subargument within coordinator
                subarg_conj = current_conj.argument
                current_conj = current_conj.result
                arg_match(subarg_arg, subarg_conj, proof)
            arg_match(current_conj, arg_type, proof)
        return head_type
