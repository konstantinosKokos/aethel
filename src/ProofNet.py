from src.Extraction import *

Proof = Dict[int, int]

class Proofify(object):
    def __init__(self, decomposer: Decompose):
        self.decomposer = decomposer

    def __call__(self, grouped: Grouped, type_sequence: WordTypes, top_type: PolarizedIndexedType):
        atomic_lexicon = reduce(lambda x, y: x.union(y.get_atomic()), type_sequence, {top_type})
        proof = dict()

        return atomic_lexicon


def iterate_top_down(grouped: Grouped, type_dict: Dict[str, WordType], decompose: Decompose):
    node_dict = {node.attrib['id']: node for node in
                 set(grouped.keys()).union(set([v[0] for v in chain.from_iterable(grouped.values())]))}
    proof = dict()

    left = list(grouped.keys())
    done = []

    fringe = decompose.fringe_heads_bottom_up(grouped, left, done)
    while fringe:
        for f in fringe:
            match_branch(f, grouped[f], type_dict, decompose, proof)
            left.remove(f)
            done += [f]
        fringe = decompose.fringe_heads_bottom_up(grouped, left, done)
    decompose.annotate_nodes(type_dict, node_dict)
    ToGraphViz()(grouped)

    return proof


def match_branch(parent: Optional[ET.Element], child_rels: ChildRels, type_dict: Dict[str, WordType],
                 decompose: Decompose, proof: Proof):
    # keep prim links
    child_rels = list(filter(lambda x: isinstance(snd(x), str) or snd(x).rank == 'primary', child_rels))
    # distinguish between args and mods
    args = list(filter(lambda x: decompose.get_rel(snd(x)) not in decompose.mod_candidates, child_rels))
    # mods are ambiguous, therefore need to be sorted
    mods = decompose.order_siblings(list(filter(lambda x: x not in args, child_rels)))

    # settle arg/head first
    head = fst(decompose.choose_head(args))
    head_type = type_dict[head.attrib['id']]
    top_type = match_branch_args(head_type, args, type_dict, decompose, proof)

    # now do modifiers
    top_type = match_branch_mods(top_type, mods, type_dict, proof)

    if parent is not None:
        type_dict[parent.attrib['id']] = top_type


def match_branch_mods(top_type: WordType, mods: ChildRels, type_dict: Dict[str, WordType], proof: Proof):
    for mod in mods[::-1]:
        node, rel = mod
        top_type = mod_match(top_type, type_dict[node.attrib['id']], proof)
    return top_type


def mod_match(result: WordType, mod: WordType, proof: Proof):
    mod_atoms = sorted(list(map(lambda x: x.index, mod.argument.get_atomic())))
    result_atoms = sorted(list(map(lambda x: x.index, result.get_atomic())))
    for m, r in zip(mod_atoms, result_atoms):
        assert m not in proof.keys()
        proof[m] = r
    return mod.result


def match_branch_args(head_type: WordType, args: ChildRels, type_dict: Dict[str, WordType],
                           decompose: Decompose, proof: Proof):
    if args:
        while isinstance(head_type, ComplexType):
            color = head_type.color
            if color in decompose.mod_candidates:
                break
            a = [fst(a) for a in args if decompose.get_rel(snd(a)) == color]
            assert len(a) == 1
            a = fst(a)

            arg_match(head_type.argument, type_dict[a.attrib['id']], proof)
            head_type = head_type.result
        return head_type
    else:
        return head_type


def arg_match(head_arg: WordType, arg: WordType, proof: Proof):
    assert head_arg.index not in proof.keys()
    proof[head_arg.index] = arg.index





