from src.Extraction import *

Pair = Tuple[int, int]
ProofNet = Set[Pair]


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
    root = decompose.get_disconnected(grouped)
    assert len(root) == 1
    root = list(root)[0]

    fringe = decompose.fringe_heads_top_down(grouped, list(grouped.keys()), [])

    for f in fringe:
        match_branch(type_dict[root.attrib['id']], grouped[f], type_dict, node_dict, decompose)
    decompose.annotate_nodes(type_dict, node_dict)
    ToGraphViz()(grouped)


def match_branch(top_type: WordType, child_rels: ChildRels, type_dict: Dict[str, WordType],
                 node_dict: Dict[str, ET.Element], decompose: Decompose):
    child_rels = list(filter(lambda x: isinstance(snd(x), str) or snd(x).rank=='primary', child_rels))
    mods = list(filter(lambda x: decompose.get_rel(snd(x)) in decompose.mod_candidates, child_rels))
    top_type = match_branch_mods(top_type, mods, type_dict)


def match_branch_mods(top_type: WordType, mods: ChildRels, type_dict: Dict[str, WordType]):
    for mod in mods[::-1]:
        node, rel = mod
        top_type = mod_match(top_type, type_dict[node.attrib['id']])
        type_dict[node.attrib['id']] = top_type
    return top_type


def mod_match(result: WordType, mod: WordType):
    return ColoredType(arguments=(mod.argument,), colors=(mod.color,), result=result)






