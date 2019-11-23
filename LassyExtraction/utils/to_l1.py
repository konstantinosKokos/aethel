from LassyExtraction.milltypes import WordType, PolarizedIndexedType, ColoredType
from LassyExtraction.graphutils import DAG, fst, snd, last, Node
from LassyExtraction.extraction import order_nodes

from LassyExtraction.proofs import ProofNet

from itertools import chain
from typing import *


class Atom(object):
    def __init__(self, atom: str, features: List[int]):
        self.type = atom
        self.features = features

    def __str__(self) -> str:
        return 'at(' + self.type + ', ' + str(self.features) + ')'

    def __repr__(self) -> str:
        return str(self)


class Implication(object):
    def __init__(self, form_a: 'Formula', form_b: 'Formula'):
        self.A = form_a
        self.B = form_b

    def __str__(self) -> str:
        return 'impl(' + str(self.A) + ', ' + str(self.B) + ')'

    def __repr__(self) -> str:
        return str(self)


Formula = Union[Atom, Implication]


class L1(NamedTuple):
    sent_id: str
    words: List[str]
    poses: List[str]
    postags: List[str]
    lemmata: List[str]
    formulas: List[Formula]
    conclusion: Formula

    def __str__(self) -> str:
        return 'lassy({}) :-\n\t{},\n\t{},\n\t{}\n\t{}\n\t{}\n\t{}\n.'.format(self.sent_id.split('/')[-1],
                                                                              self.words, self.poses,
                                                                              self.postags, self.lemmata,
                                                                              self.formulas, self.conclusion)


def project_leaf(dag: DAG, leaf: Node) -> Tuple[str, WordType, str, str, str, str]:

    def wrap_mwu(fun: Callable[[Node], str]) -> Callable[[Node], str]:
        return lambda leaf_: 'mwp' if ' ' in get_word(leaf_) else fun(leaf_)

    def get_word(leaf_: Node) -> str:
        return dag.attribs[leaf_]['word']

    def get_type(leaf_: Node) -> str:
        return dag.attribs[leaf_]['type']

    @wrap_mwu
    def get_pos(leaf_: Node) -> str:
        return dag.attribs[leaf_]['pos']

    @wrap_mwu
    def get_postag(leaf_: Node) -> str:
        return dag.attribs[leaf_]['postag']

    @wrap_mwu
    def get_pt(leaf_: Node) -> str:
        return dag.attribs[leaf_]['pt']

    @wrap_mwu
    def get_lemma(leaf_: Node) -> str:
        return dag.attribs[leaf_]['lemma']

    return tuple(map(lambda function: function(leaf), (get_word, get_type, get_pos, get_postag, get_lemma)))


def to_l1(proof: ProofNet, dag: DAG) -> L1:
    leaves = set(filter(lambda node: dag.is_leaf(node), dag.nodes))
    leaves = order_nodes(dag, leaves)
    matchings = get_matchings(proof)
    words, types, poses, postags, lemmata = list(zip(*list(map(lambda leaf: project_leaf(dag, leaf), leaves))))
    formulas = list(map(lambda type_: type_to_formula(type_, matchings), types))
    sent_id = dag.meta['src']
    return L1(sent_id, list(words), list(poses), list(postags), list(lemmata), list(formulas),
              get_conclusion(types, matchings))


def atomic_type_to_atom(inp: PolarizedIndexedType, matchings: Dict[int, int]) -> Atom:
    if inp.polarity:
        return Atom(str(inp.depolarize()).lower(), [inp.index])
    else:
        return Atom(str(inp.depolarize()).lower(), [matchings[inp.index]])


def colored_type_to_impl(inp: ColoredType, matchings: Dict[int, int]) -> Implication:
    a = inp.argument
    b = inp.result
    return Implication(atomic_type_to_atom(a, matchings) if isinstance(a, PolarizedIndexedType) else
                       colored_type_to_impl(a, matchings),
                       atomic_type_to_atom(b, matchings) if isinstance(b, PolarizedIndexedType) else
                       colored_type_to_impl(b, matchings))


def type_to_formula(type_: WordType, matchings: Dict[int, int]) -> Formula:
    return atomic_type_to_atom(type_, matchings) if isinstance(type_, PolarizedIndexedType) else \
        colored_type_to_impl(type_, matchings)


def get_matchings(proof: ProofNet) -> Dict[int, int]:
    return {v: k for k, v in proof}


def get_conclusion(types_: List[WordType], matchings: Dict[int, int]) -> Atom:
    atomic_types = map(lambda type_: type_.get_atomic(), types_)
    atomic_types = list(chain.from_iterable(atomic_types))
    atomic_indexes = set(map(lambda atomic: atomic.index, atomic_types))
    missing = fst(list(set(k for k in matchings.keys()) - atomic_indexes))
    missing = matchings[missing]
    conclusion = fst(list(filter(lambda atomic: atomic.index == missing, atomic_types)))
    return atomic_type_to_atom(conclusion, matchings)


def project_one_dag(dag: DAG) -> List[Sequence[str]]:
    leaves = set(filter(dag.is_leaf, dag.nodes))
    leaves = order_nodes(dag, leaves)
    leaves = list(map(lambda leaf: project_leaf(dag, leaf)[0:3], leaves))
    return list(map(lambda leaf: (fst(leaf), snd(leaf).depolarize().__str__(), last(leaf)), leaves))


def get_wtp_tuples(dags: List[DAG]) -> List[Sequence[str]]:
    return list(map(lambda seq: tuple(seq[0:3]),
                    list(chain.from_iterable(list(map(project_one_dag, dags))))))


def wp_to_t(wtps: List[Sequence[str]]):
    def getkey(wtp: Sequence[str]) -> Tuple[str, str]:
        return fst(wtp), last(wtp)

    def getvalue(wtp: Sequence[str]) -> str:
        return snd(wtp)

    keys = set(map(getkey, wtps))
    values = set(map(getvalue, wtps))

    outer = {k: {v: 0 for v in values} for k in keys}

    for wtp in wtps:
        key = getkey(wtp)
        value = getvalue(wtp)
        outer[key][value] = outer[key][value] + 1
    return outer


def sort_wpt(outer: Dict[Tuple[str, str], Dict[str, int]]) -> List[Tuple[Tuple[str, str], Sequence[Tuple[str, int]]]]:
    outer = {outer_key:
                 sorted(filter(lambda pair: snd(pair) > 0, inner_dict.items()),
                        key=lambda pair: snd(pair),
                        reverse=True)
             for outer_key, inner_dict in outer.items()}
    return sorted(outer.items(), key=lambda pair: sum(list(map(snd, snd(pair)))))


def print_sorted_wpt(outer: List[Tuple[Tuple[str, str], Sequence[Tuple[str, int]]]]):
    def print_one(wpt: Tuple[Tuple[str, str], Sequence[Tuple[str, int]]]) -> str:
        def print_inner(inner: Sequence[Tuple[str, int]]) -> str:
            def print_pair(pair: Tuple[str, int]) -> str:
                return fst(pair) + ' #= ' + str(snd(pair))
            return ' | '.join(list(map(print_pair, inner)))
        return fst(fst(wpt))+'\t'+snd(fst(wpt)) + '\t' + print_inner(snd(wpt))
    return '\n'.join(list(map(print_one, outer)))