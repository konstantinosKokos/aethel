from ..milltypes import *
from ..graphutils import DAG, fst, snd, last, Node
from ..extraction import order_nodes

from LassyExtraction.proofs import AxiomLinks

from itertools import chain
from typing import *


class Atom(object):
    def __init__(self, atom: str, features: List[int]):
        self.type = atom
        self.features = features

    def __str__(self) -> str:
        return f'at({self.type}, {str(self.features)})'

    def __repr__(self) -> str:
        return str(self)


class Implication(object):
    def __init__(self, form_a: 'Formula', form_b: 'Formula'):
        self.A = form_a
        self.B = form_b

    def __str__(self) -> str:
        return f'impl({str(self.A)}, {str(self.B)})'

    def __repr__(self) -> str:
        return str(self)


class ModalImplication(Implication):
    def __init__(self, form_a: 'Formula', form_b: 'Formula', modality: str, dep: str):
        super(ModalImplication, self).__init__(form_a, form_b)
        self.modality = modality
        self.dep = dep

    def __str__(self) -> str:
        return f'{self.modality}({self.dep}, {str(self.A)}, {str(self.B)})'


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
        return f'lassy({self.sent_id},' \
               f'\n\t{self.words},' \
               f'\n\t{self.poses},' \
               f'\n\t{self.postags},' \
               f'\n\t{self.lemmata},' \
               f'\n\t{self.formulas},' \
               f'\n\t{self.conclusion}).\n'


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


def to_l1(dag: DAG, proof: AxiomLinks) -> L1:
    leaves = set(filter(lambda node: dag.is_leaf(node), dag.nodes))
    leaves = order_nodes(dag, list(leaves))
    matchings = get_matchings(proof)
    words, types, poses, postags, lemmata = list(zip(*list(map(lambda leaf: project_leaf(dag, leaf), leaves))))
    if len(types) == 1:
        _, types = polarize_and_index_many(types, 0)
        conclusion = Atom(str(types[0].depolarize()).lower(), [types[0].index])
    else:
        conclusion = get_conclusion(types, matchings)
    formulas = list(map(lambda type_: type_to_formula(type_, matchings), types))
    sent_id = dag.meta['src'].split('/')[-1].split('_')
    sent_id = '.'.join(sent_id[0].split('.')[:-1]) + f'_{sent_id[1]}.xml'
    return L1(sent_id, list(words), list(poses), list(postags), list(lemmata), list(formulas), conclusion)


def atomic_type_to_atom(inp: PolarizedType, matchings: Dict[int, int]) -> Atom:
    if inp.polarity:
        return Atom(str(inp.depolarize()).lower(), [inp.index])
    else:
        return Atom(str(inp.depolarize()).lower(), [matchings[inp.index]])


@overload
def functor_to_impl(inp: DiamondType, matchings: Dict[int, int]) -> ModalImplication:
    pass


@overload
def functor_to_impl(inp: BoxType, matchings: Dict[int, int]) -> ModalImplication:
    pass


@overload
def functor_to_impl(inp: FunctorType, matchings: Dict[int, int]) -> Implication:
    pass


def functor_to_impl(inp, matchings):
    arg = inp.argument
    res = inp.result
    a = atomic_type_to_atom(arg, matchings) if isinstance(arg, PolarizedType) else functor_to_impl(arg, matchings)
    b = atomic_type_to_atom(res, matchings) if isinstance(res, PolarizedType) else functor_to_impl(res, matchings)
    if isinstance(inp, DiamondType):
        return ModalImplication(a, b, 'dia', inp.diamond)
    elif isinstance(inp, BoxType):
        return ModalImplication(a, b, 'box', inp.box)
    elif isinstance(inp, FunctorType):
        return Implication(a, b)
    else:
        raise TypeError


def type_to_formula(type_: WordType, matchings: Dict[int, int]) -> Formula:
    if isinstance(type_, PolarizedType):
        return atomic_type_to_atom(type_, matchings)
    elif isinstance(type_, FunctorType):
        return functor_to_impl(type_, matchings)
    else:
        print(type_)
        raise TypeError


def get_matchings(proof: AxiomLinks) -> Dict[int, int]:
    return {v: k for k, v in proof}


def get_conclusion(types_: List[WordType], matchings: Dict[int, int]) -> Atom:
    atomic_types = map(lambda type_: type_.get_atomic(), types_)
    atomic_types = list(chain.from_iterable(atomic_types))
    atomic_indexes = set(map(lambda atomic: atomic.index, atomic_types))
    missing = fst(list(set(k for k in matchings.keys()) - atomic_indexes))
    missing = matchings[missing]
    conclusion = fst(list(filter(lambda atomic: atomic.index == missing, atomic_types)))
    return atomic_type_to_atom(conclusion, matchings)


def print_l1s(l1s: List[L1]) -> str:
    return '\n'.join(list(map(str, l1s)))