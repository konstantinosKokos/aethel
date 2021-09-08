from .terms import *
from .milltypes import *
from .proofs import AxiomLinks
from functools import reduce
from operator import add
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Premise:
    """
        Class represeting a lexical item.
    """
    word: str
    type: WordType


@dataclass
class ProofFrame:
    """
        Class representing a judgement.
    """
    premises: List[Premise]
    conclusion: PolarizedType

    def get_words(self) -> List[str]:
        return [p.word for p in self.premises]

    def get_types(self) -> List[WordType]:
        return [p.type for p in self.premises]

    def __str__(self) -> str:
        return self.print()

    def __repr__(self) -> str:
        return str(self)

    def __len__(self) -> int:
        return len(self.get_words())

    def word_printer(self, idx: int, show_word: bool = True, show_type: bool = True) -> str:
        ret = self.premises[idx].word if show_word else f'w{subscript(idx)}'
        return ret + (f':{self.premises[idx].type.depolarize()}' if show_type else '')

    def print(self, show_words: bool = True, show_types: bool = True) -> str:
        ret = ', '.join(map(lambda idx: self.word_printer(idx, show_words, show_types), range(len(self.premises))))
        return f'{ret} ⊢ {self.conclusion.depolarize()}'

    @staticmethod
    def from_data(words: List[str], types: List[WordType]) -> 'ProofFrame':
        if len(types) == 1:
            _, types = polarize_and_index_many(types, 0)
        premises = [Premise(word, wordtype) for (word, wordtype) in zip(words, types)]
        types = [t for t in types if not isinstance(t, EmptyType)]
        atoms = list(zip(*list(map(get_polarities_and_indices, types))))
        negative, positive = list(map(lambda x: reduce(add, x), atoms))
        rem = (Counter([x[0] for x in positive]) - Counter([x[0] for x in negative]))
        assert len(rem) == 1
        assert sum(rem.values()) == 1
        conclusion = list(rem.keys())[0]
        conclusion_id = max((x[1] for x in sum(atoms[0]+atoms[1], []))) + 1
        return ProofFrame(premises, PolarizedType(_type=conclusion.type, polarity=False, index=conclusion_id))


@dataclass
class ProofNet:
    """
        Class representing a judgement together with its proof.
    """
    proof_frame: ProofFrame
    axiom_links: AxiomLinks
    name: Optional[str]

    def __len__(self) -> int:
        return len(self.proof_frame)

    def print_frame(self, show_words: bool = True, show_types: bool = True) -> str:
        return self.proof_frame.print(show_words, show_types)

    def print_term(self, show_words: bool = False, show_types: bool = False, show_decorations: bool = True) -> str:
        return print_term(self.get_term(), show_decorations,
                          lambda idx: self.proof_frame.word_printer(idx, show_words, show_types))

    def get_term(self) -> 'Term':
        vargen = iter(range(999))

        def pos_to_lambda(tree_path: Tuple[WordType, Path, Paths], idx: Optional[int]) -> Term:
            wordtype, pos_path, neg_paths = tree_path

            bodies = map(neg_to_lambda, neg_paths)

            if idx is None:
                term = Var(wordtype.depolarize(), pos_path[0].idx)
            else:
                term = Lex(wordtype.depolarize(), idx)

            for p in pos_path[:-1]:
                if isinstance(p, Cotensor):
                    continue
                if isinstance(p, Tensor):
                    body = next(bodies)
                    term = Application(term, body)
                if isinstance(p, Diamond):
                    term = DiamondElim(term)
                if isinstance(p, Box):
                    term = BoxElim(term)
            return term

        def neg_to_lambda(path: Path) -> Term:
            tgt, idx = cross(path[-1])
            fn = lambda x: x
            for p in path[:-1]:
                if isinstance(p, Cotensor):
                    p.idx = next(vargen)
                    fn = compose(fn, Abstraction.preemptive(p.idx))
                if isinstance(p, Diamond):
                    fn = compose(fn, DiamondIntro.preemptive(p.name))
                if isinstance(p, Box):
                    fn = compose(fn, BoxIntro.preemptive(p.name))
            return fn(pos_to_lambda(tgt, idx))

        def cross(neg: PolarizedType) -> Tuple[Tuple[WordType, Path, Paths], Optional[int]]:
            pos_idx = neg_to_pos[neg.index]
            for i, pps in enumerate(all_paths):
                for j, (wt, p, ps) in enumerate(pps):
                    p_atoms = list(filter(lambda atom: isinstance(atom, PolarizedType), p))
                    if pos_idx in (atom.index for atom in p_atoms):
                        return (wt, p, ps), i if j == 0 else None

        all_paths = [paths(premise.type) for premise in self.proof_frame.premises]
        neg_to_pos = {v: k for k, v in self.axiom_links}
        return neg_to_lambda([self.proof_frame.conclusion])

    @staticmethod
    def from_data(words: List[str], types: List[WordType], links: AxiomLinks, name: Optional[str] = None) -> 'ProofNet':
        if len(types) == 1:
            links = {(0, 1)}
        return ProofNet(ProofFrame.from_data(words, types), links, name)
