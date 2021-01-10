from .terms import Term, Application, Abstraction, Atom, subscript, smallcaps, print_term
from .milltypes import (WordType, WordTypes, PolarizedType, EmptyType, get_polarities_and_indices, Path, Paths, paths,
                        polarize_and_index_many, Decoration)
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

    def get_types(self) -> WordTypes:
        return [p.type for p in self.premises]

    def __str__(self):
        return self.print()

    def __repr__(self):
        return str(self)

    def word_printer(self, idx: int, show_word: bool = True, show_type: bool = True) -> str:
        ret = self.premises[idx].word if show_word else f'w{subscript(idx)}'
        return ret + (f':{smallcaps(self.premises[idx].type.depolarize())}' if show_type else '')

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
        return ProofFrame(premises, PolarizedType(wordtype=conclusion.type, polarity=False, index=conclusion_id))


@dataclass
class ProofNet:
    """
        Class representing a judgement together with its proof.
    """
    proof_frame: ProofFrame
    axiom_links: AxiomLinks
    name: Optional[str]

    def print_frame(self, show_words: bool = True, show_types: bool = True) -> str:
        return self.proof_frame.print(show_words, show_types)

    def print_term(self, show_words: bool = False, show_types: bool = False, show_decorations: bool = True) -> str:
        return print_term(self.get_term(), show_decorations,
                          lambda idx: self.proof_frame.word_printer(idx, show_words, show_types))

    def get_term(self) -> 'Term':
        def pos_to_lambda(path_pair: Tuple[Path, Paths], idx: Optional[int], varcount: int) -> Tuple[Term, int]:
            pos_path, neg_paths = path_pair
            bodies = []
            for neg_path in neg_paths:
                body, varcount = neg_to_lambda(neg_path, varcount)
                bodies.append(body)
            if idx is None:
                varcount -= 1
                app = (varcount, False)
            else:
                app = (idx, True)
            return Application.from_arglist(Atom.make(*app), bodies, pos_path[::-1]), varcount

        def neg_to_lambda(path: Path, varcount: int) -> Tuple[Term, int]:
            def fn(t: Tuple[Term, int], s: Optional[Decoration]) -> Tuple[Term, int]:
                return Abstraction(Atom.make(t[1], False), t[0], s), t[1] + 1
            tgt, idx = cross(path[-1])
            body, varcount_after = pos_to_lambda(tgt, idx, varcount + len(path) - 1)
            return reduce(fn, path[::-1][1:], (body, varcount))[0], varcount_after

        def cross(neg: int) -> Tuple[Tuple[Path, Paths], Optional[int]]:
            pos = neg_to_pos[neg]
            for i, pps in enumerate(all_paths):
                for j, (p, ps) in enumerate(pps):
                    if pos in p:
                        return (p, ps), i if j == 0 else None

        all_paths = [paths(premise.type) for i, premise in enumerate(self.proof_frame.premises)]
        neg_to_pos = {v: k for k, v in self.axiom_links}
        return neg_to_lambda([self.proof_frame.conclusion.index], 0)[0]

    @staticmethod
    def from_data(words: List[str], types: WordTypes, links: AxiomLinks, name: Optional[str] = None) -> 'ProofNet':
        if len(types) == 1:
            links = {(0, 1)}
        return ProofNet(ProofFrame.from_data(words, types), links, name)
