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

    def __str__(self):
        return self.print()

    def __repr__(self):
        return str(self)

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

    def print_frame(self, show_words: bool = True, show_types: bool = True) -> str:
        return self.proof_frame.print(show_words, show_types)

    def print_term(self, show_words: bool = False, show_types: bool = False, show_decorations: bool = True) -> str:
        return print_term(self.get_term(), show_decorations,
                          lambda idx: self.proof_frame.word_printer(idx, show_words, show_types))

    def get_term(self) -> 'Term':
        def pt(term):
            return print_term(term, True, lambda idx: self.proof_frame.word_printer(idx, True, True))

        def pos_to_lambda(path_pair: Tuple[Path, Paths], idx: Optional[int], varcount: int) -> Tuple[Term, int]:
            pos_path, neg_paths = path_pair
            bodies = []
            for neg_path in neg_paths:
                body, varcount = neg_to_lambda(neg_path, varcount)
                bodies.append(body)
            if idx is None:
                varcount -= 1
                hypothesis = pos_path[-1].depolarize()
                for p in pos_path[:-1]:
                    if isinstance(p, Diamond):
                        hypothesis = DiamondType(hypothesis, p.name)
                term = Var(hypothesis, varcount)
            else:
                term = Lex(self.proof_frame.premises[idx].type.depolarize(), idx)
            body_iter = iter(bodies)
            for p in pos_path[:-1]:
                if isinstance(p, Tensor):
                    body = next(body_iter)
                    term = Application(term, body)
                if isinstance(p, Diamond):
                    term = DiamondElim(term)
                if isinstance(p, Box):
                    term = BoxElim(term)
            return term, varcount

        def neg_to_lambda(path: Path, varcount: int) -> Tuple[Term, int]:
            tgt, idx = cross(path[-1])
            body, varcount_after = pos_to_lambda(tgt, idx, varcount + len([p for p in path if isinstance(p, Cotensor)]))
            for p in path[::-1][1:]:
                if isinstance(p, Cotensor):
                    body = Abstraction(body, varcount)
                    varcount += 1
                if isinstance(p, Diamond):
                    body = DiamondIntro(body, p.name)
                if isinstance(p, Box):
                    body = BoxIntro(body, p.name)
            return body, varcount_after

        def cross(neg: PolarizedType) -> Tuple[Tuple[Path, Paths], Optional[int]]:
            pos_idx = neg_to_pos[neg.index]
            for i, pps in enumerate(all_paths):
                for j, (p, ps) in enumerate(pps):
                    p_atoms = list(filter(lambda atom: isinstance(atom, PolarizedType), p))
                    if pos_idx in (atom.index for atom in p_atoms):
                        return (p, ps), i if j == 0 else None

        all_paths = [paths(premise.type) for i, premise in enumerate(self.proof_frame.premises)]
        neg_to_pos = {v: k for k, v in self.axiom_links}
        return neg_to_lambda([self.proof_frame.conclusion], 0)[0]

    @staticmethod
    def from_data(words: List[str], types: List[WordType], links: AxiomLinks, name: Optional[str] = None) -> 'ProofNet':
        if len(types) == 1:
            links = {(0, 1)}
        return ProofNet(ProofFrame.from_data(words, types), links, name)
