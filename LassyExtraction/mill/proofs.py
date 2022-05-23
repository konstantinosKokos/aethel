from __future__ import annotations

from .structures import Sequence, Unary
from .types import Type, Atom, Diamond, Box
from .terms import (Term, Variable, Constant, ArrowElimination, ArrowIntroduction,
                    DiamondIntroduction, BoxElimination, BoxIntroduction, DiamondElimination,
                    substitute, term_repr)
from enum import Enum
from functools import partial


class ProofError(Exception):
    pass


class Judgement:
    __match_args__ = ('structure', 'term')

    structure:  Sequence
    term:       Term

    def __init__(self, structure: Sequence, term: Term) -> None:
        self.structure = structure
        self.term = term

    def __repr__(self) -> str: return judgement_repr(self)


def judgement_repr(judgement: Judgement) -> str:
    return f'{judgement.structure} |- {term_repr(judgement.term, False)}: {judgement.term.type}'


class Rule(Enum):
    def __repr__(self) -> str: return self.name
    def __call__(self, *args, **kwargs) -> Proof: return self.value(*args, **kwargs)


class Logical(Rule):
    @staticmethod
    def _init_var(var: Variable) -> Proof:
        return Proof(rule=Logical.Variable, premises=(), conclusion=Judgement(Sequence(var), var))

    @staticmethod
    def _init_con(con: Constant) -> Proof:
        return Proof(rule=Logical.Constant, premises=(), conclusion=Judgement(Sequence(con), con))

    @staticmethod
    def _init_arrow_elim(function: Proof, argument: Proof) -> Proof:
        return Proof(rule=Logical.ArrowElimination,
                     premises=(function, argument),
                     conclusion=Judgement(Sequence(*function.structure, *argument.structure),
                                          ArrowElimination(function.term, argument.term)))

    @staticmethod
    def _init_arrow_intro(abstraction: Term, body: Proof) -> Proof:
        if abstraction not in body.structure:
            raise ProofError(f'{abstraction} does not occur free in {body.structure}')
        return Proof(rule=Logical.ArrowIntroduction,
                     premises=(body,),
                     conclusion=Judgement(Sequence(*[s for s in body.structure if s != abstraction]),
                                          ArrowIntroduction(abstraction, body.term)))

    @staticmethod
    def _init_dia_intro(body: Proof, diamond: str) -> Proof:
        succedent = DiamondIntroduction(diamond, body.term)
        return Proof(rule=Logical.DiamondIntroduction,
                     premises=(body,),
                     conclusion=Judgement(Sequence(body.structure**succedent.decoration), succedent))

    @staticmethod
    def _init_dia_elim(original: Proof, where: Proof, becomes: Proof) -> Proof:
        replacement = DiamondElimination(None, becomes.term)
        match where.structure:
            case Sequence((Variable(_, _),)): pass
            case _: raise ProofError(f'{where.structure} is not a variable')
        if where.structure**replacement.decoration not in original.structure:
            raise ProofError(f'{where.structure} does not immediately occur in {original.structure}')
        return Proof(rule=Logical.DiamondElimination,
                     premises=(original, where, becomes),
                     conclusion=Judgement(
                         Sequence(original.structure.substitute(where.structure**replacement.decoration,
                                                                becomes.structure)),
                         substitute(original.term, where.term, replacement)))

    @staticmethod
    def _init_box_intro(body: Proof, box: str) -> Proof:
        structure = body.structure
        if len(structure) != 1:
            raise ProofError(f'{body.structure} is not a singleton structure')
        unary = structure[0]
        if not isinstance(unary, Unary) or unary.brackets != box:
            raise ProofError(f'{unary} is not a {box}-bracketed structure')
        return Proof(rule=Logical.BoxIntroduction,
                     premises=(body,),
                     conclusion=Judgement(Sequence(unary.content), BoxIntroduction(box, body.term)))

    @staticmethod
    def _init_box_elim(body: Proof, box: str | None) -> Proof:
        succedent = BoxElimination(box, body.term)
        return Proof(rule=Logical.BoxElimination,
                     premises=(body,),
                     conclusion=Judgement(Sequence(body.structure**succedent.decoration), succedent))

    Variable = partial(_init_var)
    Constant = partial(_init_con)
    ArrowElimination = partial(_init_arrow_elim)
    ArrowIntroduction = partial(_init_arrow_intro)
    DiamondIntroduction = partial(_init_dia_intro)
    BoxElimination = partial(_init_box_elim)
    BoxIntroduction = partial(_init_box_intro)
    DiamondElimination = partial(_init_dia_elim)


class Proof:
    __match_args__ = ('premises', 'conclusion', 'rule')

    premises:   tuple[Proof, ...]
    conclusion: Judgement
    rule:       Rule

    def __init__(self, premises: tuple[Proof, ...], conclusion: Judgement, rule: Rule) -> None:
        self.premises = premises
        self.conclusion = conclusion
        self.rule = rule

    @property
    def structure(self) -> Sequence: return self.conclusion.structure
    @property
    def term(self) -> Term: return self.conclusion.term
    @property
    def type(self) -> Type: return self.term.type
    def __repr__(self) -> str: return judgement_repr(self.conclusion)
    def apply(self, other: Proof) -> Proof: return Logical.ArrowElimination(self, other)
    def diamond(self, diamond: str) -> Proof: return Logical.DiamondIntroduction(self, diamond)
    def box(self, box: str) -> Proof: return Logical.BoxIntroduction(self, box)
    def unbox(self, box: str | None = None) -> Proof: return Logical.BoxElimination(self, box)
    def undiamond(self, where: Proof, becomes: Proof) -> Proof: return Logical.DiamondElimination(self, where, becomes)
    def __matmul__(self, other) -> Proof: return self.apply(other)

    def vars(self) -> list[tuple[tuple[str, ...], Variable]]:
        return [(ctx, v) for ctx, v in self.structure.units() if isinstance(v, Variable)]

    def constants(self) -> list[tuple[tuple[str, ...], Constant]]:
        return [(ctx, c) for ctx, c in self.structure.units() if isinstance(c, Constant)]

    def abstract(self, var: Term) -> Proof: return Logical.ArrowIntroduction(var, self)

    def translate_lex(self, trans: dict[int, int]) -> Proof:
        def go(p: Proof) -> Proof:
            match p.rule:
                case Logical.Constant: return constant(p.type, trans.pop(c)) if (c := p.term.index) in trans else p
                case Logical.Variable: return p
                case Logical.ArrowElimination: return go(p.premises[0])@go(p.premises[1])
                case Logical.ArrowIntroduction: return go(p.premises[0]).abstract(p.term.abstraction)
                case Logical.DiamondIntroduction: return go(p.premises[0]).diamond(p.term.decoration)
                case Logical.BoxElimination: return go(p.premises[0]).unbox(p.term.decoration)
                case Logical.BoxIntroduction: return go(p.premises[0]).box(p.term.decoration)
                case Logical.DiamondElimination:
                    original, where, becomes = p.premises
                    return go(original).undiamond(where, go(becomes))
        return go(self)

    def eta_norm(self) -> Proof:
        ...

    def beta_norm(self) -> Proof:
        ...


def constant(type: Type, index: int) -> Proof: return Logical.Constant(Constant(type, index))
def variable(type: Type, index: int) -> Proof: return Logical.Variable(Variable(type, index))


########################################################################################################################
# # Examples / Tests
########################################################################################################################
A = Atom('A')

# Adjunction theorems

# ◇ □ A |- A
p0 = (x := variable(Box('a', A), 0)).unbox()
y = variable(Diamond('a', Box('a', A)), 0)
p0 = p0.undiamond(where=x, becomes=y).abstract(y.term)

# A |- □ ◇ A
p1 = (x := variable(A, 0)).diamond('a').box('a').abstract(x.term)

# □ A |- □ ◇ □ A
p2 = (x := variable(Box('a', A), 0)).diamond('a').box('a').abstract(x.term)

# ◇ A |- ◇ □ ◇ A
p3 = (x := variable(A, 0)).diamond('a').box('a').diamond('a')
y = variable(Diamond('a', A), 0)
p3 = p3.undiamond(where=x, becomes=y).abstract(y.term)

# ◇ □ ◇ A |- ◇ A
p4 = (x := variable(Box('a', Diamond('a', A)), 0)).unbox()
y = variable(Diamond('a', Box('a', Diamond('a', A))), 0)
p4 = p4.undiamond(where=x, becomes=y).abstract(y.term)

# □ ◇ □ A |- □ A
p5 = (x := variable(Box('a', A), 0)).unbox()
y = variable(Box('a', Diamond('a', Box('a', A))), 0)
p5 = p5.undiamond(where=x, becomes=y).box('a').abstract(y.term)
