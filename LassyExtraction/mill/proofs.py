from __future__ import annotations

from typing import Callable
from typing import Optional as Maybe
from .structures import Sequence, Unary, struct_repr
from .types import Type, Atom, Diamond, Box
from .terms import (TERM, Variable, Constant, ArrowElimination, ArrowIntroduction,
                    DiamondIntroduction, BoxElimination, BoxIntroduction, DiamondElimination,
                    substitute, term_repr, _word_repr)
from enum import Enum
from functools import partial


class ProofError(Exception):
    pass


########################################################################################################################
# Judgements
########################################################################################################################


class Judgement:
    __match_args__ = ('assumptions', 'term')

    assumptions:    Sequence[Variable | Constant]
    term:           TERM

    def __init__(self, assumptions: Sequence, term: TERM) -> None:
        self.assumptions = assumptions
        self.term = term

    def __repr__(self) -> str: return judgement_repr(self)


def judgement_repr(judgement: Judgement, word_repr: Callable[[int], str] = _word_repr) -> str:
    return f'{struct_repr(judgement.assumptions, item_repr=lambda _t: term_repr(_t, True, word_repr=word_repr))} |- ' \
           f'{term_repr(judgement.term, False, word_repr=word_repr)}: {judgement.term.type}'


########################################################################################################################
# Rules
########################################################################################################################

class Rule(Enum):
    def __repr__(self) -> str: return self.name
    def __str__(self) -> str: return repr(self)
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
    def _init_arrow_intro(abstraction: Variable, body: Proof) -> Proof:
        if abstraction not in body.structure:
            raise ProofError(f'{abstraction} does not occur free in {body.structure}')
        return Proof(rule=Logical.ArrowIntroduction,
                     premises=(body,),
                     conclusion=Judgement(Sequence(*[s for s in body.structure if s != abstraction]),
                                          ArrowIntroduction(abstraction, body.term)),
                     focus=abstraction)

    @staticmethod
    def _init_dia_intro(body: Proof, diamond: str) -> Proof:
        succedent = DiamondIntroduction(diamond, body.term)
        return Proof(rule=Logical.DiamondIntroduction,
                     premises=(body,),
                     conclusion=Judgement(Sequence(body.structure**succedent.decoration), succedent))

    @staticmethod
    def _init_dia_elim(original: Proof, where: Variable, becomes: Proof) -> Proof:
        replacement = DiamondElimination(None, becomes.term)
        struct = Sequence(where)**replacement.decoration
        if struct not in original.structure:
            raise ProofError(f'{struct} does not immediately occur in {original.structure}')
        return Proof(rule=Logical.DiamondElimination,
                     premises=(original, becomes),
                     conclusion=Judgement(Sequence(original.structure.substitute(struct, becomes.structure)),
                                          substitute(original.term, where, replacement)),
                     focus=where)

    @staticmethod
    def _init_box_intro(body: Proof, box: str) -> Proof:
        match body.structure:
            case Sequence((Unary(unary, b),)):
                if b == box:
                    return Proof(rule=Logical.BoxIntroduction,
                                 premises=(body,),
                                 conclusion=Judgement(Sequence(unary), BoxIntroduction(box, body.term)))
                raise ProofError(f'{body.structure} is not {box}-bracketed')
            case _: raise ProofError(f'{body.structure} is not a singleton')

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


class Structural(Rule):
    @staticmethod
    def _extract(proof: Proof,
                 var: Variable,
                 substructure: Maybe[Unary[Variable | Constant]] = None) -> Proof:
        # <Γ, x, Δ> -> <Γ, Δ>, <x>
        if substructure is None:
            substructure = Structural.extractable(proof, var)
        inner, brackets = substructure.content, substructure.brackets
        substructure = proof.structure.substitute(substructure, Sequence(*(s for s in inner if s != var)) ** brackets)
        structure = Sequence(*substructure, *Sequence(var)**brackets)
        return Proof(rule=Structural.Extract,
                     premises=(proof,),
                     conclusion=Judgement(structure, proof.term),
                     focus=var)

    @staticmethod
    def extractable(proof: Proof, var: Variable) -> Maybe[Unary[Variable | Constant]]:
        return next(iter(s for s in proof.structure if isinstance(s, Unary) and var in s.content), None)

    Extract = partial(_extract)


# shortcut for extract followed by diamond elimination
def extract_renaming(proof: Proof,
                     var: Variable,
                     context: Maybe[Unary[Variable | Constant]] = None) -> tuple[Proof, Variable]:
    proof = Structural.Extract(proof, var, context)
    extracted: Unary = proof.structure[-1]
    renamed = variable(Diamond(extracted.brackets, var.type), var.index)
    return proof.undiamond(where=proof.focus, becomes=renamed), renamed.term


# fixpoint iteration of the nested version
def deep_extract_renaming(proof: Proof, var: Variable, recurse: bool = True) -> tuple[Proof, Variable]:
    def go(_proof: Proof, _var: Variable) -> tuple[Proof, Variable]:
        return deep_extract_renaming(_proof, _var, False)

    if (substructure := Structural.extractable(proof, var)) is not None:
        return extract_renaming(proof, var, substructure)
    if not recurse:
        return proof, var
    match proof.rule:
        case Logical.DiamondElimination:
            original, becomes = proof.premises
            if var in (v for _, v in original.vars()):
                deep, renamed = deep_extract_renaming(original, var)
                return go(Logical.DiamondElimination(deep, proof.focus, becomes), renamed)
            raise ProofError
        case Logical.DiamondIntroduction:
            (body,) = proof.premises
            (struct,) = proof.structure
            deep, renamed = deep_extract_renaming(body, var)
            return go(Logical.DiamondIntroduction(deep, struct.brackets), renamed)
        case Logical.BoxElimination:
            (body,) = proof.premises
            (struct,) = proof.structure
            deep, renamed = deep_extract_renaming(body, var)
            return go(Logical.BoxElimination(deep, struct.brackets), renamed)
        case Logical.BoxIntroduction:
            (body,) = proof.premises
            (struct,) = body.structure
            deep, renamed = deep_extract_renaming(body, var)
            return go(Logical.BoxIntroduction(deep, struct.brackets), renamed)
        case Logical.ArrowElimination:
            (fn, arg) = proof.premises
            if var in (v for _, v in fn.vars()):
                deep, renamed = deep_extract_renaming(fn, var)
                return go(deep @ arg, renamed)
            elif var in (v for _, v in arg.vars()):
                deep, renamed = deep_extract_renaming(arg, var)
                return go(fn @ deep, renamed)
            else:
                raise AssertionError
        case Logical.ArrowIntroduction:
            (body,) = proof.premises
            deep, renamed = deep_extract_renaming(body, var)
            return go(deep.abstract(proof.focus), renamed)
        case Structural.Extract:
            (body,) = proof.premises
            deep, renamed = deep_extract_renaming(body, var)
            return go(Structural.Extract(deep, proof.focus), renamed)
        case _:
            raise NotImplementedError


########################################################################################################################
# Proofs
########################################################################################################################

class Proof:
    __match_args__ = ('premises', 'conclusion', 'rule')

    premises:   tuple[Proof, ...]
    conclusion: Judgement
    rule:       Rule
    focus:      Maybe[Variable]

    def __init__(self, premises: tuple[Proof, ...], conclusion: Judgement, rule: Rule, focus: Maybe[Variable] = None):
        self.premises = premises
        self.conclusion = conclusion
        self.rule = rule
        self.focus = focus

    @property
    def structure(self) -> Sequence: return self.conclusion.assumptions
    @property
    def term(self) -> TERM: return self.conclusion.term
    @property
    def type(self) -> Type: return self.term.type
    def __repr__(self) -> str: return proof_repr(self)
    def __str__(self) -> str: return repr(self)
    def apply(self, other: Proof) -> Proof: return Logical.ArrowElimination(self, other)
    def diamond(self, diamond: str) -> Proof: return Logical.DiamondIntroduction(self, diamond)
    def box(self, box: str) -> Proof: return Logical.BoxIntroduction(self, box)
    def unbox(self, box: str | None = None) -> Proof: return Logical.BoxElimination(self, box)

    def undiamond(self, where: Variable, becomes: Proof) -> Proof:
        return Logical.DiamondElimination(self, where, becomes)

    def __matmul__(self, other) -> Proof: return self.apply(other)

    def vars(self) -> list[tuple[tuple[str, ...], Variable]]:
        return [(ctx, v) for ctx, v in self.structure.units() if isinstance(v, Variable)]

    def constants(self) -> list[tuple[tuple[str, ...], Constant]]:
        return [(ctx, c) for ctx, c in self.structure.units() if isinstance(c, Constant)]

    def abstract(self, var: Variable) -> Proof: return Logical.ArrowIntroduction(var, self)
    def extract(self, var: Variable) -> Proof: return Structural.Extract(self, var)

    def translate_lex(self, trans: dict[int, int]) -> Proof:
        def go(p: Proof) -> Proof:
            match p.rule:
                case Logical.Constant:
                    return constant(p.type, trans.pop(c)) if (c := p.term.index) in trans else p
                case Logical.Variable:
                    return p
                case Logical.ArrowElimination:
                    (fn, arg) = p.premises
                    return go(fn)@go(arg)
                case Logical.ArrowIntroduction:
                    (body,) = p.premises
                    return go(body).abstract(p.focus)
                case Logical.DiamondIntroduction:
                    (body,) = p.premises
                    (struct,) = p.structure
                    return go(body).diamond(struct.brackets)
                case Logical.BoxElimination:
                    (body,) = p.premises
                    (struct,) = p.structure
                    return go(body).unbox(struct.brackets)
                case Logical.BoxIntroduction:
                    (body,) = p.premises
                    (struct,) = body.structure
                    return go(body).box(struct.brackets)
                case Logical.DiamondElimination:
                    original, becomes = p.premises
                    return go(original).undiamond(p.focus, go(becomes))
                case Structural.Extract:
                    (body,) = p.premises
                    return Structural.Extract(body, p.focus)
        return go(self)

    def eta_norm(self) -> Proof:
        ...

    def beta_norm(self) -> Proof:
        ...


def constant(_type: Type, index: int) -> Proof: return Logical.Constant(Constant(_type, index))
def variable(_type: Type, index: int) -> Proof: return Logical.Variable(Variable(_type, index))


def proof_repr(proof: Proof, word_repr: Callable[[int], str] = _word_repr) -> str:
    return judgement_repr(proof.conclusion, word_repr=word_repr)


########################################################################################################################
# Examples / Tests
########################################################################################################################
A = Atom('A')

# Adjunction theorems

# ◇ □ A |- A
p0 = (x := variable(Box('a', A), 0)).unbox()
y = variable(Diamond('a', Box('a', A)), 0)
p0 = p0.undiamond(where=x.term, becomes=y).abstract(y.term)

# A |- □ ◇ A
p1 = (x := variable(A, 0)).diamond('a').box('a').abstract(x.term)

# □ A |- □ ◇ □ A
p2 = (x := variable(Box('a', A), 0)).diamond('a').box('a').abstract(x.term)

# ◇ A |- ◇ □ ◇ A
p3 = (x := variable(A, 0)).diamond('a').box('a').diamond('a')
y = variable(Diamond('a', A), 0)
p3 = p3.undiamond(where=x.term, becomes=y).abstract(y.term)

# ◇ □ ◇ A |- ◇ A
p4 = (x := variable(Box('a', Diamond('a', A)), 0)).unbox()
y = variable(Diamond('a', Box('a', Diamond('a', A))), 0)
p4 = p4.undiamond(where=x.term, becomes=y).abstract(y.term)

# □ ◇ □ A |- □ A
p5 = (x := variable(Box('a', A), 0)).unbox()
y = variable(Box('a', Diamond('a', Box('a', A))), 0)
p5 = (p5.undiamond(where=x.term, becomes=y.unbox()).box('a')).abstract(y.term)
