from __future__ import annotations

import pdb
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
    def __eq__(self, other): return isinstance(other, Judgement) and judgement_eq(self, other)


def judgement_repr(judgement: Judgement, show_types: bool = False, word_repr: Callable[[int], str] = _word_repr) -> str:
    antecedents = struct_repr(judgement.assumptions, item_repr=lambda _t: term_repr(_t, show_types, False, word_repr))
    conclusion = term_repr(judgement.term, True, False, word_repr)
    return f'{antecedents} ⊢ {conclusion}'


def judgement_eq(left: Judgement, right: Judgement) -> bool:
    return left.assumptions == right.assumptions and left.term == right.term


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
        substructure = proof.structure.substitute(substructure,
                                                  Sequence(*(s for s in inner if s != Sequence(var)**'x')) ** brackets)
        structure = Sequence(*substructure, *Sequence(var)**'x')
        return Proof(rule=Structural.Extract,
                     premises=(proof,),
                     conclusion=Judgement(structure, proof.term),
                     focus=var)

    @staticmethod
    def extractable(proof: Proof, var: Variable) -> Maybe[Unary[Variable | Constant]]:
        return next(iter(s for s in proof.structure if isinstance(s, Unary)
                         and Sequence(var)**'x' in s.content), None)

    Extract = partial(_extract)


def make_extractable(proof: Proof, var: Variable) -> tuple[Proof, Variable]:
    match proof.rule:
        case Logical.Variable:
            return (deep := variable(Box('x', var.type), var.index)).unbox('x'), deep.term
        case Logical.ArrowElimination:
            (function, argument) = proof.premises
            if var in (v for _, v in function.vars()):
                deep, var = make_extractable(function, var)
                return deep @ argument, var
            else:
                deep, var = make_extractable(argument, var)
                return function @ deep, var
        case Logical.ArrowIntroduction:
            (body,) = proof.premises
            (deep, var) = make_extractable(body, var)
            return deep.abstract(proof.focus), var
        case Logical.DiamondIntroduction:
            (body,) = proof.premises
            (struct,) = proof.structure
            (deep, var) = make_extractable(body, var)
            return deep.diamond(struct.brackets), var
        case Logical.DiamondElimination:
            if proof.focus.index == var.index:
                (original, becomes) = proof.premises
                becomes = (var := variable(Box('x', becomes.type), var.index)).unbox('x')
                return original.undiamond(proof.focus, becomes), var.term
            (original, becomes) = proof.premises
            (deep, var) = make_extractable(original, var)
            return deep.undiamond(proof.focus, becomes), var
        case Logical.BoxIntroduction:
            (body,) = proof.premises
            (struct,) = body.structure
            (deep, var) = make_extractable(body, var)
            return deep.box(struct.brackets), var
        case Logical.BoxElimination:
            (body,) = proof.premises
            (struct,) = proof.structure
            (deep, var) = make_extractable(body, var)
            return deep.unbox(struct.brackets), var
        case Structural.Extract:
            (body,) = proof.premises
            (deep, var) = make_extractable(body, var)
            return deep.extract(proof.focus), var
        case _:
            print(proof.rule)
            raise NotImplementedError


# fixpoint iteration of the nested version
def deep_extract(proof: Proof, var: Variable) -> tuple[Proof, Variable]:
    def go(_proof: Proof, recurse: bool = True) -> Proof:
        if (substructure := Structural.extractable(_proof, var)) is not None:
            return Structural.Extract(_proof, var, substructure)
        if recurse:
            match _proof.rule:
                case Logical.DiamondElimination:
                    original, becomes = _proof.premises
                    return go(original).undiamond(_proof.focus, becomes)
                case Logical.DiamondIntroduction:
                    (body,) = _proof.premises
                    (struct,) = _proof.structure
                    return go(go(body).diamond(struct.brackets), False)
                case Logical.BoxElimination:
                    (body,) = _proof.premises
                    (struct,) = _proof.structure
                    return go(go(body).unbox(struct.brackets), False)
                case Logical.BoxIntroduction:
                    (body,) = _proof.premises
                    (struct,) = body.structure
                    return go(body).box(struct.brackets)
                case Logical.ArrowElimination:
                    (function, argument) = _proof.premises
                    if var in (v for _, v in function.vars()):
                        return go(function) @ argument
                    else:
                        return function @ go(argument)
                case Logical.ArrowIntroduction:
                    (body,) = _proof.premises
                    return go(body).abstract(_proof.focus)
                case Structural.Extract:
                    (body,) = _proof.premises
                    return go(body).extract(_proof.focus)
                case _:
                    raise ValueError
        return _proof
    renamed = variable(Diamond('x', var.type), var.index)
    return go(proof).undiamond(where=var, becomes=renamed), renamed.term


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
    def __eq__(self, other) -> bool: return isinstance(other, Proof) and proof_eq(self, other)
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
                    return Structural.Extract(go(body), p.focus)
        return go(self)

    def translate_var(self, where: int, becomes: int) -> Proof:
        def go_focus(term_var: Variable) -> Variable:
            return Variable(term_var.type, becomes) if term_var.index == where else term_var

        def go(proof: Proof) -> Proof:
            match proof.rule:
                case Logical.Constant:
                    return proof
                case Logical.Variable:
                    return variable(proof.type, becomes if proof.term.index == where else proof.term.index)
                case Logical.ArrowElimination:
                    (fn, arg) = proof.premises
                    return go(fn)@go(arg)
                case Logical.ArrowIntroduction:
                    (body,) = proof.premises
                    return go(body).abstract(go_focus(proof.focus))
                case Logical.DiamondIntroduction:
                    (body,) = proof.premises
                    (struct,) = proof.structure
                    return go(body).diamond(struct.brackets)
                case Logical.BoxElimination:
                    (body,) = proof.premises
                    (struct,) = proof.structure
                    return go(body).unbox(struct.brackets)
                case Logical.BoxIntroduction:
                    (body,) = proof.premises
                    (struct,) = body.structure
                    return go(body).box(struct.brackets)
                case Logical.DiamondElimination:
                    _original, _becomes = proof.premises
                    return go(_original).undiamond(go_focus(proof.focus), go(_becomes))
                case Structural.Extract:
                    (body,) = proof.premises
                    return Structural.Extract(go(body), go_focus(proof.focus))
                case _:
                    raise ValueError
        return go(self)

    def de_bruijn(self) -> Proof: return de_bruijn(self)

    def eta_norm(self) -> Proof: return eta_norm(self)

    def beta_norm(self) -> Proof: return beta_norm(self)


def de_bruijn(proof: Proof) -> Proof:
    def distance_to(term: TERM, var: Variable) -> int:
        def go(_term: TERM) -> int:
            match _term:
                case Variable(_):
                    assert var == _term
                    return 0
                case ArrowElimination(fn, arg):
                    if var in fn.subterms():
                        return go(fn)
                    return go(arg)
                case ArrowIntroduction(_, body):
                    return 1 + go(body)
                case (DiamondIntroduction(_, body) | BoxIntroduction(_, body)
                      | BoxElimination(_, body) | DiamondElimination(_, body)):
                    return go(body)
                case _:
                    raise ValueError
        return go(term)
    distances = {subterm.abstraction.index: distance_to(subterm.body, subterm.abstraction)
                 for subterm in proof.term.subterms() if isinstance(subterm, ArrowIntroduction)}
    while True:
        if not distances:
            return proof
        (where, becomes) = next((k, v) for k, v in distances.items() if v not in distances.keys())
        proof = proof.translate_var(where, becomes)
        del distances[where]


def _fixpoint(proof_op: Callable[[Proof], Proof]) -> Callable[[Proof], Proof]:
    def inner(proof: Proof) -> Proof: return step if (step := proof_op(proof)) == proof else inner(step)
    return inner


@_fixpoint
def eta_norm(proof: Proof) -> Proof:
    match proof.rule:
        case Logical.Variable | Logical.Constant:
            return proof
        case Logical.ArrowIntroduction:
            (body,) = proof.premises
            if body.rule == Logical.ArrowElimination:
                (fn, arg) = body.premises
                if arg.term == proof.focus:
                        return fn
            return eta_norm(body).abstract(proof.focus)
        case Logical.ArrowElimination:
            (fn, arg) = proof.premises
            return eta_norm(fn) @ eta_norm(arg)
        case Logical.DiamondIntroduction:
            (body,) = proof.premises
            (struct,) = proof.structure
            if body.rule == Logical.DiamondElimination:
                raise NotImplementedError
            return eta_norm(body).diamond(struct.brackets)
        case Logical.DiamondElimination:
            (original, becomes) = proof.premises
            return eta_norm(original).undiamond(proof.focus, eta_norm(becomes))
        case Logical.BoxIntroduction:
            (body,) = proof.premises
            if body.rule ==  Logical.BoxElimination:
                (nested,) = body.premises
                (struct,) = body.structure
                if struct.brackets == proof.type.decoration:  # type: ignore
                    return eta_norm(nested)
            return eta_norm(body).box(proof.type.decoration)  # type: ignore
        case Logical.BoxElimination:
            (body,) = proof.premises
            (struct,) = proof.structure
            return eta_norm(body).unbox(struct.brackets)
        case Structural.Extract:
            (body,) = proof.premises
            return eta_norm(body).extract(proof.focus)


@_fixpoint
def beta_norm(proof: Proof) -> Proof:
    match proof.rule:
        case Logical.Variable | Logical.Constant:
            return proof
        case Logical.ArrowIntroduction:
            (body,) = proof.premises
            return beta_norm(body).abstract(proof.focus)
        case Logical.ArrowElimination:
            (fn, arg) = proof.premises
            if fn.rule == Logical.ArrowIntroduction:
                (fn_body,) = fn.premises
                return beta_norm(fn_body) @ beta_norm(arg)
            return beta_norm(fn) @ beta_norm(arg)
        case Logical.DiamondIntroduction:
            (body,) = proof.premises
            (struct,) = proof.structure
            return beta_norm(body).diamond(struct.brackets)
        case Logical.DiamondElimination:
            # todo.
            (original, becomes) = proof.premises
            return beta_norm(original).undiamond(proof.focus, beta_norm(becomes))
        case Logical.BoxIntroduction:
            (body,) = proof.premises
            return beta_norm(body).box(proof.type.decoration)  # type: ignore
        case Logical.BoxElimination:
            (body,) = proof.premises
            (struct,) = proof.structure
            if body.rule == Logical.BoxIntroduction:
                (nested,) = body.premises
                if struct.brackets == body.type.decoration:  # type: ignore
                    return beta_norm(nested)
            return beta_norm(body).unbox(struct.brackets)
        case Structural.Extract:
            (body,) = proof.premises
            return beta_norm(body).extract(proof.focus)
        case _:
            pdb.set_trace()


def constant(_type: Type, index: int) -> Proof: return Logical.Constant(Constant(_type, index))
def variable(_type: Type, index: int) -> Proof: return Logical.Variable(Variable(_type, index))


def proof_repr(proof: Proof, show_types: bool = False, word_repr: Callable[[int], str] = _word_repr) -> str:
    return judgement_repr(proof.conclusion, show_types=show_types, word_repr=word_repr)


def proof_eq(left: Proof, right: Proof) -> bool:
    return all((left.premises == right.premises, left.conclusion == right.conclusion, left.rule == right.rule))


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
