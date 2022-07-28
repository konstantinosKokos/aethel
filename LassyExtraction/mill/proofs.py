from __future__ import annotations

import pdb
from typing import Callable, Iterator
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

    def standardize_vars(self) -> Proof: return standardize_vars(self)

    def eta_norm(self) -> Proof: return eta_norm(self)

    def beta_norm(self) -> Proof: return beta_norm(self)

    def is_linear(self) -> bool: return is_linear(self)

    def subproofs(self) -> Iterator[Proof]: return subproofs(self)


def subproofs(proof: Proof) -> Iterator[Proof]:
    match proof.rule:
        case Logical.Constant | Logical.Variable:
            yield proof
        case _:
            yield proof
            for premise in proof.premises:
                yield from subproofs(premise)


def is_linear(proof: Proof) -> bool:
    def go(_proof: Proof, free: set[int]) -> tuple[set[int], bool]:
        match _proof.rule:
            case Logical.Constant:
                return free, True
            case Logical.Variable:
                return free - {_proof.term.index}, _proof.term.index in free
            case Logical.ArrowElimination:
                (fn, arg) = _proof.premises
                (free_fn, linear_fn) = go(fn, free)
                (free_arg, linear_arg) = go(arg, free_fn)
                return free_arg, linear_fn and linear_arg
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                return go(body, free | {_proof.focus.index})
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                return go(body, free)
            case Logical.BoxElimination:
                (body,) = _proof.premises
                return go(body, free)
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                return go(body, free)
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                (free_original, linear_original) = go(original, free)
                (free_becomes, linear_becomes) = go(becomes, free_original | {_proof.focus.index})
                return free_becomes, linear_original and linear_becomes
            case Structural.Extract:
                (body,) = _proof.premises
                return go(body, free)
            case _:
                raise ValueError
    free, linear = go(proof, set())
    return free == set() and linear


def de_bruijn(proof: Proof) -> Proof:
    detours: dict[int, list[Proof]] = {}
    cuts: dict[int, list[Proof]] = {}

    def go(_proof: Proof,
           context: dict[int, int]) -> tuple[Proof, dict[int, int]]:
        nonlocal detours
        nonlocal cuts

        match _proof.rule:
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                focus = _proof.focus
                detours[focus.index] = detours.get(focus.index, []) + [becomes]
                original, resolved = go(original, context)
                becomes = cuts[focus.index].pop()
                return original.undiamond(Variable(focus.type, resolved[focus.index]), becomes), resolved
            case Logical.ArrowElimination:
                (fn, arg) = _proof.premises
                fn, res_fn = go(fn, context)
                arg, res_arg = go(arg, context)
                return fn @ arg, res_fn | res_arg
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                focus = _proof.focus
                context = {k: v + 1 for k, v in context.items()}
                context[focus.index] = 0
                body, resolved = go(body, context)
                return (body.abstract(Variable(focus.type, resolved[focus.index])),
                        {k: v for k, v in resolved.items() if k != focus.index})
            case Logical.Variable:
                index = _proof.term.index
                if index in detours.keys() and len(detours[index]):
                    cut, resolved = go(detours[index].pop(), context)
                    cuts[index] = cuts.get(index, []) + [cut]
                else:
                    resolved = {index: context[index]}
                return variable(_proof.type, context[index]), resolved
            case Logical.Constant:
                return constant(_proof.type, _proof.term.index), {}
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                body, resolved = go(body, context)
                return body.diamond(struct.brackets), resolved
            case Logical.BoxElimination:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                body, resolved = go(body, context)
                return body.unbox(struct.brackets), resolved
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                (struct,) = body.structure
                body, resolved = go(body, context)
                return body.box(struct.brackets), resolved
            case Structural.Extract:
                (body,) = _proof.premises
                focus = _proof.focus
                body, resolved = go(body, context)
                return body.extract(Variable(focus.type, resolved[focus.index])), resolved
            case _:
                raise ValueError
    return go(proof, {})[0]


def standardize_vars(proof: Proof) -> Proof:
    detours: dict[int, list[Proof]] = {}
    cuts: dict[int, list[Proof]] = {}

    def go(_proof: Proof,
           counter: int,
           trans: dict[int, int]) -> tuple[Proof, int, dict[int, int]]:

        match _proof.rule:
            case Logical.ArrowIntroduction:
                (body,) = _proof.premises
                focus = _proof.focus
                trans[focus.index] = (counter := counter + 1)
                body, counter, trans = go(body, counter, trans)
                return body.abstract(Variable(focus.type, trans[focus.index])), counter, trans
            case Logical.DiamondElimination:
                (original, becomes) = _proof.premises
                focus = _proof.focus
                detours[focus.index] = detours.get(focus.index, []) + [becomes]
                original, counter, trans = go(original, counter, trans)
                becomes = cuts[focus.index].pop()
                return (original.undiamond(Variable(focus.type, trans[focus.index]), becomes),
                        counter,
                        trans)
            case Logical.Variable:
                index = _proof.term.index
                if index in detours:
                    if index in detours.keys() and len(detours[index]):
                        cut, counter, trans = go(detours[index].pop(), counter, trans)
                        cuts[index] = cuts.get(index, []) + [cut]
                return variable(_proof.type, trans[index]), counter, trans
            case Logical.Constant:
                return _proof, counter, trans
            case Logical.ArrowElimination:
                (fn, arg) = _proof.premises
                fn, counter, trans = go(fn, counter, trans)
                arg, counter, trans = go(arg, counter, trans)
                return fn @ arg, counter, trans
            case Logical.DiamondIntroduction:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                body, counter, trans = go(body, counter, trans)
                return body.diamond(struct.brackets), counter, trans
            case Logical.BoxElimination:
                (body,) = _proof.premises
                (struct,) = _proof.structure
                body, counter, trans = go(body, counter, trans)
                return body.unbox(struct.brackets), counter, trans
            case Logical.BoxIntroduction:
                (body,) = _proof.premises
                (struct,) = body.structure
                body, counter, trans = go(body, counter, trans)
                return body.box(struct.brackets), counter, trans
            case Structural.Extract:
                (body,) = _proof.premises
                focus = _proof.focus
                body, counter, trans = go(body, counter, trans)
                return body.extract(Variable(focus.type, trans[focus.index])), counter, trans
            case _:
                raise ValueError

    _proof, _, _ = go(proof, -1, {})
    return _proof


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
            if body.rule == Logical.DiamondElimination and struct.brackets == body.premises[1].type.decoration:
                raise NotImplementedError
            return eta_norm(body).diamond(struct.brackets)
        case Logical.DiamondElimination:
            (original, becomes) = proof.premises
            return eta_norm(original).undiamond(proof.focus, eta_norm(becomes))
        case Logical.BoxIntroduction:
            (body,) = proof.premises
            if body.rule == Logical.BoxElimination:
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
