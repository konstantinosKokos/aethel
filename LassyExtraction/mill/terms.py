from __future__ import annotations
from .types import Type, TypeInference, Functor, Diamond, Box
from abc import ABC
from typing import Callable, Iterable


class TermError(Exception):
    pass


class Term(ABC):
    type:   Type
    def __repr__(self) -> str: return term_repr(self)
    def __matmul__(self, other) -> ArrowElimination: return ArrowElimination(self, other)
    def subterms(self) -> list[Term]: return subterms(self)
    def vars(self) -> Iterable[Variable]: return term_vars(self)
    def constants(self) -> Iterable[Constant]: return term_constants(self)


class Variable(Term):
    __match_args__ = ('type', 'index')

    def __init__(self, _type: Type, index: int):
        self.index = index
        self.type = _type


class Constant(Term):
    __match_args__ = ('type', 'index')

    def __init__(self, type: Type, index: int):
        self.index = index
        self.type = type


class ArrowElimination(Term):
    __match_args__ = ('function', 'argument')

    def __init__(self, function: Term, argument: Term):
        self.function = function
        self.argument = argument
        self.type = TypeInference.arrow_elim(function.type, argument.type)


class ArrowIntroduction(Term):
    __match_args__ = ('abstraction', 'body')

    def __init__(self, abstraction: Term, body: Term):
        if not isinstance(abstraction, Variable):
            raise TypeInference.TypeCheckError(f'{abstraction} of type {type(abstraction)} is not a variable')
        self.abstraction = abstraction
        self.body = body
        self.type = Functor(abstraction.type, body.type)


class DiamondIntroduction(Term):
    __match_args__ = ('decoration', 'body')

    def __init__(self, diamond: str, body: Term):
        self.decoration = diamond
        self.body = body
        self.type = Diamond(diamond, body.type)


class BoxElimination(Term):
    __match_args__ = ('decoration', 'body')

    def __init__(self, box: str | None, body: Term):
        self.type, self.decoration = TypeInference.box_elim(body.type, box)
        self.body = body


class BoxIntroduction(Term):
    __match_args__ = ('decoration', 'body')

    def __init__(self, box: str, body: Term):
        self.decoration = box
        self.body = body
        self.type = Box(box, body.type)


class DiamondElimination(Term):
    __match_args__ = ('decoration', 'body')

    def __init__(self, diamond: str | None, body: Term):
        self.type, self.decoration = TypeInference.dia_elim(body.type, diamond)
        self.body = body


########################################################################################################################
# Meta-Rules and Term Rewrites
########################################################################################################################
def substitute(term: Term, replace: Term, with_: Term) -> Term:
    TypeInference.assert_equal(replace.type, with_.type)
    if (c := term.subterms().count(replace)) != 1:
        raise TypeInference.TypeCheckError(f"Expected exactly one occurrence of {replace} in {term}, but found {c}")

    def go(_term: Term) -> Term:
        if _term == replace:
            return with_
        match _term:
            case Variable(_, _) | Constant(_, _): return _term
            case ArrowElimination(fn, arg): return ArrowElimination(go(fn), go(arg))
            case ArrowIntroduction(abst, body): return ArrowIntroduction(go(abst), go(body))
            case DiamondIntroduction(dia, body): return DiamondIntroduction(dia, go(body))
            case BoxElimination(box, body): return BoxElimination(box, go(body))
            case BoxIntroduction(box, body): return BoxIntroduction(box, go(body))
            case DiamondElimination(dia, body): return DiamondElimination(dia, go(body))
    return go(term)


def subterms(term: Term) -> list[Term]:
    match term:
        case Variable(_, _) | Constant(_, _): return [term]
        case ArrowElimination(fn, arg): return [term, *fn.subterms(), *arg.subterms()]
        case ArrowIntroduction(_, body): return [term, *body.subterms()]
        case DiamondIntroduction(_, body): return [term, *body.subterms()]
        case BoxElimination(_, body): return [term, *body.subterms()]
        case BoxIntroduction(_, body): return [term, *body.subterms()]
        case DiamondElimination(_, body): return [term, *body.subterms()]


def term_repr(term: Term,
              show_types: bool = True,
              word_printer: Callable[[int], str] = lambda i: f'c{i}') -> str:
    def needs_par(_term: Term) -> bool:
        match _term:
            case Variable(_, _) | Constant(_, _): return False
            case _: return True

    def f(_term: Term) -> str: return term_repr(_term, show_types, word_printer)
    def v(_term: Term) -> str: return term_repr(_term, False)

    match term:
        case Variable(_type, index): return f'x{index}' + f': {_type}' * show_types
        case Constant(_type, index): return f'{word_printer(index)}' + f': {_type}' * show_types
        case ArrowElimination(fn, arg): return f'{f(fn)} ({f(arg)})' if needs_par(arg) else f'{f(fn)} {f(arg)}'
        case ArrowIntroduction(var, body): return f'λ{v(var)}.({f(body)})'
        case DiamondIntroduction(decoration, body): return f'▵{decoration}({f(body)})'
        case BoxElimination(decoration, body): return f'▾{decoration}({f(body)})'
        case BoxIntroduction(decoration, body): return f'▴{decoration}({f(body)})'
        case DiamondElimination(decoration, body): return f'▿{decoration}({f(body)})'
        case _: raise NotImplementedError


def term_vars(term: Term) -> Iterable[Variable]:
    match term:
        case Variable(_, _): yield term
        case Constant(_, _): yield from ()
        case ArrowElimination(fn, arg):
            yield from fn.vars()
            yield from arg.vars()
        case _:
            yield from term.body.vars()  # type: ignore


def term_constants(term: Term) -> Iterable[Constant]:
    match term:
        case Variable(_, _): yield from ()
        case Constant(_, _): yield term
        case ArrowElimination(fn, arg):
            yield from fn.constants()
            yield from arg.constants()
        case _:
            yield from term.body.constants()  # type: ignore
