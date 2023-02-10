from __future__ import annotations
from .types import Type, TypeInference, Functor, Diamond, Box, TypeVar
from abc import ABC, abstractmethod
from typing import Callable, Iterable


class TermError(Exception):
    pass


class Term(ABC):
    def __repr__(self) -> str: return term_repr(self)
    def __matmul__(self, other) -> ArrowElimination: return ArrowElimination(self, other)
    def vars(self) -> Iterable[Variable]: return term_vars(self)
    def constants(self) -> Iterable[Constant]: return term_constants(self)
    def __eq__(self, other) -> bool: return isinstance(other, Term) and term_eq(self, other)

    @property
    @abstractmethod
    def type(self) -> Type: ...


TERM = TypeVar('TERM', bound=Term)


class Variable(Term):
    type: Type
    index: int
    __match_args__ = ('type', 'index')

    def __init__(self, _type: Type, index: int):
        self.index = index
        self._type = _type

    @property
    def type(self) -> Type:
        return self._type


class Constant(Term):
    type: Type
    index: int
    __match_args__ = ('type', 'index')

    def __init__(self, _type: Type, index: int):
        self.index = index
        self._type = _type

    @property
    def type(self) -> Type:
        return self._type


class ArrowElimination(Term):
    function: Term
    argument: Term
    __match_args__ = ('function', 'argument')

    def __init__(self, function: Term, argument: Term):
        self.function = function
        self.argument = argument
        _ = self.type

    @property
    def type(self) -> Type:
        return TypeInference.arrow_elim(self.function.type, self.argument.type)


class ArrowIntroduction(Term):
    abstraction: Variable
    body: Term
    __match_args__ = ('abstraction', 'body')

    def __init__(self, abstraction: Variable, body: Term):
        self.abstraction = abstraction
        self.body = body

    @property
    def type(self) -> Type:
        return Functor(self.abstraction.type, self.body.type)


class DiamondIntroduction(Term):
    decoration: str
    body: Term
    __match_args__ = ('decoration', 'body')

    def __init__(self, diamond: str, body: Term):
        self.decoration = diamond
        self.body = body

    @property
    def type(self) -> Type:
        return Diamond(self.decoration, self.body.type)


class BoxElimination(Term):
    decoration: str
    body: Term
    __match_args__ = ('decoration', 'body')

    def __init__(self, box: str | None, body: Term):
        _, self.decoration = TypeInference.box_elim(body.type, box)
        self.body = body

    @property
    def type(self) -> Type:
        return TypeInference.box_elim(self.body.type, self.decoration)[0]


class BoxIntroduction(Term):
    decoration: str
    body: Term
    __match_args__ = ('decoration', 'body')

    def __init__(self, box: str, body: Term):
        self.decoration = box
        self.body = body

    @property
    def type(self) -> Type:
        return Box(self.decoration, self.body.type)


class DiamondElimination(Term):
    decoration: str
    body: Term
    __match_args__ = ('decoration', 'body')

    def __init__(self, diamond: str | None, body: Term):
        _, self.decoration = TypeInference.dia_elim(body.type, diamond)
        self.body = body

    @property
    def type(self) -> Type:
        return TypeInference.dia_elim(self.body.type, self.decoration)[0]


class CaseOf(Term):
    becomes: Term
    where: Term
    original: Term
    __match_args__ = ('becomes', 'where', 'original')

    def __init__(self, becomes: Term, where: Term, original: Term):
        self.becomes = becomes
        self.where = where
        self.original = original

    @property
    def type(self) -> Type:
        return self.original.type


########################################################################################################################
# Meta-Rules and Term Rewrites
########################################################################################################################
def _word_repr(idx: int) -> str: return f'c{idx}'


def needs_par(term: Term) -> bool:
    match term:
        case ArrowElimination(_, _): return True
        case _: return False


def term_repr(term: Term,
              show_type: bool = True,
              show_intermediate_types: bool = False,
              word_repr: Callable[[int], str] = _word_repr) -> str:

    def f(_term: Term) -> str: return term_repr(_term, False, show_intermediate_types, word_repr)
    def v(_term: Term) -> str: return term_repr(_term, False, False)
    def type_hint(s: str) -> str: return f'{s} : {term.type}' if show_type ^ show_intermediate_types else s

    def par(_term: Term) -> str:
        p = f(_term)
        return f'({p})' if needs_par(_term) else p

    match term:
        case Variable(_type, index): ret = f'x{index}'
        case Constant(_type, index): ret = f'{word_repr(index)}'
        case ArrowElimination(fn, arg): ret = f'{f(fn)} {par(arg)}'
        case ArrowIntroduction(var, body): ret = f'(λ{v(var)}.{f(body)})'
        case DiamondIntroduction(decoration, body): ret = f'▵{decoration}({f(body)})'
        case BoxElimination(decoration, body): ret = f'▾{decoration}({f(body)})'
        case BoxIntroduction(decoration, body): ret = f'▴{decoration}({f(body)})'
        case DiamondElimination(decoration, body): ret = f'▿{decoration}({f(body)})'
        case CaseOf(becomes, where, original): ret = f'case {f(becomes)} of {v(where)} in {par(original)}'
        case _: raise NotImplementedError
    return type_hint(ret)


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
        case CaseOf(becomes, _, original):
            yield from term_constants(becomes)
            yield from term_constants(original)
        case _:
            yield from term.body.constants()  # type: ignore


def term_eq(left: Term, right: Term) -> bool:
    match left, right:
        case Variable(left_type, left_index), Variable(right_type, right_index):
            return left_index == right_index and left_type == right_type
        case Constant(left_type, left_index), Constant(right_type, right_index):
            return left_index == right_index and left_type == right_type
        case ArrowElimination(left_fn, left_arg), ArrowElimination(right_fn, right_arg):
            return left_fn == right_fn and left_arg == right_arg
        case ArrowIntroduction(left_var, left_body), ArrowIntroduction(right_var, right_body):
            return left_var == right_var and left_body == right_body
        case DiamondIntroduction(left_decoration, left_body), DiamondIntroduction(right_decoration, right_body):
            return left_decoration == right_decoration and left_body == right_body
        case BoxElimination(left_decoration, left_body), BoxElimination(right_decoration, right_body):
            return left_decoration == right_decoration and left_body == right_body
        case BoxIntroduction(left_decoration, left_body), BoxIntroduction(right_decoration, right_body):
            return left_decoration == right_decoration and left_body == right_body
        case DiamondElimination(left_decoration, left_body), DiamondElimination(right_decoration, right_body):
            return left_decoration == right_decoration and left_body == right_body
        case CaseOf(left_becomes, left_where, left_original), CaseOf(right_becomes, right_where, right_original):
            return left_becomes == right_becomes and left_where == right_where and left_original == right_original
        case _:
            return False
