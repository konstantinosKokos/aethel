from abc import ABC, abstractmethod
from .milltypes import WordType, DiamondType, BoxType, FunctorType
from .utils.printing import cap, cup, wedge, vee, subscript
from typing import Callable, Set


class Term(ABC):
    @property
    @abstractmethod
    def type(self) -> WordType:
        pass

    @abstractmethod
    def free(self) -> Set['Atom']:
        pass


class Atom(Term):
    def __init__(self, _type: WordType, idx: int):
        self._type = _type
        self.idx = idx

    def type(self) -> WordType:
        return self._type

    @staticmethod
    def make(idx: int, _type: WordType, lex: bool) -> 'Atom':
        return (Lex if lex else Var)(_type, idx)

    def free(self) -> Set['Term']:
        return {self} if isinstance(self, Var) else set()


class Var(Atom):
    pass


class Lex(Atom):
    pass


class BoxElim(Term):
    def __init__(self, body: Term):
        if not isinstance(body.type(), BoxType):
            raise TypeError(f'No box to eliminate in {body.type()}')
        self.body = body
        self.box = body.type().modality

    def type(self) -> WordType:
        return self.body.type().content

    def free(self) -> Set['Term']:
        return self.body.free()


class BoxIntro(Term):
    def __init__(self, body: Term, box: str):
        self.body = body
        self.box = box

    def type(self) -> BoxType:
        return BoxType(self.body.type(), modality=self.box)

    def free(self) -> Set['Term']:
        return self.body.free()

    @staticmethod
    def preemptive(box: str) -> Callable[[Term], 'BoxIntro']:
        def make(body: Term):
            return BoxIntro(body, box)
        return make


class DiamondElim(Term):
    def __init__(self, body: Term):
        if not isinstance(body.type(), DiamondType):
            raise TypeError(f'No diamond to eliminate in {body.type()}')
        self.body = body
        self.diamond = body.type().modality

    def type(self) -> WordType:
        return self.body.type().content

    def free(self) -> Set['Term']:
        return self.body.free()


class DiamondIntro(Term):
    def __init__(self, body: Term, diamond: str):
        self.body = body
        self.diamond = diamond

    def type(self) -> DiamondType:
        return DiamondType(self.body.type(), modality=self.diamond)

    def free(self) -> Set['Term']:
        return self.body.free()

    @staticmethod
    def preemptive(diamond: str) -> Callable[[Term], 'DiamondIntro']:
        def make(body: Term):
            return DiamondIntro(body, diamond)
        return make


class Application(Term):
    def __init__(self, functor: Term, argument: Term):
        if not isinstance(functor.type(), FunctorType) or functor.type().argument != argument.type():
            raise TypeError(f'Argument of type {argument.type()} is not valid for a functor of type {functor.type()}')
        self.functor = functor
        self.argument = argument

    def type(self) -> WordType:
        return self.functor.type().result

    def free(self) -> Set['Term']:
        return self.functor.free().union(self.argument.free())


class Abstraction(Term):
    def __init__(self, body: Term, abstraction: int):
        self.body = body
        free = self.body.free()
        bound = [f for f in free if f.idx == abstraction]
        if len(bound) != 1:
            raise AssertionError(f'{len(bound)}')
        self.abstraction = bound[0]

    def type(self) -> FunctorType:
        return FunctorType(argument=self.abstraction.type(), result=self.body.type())

    def free(self) -> Set['Term']:
        return self.body.free().difference({self.abstraction})

    @staticmethod
    def preemptive(abstraction: int) -> Callable[[Term], 'Abstraction']:
        def make(body: Term) -> Abstraction:
            return Abstraction(body, abstraction)
        return make


def print_term(term: Term, show_decorations: bool, word_printer: Callable[[int], str]) -> str:
    def pt(_term: Term) -> str:
        return print_term(_term, show_decorations, word_printer)

    if isinstance(term, Atom):
        return word_printer(term.idx) if isinstance(term, Lex) else f'x{subscript(term.idx)}'
    if isinstance(term, Application):
        return f'({pt(term.functor)} {pt(term.argument)})'
    if isinstance(term, BoxIntro):
        return f'{cap(term.box)}({pt(term.body)})' if show_decorations else pt(term.body)
    if isinstance(term, DiamondIntro):
        return f'{wedge(term.diamond)}({pt(term.body)})' if show_decorations else pt(term.body)
    if isinstance(term, BoxElim):
        return f'{cup(term.box)}({pt(term.body)})' if show_decorations else pt(term.body)
    if isinstance(term, DiamondElim):
        return f'{vee(term.diamond)}({pt(term.body)})' if show_decorations else pt(term.body)
    if isinstance(term, Abstraction):
        return f'λ{pt(term.abstraction)}.{pt(term.body)}'
    raise TypeError(f'Unexpected term of type {type(term)}')


def compose(f: Callable[[Term], Term], g: Callable[[Term], Term]) -> Callable[[Term], Term]:
    def h(x: Term) -> Term:
        return f(g(x))
    return h
