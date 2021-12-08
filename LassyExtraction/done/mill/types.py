from __future__ import annotations
from abc import abstractmethod, ABCMeta
from typing import TypeVar, Callable
from functools import partial
from enum import Enum


########################################################################################################################
# Type Syntax
########################################################################################################################

T = TypeVar('T', bound='Type')


class Type(ABCMeta):
    _registry: dict[str, T]

    @abstractmethod
    def __repr__(cls) -> str: ...

    @abstractmethod
    def __eq__(cls, other) -> bool: ...

    @abstractmethod
    def __hash__(cls) -> int: ...

    @abstractmethod
    def order(cls) -> int: ...

    @classmethod
    def __init_subclass__(mcs, **kwargs):
        super(Type, mcs).__init_subclass__(**kwargs)
        mcs._registry = {}

    def __new__(mcs, name) -> T:
        if name in mcs._registry:
            return mcs._registry[name]
        ret = super(Type, mcs).__new__(mcs, name, (Proof,), {})
        mcs._registry[name] = ret
        return ret


class Atom(Type):
    _registry: dict[str, Atom]
    sign: str

    __match_args__ = ('sign',)

    def __new__(mcs, sign: str) -> Atom:
        return super(Atom, mcs).__new__(mcs, sign)

    def __init__(cls, sign: str) -> None:
        super().__init__(cls)
        cls.sign = sign

    def __repr__(cls) -> str: return cls.sign

    def __eq__(cls, other) -> bool: return isinstance(other, Atom) and cls.sign == other.sign

    def __hash__(cls) -> int: return hash(cls.sign)

    def order(cls): return 0


class Functor(Type):
    _registry: dict[str, Functor]
    argument: Type
    result: Type

    __match_args__ = ('argument', 'result')

    def __new__(mcs, argument: Type, result: Type) -> Functor:
        return super(Functor, mcs).__new__(mcs, Functor.print(argument, result))

    def __init__(cls, argument: Type, result: Type) -> None:
        super().__init__(cls)
        cls.argument = argument
        cls.result = result

    def __repr__(cls) -> str: return Functor.print(cls.argument, cls.result)

    def __eq__(cls, other):
        return isinstance(other, Functor) and cls.argument == other.argument and cls.result == other.result

    def __hash__(cls): return hash((cls.argument, cls.result))

    def order(cls): return max(cls.argument.order() + 1, cls.result.order())

    @staticmethod
    def print(argument: Type, result: Type) -> str:
        def par(x: Type) -> str: return f"({x})" if x.order() > 0 else f'{x}'
        return f'{par(argument)}->{result}'


class Modal(Type):
    _registry: dict[str, Modal]
    content: Type
    decoration: str

    # this is an ABC: await pr @ https: // github.com / python / cpython / pull / 27648
    __match_args__ = ('content', 'decoration')

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.content == other.content and self.decoration == other.decoration

    def __hash__(cls) -> int: return hash((cls.content, cls.decoration, type(cls)))

    def order(cls): return cls.content.order()


class Box(Modal):
    _registry: dict[str, Box]
    content: Type
    decoration: str

    def __new__(mcs, decoration: str, content: Type,) -> Box:
        return super(Box, mcs).__new__(mcs, f'□{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super().__init__(cls)
        cls.content = content
        cls.decoration = decoration

    def __repr__(cls) -> str: return f'□{cls.decoration}({cls.content})'


class Diamond(Modal):
    _registry: dict[str, Diamond]
    content: Type
    decoration: str

    def __new__(mcs, decoration: str, content: Type) -> Diamond:
        return super(Diamond, mcs).__new__(mcs, f'◇{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super(Diamond, cls).__init__(f'◇{decoration}({content})')
        cls.content = content
        cls.decoration = decoration

    def __repr__(cls) -> str: return f'◇{cls.decoration}({cls.content})'

########################################################################################################################
# Type Inference
########################################################################################################################


class TypeInference:
    class TypeCheckError(Exception):
        pass

    @staticmethod
    def assert_equal(a: Type, b: Type) -> None:
        if a != b:
            raise TypeInference.TypeCheckError(f'{a} != {b}')

    @staticmethod
    def arrow_elim(functor: Type, argument: Type) -> Type:
        if not isinstance(functor, Functor) or functor.argument != argument:
            raise TypeInference.TypeCheckError(f'{functor} is not a functor of {argument}')
        return functor.result

    @staticmethod
    def arrow_intro(variable: Type, body: Type) -> Type:
        return Functor(variable, body)

    @staticmethod
    def box_elim(wrapped: Type, key: str | None = None) -> Type:
        if not isinstance(wrapped, Box):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a box')
        if key is not None and key != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a box with {key}')
        return wrapped.content

    @staticmethod
    def diamond_elim(wrapped: Type, key: str | None = None) -> Type:
        if not isinstance(wrapped, Diamond):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a diamond')
        if key is not None and key != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a diamond with {key}')
        return wrapped.content

    @staticmethod
    def box_intro(decoration: str, content: Type) -> Type:
        return Box(decoration, content)

    @staticmethod
    def diamond_intro(decoration: str, content: Type) -> Type:
        return Diamond(decoration, content)


########################################################################################################################
# Term Syntax
########################################################################################################################


class Proof:
    rule: Rule
    constant: int
    variable: int
    function: Proof
    argument: Proof
    box: str
    diamond: str
    abstraction: Proof
    body: Proof

    class Rule(Enum):
        @staticmethod
        def lex(self, constant: int):
            self.constant = constant

        @staticmethod
        def ax(self, variable: int):
            self.variable = variable

        @staticmethod
        def le(self, function: Proof, argument: Proof):
            TypeInference.assert_equal(TypeInference.arrow_elim(type(function), type(argument)), type(self))
            self.function = function
            self.argument = argument

        @staticmethod
        def li(self, variable: Proof, body: Proof):
            TypeInference.assert_equal(TypeInference.arrow_intro(type(variable), type(body)), type(self))
            self.variable = variable
            self.body = body

        @staticmethod
        def be(self, box: str, body: Proof):
            TypeInference.assert_equal(Box(box, type(self)), type(body))
            self.body = body
            self.box = box

        @staticmethod
        def de(self, diamond: str, body: Proof):
            TypeInference.assert_equal(Diamond(diamond, type(self)), type(body))
            self.body = body
            self.diamond = diamond

        @staticmethod
        def di(self, diamond: str, body: Proof):
            TypeInference.assert_equal(Diamond(diamond, type(body)), type(self))
            self.body = body
            self.diamond = diamond

        @staticmethod
        def bi(self, box: str, body: Proof):
            TypeInference.assert_equal(Box(box, type(body)), type(self))
            self.body = body
            self.box = box

        Lex = partial(lex)
        Ax = partial(ax)
        lE = partial(le)
        lI = partial(li)
        bE = partial(be)
        dE = partial(de)
        dI = partial(di)
        bI = partial(bi)

    def __init__(self, *args, rule: Rule, **kwargs):
        self.rule = rule
        self.rule.value(self, *args, **kwargs)

    @classmethod
    def con(cls, c: int) -> Proof:
        # should avoid accessing the registry directly, but getitem does not work for metaclasses, see
        # https: // bugs.python.org / issue35992
        if isinstance(cls, Type):
            return type(cls)._registry[cls.__name__](c, rule=Proof.Rule.Lex)
        return Proof(c, rule=Proof.Rule.Lex)

    @classmethod
    def var(cls, v: int) -> Proof:
        if isinstance(cls, Type):
            return type(cls)._registry[cls.__name__](v, rule=Proof.Rule.Ax)
        return Proof(v, rule=Proof.Rule.Ax)

    @staticmethod
    def apply(functor: Proof | Type, argument: Proof | Type) -> Proof | Type:
        # in reality an intersection type; awaiting implementation
        return TypeInference.arrow_elim(type(functor), type(argument))(functor, argument, rule=Proof.Rule.lE)

    @staticmethod
    def abstract(variable: Proof | Type, body: Proof | Type) -> Proof | Type:
        return TypeInference.arrow_intro(type(variable), type(body))(variable, body, rule=Proof.Rule.lI)

    @staticmethod
    def box(box: str, body: Proof | Type) -> Proof | Type:
        return TypeInference.box_intro(box, type(body))(body, rule=Proof.Rule.bI)

    @staticmethod
    def diamond(diamond: str, body: Proof | Type) -> Proof | Type:
        return TypeInference.diamond_intro(diamond, type(body))(body, rule=Proof.Rule.dI)

    @staticmethod
    def unbox(box: str, body: Proof | Type) -> Proof | Type:
        return TypeInference.box_elim(body, box)(body, rule=Proof.Rule.bE)

    @staticmethod
    def undiamond(diamond: str, body: Proof | Type) -> Proof | Type:
        return TypeInference.diamond_elim(body, diamond)(body, rule=Proof.Rule.dE)

    def __repr__(self) -> str: return show_term(self)


def show_term(proof: Proof, show_decorations: bool = True, show_types: bool = True,
              word_printer: Callable[[int], str] = str) -> str:
    def f(_proof: Proof) -> str: return show_term(_proof, show_decorations, show_types)
    def v(_proof: Proof) -> str: return show_term(_proof, show_decorations, False)

    def needs_par(_proof: Proof) -> bool:
        if _proof.rule in {Proof.Rule.Ax, Proof.Rule.Lex}:
            return False
        if _proof.rule in {Proof.Rule.dE, Proof.Rule.dI, Proof.Rule.bE, Proof.Rule.bI}:
            return not show_decorations and needs_par(_proof.body)
        return True

    match proof.rule:
        case Proof.Rule.Lex:
            return f'{word_printer(proof.constant)}' if not show_types else f'{proof.constant}::{type(proof)}'
        case Proof.Rule.Ax:
            return f'x{proof.variable}' if not show_types else f'x{proof.variable}::{type(proof)}'
        case Proof.Rule.lE:
            fn, arg = proof.function, proof.argument
            return f'{f(fn)} ({f(arg)})' if needs_par(arg) else f'{f(fn)} {f(arg)}'
        case Proof.Rule.lI:
            var, body = proof.abstraction, proof.body
            return f'λ{v(var)}.({f(body)})' if needs_par(body) else f'λ{v(var)}.({f(body)})'
        case Proof.Rule.dI:
            return f'▵{proof.diamond}({f(proof.body)})'
        case Proof.Rule.dE:
            return f'▿{proof.diamond}({f(proof.body)})'
        case Proof.Rule.bI:
            return f'▴{proof.box}({f(proof.body)})'
        case Proof.Rule.bE:
            return f'▾{proof.box}({f(proof.body)})'
        case _:
            raise ValueError(f'unknown rule {proof.rule}')