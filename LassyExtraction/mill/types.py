from __future__ import annotations

from abc import ABC
from typing import TypeVar

########################################################################################################################
# Type Syntax
########################################################################################################################

T = TypeVar('T', bound='Type')


class Type(ABC):
    def __repr__(self) -> str: return type_repr(self)
    def order(self) -> int: return type_order(self)
    def __eq__(self, other) -> bool: return type_eq(self, other)
    def __hash__(self) -> int: return type_hash(self)
    def prefix(self) -> str: return type_prefix(self)

    @staticmethod
    def parse_prefix(prefix: str) -> Type: return parse_prefix(prefix)


class Atom(Type):
    sign: str
    __match_args__ = ('sign',)

    def __init__(self, sign: str) -> None:
        self.sign = sign


class Functor(Type):
    argument:   Type
    result:     Type
    __match_args__ = ('argument', 'result')

    def __init__(self, argument: Type, result: Type) -> None:
        self.argument = argument
        self.result = result

    @staticmethod
    def repr(argument: Type, result: Type) -> str:
        def par(x: Type) -> str: return f"({x})" if x.order() > 0 else f'{x}'
        return f'{par(argument)}⟶{result}'


class Modal(Type, ABC):
    content:    Type
    decoration: str
    __match_args__ = ('decoration', 'content')


class Box(Modal):
    def __init__(self, decoration: str, content: Type) -> None:
        self.content = content
        self.decoration = decoration


class Diamond(Modal):
    content: Type
    decoration: str
    __match_args__ = ('decoration', 'content')

    def __init__(self, decoration: str, content: Type) -> None:
        self.content = content
        self.decoration = decoration


########################################################################################################################
# Type utilities
########################################################################################################################
def type_order(type_: Type) -> int:
    match type_:
        case Atom(_): return 0
        case Functor(argument, result): return max(type_order(argument) + 1, type_order(result))
        case Modal(_, content): return type_order(content)
        case _: raise ValueError(f'Unknown type: {type_}')


def needs_par(_type: Type) -> bool:
    return isinstance(_type, Functor)


def type_repr(_type: Type) -> str:
    match _type:
        case Atom(sign): return sign
        case Functor(argument, result): return Functor.repr(argument, result)
        case Box(decoration, content): return f'□{decoration}({type_repr(content)})'
        case Diamond(decoration, content): return f'◇{decoration}({type_repr(content)})'
        case _: raise ValueError(f'Unknown type: {_type}')


def type_eq(_type: Type, other: Type) -> bool:
    match _type:
        case Atom(sign):
            return isinstance(other, Atom) and sign == other.sign
        case Functor(argument, result):
            return isinstance(other, Functor) and type_eq(argument, other.argument) and type_eq(result, other.result)
        case Box(decoration, content):
            return isinstance(other, Box) and decoration == other.decoration and type_eq(content, other.content)
        case Diamond(decoration, content):
            return isinstance(other, Diamond) and decoration == other.decoration and type_eq(content, other.content)
        case _: raise ValueError(f'Unknown type: {_type}')


def type_prefix(_type: Type) -> str:
    match _type:
        case Atom(sign): return sign
        case Functor(argument, result): return f'⟶ {type_prefix(argument)} {type_prefix(result)}'
        case Box(decoration, content): return f'□{decoration} {type_prefix(content)}'
        case Diamond(decoration, content): return f'◇{decoration} {type_prefix(content)}'
        case _: raise ValueError(f'Unknown type: {_type}')


def parse_prefix(string: str) -> Type:
    symbols = string.split()
    stack: list[Type] = []
    for symbol in reversed(symbols):
        if symbol == '⟶':
            stack.append(Functor(stack.pop(), stack.pop()))
        elif symbol.startswith('□'):
            stack.append(Box(symbol.lstrip('□'), stack.pop()))
        elif symbol.startswith('◇'):
            stack.append(Diamond(symbol.lstrip('◇'), stack.pop()))
        else:
            stack.append(Atom(symbol))
    return stack.pop()


def type_hash(_type: Type) -> int:
    match _type:
        case Atom(sign): return hash((sign,))
        case Functor(argument, result): return hash((type_hash(argument), type_hash(result)))
        case Box(decoration, content): return hash((f'□{decoration}', type_hash(content)))
        case Diamond(decoration, content): return hash((f'◇{decoration}', type_hash(content)))
        case _: raise ValueError(f'Unknown type: {_type}')


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
    def box_elim(wrapped: Type, box: str | None = None) -> tuple[Type, str]:
        if not isinstance(wrapped, Box):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a box')
        if box is not None and box != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a {box}-box')
        return wrapped.content, wrapped.decoration

    @staticmethod
    def dia_elim(wrapped: Type, dia: str | None = None) -> tuple[Type, str]:
        if not isinstance(wrapped, Diamond):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a diamond')
        if dia is not None and dia != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a {dia}-diamond')
        return wrapped.content, wrapped.decoration


def decolor_type(_type: Type) -> Type:
    match _type:
        case Atom(_): return _type
        case Functor(arg, res): return Functor(decolor_type(arg), decolor_type(res))
        case Modal(_, content): return decolor_type(content)
