from __future__ import annotations

from abc import ABCMeta
from typing import TypeVar
from typing import Type as TYPE
from typing import Optional as Maybe
from functools import reduce
from enum import Enum
from itertools import product as prod
from collections import Counter as Bag

########################################################################################################################
# Type Syntax
########################################################################################################################

T = TypeVar('T', bound='Type')

SerializedType = tuple[TYPE, tuple[str]] | \
                 tuple[TYPE, tuple['SerializedType', 'SerializedType']] | \
                 tuple[TYPE, tuple[str, 'SerializedType']]


class Type(ABCMeta):
    _registry: dict[str, T]
    def __repr__(cls) -> str: return type_repr(cls)
    def order(cls) -> int: return type_order(cls)
    def __eq__(cls, other) -> bool: return type_eq(cls, other)
    def __hash__(cls) -> int: return type_hash(cls)
    def prefix(cls) -> str: return type_prefix(cls)
    def serialize_type(cls) -> SerializedType: return serialize_type(cls)

    @classmethod
    def __init_subclass__(mcs, **kwargs):
        super(Type, mcs).__init_subclass__(**kwargs)
        mcs._registry = {}

    def __new__(mcs, name, bases: tuple[Type, ...] = ()) -> T:
        if name in mcs._registry:
            return mcs._registry[name]
        ret = super(Type, mcs).__new__(mcs, name, bases, {})
        mcs._registry[name] = ret
        return ret

    @staticmethod
    def parse_prefix(string: str) -> Type: return parse_prefix(string)


class Atom(Type):
    _registry: dict[str, Atom]
    sign: str
    __match_args__ = ('sign',)

    def __new__(mcs, sign: str, bases: tuple[Type, ...] = ()) -> Atom:
        return super(Atom, mcs).__new__(mcs, sign, bases)

    def __init__(cls, sign: str, _: tuple[Type, ...] = ()) -> None:
        super(Atom, cls).__init__(cls)
        cls.sign = sign


class Functor(Type):
    _registry: dict[str, Functor]
    argument: Type
    result: Type

    __match_args__ = ('argument', 'result')

    def __new__(mcs, argument: Type, result: Type) -> Functor:
        return super(Functor, mcs).__new__(mcs, Functor.repr(argument, result))

    def __init__(cls, argument: Type, result: Type) -> None:
        super(Functor, cls).__init__(cls)
        cls.argument = argument
        cls.result = result

    @staticmethod
    def repr(argument: Type, result: Type) -> str:
        def par(x: Type) -> str: return f"({x})" if x.order() > 0 else f'{x}'
        return f'{par(argument)}⟶{result}'


class Modal(Type):
    _registry: dict[str, Modal]
    content: Type
    decoration: str

    __match_args__ = ('decoration', 'content')


class Box(Modal):
    _registry: dict[str, Box]
    content: Type
    decoration: str

    __match_args__ = ('decoration', 'content')

    def __new__(mcs, decoration: str, content: Type,) -> Box:
        return super(Box, mcs).__new__(mcs, f'□{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super(Box, cls).__init__(cls)
        cls.content = content
        cls.decoration = decoration


class Diamond(Modal):
    _registry: dict[str, Diamond]
    content: Type
    decoration: str

    __match_args__ = ('decoration', 'content')

    def __new__(mcs, decoration: str, content: Type) -> Diamond:
        return super(Diamond, mcs).__new__(mcs, f'◇{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super(Diamond, cls).__init__(cls)
        cls.content = content
        cls.decoration = decoration


########################################################################################################################
# Type utilities
########################################################################################################################
def type_order(type_: Type) -> int:
    match type_:
        case Atom(_): return 0
        case Functor(argument, result): return max(type_order(argument) + 1, type_order(result))
        case Modal(_, content): return type_order(content)
        case _: raise ValueError(f'Unknown type: {type_}')


def type_repr(type_: Type) -> str:
    match type_:
        case Atom(sign): return sign
        case Functor(argument, result): return Functor.repr(argument, result)
        case Box(decoration, content): return f'□{decoration}({type_repr(content)})'
        case Diamond(decoration, content): return f'◇{decoration}({type_repr(content)})'
        case _: raise ValueError(f'Unknown type: {type_}')


def type_eq(type_: Type, other: Type) -> bool:
    match type_:
        case Atom(sign):
            return isinstance(other, Atom) and sign == other.sign and type_.__bases__ == other.__bases__
        case Functor(argument, result):
            return isinstance(other, Functor) and type_eq(argument, other.argument) and type_eq(result, other.result)
        case Box(decoration, content):
            return isinstance(other, Box) and decoration == other.decoration and type_eq(content, other.content)
        case Diamond(decoration, content):
            return isinstance(other, Diamond) and decoration == other.decoration and type_eq(content, other.content)
        case _: raise ValueError(f'Unknown type: {type_}')


def type_prefix(type_: Type) -> str:
    match type_:
        case Atom(sign): return sign
        case Functor(argument, result): return f'⊸ {type_prefix(argument)} {type_prefix(result)}'
        case Box(decoration, content): return f'□{decoration} {type_prefix(content)}'
        case Diamond(decoration, content): return f'◇{decoration} {type_prefix(content)}'
        case _: raise ValueError(f'Unknown type: {type_}')


def parse_prefix(string: str) -> Type:
    symbols = string.split()
    stack: list[Type] = []
    for symbol in reversed(symbols):
        if symbol == '⟶':
            return Functor(stack.pop(), stack.pop())
        if symbol.startswith('□'):
            return Box(symbol.lstrip('□'), stack.pop())
        if symbol.startswith('◇'):
            return Diamond(symbol.lstrip('◇'), stack.pop())
        stack.append(Atom(symbol))
    return stack.pop()


def type_hash(type_: Type) -> int:
    match type_:
        case Atom(sign): return hash((sign,))
        case Functor(argument, result): return hash((type_hash(argument), type_hash(result)))
        case Box(decoration, content): return hash((f'□{decoration}', type_hash(content)))
        case Diamond(decoration, content): return hash((f'◇{decoration}', type_hash(content)))
        case _: raise ValueError(f'Unknown type: {type_}')


def serialize_type(type_: Type) -> SerializedType:
    match type_:
        case Atom(sign): return Atom, (sign,)
        case Functor(argument, result): return Functor, (serialize_type(argument), serialize_type(result))
        case Box(decoration, content): return Box, (decoration, serialize_type(content))
        case Diamond(decoration, content): return Diamond, (decoration, serialize_type(content))
        case _: raise ValueError(f'Unknown type: {type_}')


def deserialize_type(serialized: SerializedType) -> Type:
    cls, args = serialized
    if cls == Atom:
        (sign,) = args
        return Atom(sign)
    if cls == Functor:
        (left, right) = args
        return Functor(deserialize_type(left), deserialize_type(right))
    if cls == Box:
        (decoration, content) = args
        return Box(decoration, deserialize_type(content))
    if cls == Diamond:
        (decoration, content) = args
        return Diamond(decoration, deserialize_type(content))
    raise ValueError(f'Unknown type: {cls}')


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
