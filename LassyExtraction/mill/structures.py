from __future__ import annotations
from typing import Iterable, Iterator, Self, Generic, TypeVar
from typing import Sequence as SequenceType
from abc import ABC, abstractmethod


class StructureError(Exception):
    pass


_T = TypeVar('_T')


class Structure(ABC, Iterable, Generic[_T]):
    @abstractmethod
    def __iter__(self) -> Iterator[Self]: ...
    def __repr__(self) -> str: return struct_repr(self)
    @abstractmethod
    def __contains__(self, item: Structure) -> bool: ...
    def __pow__(self, brackets: str) -> Unary[_T]: return Unary(self, brackets)
    def __eq__(self, other) -> bool: return struct_eq(self, other)
    @abstractmethod
    def __abs__(self) -> Structure[_T]: ...
    @abstractmethod
    def replace(self, old: Structure, new: Structure) -> Structure[_T]: ...
    @abstractmethod
    def vars(self) -> list[_T]: ...


class Unary(Structure[_T]):
    __match_args__ = ('content', 'brackets')

    def __init__(self, content: Structure[_T], brackets: str):
        self.content = content
        self.brackets = brackets

    def __iter__(self) -> Iterator[Unary]: yield self
    def __contains__(self, item: Structure[_T]) -> bool: return struct_eq(self, item)
    def __abs__(self) -> Structure[_T]: return self

    def replace(self, old: Structure[_T], new: Structure[_T]) -> Structure[_T]:
        if self == old:
            return new
        raise StructureError(f'{old} != {self}')

    def vars(self) -> list[_T]: return self.content.vars()


class Sequence(SequenceType, Structure[_T]):
    __match_args__ = ('structures',)

    def __init__(self, *structures: Structure[_T] | _T):
        self.structures = structures

    def __len__(self) -> int: return len(self.structures)
    def __iter__(self) -> Iterator[Sequence[_T]]: yield from self.structures
    def __contains__(self, item: Structure[_T]) -> bool: return item in self.structures
    def __getitem__(self, item: int) -> Structure[_T]: return self.structures[item]
    def __abs__(self) -> Structure[_T]: return self if len(self) > 1 else self[0] if len(self) == 1 else Sequence()

    def replace(self, item: Structure[_T], new_item: Structure[_T]) -> Structure[_T]:
        if (cs := self.structures.count(item)) != 1:
            raise StructureError(f'Expected exactly one occurrence of {item} in {self}, but found {cs}')
        return abs(Sequence(*(new_item if struct == item else struct for struct in self.structures)))

    def vars(self) -> list[_T]:
        return [struct.vars() if isinstance(struct, Structure) else struct for struct in self.structures]


def struct_repr(structure: Structure[_T]) -> str:
    match structure:
        case Sequence(xs): return ', '.join(map(repr, xs))
        case Unary(x, bs): return f'〈{repr(x)}〉{bs}'


def struct_eq(structure1: Structure[_T], structure2: Structure[_T]) -> bool:
    match structure1, structure2:
        case Sequence(xs), Sequence(ys): return xs == ys
        case Unary(x, lb), Unary(y, rb): return x == y and lb == rb
        case _: return False


def bracket(structure: Structure[_T] | _T, brackets: str) -> Unary[_T]:
    return structure**brackets if isinstance(structure, (Structure, Unary)) else Sequence(structure)**brackets
