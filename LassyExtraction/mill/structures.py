from __future__ import annotations
from typing import Iterable, Iterator, Self, Generic, TypeVar
from typing import Sequence as SequenceType
from abc import ABC, abstractmethod


class StructureError(Exception):
    pass


_T = TypeVar('_T')


class Structure(ABC, Generic[_T], Iterable[_T]):
    def __repr__(self) -> str: return struct_repr(self)
    def __eq__(self, other) -> bool: return struct_eq(self, other)
    @abstractmethod
    def __contains__(self, item) -> bool: ...
    @abstractmethod
    def units(self) -> Iterable[tuple[tuple[str, ...], _T]]: ...
    @abstractmethod
    def __iter__(self) -> Iterator[_T]: ...
    @abstractmethod
    def substitute(self, var: Structure[_T], value: Structure[_T]) -> Structure[_T]: ...


class Unary(Structure[_T]):
    __match_args__ = ('content', 'brackets')

    def __init__(self, content: Sequence[_T], brackets: str) -> None:
        self.content = Sequence(content)
        self.brackets = brackets

    def __contains__(self, item) -> bool: return struct_eq(self, item)

    def units(self) -> Iterable[tuple[tuple[str, ...], _T]]:
        for (ctx, item) in self.content.units():
            yield (self.brackets, *ctx), item

    def __iter__(self) -> Iterator[Self]: yield self

    def substitute(self, var: Structure[_T], value: Structure[_T]) -> Structure[_T]:
        if struct_eq(self, var):
            return value
        raise ValueError(f'{var} != {self}')


class Sequence(Structure[_T], SequenceType[_T]):
    __match_args__ = ('structures',)

    def __init__(self, *structures: _T | Unary[_T]) -> None:
        self.structures = sum((s.structures if isinstance(s, Sequence) else (s,) for s in structures), ())

    def __getitem__(self, item: int | slice) -> Structure[_T]: return self.structures[item]
    def __len__(self) -> int: return len(self.structures)
    def __contains__(self, item): return item in self.structures
    def __pow__(self, brackets: str) -> Unary[_T]: return Unary(self, brackets)

    def units(self) -> Iterable[tuple[tuple[str, ...], _T]]:
        for s in self.structures:
            match s:
                case Unary(_, _): yield from s.units()
                case _: yield (), s

    def __iter__(self) -> Iterator[Structure[_T]]: yield from self.structures

    def substitute(self, var: Structure[_T], value: Structure[_T]) -> Structure[_T]:
        if (cs := self.structures.count(var)) != 1:
            raise StructureError(f'Expected exactly one occurrence of {var} in {self}, but found {cs}')
        return Sequence(*(s if s != var else value for s in self.structures))


def struct_repr(structure: Structure[_T]) -> str:
    match structure:
        case Sequence(xs): return ', '.join(map(repr, xs))
        case Unary(x, bs): return f'〈{repr(x)}〉{bs}'


def struct_eq(structure1: Structure[_T], structure2: Structure[_T]) -> bool:
    match structure1, structure2:
        case Sequence(xs), Sequence(ys): return xs == ys
        case Unary(x, lb), Unary(y, rb): return x == y and lb == rb
        case _: return False