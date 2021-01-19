from abc import ABC, abstractmethod
from .utils.printing import *
from collections import Counter as Multiset
from functools import reduce
from operator import add
from typing import Set, Sequence, Tuple, List, overload, TypeVar, Mapping, Union, Optional
from dataclasses import dataclass


class WordType(ABC):
    def __repr__(self) -> str:
        return str(self)

    def __call__(self) -> str:
        return repr(self)

    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        return hash(str(self))

    @abstractmethod
    def polish(self) -> str:
        pass

    @abstractmethod
    def order(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def decolor(self) -> 'WordType':
        pass

    @abstractmethod
    def depolarize(self) -> 'WordType':
        pass

    @abstractmethod
    def atoms(self) -> Set['AtomicType']:
        pass


T_co = TypeVar('T_co', covariant=True, bound=WordType)


class AtomicType(WordType):
    def __init__(self, _type: str) -> None:
        if not isinstance(_type, str):
            raise TypeError(f'Expected result to be of type str, received {type(_type)} instead.')
        self.type = _type

    def __str__(self) -> str:
        return smallcaps(self.type)

    def __hash__(self):
        return hash(str(self))

    def polish(self) -> str:
        return str(self)

    def order(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AtomicType) and self.type == other.type

    def decolor(self) -> 'AtomicType':
        return self

    def depolarize(self) -> 'AtomicType':
        return self

    def atoms(self) -> Set['AtomicType']:
        return {self}


class FunctorType(WordType):
    def __init__(self, argument: WordType, result: WordType) -> None:
        self.result = result
        self.argument = argument

    def __str__(self) -> str:
        if self.argument.order() > 0 and not isinstance(self.argument, ModalType):
            return f'({str(self.argument)}) → {str(self.result)}'
        return f'{str(self.argument)} → {str(self.result)}'

    def __hash__(self):
        return hash(str(self))

    def polish(self) -> str:
        return f'→ {self.argument.polish()} {self.result.polish()}'

    def order(self) -> int:
        return max(self.argument.order() + 1, self.result.order())

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FunctorType) and self.argument == other.argument and self.result == other.result

    def decolor(self) -> 'FunctorType':
        return FunctorType(argument=self.argument.decolor(), result=self.result.decolor())

    def depolarize(self) -> 'FunctorType':
        return FunctorType(argument=self.argument.depolarize(), result=self.result.depolarize())

    def atoms(self) -> Set[AtomicType]:
        return self.argument.atoms().union(self.result.atoms())


class ModalType(WordType, ABC):
    def __init__(self, content: T_co, modality: str):
        self.content = content
        self.modality = modality

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.content == other.content and self.modality == other.modality

    def order(self) -> int:
        return self.content.order()

    def decolor(self) -> WordType:
        return self.content.decolor()

    def depolarize(self) -> WordType:
        return type(self)(self.content.depolarize(), self.modality)

    def atoms(self) -> Set[AtomicType]:
        return self.content.atoms()


class BoxType(ModalType):
    def __str__(self):
        return f'{print_box(self.modality)}({str(self.content)})'

    def __hash__(self):
        return hash(str(self))

    def polish(self) -> str:
        return f'{print_box(self.modality)} {self.content.polish()}'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, BoxType) and self.modality == other.modality and self.content == other.content


class DiamondType(ModalType):
    def __str__(self):
        return print_diamond(self.modality) + (str(self.content) if self.content.order() == 0 else f'({str(self.content)})')

    def __hash__(self):
        return hash(str(self))

    def polish(self) -> str:
        return f'{print_box(self.modality)} {self.content.polish()}'

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiamondType) and self.modality == other.modality and self.content == other.content


class PolarizedType(AtomicType):
    def __init__(self, _type: str, polarity: bool, index: int):
        super(PolarizedType, self).__init__(_type=_type)
        self.polarity = polarity
        self.index = index

    def __str__(self) -> str:
        return super(PolarizedType, self).__str__() + f'({"+" if self.polarity else "-"},{str(self.index)})'

    def __hash__(self):
        return hash(str(self))

    def depolarize(self) -> 'AtomicType':
        return AtomicType(_type=self.type)


class EmptyType(AtomicType):
    def __init__(self):
        super(EmptyType, self).__init__('_')


def polarize_and_index(wordtype: T_co, polarity: bool = True, index: int = 0) -> Tuple[int, T_co]:
    if isinstance(wordtype, EmptyType):
        return index, wordtype
    if isinstance(wordtype, AtomicType):
        return index + 1, PolarizedType(_type=wordtype.type, polarity=polarity, index=index)
    if isinstance(wordtype, ModalType):
        index, content = polarize_and_index(wordtype.content, polarity=polarity, index=index)
        return index, type(wordtype)(content=content, modality=wordtype.modality)
    if isinstance(wordtype, FunctorType):
        index, arg = polarize_and_index(wordtype.argument, polarity=not polarity, index=index)
        index, res = polarize_and_index(wordtype.result, polarity=polarity, index=index)
        return index, FunctorType(argument=arg, result=res)
    raise TypeError(f'Unexpected argument {wordtype} of type {type(wordtype)}')


def polarize_and_index_many(wordtypes: Sequence[T_co], index: int = 0) -> Tuple[int, List[T_co]]:
    ret = []
    for w in wordtypes:
        index, x = polarize_and_index(w, True, index)
        ret.append(x)
    return index, ret


def get_polarities(wordtype, get_indices: bool = False):
    def f(x: PolarizedType):
        return (x.depolarize(), x.index) if get_indices else x.depolarize()

    if isinstance(wordtype, EmptyType):
        return [], []
    if isinstance(wordtype, PolarizedType):
        return [], [f(wordtype)]
    if isinstance(wordtype, AtomicType) and not get_indices:
        return [], [wordtype]
    if isinstance(wordtype, FunctorType):
        argpos, argneg = get_polarities(wordtype.argument, get_indices)
        resneg, respos = get_polarities(wordtype.result, get_indices)
        return argneg + resneg, argpos + respos
    if isinstance(wordtype, ModalType):
        return get_polarities(wordtype.content, get_indices)
    else:
        raise TypeError(f'Unexpected argument {wordtype} of type {type(wordtype)}')


def get_polarities_and_indices(wordtype: T_co) -> Tuple[List[Tuple[AtomicType, int]], List[Tuple[AtomicType, int]]]:
    return get_polarities(wordtype, True)


@overload
def decolor(x: T_co) -> T_co:
    pass


@overload
def decolor(x: List[T_co]) -> List[T_co]:
    pass


def decolor(x):
    if isinstance(x, WordType):
        return x.decolor()
    else:
        return list(map(decolor, x))


@overload
def depolarize(x: T_co) -> T_co:
    pass


@overload
def depolarize(x: List[T_co]) -> List[T_co]:
    pass


def depolarize(x):
    if isinstance(x, WordType):
        return x.depolarize()
    else:
        return list(map(depolarize, x))


def literal_invariance(premises: Sequence[T_co]):
    seqpos, seqneg = list(map(lambda x: reduce(add, x), tuple(zip(*map(get_polarities, premises)))))
    return Multiset(seqneg) - Multiset(seqpos)


def operator_count(wordtype: T_co) -> int:
    if isinstance(wordtype, AtomicType):
        return 0
    if isinstance(wordtype, FunctorType):
        return operator_count(wordtype.result) - operator_count(wordtype.argument) - 1
    if isinstance(wordtype, ModalType):
        return operator_count(wordtype.content)


def operator_invariance(premises: Sequence[T_co]) -> int:
    return reduce(add, map(operator_count, premises)) + len(premises) - 1


def invariance_check(premises: Sequence[T_co], goal: WordType) -> bool:
    premises = list(filter(lambda type_: not isinstance(type_, EmptyType), premises))
    inferred = literal_invariance(premises)
    if list(inferred.values()) != [1]:
        return False
    elif list(inferred.keys()) != [goal]:
        return False
    if operator_invariance(premises) != operator_count(goal):
        return False
    return True


class Connective(ABC):
    def __repr__(self):
        return str(self)

    def __call__(self):
        return repr(self)

    @abstractmethod
    def __str__(self):
        pass


class Tensor(Connective):
    def __str__(self):
        return 'x'


class Cotensor(Connective):
    def __str__(self):
        return '+'


@dataclass(repr=False)
class Diamond(Connective):
    name: str

    def __str__(self):
        return print_diamond(self.name)


@dataclass(repr=False)
class Box(Connective):
    name: str

    def __str__(self):
        return print_box(self.name)


Path = List[Union[PolarizedType, Connective]]
Paths = List[Path]


def paths(wordtype: WordType) -> List[Tuple[Path, Paths]]:
    if isinstance(wordtype, EmptyType):
        return []
    return [(p, ps) for p, ps in traverse_pos(wordtype, [])]


def traverse_pos(wordtype: WordType, history: Path) -> List[Tuple[Path, Paths]]:
    if isinstance(wordtype, PolarizedType):
        return [(history + [wordtype], [])]
    if isinstance(wordtype, FunctorType):
        pcont = traverse_pos(wordtype.result, history + [Tensor()])
        neg, rest = traverse_neg(wordtype.argument, [])
        pc = pcont[0]
        return [(pc[0], [neg] + pc[1])] + pcont[1:] + rest
    if isinstance(wordtype, DiamondType):
        return traverse_pos(wordtype.content, history + [Diamond(wordtype.modality)])
    if isinstance(wordtype, BoxType):
        return traverse_pos(wordtype.content, history + [Box(wordtype.modality)])
    raise TypeError(f'Cannot traverse {wordtype} of type {type(wordtype)}')


def traverse_neg(wordtype: WordType, hist: Path) -> Tuple[Path, List[Tuple[Path, Paths]]]:
    if isinstance(wordtype, PolarizedType):
        return hist + [wordtype], []
    if isinstance(wordtype, FunctorType):
        ncont, rest = traverse_neg(wordtype.result, hist + [Cotensor()])
        return ncont, traverse_pos(wordtype.argument, []) + rest
    if isinstance(wordtype, DiamondType):
        return traverse_neg(wordtype.content, hist + [Diamond(wordtype.modality)])
    if isinstance(wordtype, BoxType):
        return traverse_neg(wordtype.content, hist + [Box(wordtype.modality)])
    raise TypeError(f'Cannot traverse {wordtype} of type {type(wordtype)}')



# def polish_to_type(symbols: strings, operators: Set[str],
#                    operator_classes: Mapping[str, type]) -> WordType:
#     stack = list()
#
#     if len(symbols) == 1:
#         return AtomicType(symbols[0]) if symbols[0] != '_' else EmptyType()
#
#     for symbol in reversed(symbols):
#         if symbol in operators:
#             _arg = stack.pop()
#             _res = stack.pop()
#             arg = _arg if isinstance(_arg, WordType) else AtomicType(_arg)
#             res = _res if isinstance(_res, WordType) else AtomicType(_res)
#             op_class = operator_classes[symbol]
#             if op_class == BoxType or op_class == DiamondType:
#                 if isinstance(arg, BoxType) and res != arg:  # case of embedded modifier
#                     stack.append(DiamondType(arg, res, symbol))
#                 else:
#                     stack.append(op_class(arg, res, symbol))
#             else:
#                 stack.append(op_class(arg, res))
#         else:
#             stack.append(symbol)
#     ret = stack.pop()
#     assert not stack
#     assert isinstance(ret, WordType)
#     return ret
#
#

#
#

#
#
#
#


SUB = str.maketrans('0123456789', '₀₁₂₃₄₅₆₇₈₉')
SUP = str.maketrans('abcdefghijklmnoprstuvwxyz1', 'ᵃᵇᶜᵈᵉᶠᵍʰⁱʲᵏˡᵐⁿᵒᵖʳˢᵗᵘᵛʷˣʸᶻ¹')
SC = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ1→', 'ᴀʙᴄᴅᴇғɢʜɪᴊᴋʟᴍɴᴏᴘǫʀsᴛᴜᴠᴡxʏᴢ1→')


def subscript(x: Any) -> str:
    return str(x).translate(SUB)


def superscript(x: Any) -> str:
    return str(x).translate(SUP)


def smallcaps(x: Any) -> str:
    return str(x).translate(SC)
