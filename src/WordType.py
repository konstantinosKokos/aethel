from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Set, Sequence


class WordType(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __hash__(self) -> int:
        pass

    @abstractmethod
    def get_arity(self) -> int:
        pass

    @abstractmethod
    def __call__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: 'WordType') -> bool:
        pass

    @abstractmethod
    def decolor(self) -> Union['AtomicType', 'ComplexType', 'CombinatorType']:
        pass

    @abstractmethod
    def retrieve_atomic(self) -> Set[str]:
        pass

WordTypes = Sequence[WordType]
strings = Sequence[str]

class AtomicType(WordType):
    def __init__(self, result: str) -> None:
        if not isinstance(result, str):
            raise TypeError('Expected result to be of type str, received {} instead.'.format(type(result)))
        self.result = result

    def __str__(self) -> str:
        return self.result

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def get_arity(self) -> int:
        return 0

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, AtomicType):
            return False
        else:
            return self.result == other.result

    def decolor(self) -> 'AtomicType':
        return self

    def retrieve_atomic(self) -> Set[str]:
        return {self.__repr__()}


class ModalType(WordType):
    def __init__(self, result: WordType, modality: str) -> None:
        if not isinstance(result, WordType):
            raise TypeError('Expected result to be of type WordType, received {} instead.'.format(type(result)))
        self.result = result
        if not isinstance(modality, str):
            raise TypeError('Expected modality to be of type str, received {} instead.'.format(type(modality)))
        self.modality = modality

    def __str__(self) -> str:
        if self.result.get_arity():
            return self.modality + '(' + str(self.result) + ')'
        else:
            return self.modality + str(self.result)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def get_arity(self) -> int:
        return self.result.get_arity()

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: WordType) -> bool:
        if isinstance(other, ModalType):
            return self.modality == other.modality and self.result == other.result
        else:
            return False

    def decolor(self) -> 'ModalType':
        return ModalType(result=(self.result.decolor()), modality=self.modality)

    def retrieve_atomic(self) -> Set[str]:
        return self.result.retrieve_atomic()


class ComplexType(WordType):
    def __init__(self, arguments: WordTypes, result: WordType) -> None:
        if not isinstance(result, WordType):
            raise TypeError('Expected result to be of type WordType, received {} instead.'.format(type(result)))
        self.result = result
        if not isinstance(arguments, Sequence):
            raise TypeError('Expected arguments to be a Sequence of WordTypes, received {} instead.'.
                            format(type(arguments)))
        if not all(map(lambda x: isinstance(x, WordType), arguments)) or len(arguments) == 0:
            raise TypeError('Expected arguments to be a non-empty Sequence of WordTypes, '
                            'received a Sequence containing {} instead.'.format(list(map(type, arguments))))
        self.arguments = sorted(arguments, key=lambda x: x.__repr__())

    def __str__(self) -> str:
        if len(self.arguments) > 1:
            return '(' + ', '.join(map(str, self.arguments)) + ') → ' + str(self.result)
        else:
            return str(self.arguments[0]) + ' → ' + str(self.result)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def get_arity(self) -> int:
        return max(map(lambda x: x.get_arity(), self.arguments)) + 1 + self.result.get_arity()

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, ComplexType):
            return False
        else:
            return self.arguments == other.arguments and self.result == other.result

    def decolor(self) -> 'ComplexType':
        return ComplexType(arguments=tuple(map(lambda x: x.decolor(), self.arguments)), result=self.result.decolor())

    def retrieve_atomic(self) -> Set[str]:
        if len(self.arguments) == 1:
            return set.union(self.arguments[0].retrieve_atomic(), self.result.retrieve_atomic())
        else:
            return reduce(set.union, [a.retrieve_atomic() for a in self.arguments])


class ColoredType(ComplexType):
    def __init__(self, arguments: WordTypes, result: WordType, colors: strings):
        if not isinstance(colors, tuple):
            raise TypeError('Expected color to be  a Sequence of strings, received {} instead.'.format(type(colors)))
        if not all(map(lambda x: isinstance(x, str), colors)) or len(colors) == 0:
            raise TypeError('Expected arguments to be a non-empty Sequence of strings,'
                            ' received a Sequence containing {} instead.'.format(list(map(type, colors))))
        if len(colors) != len(arguments):
            raise ValueError('Uneven amount of arguments ({}) and colors ({}).'.format(len(arguments), len(colors)))
        sorted_ac = sorted([ac for ac in zip(arguments, colors)], key=lambda x: x[0].__repr__() + x[1].__repr__())
        arguments, colors = list(zip(*sorted_ac))
        super(ColoredType, self).__init__(arguments, result)
        self.colors = colors

    def __str__(self) -> str:
        if len(self.arguments) > 1:
            return '(' + ', '.join(map(lambda x: '{' + x[0].__repr__() + ': ' + x[1].__repr__() + '}',
                                       zip(self.arguments, self.colors))) + ') → ' + str(self.result)
        else:
            return '{' + str(self.arguments[0]) + ': ' + self.colors[0] + '} → ' + str(self.result)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, ColoredType):
            return False
        else:
            return (self.arguments, self.colors) == (other.arguments, other.colors) and self.result == other.result

    def decolor(self) -> 'ComplexType':
        return ComplexType(arguments=tuple(map(lambda x: x.decolor(), self.arguments)), result=self.result.decolor())


class CombinatorType(WordType):
    def __init__(self, types: WordTypes, combinator: str):
        if not isinstance(types, tuple):
            raise TypeError('Expected types to be  a Sequence of WordTypes, received {} instead.'.format(type(types)))
        if not all(map(lambda x: isinstance(x, WordType), types)) or len(types) < 1:
            raise TypeError('Expected types to be a non-empty Sequence of WordTypes,'
                            ' received a Sequence containing {} instead.'.format(list(map(type, types))))
        if not isinstance(combinator, str):
            raise TypeError('Expected combinator to be of type str, received {} instead.'.format(type(combinator)))
        self.types = sorted(types, key=lambda x: x.__repr__())
        self.combinator = combinator

    def __str__(self) -> str:
        return (' ' + self.combinator + ' ').join(t.__repr__() for t in self.types)

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def get_arity(self) -> int:
        return max(t.get_arity() for t in self.types)

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: WordType):
        if isinstance(other, CombinatorType) and self.types == other.types and self.combinator == other.combinator:
                return True
        return False

    def decolor(self) -> 'CombinatorType':
        return CombinatorType(types=tuple(map(lambda x: x.decolor(), self.types)), combinator=self.combinator)

    def retrieve_atomic(self) -> Set[WordType]:
        return reduce(set.union, [a.retrieve_atomic() for a in self.types])


def compose(base_types: WordTypes, base_colors: strings, result: WordType):
    if len(base_types) != len(base_colors):
        raise ValueError('Uneven number of types ({}) and colors ({}).'.format(len(base_types), len(base_colors)))
    return reduce(lambda x, y: ColoredType(result=x, arguments=y[0], colors=y[1]),
                  zip(base_types[::-1], base_colors[::-1]),
                  result)


def decolor(colored_type: WordType) -> Union[AtomicType, ModalType, ComplexType]:
    return colored_type.decolor()


def retrieve_atomic(something: Union[WordTypes, WordType]):
    if isinstance(something, Sequence):
        return reduce(set.union, [retrieve_atomic(s) for s in something])
    else:
        return something.retrieve_atomic()


def flat_colored_type_constructor(arguments: WordTypes, result: WordType, colors: strings) -> ColoredType:
    if isinstance(result, ColoredType):
        all_args = tuple([x for x in list(arguments)+list(result.arguments)])
        all_colors = tuple([x for x in list(colors)+list(result.colors)])
        return ColoredType(all_args, result.result, all_colors)
    else:
        return ColoredType(arguments, result, colors)
