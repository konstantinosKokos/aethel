from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from operator import add
from typing import Union, Set, Sequence, Tuple, Iterable, List, Callable


class WordType(ABC):
    @abstractmethod
    def __str__(self) -> str:
        pass

    @abstractmethod
    def polish(self) -> str:
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
    def get_atomic(self) -> Set['AtomicType']:
        pass

    @abstractmethod
    def get_colors(self) -> Set[str]:
        pass

    @abstractmethod
    def depolarize(self) -> 'WordType':
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

    def polish(self) -> str:
        return self.__str__()

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

    def get_atomic(self) -> Set['AtomicType']:
        return {self}

    def get_colors(self) -> Set[str]:
        return set()

    def depolarize(self) -> 'AtomicType':
        return self


class ComplexType(WordType):
    def __init__(self, argument: WordType, result: WordType) -> None:
        self.result = result
        self.argument = argument

    def __str__(self) -> str:
        return str(self.argument) + ' → ' + str(self.result)

    def polish(self) -> str:
        return '→ ' + self.argument.polish() + ' ' + self.result.polish()

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def get_arity(self) -> int:
        return max(self.argument.get_arity() + 1, self.result.get_arity())

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, ComplexType):
            return False
        else:
            return self.argument == other.argument and self.result == other.result

    def decolor(self) -> 'ComplexType':
        return ComplexType(argument=self.argument.decolor(), result=self.result.decolor())

    def get_atomic(self) -> Set[AtomicType]:
        return set.union(self.argument.get_atomic(), self.result.get_atomic())

    def get_colors(self) -> Set[str]:
        return set.union(self.argument.get_colors(), self.result.get_colors())

    def depolarize(self) -> 'ComplexType':
        return ComplexType(argument=self.argument.depolarize(), result=self.result.depolarize())


class ColoredType(ComplexType):
    def __init__(self, argument: WordType, result: WordType, color: str):
        super(ColoredType, self).__init__(argument, result)
        assert(isinstance(argument, WordType))
        assert(isinstance(result, WordType))
        assert(isinstance(color, str))
        self.color = color

    def __str__(self) -> str:
        return '<' + str(self.argument) + '> ' + self.color + ' → ' + str(self.result)

    def polish(self) -> str:
        return '→ ' + '<' + self.argument.polish() + '> ' + self.color + ' ' + self.result.polish()

    def polish_short(self) -> str:
        argstr = self.argument.polish_short() if isinstance(self.argument, ColoredType) else self.argument.polish()
        resstr = self.result.polish_short() if isinstance(self.result, ColoredType) else self.result.polish()
        return self.color + ' ' + argstr + ' ' + resstr

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, ColoredType):
            return False
        else:
            return super(ColoredType, self).__eq__(other) and self.color == other.color

    def decolor(self) -> ComplexType:
        return ComplexType(argument=self.argument.decolor(), result=self.result.decolor())

    def get_colors(self) -> Set[str]:
        return set.union(super(ColoredType, self).get_colors(), {self.color})

    def depolarize(self) -> 'ColoredType':
        return ColoredType(argument=self.argument.depolarize(), result=self.result.depolarize(), color=self.color)


class PolarizedIndexedType(AtomicType):
    def __init__(self, result: str, polarity: bool, index: int) -> None:
        super(PolarizedIndexedType, self).__init__(result=result)
        self.polarity = polarity
        self.index = index

    def __str__(self) -> str:
        return super(PolarizedIndexedType, self).__str__() + \
               '(' + ('+' if self.polarity else '-') + ', ' + str(self.index) + ')'

    def depolarize(self) -> AtomicType:
        return AtomicType(self.result)


def polarize_and_index(wordtype: WordType, polarity: bool = True, index: int = 0) -> Tuple[int, WordType]:
    if isinstance(wordtype, AtomicType):
        return index+1, PolarizedIndexedType(result=wordtype.result, polarity=polarity, index=index)
    elif isinstance(wordtype, ComplexType):
        index, arg = polarize_and_index(wordtype.argument, not polarity, index)
        index, res = polarize_and_index(wordtype.result, polarity, index)
        if isinstance(wordtype, ColoredType):
            return index, ColoredType(argument=arg, result=res, color=wordtype.color)
        else:
            return index, ComplexType(argument=arg, result=res)


def polarize_and_index_many(wordtypes: Sequence[WordType], index: int = 0) -> Tuple[int, List[WordType]]:
    ret = []
    for w in wordtypes:
        index, x = polarize_and_index(w, True, index)
        ret.append(x)
    return index, ret


def binarize(sorting_fn: Callable[[Iterable[Tuple[WordType, str]]], List[Tuple[WordType, str]]],
             arguments: WordTypes, colors: strings, result: WordType) -> ColoredType:
    argcolors = zip(arguments, colors)
    argcolors = list(sorting_fn(argcolors))
    return reduce(lambda x, y: ColoredType(result=x, argument=y[0], color=y[1]), argcolors, result)


def decolor(colored_type: WordType) -> Union[AtomicType, ComplexType]:
    return colored_type.decolor()


def depolarize(polar_type: WordType) -> WordType:
    return polar_type.depolarize()


def get_atomic(something: Union[WordTypes, WordType]) -> Set[AtomicType]:
    if isinstance(something, Sequence):
        return set.union(*map(get_atomic, something))
    else:
        return something.get_atomic()


def get_colors(something: Union[WordTypes, WordType]) -> Set[str]:
    if isinstance(something, Sequence):
        return set.union(*map(get_colors, something))
    else:
        return something.get_colors()


def polish(wordtype: WordType) -> str:
    return wordtype.polish()


def polish_short(wordtype: WordType) -> str:
    return wordtype.polish() if isinstance(wordtype, AtomicType) else wordtype.polish_short()


def get_polarities(wordtype: WordType) -> Tuple[List[AtomicType], List[AtomicType]]:
    if isinstance(wordtype, AtomicType):
        if str(wordtype)[0] == '_':
            return [], []
        return [], [wordtype.depolarize()]
    elif isinstance(wordtype, ComplexType):
        argneg, argpos = get_polarities(wordtype.argument)
        respos, resneg = get_polarities(wordtype.result)
        return argpos + respos, argneg + resneg
    else:
        raise TypeError


def get_polarities_and_indices(wordtype: WordType) -> Tuple[List[Tuple[AtomicType, int]], List[Tuple[AtomicType, int]]]:
    if isinstance(wordtype, AtomicType):
        if str(wordtype)[0] == '_':
            return [], []
        return [], [(wordtype.depolarize(), wordtype.index)]
    elif isinstance(wordtype, ComplexType):
        argneg, argpos = get_polarities_and_indices(wordtype.argument)
        respos, resneg = get_polarities_and_indices(wordtype.result)
        return argpos + respos, argneg + resneg
    else:
        raise TypeError


def literal_invariance(premises: WordTypes):
    seqpos, seqneg = list(map(lambda x: reduce(add, x), tuple(zip(*map(get_polarities, premises)))))
    return Counter(seqneg) - Counter(seqpos)


def operator_count(wt: WordType) -> int:
    return 0 if isinstance(wt, AtomicType) else operator_count(wt.result) - operator_count(wt.argument) - 1


def operator_invariance(premises: WordTypes) -> int:
    return reduce(add, map(operator_count, premises)) + len(premises) - 1


def invariance_check(premises: WordTypes, goal: WordType) -> bool:
    premises = list(filter(lambda type_: type_ != AtomicType('_'), premises))
    inferred = literal_invariance(premises)
    if list(inferred.values()) != [1]:
        return False
    elif list(inferred.keys()) != [goal]:
        return False
    if operator_invariance(premises) != operator_count(goal):
        return False
    return True


def polish_to_type(x: Sequence[str], colors: Set[str]) -> Tuple[WordType, Sequence[str]]:
    if x[0] in colors:
        color = x[0]
        arg, rem = polish_to_type(x[1:], colors)
        res, rem = polish_to_type(rem, colors)
        return ColoredType(color=color, argument=arg, result=res), rem
    else:
        return AtomicType(x[0]), x[1:]


class StrToType(object):
    def __init__(self, colors: Set[str]):
        self.colors = colors

    def __call__(self, x: Sequence[str]) -> WordType:
        return polish_to_type(x, self.colors)[0]
