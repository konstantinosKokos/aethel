from abc import ABC, abstractmethod
from collections import Counter
from functools import reduce
from operator import add
from typing import Set, Sequence, Tuple, List, overload


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
    def arity(self) -> int:
        pass

    @abstractmethod
    def __call__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def decolor(self) -> 'WordType':
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
    def __init__(self, wordtype: str) -> None:
        if not isinstance(wordtype, str):
            raise TypeError('Expected result to be of type str, received {} instead.'.format(type(wordtype)))
        self.type = wordtype

    def __str__(self) -> str:
        return self.type

    def polish(self) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()

    def __hash__(self) -> int:
        return self.__str__().__hash__()

    def arity(self) -> int:
        return 0

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicType):
            return False
        else:
            return self.type == other.type

    def decolor(self) -> 'AtomicType':
        return self

    def get_atomic(self) -> Set['AtomicType']:
        return {self}

    def get_colors(self) -> Set[str]:
        return set()

    def depolarize(self) -> 'AtomicType':
        return self


class FunctorType(WordType):
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

    def arity(self) -> int:
        return max(self.argument.arity() + 1, self.result.arity())

    def __call__(self) -> str:
        return self.__str__()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FunctorType):
            return False
        else:
            return self.argument == other.argument and self.result == other.result

    def decolor(self) -> 'FunctorType':
        return FunctorType(argument=self.argument.decolor(), result=self.result.decolor())

    def get_atomic(self) -> Set[AtomicType]:
        return set.union(self.argument.get_atomic(), self.result.get_atomic())

    def get_colors(self) -> Set[str]:
        return set.union(self.argument.get_colors(), self.result.get_colors())

    def depolarize(self) -> 'FunctorType':
        return FunctorType(argument=self.argument.depolarize(), result=self.result.depolarize())


class DiamondType(FunctorType):
    def __init__(self, argument: WordType, result: WordType, diamond: str):
        super(DiamondType, self).__init__(argument, result)
        self.diamond = diamond

    def __str__(self):
        return '<' + self.argument.__str__() + '> ' + self.diamond + ' → ' + self.result.__str__()

    def polish(self) -> str:
        return self.diamond + ' ' + self.argument.polish() + ' ' + self.result.polish()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DiamondType):
            return False
        else:
            return self.diamond == other.diamond and super(DiamondType, self).__eq__(other)

    def depolarize(self) -> 'DiamondType':
        return DiamondType(argument=self.argument.depolarize(), result=self.result.depolarize(),
                           diamond=self.diamond)

    def __hash__(self):
        return super(DiamondType, self).__hash__()


class BoxType(FunctorType):
    def __init__(self, argument: WordType, result: WordType, box: str):
        super(BoxType, self).__init__(argument, result)
        self.box = box

    def __str__(self):
        return '[' + self.argument.__str__() + ' → ' + self.result.__str__() + '] ' + self.box

    def polish(self) -> str:
        return self.box + ' ' + self.argument.polish() + ' ' + self.result.polish()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BoxType):
            return False
        else:
            return self.box == other.box and super(BoxType, self).__eq__(other)

    def depolarize(self) -> 'FunctorType':
        return BoxType(argument=self.argument.depolarize(), result=self.result.depolarize(),
                       box=self.box)

    def __hash__(self):
        return super(BoxType, self).__hash__()


class PolarizedType(AtomicType):
    def __init__(self, wordtype: str, polarity: bool, index: int):
        super(PolarizedType, self).__init__(wordtype=wordtype)
        self.polarity = polarity
        self.index = index

    def __str__(self) -> str:
        return super(PolarizedType, self).__str__() + '(' + ('+' if self.polarity else '-') + str(self.index) + ')'

    def depolarize(self) -> 'AtomicType':
        return AtomicType(wordtype=self.type)


def polarize_and_index(wordtype: WordType, polarity: bool = True, index: int = 0) -> Tuple[int, WordType]:
    if isinstance(wordtype, AtomicType):
        return index + 1, PolarizedType(wordtype=wordtype.type, polarity=polarity, index=index)
    elif isinstance(wordtype, DiamondType):
        index, arg = polarize_and_index(wordtype.argument, not polarity, index)
        index, res = polarize_and_index(wordtype.result, polarity, index)
        return index, DiamondType(argument=arg, result=res, diamond=wordtype.diamond)
    elif isinstance(wordtype, BoxType):
        index, arg = polarize_and_index(wordtype.argument, not polarity, index)
        index, res = polarize_and_index(wordtype.result, polarity, index)
        return index, BoxType(argument=arg, result=res, box=wordtype.box)
    elif isinstance(wordtype, FunctorType):
        index, arg = polarize_and_index(wordtype.argument, not polarity, index)
        index, res = polarize_and_index(wordtype.result, polarity, index)
        return index, FunctorType(argument=arg, result=res)
    else:
        raise TypeError('Expected wordtype to be of type WordType, received {} instead.'.format(type(wordtype)))


def polarize_and_index_many(wordtypes: Sequence[WordType], index: int = 0) -> Tuple[int, List[WordType]]:
    ret = []
    for w in wordtypes:
        index, x = polarize_and_index(w, True, index)
        ret.append(x)
    return index, ret


@overload
def decolor(x: WordType) -> WordType:
    pass


@overload
def decolor(x: WordTypes) -> WordTypes:
    pass


def decolor(x):
    if isinstance(x, WordType):
        return x.decolor()
    else:
        return list(map(decolor, x))


@overload
def get_atomic(x: WordType) -> Set[AtomicType]:
    pass


@overload
def get_atomic(x: WordTypes) -> Set[AtomicType]:
    pass


def get_atomic(x):
    if isinstance(x, WordType):
        return x.get_atomic()
    else:
        return set.union(*map(get_atomic, x))


@overload
def get_colors(x: WordType) -> Set[str]:
    pass


@overload
def get_colors(x: WordTypes) -> Set[str]:
    pass


def get_colors(x):
    if isinstance(x, WordType):
        return x.get_colors()
    else:
        set.union(*map(get_colors, x))


def get_polarities_and_indices(wordtype: WordType) -> Tuple[List[Tuple[AtomicType, int]], List[Tuple[AtomicType, int]]]:
    if isinstance(wordtype, PolarizedType):
        if str(wordtype)[0] == '_':
            return [], []
        return [], [(wordtype.depolarize(), wordtype.index)]
    elif isinstance(wordtype, FunctorType):
        argneg, argpos = get_polarities_and_indices(wordtype.argument)
        respos, resneg = get_polarities_and_indices(wordtype.result)
        return argpos + respos, argneg + resneg
    else:
        raise TypeError('Expected wordtype to be of type Union[PolarizedType, FunctorType],'
                        ' received {} instead'.format(type(wordtype)))


def get_polarities(wordtype: WordType) -> Tuple[List[AtomicType], List[AtomicType]]:
    if isinstance(wordtype, AtomicType):
        if str(wordtype)[0] == '_':
            return [], []
        return [], [wordtype.depolarize()]
    elif isinstance(wordtype, FunctorType):
        argneg, argpos = get_polarities(wordtype.argument)
        respos, resneg = get_polarities(wordtype.result)
        return argpos + respos, argneg + resneg
    else:
        raise TypeError('Expected wordtype to be of type Union[PolarizedType, FunctorType],'
                        ' received {} instead'.format(type(wordtype)))


def literal_invariance(premises: WordTypes):
    seqpos, seqneg = list(map(lambda x: reduce(add, x), tuple(zip(*map(get_polarities, premises)))))
    return Counter(seqneg) - Counter(seqpos)


def operator_count(wt: WordType) -> int:
    if isinstance(wt, AtomicType):
        return 0
    elif isinstance(wt, FunctorType):
        return operator_count(wt.result) - operator_count(wt.argument) - 1
    else:
        raise TypeError('Expected wt to be of type WordType, received {} instead.'.format(type(wt)))


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
