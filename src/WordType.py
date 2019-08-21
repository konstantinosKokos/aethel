from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Set, Sequence, Tuple, Iterable, List
from collections import defaultdict, Counter
from itertools import chain
from operator import add


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
    def get_atomic(self) -> Set['AtomicType']:
        pass

    @abstractmethod
    def get_colors(self) -> Set[str]:
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

    def get_atomic(self) -> Set['AtomicType']:
        return {self}

    def get_colors(self) -> Set[str]:
        return set()

    def depolarize(self) -> 'AtomicType':
        return self


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

    def get_atomic(self) -> Set[AtomicType]:
        return self.result.get_atomic()

    def get_colors(self) -> Set[str]:
        return self.result.get_colors()


class ComplexType(WordType):
    def __init__(self, argument: WordType, result: WordType) -> None:
        self.result = result
        self.argument = argument

    def __str__(self) -> str:
        return str(self.argument) + ' → ' + str(self.result)

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


class ColoredType(ComplexType):
    def __init__(self, argument: WordType, result: WordType, color: str):
        super(ColoredType, self).__init__(argument, result)
        assert(isinstance(argument, WordType))
        assert(isinstance(result, WordType))
        assert(isinstance(color, str))
        self.color = color

    def __str__(self) -> str:
        return '<' + str(self.argument) + '> ' + self.color + ' → ' + str(self.result)

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


class CombinatorType(WordType):
    def __init__(self, types: WordTypes, combinator: str):
        self.types = types
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

    def get_atomic(self) -> Set[AtomicType]:
        return reduce(set.union, [a.get_atomic() for a in self.types])

    def get_colors(self) -> Set[str]:
        return reduce(set.union, [a.get_colors() for a in self.types])


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


def polarize_and_index(w: WordType, polarity: bool = True, index: int = 0) -> Tuple[int, WordType]:
    if isinstance(w, AtomicType):
        return index+1, PolarizedIndexedType(result=w.result, polarity=polarity, index=index)
    elif isinstance(w, ComplexType):
        index, arg = polarize_and_index(w.argument, not polarity, index)
        index, res = polarize_and_index(w.result, polarity, index)
        if isinstance(w, ColoredType):
            return index, ColoredType(argument=arg, result=res, color=w.color)
        else:
            return index, ComplexType(argument=arg, result=res)


def polarize_and_index_many(W: Sequence[WordType], index: int=0) -> Tuple[int, Sequence[WordType]]:
    ret = []
    for w in W:
        index, x = polarize_and_index(w, True, index)
        ret.append(x)
    return index, ret


def binarizer(arguments: WordTypes, result: WordType, colors: strings) -> ColoredType:
    argcolors = zip(arguments, colors)
    argcolors = list(dependency_sort(argcolors))
    return reduce(lambda x, y: ColoredType(result=x, argument=y[0], color=y[1]), argcolors, result)


def flatten_binary(arguments: WordTypes, result: WordType, colors: strings) -> ColoredType:
    x = result
    while isinstance(x, ColoredType) and x.color not in ('mod', 'predm', 'app'):
        arguments += (x.argument,)
        colors += (x.color,)
        x = x.result
    return binarizer(arguments, x, colors)


def dependency_sort(argcolors: Iterable[Tuple[WordType, str]]) -> Sequence[Tuple[WordType, str]]:
    obliqueness_order = (  # upper is inner
        ('mod', 'app', 'predm'),  # modifiers
        ('body', 'rhd_body', 'whd_body'),  # clause bodies
        ('svp',),  # phrasal verb part
        ('ld', 'me', 'vc'),  # verb complements
        ('predc', 'obj2', 'se', 'pc', 'hdf'),  # verb secondary arguments
        ('obj1',),  # primary object
        ('pobj',),  # preliminary object
        ('su',),  # primary subject
        ('sup',),  # preliminary subject
        ('invdet',),  # NP head
    )
    priority = {k: i for i, k in enumerate(reversed(list(chain.from_iterable(obliqueness_order))))}
    priority = defaultdict(lambda: -1, {**priority, **{'cnj': -2}})
    return sorted(argcolors, key=lambda x: (priority[x[1]], str(x[0])), reverse=True)


def decolor(colored_type: WordType) -> Union[AtomicType, ModalType, ComplexType]:
    return colored_type.decolor()


def get_atomic(something: Union[WordTypes, WordType]) -> Set[AtomicType]:
    if isinstance(something, Sequence):
        return reduce(set.union, [get_atomic(s) for s in something])
    else:
        return something.get_atomic()


def get_colors(something: Union[WordTypes, WordType]) -> Set[str]:
    if isinstance(something, Sequence):
        return reduce(set.union, [get_colors(s) for s in something])
    else:
        return something.get_colors()


def kleene_star_type_constructor(arguments: WordTypes, result: WordType, colors: strings) -> ColoredType:
    if all(list(map(lambda x: x == 'cnj', colors))) and len(set(arguments)) == 1:
        return ColoredType(ModalType(arguments[0], modality='*'), result, 'cnj')
    else:
        return binarizer(arguments, result, colors)


def multi_arg_kleene_star_constructor(arguments: WordTypes, result: WordType, colors: strings) -> ColoredType:
    if all(list(map(lambda x: x == 'cnj', colors))):
        arguments = tuple((set(arguments)))
        arguments = tuple(map(lambda x: ModalType(x, modality='*'), arguments))
    return flatten_binary(arguments, result, colors)


def polish(wordtype: WordType) -> str:
    if isinstance(wordtype, ColoredType):
        return wordtype.color + ' ' + polish(wordtype.argument) + ' ' + polish(wordtype.result)
    elif isinstance(wordtype, AtomicType):
        return str(wordtype)
    elif isinstance(wordtype, CombinatorType):
        if len(wordtype.types) != 2:
            raise NotImplementedError('Polish not implemented for {}-ary CombinatorTypes'.format(len(wordtype.types)))
        return wordtype.combinator + ' ' + polish(wordtype.types[0]) + ' ' + polish(wordtype.types[1])
    elif isinstance(wordtype, ModalType):
        return wordtype.modality + ' ' + polish(wordtype.result)
    else:
        raise NotImplementedError('Polish not implemented for {}'.format(type(wordtype)))


def associative_combinator(types: WordTypes, combinator: str) -> CombinatorType:
    new_types = tuple(set(chain.from_iterable([t.types if isinstance(t, CombinatorType) and t.combinator == combinator
                                              else [t] for t in types])))
    new_types = sorted(new_types, key=lambda x: str(x))
    return CombinatorType(new_types, combinator)


def rightwards_inclusion(left: WordType, right: WordType) -> bool:
    if isinstance(right, CombinatorType):
        return any([left == t for t in right.types])
    else:
        return left == right


def get_polarities(wordtype: WordType) -> Tuple[List[AtomicType], List[AtomicType]]:
    if isinstance(wordtype, AtomicType):
        if str(wordtype)[0] == '_':
            return [], []
        return [], [wordtype.depolarize()]
    elif isinstance(wordtype, ComplexType):
        argneg, argpos = get_polarities(wordtype.argument)
        respos, resneg = get_polarities(wordtype.result)
        return argpos + respos, argneg + resneg
    elif isinstance(wordtype, ModalType):
        return get_polarities(wordtype.result)


def literal_invariance(premises: Sequence[WordType]):
    seqpos, seqneg = list(map(lambda x: reduce(add, x), tuple(zip(*map(get_polarities, premises)))))
    return Counter(seqneg) - Counter(seqpos)


def operator_count(wt: WordType) -> int:
    return 0 if isinstance(wt, AtomicType) else operator_count(wt.result) - operator_count(wt.argument) - 1


def operator_invariance(premises: Sequence[WordType]) -> int:
    return reduce(add, map(operator_count, premises)) + len(premises) - 1


def typecheck(premises: Sequence[WordType], goal: WordType) -> bool:
    inferred = literal_invariance(premises)
    if list(inferred.values()) != [1]:
        return False
    elif list(inferred.keys()) != [goal]:
        return False
    return True


def str_to_type(x: Sequence[str], colors: Set[str]) -> Tuple[WordType, Sequence[str]]:
    if x[0] in colors:
        color = x[0]
        arg, rem = str_to_type(x[1:], colors)
        res, rem = str_to_type(rem, colors)
        return ColoredType(color=color, argument=arg, result=res), rem
    else:
        return AtomicType(x[0]), x[1:]


class StrToType(object):
    def __init__(self, colors: Set[str]):
        self.colors = colors

    def __call__(self, x: Sequence[str]) -> WordType:
        return str_to_type(x, self.colors)[0]

