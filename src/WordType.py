from abc import ABC, abstractmethod
from functools import reduce
from typing import Union, Set, Sequence, Tuple, Iterable
from collections import defaultdict
from itertools import chain


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
    def get_atomic(self) -> Set[str]:
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

    def get_atomic(self) -> Set[str]:
        return {self.__repr__()}

    def get_colors(self) -> Set[str]:
        return set()


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

    def get_atomic(self) -> Set[str]:
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

    def get_atomic(self) -> Set[str]:
        return set.union(self.argument.get_atomic(), self.result.get_atomic())

    def get_colors(self) -> Set[str]:
        return set.union(self.argument.get_colors(), self.result.get_colors())


class DirectedComplexType(ComplexType):
    def __init__(self, argument: WordType, result: WordType, direction: str) -> None:
        super(DirectedComplexType, self).__init__(argument, result)
        if direction == 'left':
            self.dir_symbol = '\\'
        elif direction == 'right':
            self.dir_symbol = '/'
        else:
            raise ValueError('Invalid direction given ({}). Expected one of "left", "right"'.format(direction))
        self.direction = direction

    def __str__(self) -> str:
        if self.direction == 'right':
            return str(self.argument) + ' ' + self.dir_symbol + ' ' + str(self.result)
        else:
            return str(self.result) + ' ' + self.dir_symbol + ' ' + str(self.argument)

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, DirectedComplexType):
            return False
        else:
            return super(DirectedComplexType, self).__eq__(other) and self.direction == other.direction


class ColoredType(ComplexType):
    def __init__(self, argument: WordType, result: WordType, color: str):
        super(ColoredType, self).__init__(argument, result)
        assert(isinstance(argument, WordType))
        assert(isinstance(result, WordType))
        assert(isinstance(color, str))
        self.color = color

    def __str__(self) -> str:
        return '{' + str(self.argument) + ': ' + self.color + '} → ' + str(self.result)

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


class DirectedColoredType(DirectedComplexType):
    def __init__(self, argument: WordType, result: WordType, color: str, direction: str):
        super(DirectedColoredType, self).__init__(argument, result, direction)
        self.color = color

    def __str__(self) -> str:
        if self.direction == 'right':
            compact = '{' + str(self.argument) + '}: ' + self.color + ' ' + self.dir_symbol + ' ' + str(self.result)
            return compact if isinstance(self.argument, AtomicType) else '(' + compact + ')'
        else:
            compact = str(self.result) + ' ' + self.dir_symbol + '{' + str(self.argument) + '}: ' + self.color
            return compact if isinstance(self.argument, AtomicType) else '(' + compact + ')'

    def __eq__(self, other: WordType) -> bool:
        if not isinstance(other, DirectedColoredType):
            return False
        else:
            return super(DirectedColoredType, self).__eq__(other) and self.color == other.color

    def decolor(self) -> DirectedComplexType:
        return DirectedComplexType(argument=self.argument.decolor(), result=self.result.decolor(),
                                   direction=self.direction)

    def get_atomic(self):
        return ComplexType.get_atomic(self)

    def get_colors(self):
        return ColoredType.get_colors(self)


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

    def get_atomic(self) -> Set[str]:
        return reduce(set.union, [a.get_atomic() for a in self.types])

    def get_colors(self) -> Set[str]:
        return reduce(set.union, [a.get_colors() for a in self.types])


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


def get_atomic(something: Union[WordTypes, WordType]) -> Set[str]:
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


def non_poly_kleene_star_type_constructor(arguments: WordTypes, result: WordType, colors: strings) -> ColoredType:
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

