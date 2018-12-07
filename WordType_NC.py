from abc import ABC, abstractmethod
from functools import reduce

class WordType(ABC):

    @abstractmethod
    def __str__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def __hash__(self):
        pass

    @abstractmethod
    def get_arity(self):
        pass

    @abstractmethod
    def __call__(self):
        pass

    @abstractmethod
    def __eq__(self, other):
        pass


class AtomicType(WordType):
    def __init__(self, result):
        if not isinstance(result, str):
            raise TypeError('Expected result to be of type str, received {} instead.'.format(type(result)))
        self.result = result

    def __str__(self):
        return self.result

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__()

    def get_arity(self):
        return 0

    def __call__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, AtomicType):
            return False
        else:
            return self.result == other.result


class ModalType(WordType):
    def __init__(self, result, modality):
        if not isinstance(result, WordType):
            raise TypeError('Expected result to be of type WordType, received {} instead.'.format(type(result)))
        self.result = result
        if not isinstance(modality, str):
            raise TypeError('Expected modality to be of type str, received {} instead.'.format(type(modality)))
        self.modality = modality

    def __str__(self):
        if self.result.get_arity():
            return self.modality + '(' + str(self.result) + ')'
        else:
            return self.modality + str(self.result)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__()

    def get_arity(self):
        return self.result.get_arity()

    def __call__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, ModalType):
            return False
        else:
            return self.modality == other.modality and self.result == other.result


class ComplexType(WordType):
    def __init__(self, argument, result):
        if not isinstance(result, WordType):
            raise TypeError('Expected result to be of type WordType, received {} instead.'.format(type(result)))
        self.result = result
        if not isinstance(argument, WordType):
            raise TypeError('Expected argument to be of type WordType, received {} instead.'.format(type(argument)))
        self.argument = argument

    def __str__(self):
        if self.argument.get_arity():
            return '(' + str(self.argument) + ')' + ' → ' + str(self.result)
        else:
            return str(self.argument) + ' → ' + str(self.result)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__()

    def get_arity(self):
        return self.argument.get_arity() + 1 + self.result.get_arity()

    def __call__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, ComplexType):
            return False
        else:
            return self.argument == other.argument and self.result == other.result


class ColoredType(ComplexType):
    def __init__(self, argument, result, color):
        super(ColoredType, self).__init__(argument, result)
        if not isinstance(color, str):
            raise TypeError('Expected color to be of type str, received {} instead.'.format(type(color)))
        self.color = color

    def __str__(self):
        return '{' + str(self.argument) + ': ' + self.color + '} → ' + str(self.result)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return self.__str__()

    def __eq__(self, other):
        if not isinstance(other, ColoredType):
            return False
        else:
            return self.color == other.color and self.argument == other.argument and self.result == other.result


def compose(base_types, base_colors, result):
    if len(base_types) != len(base_colors):
        raise ValueError('Uneven number of types ({}) and colors ({}).'.format(len(base_types), len(base_colors)))
    return reduce(lambda x, y: ColoredType(result=x, argument=y[0], color=y[1]),
                  zip(base_types[::-1], base_colors[::-1]),
                  result)


def decolor(colored_type):
    if not isinstance(colored_type, WordType):
        raise TypeError('Expected input of type WordType, received {} instead.'.format(type(colored_type)))
    if isinstance(colored_type, ColoredType) or isinstance(colored_type, ComplexType):
        return ComplexType(decolor(colored_type.argument), decolor(colored_type.result))
    elif isinstance(colored_type, ModalType):
        return ModalType(decolor(colored_type.result), modality=colored_type.modality)
    elif isinstance(colored_type, AtomicType):
        return colored_type
