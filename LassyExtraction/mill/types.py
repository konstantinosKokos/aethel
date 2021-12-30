# todo: missing correct annotation for intersection types; see https://github.com/python/typing/issues/213
# todo: accessing _registry due to __getitem__ not working in metaclasses; see https://bugs.python.org/issue35992
# todo: class-level methods for proof construction take redundant arguments
# todo: beta-reduction

from __future__ import annotations
from abc import abstractmethod, ABCMeta
from typing import TypeVar, Callable, overload
from functools import partial
from enum import Enum


########################################################################################################################
# Type Syntax
########################################################################################################################

T = TypeVar('T', bound='Type')


class Type(ABCMeta):
    _registry: dict[str, T]

    @abstractmethod
    def __repr__(cls) -> str: ...

    @abstractmethod
    def __eq__(cls, other) -> bool: ...

    @abstractmethod
    def __hash__(cls) -> int: ...

    @abstractmethod
    def order(cls) -> int: ...

    @classmethod
    def __init_subclass__(mcs, **kwargs):
        super(Type, mcs).__init_subclass__(**kwargs)
        mcs._registry = {}

    def __new__(mcs, name, bases: tuple[Type, ...] = ()) -> T:
        if name in mcs._registry:
            return mcs._registry[name]
        ret = super(Type, mcs).__new__(mcs, name, (*bases, Proof), {})
        mcs._registry[name] = ret
        return ret


class Atom(Type):
    _registry: dict[str, Atom]
    sign: str

    __match_args__ = ('sign',)

    def __new__(mcs, sign: str, bases: tuple[Type, ...] = ()) -> Atom:
        return super(Atom, mcs).__new__(mcs, sign, bases)

    def __init__(cls, sign: str, bases: tuple[Type, ...] = ()) -> None:
        super().__init__(cls)
        cls.sign = sign

    def __repr__(cls) -> str: return cls.sign

    def __eq__(cls, other) -> bool: return isinstance(other, Atom) and cls.sign == other.sign

    def __hash__(cls) -> int: return hash(cls.sign)

    def order(cls): return 0


class Functor(Type):
    _registry: dict[str, Functor]
    argument: Type
    result: Type

    __match_args__ = ('argument', 'result')

    def __new__(mcs, argument: Type, result: Type) -> Functor:
        return super(Functor, mcs).__new__(mcs, Functor.print(argument, result))

    def __init__(cls, argument: Type, result: Type) -> None:
        super().__init__(cls)
        cls.argument = argument
        cls.result = result

    def __repr__(cls) -> str: return Functor.print(cls.argument, cls.result)

    def __eq__(cls, other):
        return isinstance(other, Functor) and cls.argument == other.argument and cls.result == other.result

    def __hash__(cls): return hash((cls.argument, cls.result))

    def order(cls): return max(cls.argument.order() + 1, cls.result.order())

    @staticmethod
    def print(argument: Type, result: Type) -> str:
        def par(x: Type) -> str: return f"({x})" if x.order() > 0 else f'{x}'
        return f'{par(argument)}⊸{result}'


class Modal(Type):
    _registry: dict[str, Modal]
    content: Type
    decoration: str

    # this is an ABC: await pr @ https: // github.com / python / cpython / pull / 27648
    __match_args__ = ('content', 'decoration')

    def __eq__(self, other) -> bool:
        return isinstance(other, type(self)) and self.content == other.content and self.decoration == other.decoration

    def __hash__(cls) -> int: return hash((cls.content, cls.decoration, type(cls)))

    def order(cls): return cls.content.order()


class Box(Modal):
    _registry: dict[str, Box]
    content: Type
    decoration: str

    def __new__(mcs, decoration: str, content: Type,) -> Box:
        return super(Box, mcs).__new__(mcs, f'□{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super().__init__(cls)
        cls.content = content
        cls.decoration = decoration

    def __repr__(cls) -> str: return f'□{cls.decoration}({cls.content})'


class Diamond(Modal):
    _registry: dict[str, Diamond]
    content: Type
    decoration: str

    def __new__(mcs, decoration: str, content: Type) -> Diamond:
        return super(Diamond, mcs).__new__(mcs, f'<>{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super().__init__(cls)
        cls.content = content
        cls.decoration = decoration

    def __repr__(cls) -> str: return f'◇{cls.decoration}({cls.content})'


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
    def arrow_elim(functor: Type, argument: Type) -> Type | Proof:
        if not isinstance(functor, Functor) or functor.argument != argument:
            raise TypeInference.TypeCheckError(f'{functor} is not a functor of {argument}')
        return functor.result

    @staticmethod
    def arrow_intro(variable: Type, body: Type) -> Type | Proof:
        return Functor(variable, body)

    @staticmethod
    def box_elim(wrapped: Type, key: str | None = None) -> Type | Proof:
        if not isinstance(wrapped, Box):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a box')
        if key is not None and key != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a box with {key}')
        return wrapped.content

    @staticmethod
    def diamond_elim(wrapped: Type, key: str | None = None) -> Type | Proof:
        if not isinstance(wrapped, Diamond):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a diamond')
        if key is not None and key != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a diamond with {key}')
        return wrapped.content

    @staticmethod
    def box_intro(decoration: str, content: Type) -> Type | Proof:
        return Box(decoration, content)

    @staticmethod
    def diamond_intro(decoration: str, content: Type) -> Type | Proof:
        return Diamond(decoration, content)


########################################################################################################################
# Term Syntax
########################################################################################################################


class Proof:
    rule: Rule
    constant: int
    variable: int
    function: Proof
    argument: Proof
    decoration: str
    abstraction: Proof
    body: Proof

    class Rule(Enum):
        @staticmethod
        def _init_lexicon(self, constant: int):
            self.constant = constant

        @staticmethod
        def _init_axiom(self, variable: int):
            self.variable = variable

        @staticmethod
        def _init_arrow_elim(self, function: Proof, argument: Proof):
            TypeInference.assert_equal(TypeInference.arrow_elim(type(function), type(argument)), type(self))
            self.function = function
            self.argument = argument

        @staticmethod
        def _init_arrow_intro(self, abstraction: Proof, body: Proof):
            TypeInference.assert_equal(TypeInference.arrow_intro(type(abstraction), type(body)), type(self))
            if abstraction not in body.free():
                raise TypeInference.TypeCheckError(f'{abstraction} is free in {body}')
            self.abstraction = abstraction
            self.body = body

        @staticmethod
        def _init_box_elim(self, box: str, body: Proof):
            TypeInference.assert_equal(Box(box, type(self)), type(body))
            self.body = body
            self.decoration = box

        @staticmethod
        def _init_box_intro(self, box: str, body: Proof):
            TypeInference.assert_equal(Box(box, type(body)), type(self))
            self.body = body
            self.decoration = box

        @staticmethod
        def _init_diamond_elim(self, diamond: str, body: Proof):
            TypeInference.assert_equal(Diamond(diamond, type(self)), type(body))
            self.body = body
            self.decoration = diamond

        @staticmethod
        def _init_diamond_intro(self, diamond: str, body: Proof):
            TypeInference.assert_equal(Diamond(diamond, type(body)), type(self))
            self.body = body
            self.decoration = diamond

        Lexicon = partial(_init_lexicon)
        Axiom = partial(_init_axiom)
        ArrowElimination = partial(_init_arrow_elim)
        ArrowIntroduction = partial(_init_arrow_intro)
        BoxElimination = partial(_init_box_elim)
        BoxIntroduction = partial(_init_box_intro)
        DiamondElimination = partial(_init_diamond_elim)
        DiamondIntroduction = partial(_init_diamond_intro)

    def __init__(self: Proof | Type, *args, rule: Rule, **kwargs):
        self.rule = rule
        self.rule.value(self, *args, **kwargs)

    def __eq__(self: Proof, other: Proof):
        if type(self) == type(other) and self.rule == other.rule:
            match self.rule:
                case Proof.Rule.Lexicon:
                    return self.constant == other.constant
                case Proof.Rule.Axiom:
                    return self.variable == other.variable
                case Proof.Rule.ArrowElimination:
                    return self.function == other.function and self.argument == other.argument
                case Proof.Rule.ArrowIntroduction:
                    return self.variable == other.variable and self.body == other.body
                case Proof.Rule.BoxElimination | self.rule.BoxIntroduction:
                    return self.decoration == other.decoration and self.body == other.body
                case Proof.Rule.DiamondElimination | self.rule.DiamondIntroduction:
                    return self.decoration == other.decoration and self.body == other.body
                case _:
                    return ValueError(f"Unrecognized rule: {self.rule}")
        return False

    def __hash__(self) -> int:
        match self.rule:
            case Proof.Rule.Lexicon: return hash((self.rule, type(self), self.constant))
            case Proof.Rule.Axiom: return hash((self.rule, type(self), self.variable))
            case Proof.Rule.ArrowElimination: return hash((self.rule, type(self), self.function, self.argument))
            case Proof.Rule.ArrowIntroduction: return hash((self.rule, type(self), self.variable, self.body))
            case Proof.Rule.BoxElimination | self.rule.BoxIntroduction:
                return hash((self.rule, type(self), self.decoration, self.body))
            case Proof.Rule.DiamondElimination | self.rule.DiamondIntroduction:
                return hash((self.rule, type(self), self.decoration, self.body))
            case _: raise ValueError(f"Unrecognized rule: {self.rule}")

    @classmethod
    def con(cls: T, c: int) -> T:
        if isinstance(cls, Type):
            return type(cls)._registry[cls.__name__](c, rule=Proof.Rule.Lexicon)
        raise TypeInference.TypeCheckError('Cannot instantiate an untyped constant')

    @classmethod
    def var(cls: T, v: int) -> T:
        if isinstance(cls, Type):
            return type(cls)._registry[cls.__name__](v, rule=Proof.Rule.Axiom)
        raise TypeInference.TypeCheckError('Cannot instantiate an untyped variable')

    @classmethod
    def apply(cls: T, function: T, argument: T) -> T:
        if isinstance(cls, Type):
            return cls(function, argument, rule=Proof.Rule.ArrowElimination)
        return TypeInference.arrow_elim(type(function), type(argument)).apply(function, argument)

    @classmethod
    def abstract(cls: T, variable: Proof | Type | int, body: Proof | Type) -> Proof | Type:
        if isinstance(cls, Type):
            return cls(variable, body, rule=Proof.Rule.ArrowIntroduction)
        return TypeInference.arrow_intro(type(variable), type(body)).abstract(variable, body)

    @classmethod
    def box(cls: Proof | Type, box: str, body: Proof | Type) -> Proof | Type:
        if isinstance(cls, Type):
            return cls(box, body, rule=Proof.Rule.BoxIntroduction)
        return TypeInference.box_intro(box, type(body)).box(box, body)

    @classmethod
    def diamond(cls: Proof | Type, diamond: str, body: Proof | Type) -> Proof | Type:
        if isinstance(cls, Type):
            return cls(diamond, body, rule=Proof.Rule.DiamondIntroduction)
        return TypeInference.diamond_intro(diamond, type(body)).diamond(diamond, body)

    @classmethod
    def unbox(cls: Proof | Type, body: Proof | Type) -> Proof | Type:
        if isinstance(cls, Type):
            return cls(type(body).decoration, body, rule=Proof.Rule.BoxElimination)
        return TypeInference.box_elim(type(body)).unbox(body)

    @classmethod
    def undiamond(cls: Proof | Type, body: Proof | Type) -> Proof | Type:
        if isinstance(cls, Type):
            return cls(type(body).decoration, body, rule=Proof.Rule.DiamondElimination)
        return TypeInference.diamond_elim(type(body)).undiamond(body)

    def __repr__(self) -> str: return show_term(self)

    def free(self: T) -> list[T]:
        match self.rule:
            case Proof.Rule.Lexicon: return []
            case Proof.Rule.Axiom: return [self]
            case Proof.Rule.ArrowElimination: return self.function.free() + self.argument.free()
            case Proof.Rule.ArrowIntroduction: return [f for f in self.body.free() if f != self.abstraction]
            case Proof.Rule.BoxElimination | self.rule.BoxIntroduction: return self.body.free()
            case Proof.Rule.DiamondElimination | self.rule.DiamondIntroduction: return self.body.free()
            case _: raise ValueError(f"Unrecognized rule: {self.rule}")


def show_term(proof: Proof, show_decorations: bool = True, show_types: bool = True,
              word_printer: Callable[[int], str] = str) -> str:
    def f(_proof: Proof) -> str: return show_term(_proof, show_decorations, show_types)
    def v(_proof: Proof) -> str: return show_term(_proof, show_decorations, False)

    def needs_par(_proof: Proof) -> bool:
        match _proof.rule:
            case Proof.Rule.Axiom | Proof.Rule.Lexicon: return False
            case (Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction | Proof.Rule.DiamondElimination |
                  Proof.Rule.DiamondIntroduction): return not show_decorations and needs_par(_proof.body)
            case _: return True

    match proof.rule:
        case Proof.Rule.Lexicon:
            return f'{word_printer(proof.constant)}' if not show_types else f'{proof.constant}::{type(proof)}'
        case Proof.Rule.Axiom:
            return f'x{proof.variable}' if not show_types else f'x{proof.variable}::{type(proof)}'
        case Proof.Rule.ArrowElimination:
            fn, arg = proof.function, proof.argument
            return f'{f(fn)} ({f(arg)})' if needs_par(arg) else f'{f(fn)} {f(arg)}'
        case Proof.Rule.ArrowIntroduction:
            var, body = proof.abstraction, proof.body
            return f'λ{v(var)}.({f(body)})' if needs_par(body) else f'λ{v(var)}.({f(body)})'
        case Proof.Rule.BoxElimination:
            return f'▾{proof.decoration}({f(proof.body)})'
        case Proof.Rule.BoxIntroduction:
            return f'▴{proof.decoration}({f(proof.body)})'
        case Proof.Rule.DiamondElimination:
            return f'▿{proof.decoration}({f(proof.body)})'
        case Proof.Rule.DiamondIntroduction:
            return f'▵{proof.decoration}({f(proof.body)})'
        case _:
            raise ValueError(f'Unrecognized rule: {proof.rule}')
