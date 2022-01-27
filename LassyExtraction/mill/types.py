# todo: missing correct annotation for intersection types; see https://github.com/python/typing/issues/213
# todo: accessing _registry due to __getitem__ not working in metaclasses; see https://bugs.python.org/issue35992
# todo: class-level methods for proof construction take redundant arguments
# todo: Modal is itself ABC -- await pr @ https://github.com/python/cpython/pull/27648
# todo: beta-reduction


from __future__ import annotations

from abc import ABCMeta
from typing import TypeVar, Callable
from typing import Type as TYPE
from functools import partial
from enum import Enum

########################################################################################################################
# Type Syntax
########################################################################################################################

T = TypeVar('T', bound='Type')
SerializedType = tuple[TYPE, tuple[str]] | \
                 tuple[TYPE, tuple['SerializedType', 'SerializedType']] | \
                 tuple[TYPE, tuple[str, 'SerializedType']]
SerializedProof = tuple[str, tuple[SerializedType, int]] | \
                  tuple[str, tuple['SerializedProof', 'SerializedProof']] | \
                  tuple[str, tuple[str, 'SerializedProof']] | \
                  tuple[str, tuple['SerializedProof']]


class Type(ABCMeta):
    _registry: dict[str, T]
    def __repr__(cls) -> str: return type_repr(cls)
    def order(cls) -> int: return type_order(cls)
    def __eq__(cls, other) -> bool: return type_eq(cls, other)
    def __hash__(cls) -> int: return type_hash(cls)
    def prefix(cls) -> str: return type_prefix(cls)
    def serialize_type(cls) -> SerializedType: return serialize_type(cls)

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

    @staticmethod
    def parse_prefix(string: str) -> Type: return parse_prefix(string)


class Atom(Type):
    _registry: dict[str, Atom]
    sign: str
    __match_args__ = ('sign',)

    def __new__(mcs, sign: str, bases: tuple[Type, ...] = ()) -> Atom:
        return super(Atom, mcs).__new__(mcs, sign, bases)

    def __init__(cls, sign: str, _: tuple[Type, ...] = ()) -> None:
        super(Atom, cls).__init__(cls)
        cls.sign = sign


class Functor(Type):
    _registry: dict[str, Functor]
    argument: Type
    result: Type

    __match_args__ = ('argument', 'result')

    def __new__(mcs, argument: Type, result: Type) -> Functor:
        return super(Functor, mcs).__new__(mcs, Functor.repr(argument, result))

    def __init__(cls, argument: Type, result: Type) -> None:
        super(Functor, cls).__init__(cls)
        cls.argument = argument
        cls.result = result

    @staticmethod
    def repr(argument: Type, result: Type) -> str:
        def par(x: Type) -> str: return f"({x})" if x.order() > 0 else f'{x}'
        return f'{par(argument)}⊸{result}'


class Modal(Type):
    _registry: dict[str, Modal]
    content: Type
    decoration: str

    __match_args__ = ('decoration', 'content')


class Box(Modal):
    _registry: dict[str, Box]
    content: Type
    decoration: str

    __match_args__ = ('decoration', 'content')

    def __new__(mcs, decoration: str, content: Type,) -> Box:
        return super(Box, mcs).__new__(mcs, f'□{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super(Box, cls).__init__(cls)
        cls.content = content
        cls.decoration = decoration


class Diamond(Modal):
    _registry: dict[str, Diamond]
    content: Type
    decoration: str

    __match_args__ = ('decoration', 'content')

    def __new__(mcs, decoration: str, content: Type) -> Diamond:
        return super(Diamond, mcs).__new__(mcs, f'◇{decoration}({content})')

    def __init__(cls, decoration: str, content: Type) -> None:
        super(Diamond, cls).__init__(cls)
        cls.content = content
        cls.decoration = decoration


def type_order(type_: Type) -> int:
    match type_:
        case Atom(_): return 0
        case Functor(argument, result): return max(type_order(argument) + 1, type_order(result))
        case Modal(_, content): return type_order(content)
        case _: raise ValueError(f'Unknown type: {type_}')


def type_repr(type_: Type) -> str:
    match type_:
        case Atom(sign): return sign
        case Functor(argument, result): return Functor.repr(argument, result)
        case Box(decoration, content): return f'□{decoration}({type_repr(content)})'
        case Diamond(decoration, content): return f'◇{decoration}({type_repr(content)})'
        case _: raise ValueError(f'Unknown type: {type_}')


def type_eq(type_: Type, other: Type) -> bool:
    match type_:
        case Atom(sign):
            return isinstance(other, Atom) and sign == other.sign and type_.__bases__ == other.__bases__
        case Functor(argument, result):
            return isinstance(other, Functor) and type_eq(argument, other.argument) and type_eq(result, other.result)
        case Box(decoration, content):
            return isinstance(other, Box) and decoration == other.decoration and type_eq(content, other.content)
        case Diamond(decoration, content):
            return isinstance(other, Diamond) and decoration == other.decoration and type_eq(content, other.content)
        case _: raise ValueError(f'Unknown type: {type_}')


def type_prefix(type_: Type) -> str:
    match type_:
        case Atom(sign): return sign
        case Functor(argument, result): return f'⊸ {type_prefix(argument)} {type_prefix(result)}'
        case Box(decoration, content): return f'□{decoration} {type_prefix(content)}'
        case Diamond(decoration, content): return f'◇{decoration} {type_prefix(content)}'
        case _: raise ValueError(f'Unknown type: {type_}')


def parse_prefix(string: str) -> Type:
    symbols = string.split()
    stack: list[Type] = []
    for symbol in reversed(symbols):
        if symbol == '⊸':
            return Functor(stack.pop(), stack.pop())
        if symbol.startswith('□'):
            return Box(symbol.lstrip('□'), stack.pop())
        if symbol.startswith('◇'):
            return Diamond(symbol.lstrip('◇'), stack.pop())
        stack.append(Atom(symbol))
    return stack.pop()


def type_hash(type_: Type) -> int:
    match type_:
        case Atom(sign): return hash((sign,))
        case Functor(argument, result): return hash((type_hash(argument), type_hash(result)))
        case Box(decoration, content): return hash((f'□{decoration}', type_hash(content)))
        case Diamond(decoration, content): return hash((f'◇{decoration}', type_hash(content)))
        case _: raise ValueError(f'Unknown type: {type_}')


def serialize_type(type_: Type) -> SerializedType:
    match type_:
        case Atom(sign): return Atom, (sign,)
        case Functor(argument, result): return Functor, (serialize_type(argument), serialize_type(result))
        case Box(decoration, content): return Box, (decoration, serialize_type(content))
        case Diamond(decoration, content): return Diamond, (decoration, serialize_type(content))
        case _: raise ValueError(f'Unknown type: {type_}')


def deserialize_type(serialized: SerializedType) -> Type:
    cls, args = serialized
    if cls == Atom:
        (sign,) = args
        return Atom(sign)
    if cls == Functor:
        (left, right) = args
        return Functor(deserialize_type(left), deserialize_type(right))
    if cls == Box:
        (decoration, content) = args
        return Box(decoration, deserialize_type(content))
    if cls == Diamond:
        (decoration, content) = args
        return Diamond(decoration, deserialize_type(content))
    raise ValueError(f'Unknown type: {cls}')


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
                raise TypeInference.TypeCheckError(f'{abstraction} is not free in {body}')
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

    def __eq__(self: Proof, other: Proof) -> bool:
        if type(self) == type(other) and self.rule == other.rule:
            match self.rule:
                case Proof.Rule.Lexicon:
                    return self.constant == other.constant
                case Proof.Rule.Axiom:
                    return self.variable == other.variable
                case Proof.Rule.ArrowElimination:
                    return self.function == other.function and self.argument == other.argument
                case Proof.Rule.ArrowIntroduction:
                    return self.abstraction == other.abstraction and self.body == other.body
                case Proof.Rule.BoxElimination | self.rule.BoxIntroduction:
                    return self.decoration == other.decoration and self.body == other.body
                case Proof.Rule.DiamondElimination | self.rule.DiamondIntroduction:
                    return self.decoration == other.decoration and self.body == other.body
                case _:
                    raise ValueError(f'Unrecognized rule: {self.rule}')
        return False

    def __hash__(self) -> int:
        match self.rule:
            case Proof.Rule.Lexicon: return hash((self.rule, type(self), self.constant))
            case Proof.Rule.Axiom: return hash((self.rule, type(self), self.variable))
            case Proof.Rule.ArrowElimination: return hash((self.rule, type(self), self.function, self.argument))
            case Proof.Rule.ArrowIntroduction: return hash((self.rule, type(self), self.abstraction, self.body))
            case Proof.Rule.BoxElimination | self.rule.BoxIntroduction:
                return hash((self.rule, type(self), self.decoration, self.body))
            case Proof.Rule.DiamondElimination | self.rule.DiamondIntroduction:
                return hash((self.rule, type(self), self.decoration, self.body))
            case _: raise ValueError(f'Unrecognized rule: {self.rule}')

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
    def abstract(cls: T, variable: Proof | Type, body: Proof | Type) -> Proof | Type:
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
            case Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction: return self.body.free()
            case Proof.Rule.DiamondElimination | Proof.Rule.DiamondIntroduction: return self.body.free()

    def vars(self: T) -> list[T]:
        match self.rule:
            case Proof.Rule.Lexicon: return []
            case Proof.Rule.Axiom: return [self]
            case Proof.Rule.ArrowElimination: return self.function.vars() + self.argument.vars()
            case Proof.Rule.ArrowIntroduction: return self.abstraction.vars() + self.body.vars()
            case Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction: return self.body.vars()
            case Proof.Rule.DiamondElimination | Proof.Rule.DiamondIntroduction: return self.body.vars()

    def constants(self: T) -> set[T]:
        match self.rule:
            case Proof.Rule.Lexicon: return {self}
            case Proof.Rule.Axiom: return set()
            case Proof.Rule.ArrowElimination: return self.function.constants() | self.argument.constants()
            case Proof.Rule.ArrowIntroduction: return self.body.constants()
            case Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction: return self.body.constants()
            case Proof.Rule.DiamondElimination | Proof.Rule.DiamondIntroduction: return self.body.constants()

    def translate_lex(self: T, trans: dict[int, int]) -> T:
        f = lambda x: x.translate_lex(trans)
        match self.rule:
            case Proof.Rule.Lexicon: return type(self).con(trans[c]) if (c := self.constant) in trans.keys() else self
            case Proof.Rule.Axiom: return self
            case Proof.Rule.ArrowElimination: return Proof.apply(f(self.function), f(self.argument))
            case Proof.Rule.ArrowIntroduction: return Proof.abstract(f(self.abstraction), f(self.body))
            case Proof.Rule.BoxElimination: return Proof.unbox(f(self.body))
            case Proof.Rule.BoxIntroduction: return Proof.box(self.decoration, f(self.body))
            case Proof.Rule.DiamondElimination: return Proof.undiamond(f(self.body))
            case Proof.Rule.DiamondIntroduction: return Proof.diamond(self.decoration, f(self.body))

    def canonicalize_var_names(self: T) -> set[T]:
        def translate(x: T, trans: dict[int, int]) -> tuple[T, dict[int, int]]:
            match x.rule:
                case Proof.Rule.Lexicon: return x, trans
                case Proof.Rule.Axiom:
                    return type(x).var(trans.pop(x.variable)), trans
                case Proof.Rule.ArrowElimination:
                    fn, trans = translate(x.function, trans)
                    arg, trans = translate(x.argument, trans)
                    return type(x).apply(fn, arg), trans
                case Proof.Rule.ArrowIntroduction:
                    trans |= {x.abstraction.variable:
                                  (var := next(k for k in range(999) if k not in trans.values()))}
                    body, trans = translate(x.body, trans)
                    return type(x).abstract(type(x.abstraction).var(var), body), trans
                case Proof.Rule.BoxIntroduction:
                    wrapped, trans = translate(x.body, trans)
                    return type(x).box(x.decoration, wrapped), trans
                case Proof.Rule.DiamondIntroduction:
                    wrapped, trans = translate(x.body, trans)
                    return type(x).diamond(x.decoration, wrapped), trans
                case Proof.Rule.BoxElimination:
                    wrapped, trans = translate(x.body, trans)
                    return type(x).unbox(wrapped), trans
                case Proof.Rule.DiamondElimination:
                    wrapped, trans = translate(x.body, trans)
                    return type(x).undiamond(wrapped), trans
        return translate(self, {})[0]

    def eta_norm(self: T) -> T:
        match self.rule:
            case Proof.Rule.Lexicon | Proof.Rule.Axiom: return self
            case Proof.Rule.ArrowElimination:
                return Proof.apply(self.function.eta_norm(), self.argument.eta_norm())
            case Proof.Rule.ArrowIntroduction:
                body = self.body.eta_norm()
                if body.rule == Proof.Rule.ArrowElimination and body.argument == self.abstraction:
                    return body.function
                return Proof.abstract(self.abstraction, body)
            case Proof.Rule.BoxElimination:
                return Proof.unbox(self.body.eta_norm())
            case Proof.Rule.BoxIntroduction:
                body = self.body.eta_norm()
                if body.rule == Proof.Rule.BoxElimination and body.decoration == self.decoration:
                    return body.body
                return Proof.box(self.decoration, body)
            case Proof.Rule.DiamondElimination:
                return Proof.undiamond(self.body.eta_norm())
            case Proof.Rule.DiamondIntroduction:
                body = self.body.eta_norm()
                if body.rule == Proof.Rule.DiamondElimination and body.decoration == self.decoration:
                    return body.body
                return Proof.diamond(self.decoration, body)

    def inplace_apply(self: T, arg: T) -> T: return Proof.apply(self, arg)
    def __matmul__(self: T, other: T) -> T: return self.inplace_apply(other)
    def inplace_abstract(self: T, abstraction: T) -> T: return Proof.abstract(abstraction, self)
    def __sub__(self: T, other: T) -> T: return self.inplace_abstract(other)

    def serialize(self: T) -> SerializedProof:
        name = self.rule.name
        match self.rule:
            case Proof.Rule.Lexicon: return name, (serialize_type(type(self)), self.constant,)
            case Proof.Rule.Axiom: return name, (serialize_type(type(self)), self.variable,)
            case Proof.Rule.ArrowElimination: return name, (self.function.serialize(), self.argument.serialize())
            case Proof.Rule.ArrowIntroduction: return name, (self.abstraction.serialize(), self.body.serialize())
            case Proof.Rule.BoxElimination: return name, (self.body.serialize(),)
            case Proof.Rule.BoxIntroduction: return name, (self.decoration, self.body.serialize())
            case Proof.Rule.DiamondElimination: return name, (self.body.serialize(),)
            case Proof.Rule.DiamondIntroduction: return name, (self.decoration, self.body.serialize())


def deserialize_proof(args):
    match args:
        case Proof.Rule.Lexicon.name, (wordtype, idx):
            return deserialize_type(wordtype).con(idx)
        case Proof.Rule.Axiom.name, (wordtype, idx):
            return deserialize_type(wordtype).var(idx)
        case Proof.Rule.ArrowElimination.name, (left, right):
            return Proof.apply(deserialize_proof(left), deserialize_proof(right))
        case Proof.Rule.ArrowIntroduction.name, (left, right):
            return Proof.abstract(deserialize_proof(left), deserialize_proof(right))
        case Proof.Rule.BoxElimination.name, (body,):
            return Proof.unbox(deserialize_proof(body))
        case Proof.Rule.BoxIntroduction.name, (decoration, body,):
            return Proof.box(decoration, deserialize_proof(body))
        case Proof.Rule.DiamondElimination.name, (body,):
            return Proof.undiamond(deserialize_proof(body))
        case Proof.Rule.DiamondIntroduction.name, (decoration, body,):
            return Proof.diamond(decoration, deserialize_proof(body))
        case _:
            raise ValueError(f'Cannot deserialize {args}')


def show_term(
        proof: Proof,
        show_decorations: bool = True,
        show_types: bool = True,
        word_printer: Callable[[int], str] = str) -> str:
    def f(_proof: Proof) -> str: return show_term(_proof, show_decorations, show_types, word_printer)
    def v(_proof: Proof) -> str: return show_term(_proof, show_decorations, False)
    wp = word_printer

    def needs_par(_proof: Proof) -> bool:
        match _proof.rule:
            case Proof.Rule.Axiom | Proof.Rule.Lexicon: return False
            case (Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction | Proof.Rule.DiamondElimination |
                  Proof.Rule.DiamondIntroduction): return not show_decorations and needs_par(_proof.body)
            case _: return True

    match proof.rule:
        case Proof.Rule.Lexicon:
            return f'{wp(proof.constant)}' if not show_types else f'{wp(proof.constant)}::{type(proof)}'
        case Proof.Rule.Axiom:
            return f'x{proof.variable}' if not show_types else f'x{proof.variable}::{type(proof)}'
        case Proof.Rule.ArrowElimination:
            fn, arg = proof.function, proof.argument
            return f'{f(fn)} ({f(arg)})' if needs_par(arg) else f'{f(fn)} {f(arg)}'
        case Proof.Rule.ArrowIntroduction:
            var, body = proof.abstraction, proof.body
            return f'λ{v(var)}.({f(body)})' if needs_par(body) else f'λ{v(var)}.({f(body)})'
        case Proof.Rule.BoxElimination:
            return f'▾{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
        case Proof.Rule.BoxIntroduction:
            return f'▴{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
        case Proof.Rule.DiamondElimination:
            return f'▿{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
        case Proof.Rule.DiamondIntroduction:
            return f'▵{proof.decoration}({f(proof.body)})' if show_decorations else f(proof.body)
