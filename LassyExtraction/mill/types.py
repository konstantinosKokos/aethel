# todo: missing correct annotation for intersection types; see https://github.com/python/typing/issues/213
# todo: accessing _registry due to __getitem__ not working in metaclasses; see https://bugs.python.org/issue35992
# todo: class-level methods for proof construction take redundant arguments
# todo: Modal is itself ABC -- await pr @ https://github.com/python/cpython/pull/27648


from __future__ import annotations

import pdb
from abc import ABCMeta
from typing import TypeVar, Callable
from typing import Type as TYPE
from typing import Optional as Maybe
from functools import partial, reduce
from enum import Enum
from itertools import product as prod

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

    ####################################################################################################################
    # Type-level shortcuts
    ####################################################################################################################
    def var(self: Type, var: int) -> Proof:
        return self(Proof.Rule.Variable, variable=var)

    def lex(self: Type, lex: int) -> Proof:
        return self(Proof.Rule.Lexicon, constant=lex)


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


########################################################################################################################
# Type utilities
########################################################################################################################
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
    def arrow_elim(functor: Type, argument: Type) -> Type:
        if not isinstance(functor, Functor) or functor.argument != argument:
            raise TypeInference.TypeCheckError(f'{functor} is not a functor of {argument}')
        return functor.result

    @staticmethod
    def box_elim(wrapped: Type, box: str | None = None) -> Type:
        if not isinstance(wrapped, Box):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a box')
        if box is not None and box != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a {box}-box')
        return wrapped.content

    @staticmethod
    def dia_elim(wrapped: Type, dia: str | None = None) -> Type:
        if not isinstance(wrapped, Diamond):
            raise TypeInference.TypeCheckError(f'{wrapped} is not a diamond')
        if dia is not None and dia != wrapped.decoration:
            raise TypeInference.TypeCheckError(f'{wrapped} is not a {dia}-diamond')
        return wrapped.content


########################################################################################################################
# Structural brackets
########################################################################################################################
class Bracket(Enum):
    Lock = '<>'
    Inner = ']['
    Outer = '[]'
    def __repr__(self) -> str: return self.name


IndexedBracket = tuple[Bracket, str]
def indexed_bracket(x: _TreeOfTs[IndexedBracket]) -> bool: return isinstance(x[0], Bracket)


def bracket_cancellations(outer: IndexedBracket, inner: IndexedBracket) -> bool:
    match outer, inner:
        case (Bracket.Lock, outer_mode), (Bracket.Inner, inner_mode):
            return outer_mode == inner_mode
        case (Bracket.Outer, outer_mode), (Bracket.Lock, inner_mode):
            return outer_mode == inner_mode
        case _:
            return False


_T = TypeVar('_T')
_TreeOfTs = _T | tuple['_TreeOfTs', '_TreeOfTs']


def associahedron(xs: list[_T]) -> list[_TreeOfTs]:
    def splits(_xs: list[_T]) -> list[tuple[list[_T], list[_T]]]:
        match _xs:
            case []: return [([], [])]
            case [_x]: return [([_x], []), ([], [_x])]
            case _: return [(_xs[:i], _xs[i:]) for i in range(1, len(_xs))]
    match xs:
        case []: return []
        case [x]: return [x]
        case _: return reduce(list.__add__, [list(prod(associahedron(s[0]), associahedron(s[1]))) for s in splits(xs)])


def _collapse(_tree: _TreeOfTs[IndexedBracket]) -> Maybe[_TreeOfTs[IndexedBracket]]:
    if indexed_bracket(_tree):
        return _tree
    left, right = _tree
    col_left, col_right = _collapse(left), _collapse(right)
    match col_left, col_right:
        case None, None:
            return None
        case None, _:
            return col_right
        case _, None:
            return col_left
        case _:
            if indexed_bracket(col_left) and indexed_bracket(col_right) and bracket_cancellations(col_left, col_right):
                return None
            return col_left, col_right


def _flatten(_tree: _TreeOfTs[IndexedBracket]) -> list[IndexedBracket]:
    if indexed_bracket(_tree):
        return [_tree]
    left, right = _tree
    return _flatten(left) + _flatten(right)


def cancels_out(tree: _TreeOfTs[IndexedBracket]) -> bool:
    return _collapse(tree) is None


def is_positive(tree: _TreeOfTs[IndexedBracket]) -> bool:
    collapsed = _collapse(tree)

    def go(_tree: _TreeOfTs[IndexedBracket]) -> bool:
        left, right = _tree
        return tree[0] == Bracket.Lock if indexed_bracket(_tree) else go(left) and go(right)
    return go(collapsed) if collapsed is not None else True


########################################################################################################################
# Term Syntax
########################################################################################################################
class Proof:
    rule: Rule
    decoration: str
    constant: int
    variable: int
    abstraction: Proof
    function: Proof
    argument: Proof
    body: Proof
    abstraction: Proof

    ####################################################################################################################
    # Constructors
    ####################################################################################################################
    class Rule(Enum):
        @staticmethod
        def _init_lex(proof: Proof, constant: int) -> None:
            proof.constant = constant

        @staticmethod
        def _init_var(proof: Proof, variable: int) -> None:
            proof.variable = variable

        @staticmethod
        def _init_arrow_elim(proof: Proof, function: Proof, argument: Proof) -> None:
            rest = TypeInference.arrow_elim(function.type, argument.type)
            TypeInference.assert_equal(rest, proof.type)
            if function.is_negative():
                raise TypeInference.TypeCheckError(f'{function} is a negative structure')
            if argument.is_negative():
                raise TypeInference.TypeCheckError(f'{argument} is a negative structure')
            proof.function = function
            proof.argument = argument

        @staticmethod
        def _init_arrow_intro(proof: Proof, abstraction: Proof, body: Proof) -> None:
            TypeInference.assert_equal(Functor(abstraction.type, body.type), proof.type)
            if abstraction not in body.vars():
                raise TypeInference.TypeCheckError(f'{abstraction} does not occur in {body}')
            if abstraction not in body.free():
                raise TypeInference.TypeCheckError(f'{abstraction} structurally locked in {body}')
            proof.abstraction = abstraction
            proof.body = body

        @staticmethod
        def _init_box_elim(proof: Proof, box: str, body: Proof) -> None:
            TypeInference.assert_equal(Box(box, proof.type), body.type)
            proof.decoration = box
            proof.body = body

        @staticmethod
        def _init_dia_intro(proof: Proof, diamond: str, body: Proof) -> None:
            TypeInference.assert_equal(Diamond(diamond, body.type), proof.type)
            proof.decoration = diamond
            proof.body = body

        @staticmethod
        def _init_box_intro(proof: Proof, box: str, body: Proof) -> None:
            TypeInference.assert_equal(Box(box, body.type), proof.type)
            proof.decoration = box
            proof.body = body

        @staticmethod
        def _init_dia_elim(proof: Proof, diamond: str, body: Proof) -> None:
            TypeInference.assert_equal(Diamond(diamond, proof.type), body.type)
            proof.decoration = diamond
            proof.body = body

        # Enumeration of rules
        Lexicon = partial(_init_lex)
        Variable = partial(_init_var)
        ArrowElimination = partial(_init_arrow_elim)
        ArrowIntroduction = partial(_init_arrow_intro)
        BoxElimination = partial(_init_box_elim)
        DiamondIntroduction = partial(_init_dia_intro)
        BoxIntroduction = partial(_init_box_intro)
        DiamondElimination = partial(_init_dia_elim)

    def __init__(self: Proof, rule: Rule, **kwargs) -> None:
        self.rule = rule
        self.rule.value(self, **kwargs)

    ####################################################################################################################
    # Term-level computations
    ####################################################################################################################
    def apply(self: Proof, argument: Proof) -> Proof:
        res_type = TypeInference.arrow_elim(self.type, argument.type)
        return res_type(rule=Proof.Rule.ArrowElimination, function=self, argument=argument)
       
    def abstract(self: Proof, variable: Proof) -> Proof:
        res_type = Functor(variable.type, self.type)
        return res_type(rule=Proof.Rule.ArrowIntroduction, abstraction=variable, body=self)

    def box(self: Proof, box: str) -> Proof:
        res_type = Box(box, self.type)
        return res_type(rule=Proof.Rule.BoxIntroduction, box=box, body=self)

    def diamond(self: Proof, diamond: str) -> Proof:
        res_type = Diamond(diamond, self.type)
        return res_type(rule=Proof.Rule.DiamondIntroduction, diamond=diamond, body=self)

    def unbox(self: Proof, decoration: str | None = None) -> Proof:
        res_type = TypeInference.box_elim(self.type, decoration)
        return res_type(rule=Proof.Rule.BoxElimination, box=self.decoration, body=self)

    def undiamond(self: Proof, decoration: str | None = None) -> Proof:
        res_type = TypeInference.dia_elim(self.type, decoration)
        return res_type(rule=Proof.Rule.DiamondElimination, diamond=self.decoration, body=self)

    ####################################################################################################################
    # Utilities
    ####################################################################################################################
    def nested_body(self: Proof) -> Proof:
        match self.rule:
            case Proof.Rule.BoxElimination: return self.body.nested_body()
            case Proof.Rule.DiamondElimination: return self.body.nested_body()
            case Proof.Rule.DiamondIntroduction: return self.body.nested_body()
            case Proof.Rule.BoxIntroduction: return self.body.nested_body()
            case _: return self

    def brackets(self: Proof) -> list[IndexedBracket]:
        def go(proof: Proof, context: list[IndexedBracket]) -> list[IndexedBracket]:
            match proof.rule:
                case Proof.Rule.BoxElimination: return go(proof.body, context + [(Bracket.Lock, proof.decoration)])
                case Proof.Rule.DiamondElimination: return go(proof.body, context + [(Bracket.Inner, proof.decoration)])
                case Proof.Rule.DiamondIntroduction: return go(proof.body, context + [(Bracket.Lock, proof.decoration)])
                case Proof.Rule.BoxIntroduction: return go(proof.body, context + [(Bracket.Outer, proof.decoration)])
                case _: return context
        return go(self, [])

    def minimal_brackets(self: Proof) -> list[IndexedBracket]:
        return min((_flatten(_collapse(tree)) for tree in associahedron(self.brackets())), key=len, default=[])

    def is_negative(self: Proof) -> bool:
        # ...
        return False if (bs := self.brackets()) == [] else not(any(map(is_positive, associahedron(bs))))

    def unbracketed(self: Proof) -> bool:
        return any(map(cancels_out, associahedron(self.brackets())))

    def __repr__(self) -> str: return show_term(self)
    def __str__(self) -> str: return show_term(self)

    def __eq__(self, other):
        if self.type != other.type or self.rule != other.rule: return False
        match self.rule:
            case Proof.Rule.Lexicon: return self.constant == other.constant
            case Proof.Rule.Variable: return self.variable == other.variable
            case Proof.Rule.ArrowElimination: return self.abstraction == other.abstraction and self.body == other.body
            case Proof.Rule.ArrowIntroduction: return self.abstraction == other.abstraction and self.body == other.body
            case _: return self.decoration == other.decoration and self.body == other.body

    def __hash__(self) -> int:
        match self.rule:
            case Proof.Rule.Lexicon: return hash((self.rule, self.type, self.constant))
            case Proof.Rule.Variable: return hash((self.rule, self.type, self.variable))
            case Proof.Rule.ArrowElimination: return hash((self.rule, self.type, self.function, self.argument))
            case Proof.Rule.ArrowIntroduction: return hash((self.rule, self.type, self.abstraction, self.body))
            case _: return hash((self.rule, self.type, self.decoration, self.body))

    def vars(self) -> list[Proof]:
        match self.rule:
            case Proof.Rule.Variable: return [self]
            case Proof.Rule.Lexicon: return []
            case Proof.Rule.ArrowElimination: return self.function.vars() + self.argument.vars()
            case Proof.Rule.ArrowIntroduction: return [f for f in self.body.vars() if f != self.abstraction]
            case _: return self.body.vars()

    def free(self: Proof) -> list[Proof]:
        match self.rule:
            case Proof.Rule.Variable: return [self]
            case Proof.Rule.Lexicon: return []
            case Proof.Rule.ArrowElimination: return self.function.free() + self.argument.free()
            case Proof.Rule.ArrowIntroduction: return [f for f in self.body.free() if self.abstraction != f]
            case _: return self.nested_body().free() if self.brackets() == [] else []

    def serialize(self) -> SerializedProof:
        return serialize_proof(self)

    @property
    def type(self) -> Type:
        return type(self)  # type: ignore

    ####################################################################################################################
    # Meta-rules and rewrites
    ####################################################################################################################
    def eta_norm(self: Proof) -> Proof:
        if self.is_negative():
            return self

        match self.rule:
            case Proof.Rule.Variable | Proof.Rule.Lexicon:
                return self
            case Proof.Rule.ArrowElimination:
                return self.function.eta_norm().apply(self.argument.eta_norm())
            case Proof.Rule.ArrowIntroduction:
                body = self.body.eta_norm()
                if body.rule == Proof.Rule.ArrowElimination and body.argument == self.abstraction:
                    return body.function
                return body.abstract(self.abstraction)
            case Proof.Rule.BoxIntroduction:
                body = self.body.eta_norm()
                if body.rule == Proof.Rule.BoxElimination and body.decoration == self.decoration:
                    return body.body
                return body.box(self.decoration)
            case Proof.Rule.DiamondIntroduction:
                body = self.body.eta_norm()
                if body.rule == Proof.Rule.DiamondElimination and body.decoration == self.decoration:
                    return body.body
                return body.diamond(self.decoration)
            case Proof.Rule.DiamondElimination:
                return self.body.eta_norm().undiamond()
            case Proof.Rule.BoxElimination:
                return self.body.eta_norm().unbox()

    def beta_norm(self: Proof) -> Proof:
        match self.rule:
            case Proof.Rule.Variable | Proof.Rule.Lexicon:
                return self
            case Proof.Rule.ArrowElimination:
                if self.function.rule == Proof.Rule.ArrowIntroduction:
                    return self.function.body.substitute(self.function.abstraction, self.argument).beta_norm()
                return self.function.beta_norm().apply(self.argument.beta_norm())
            case Proof.Rule.ArrowIntroduction:
                return self.body.beta_norm().abstract(self.abstraction)
            case Proof.Rule.BoxIntroduction:
                return self.body.beta_norm().box(self.decoration)
            case Proof.Rule.DiamondIntroduction:
                return self.body.beta_norm().diamond(self.decoration)
            case Proof.Rule.DiamondElimination:
                if self.body.rule == Proof.Rule.DiamondIntroduction:
                    return self.body.body
                return self.body.beta_norm().undiamond()
            case Proof.Rule.BoxElimination:
                if self.body.rule == Proof.Rule.BoxIntroduction and self.body.decoration == self.decoration:
                    return self.body.body
                return self.body.beta_norm().unbox()

    def translate_lex(self: Proof, trans: dict[int, int]) -> Proof:
        def go(p: Proof) -> Proof:
            match p.rule:
                case Proof.Rule.Lexicon: return p.type.lex(trans[c]) if (c := p.constant) in trans.keys() else p
                case Proof.Rule.Variable: return p
                case Proof.Rule.ArrowElimination: return go(p.function).apply(go(p.argument))
                case Proof.Rule.ArrowIntroduction: return go(p.body).abstract(p.abstraction)
                case Proof.Rule.BoxElimination: return go(p.body).unbox()
                case Proof.Rule.BoxIntroduction: return go(p.body).box(p.decoration)
                case Proof.Rule.DiamondElimination: return go(p.body).undiamond()
                case Proof.Rule.DiamondIntroduction: return go(p.body).diamond(p.decoration)
        return go(self)

    def subproofs(self: Proof) -> list[Proof]:
        match self.rule:
            case Proof.Rule.Variable | Proof.Rule.Lexicon: return [self]
            case Proof.Rule.ArrowElimination: return [self, self.function, self.argument]
            case Proof.Rule.ArrowIntroduction: return [self, self.body]
            case _: return [self] + self.body.subproofs()

    def canonicalize_var_names(self: Proof) -> Proof:
        def de_bruijn(proof: Proof, variable: Proof) -> int:
            match proof.rule:
                case Proof.Rule.Variable: return 0
                case Proof.Rule.ArrowIntroduction: return 1 + de_bruijn(proof.body, variable)
                case Proof.Rule.ArrowElimination:
                    if variable in proof.function.free():
                        return de_bruijn(proof.function, variable)
                    return de_bruijn(proof.argument, variable)
                case _: return de_bruijn(proof.body, variable)

        def go(proof: Proof, trans: dict[int, int]) -> tuple[Proof, dict[int, int]]:
            match proof.rule:
                case Proof.Rule.Lexicon:
                    return proof, trans
                case Proof.Rule.Variable:
                    return proof.type.var(trans.pop(proof.variable)), trans
                case Proof.Rule.ArrowElimination:
                    fn, trans = go(proof.function, trans)
                    arg, trans = go(proof.argument, trans)
                    return fn.apply(arg), trans
                case Proof.Rule.ArrowIntroduction:
                    trans |= {proof.abstraction.variable: (db := de_bruijn(proof.body, proof.abstraction))}
                    body, trans = go(proof.body, trans)
                    return body.abstract(proof.abstraction.type.var(db)), trans
                case Proof.Rule.BoxElimination:
                    body, trans = go(proof.body, trans)
                    return body.unbox(), trans
                case Proof.Rule.BoxIntroduction:
                    body, trans = go(proof.body, trans)
                    return body.box(proof.decoration), trans
                case Proof.Rule.DiamondIntroduction:
                    body, trans = go(proof.body, trans)
                    return body.diamond(proof.decoration), trans
                case Proof.Rule.DiamondElimination:
                    body, trans = go(proof.body, trans)
                    return body.undiamond(), trans

        return go(self, {})[0]

    def substitute(self: Proof, replace: Proof, with_: Proof) -> Proof:
        TypeInference.assert_equal(replace.type, with_.type)
        if (c := self.subproofs().count(replace)) != 1:
            raise TypeInference.TypeCheckError(f"Expected to find one occurrence of {replace} in {self}, but found {c}")

        def go(_proof: Proof) -> Proof:
            if _proof == replace:
                return with_
            match _proof.rule:
                case Proof.Rule.Variable | Proof.Rule.Lexicon:
                    return _proof
                case Proof.Rule.ArrowElimination:
                    return go(_proof.function).apply(go(_proof.argument))
                case Proof.Rule.ArrowIntroduction:
                    return go(_proof.body).abstract(go(_proof.abstraction))
                case Proof.Rule.BoxIntroduction:
                    return go(_proof.body).box(_proof.decoration)
                case Proof.Rule.DiamondIntroduction:
                    return go(_proof.body).diamond(_proof.decoration)
                case Proof.Rule.DiamondElimination:
                    return go(_proof.body).undiamond()
                case Proof.Rule.BoxElimination:
                    return go(_proof.body).unbox()
        return go(self)

    def dia_cut(self: Proof, replace: Proof, diamond: Proof) -> Proof:
        return self.substitute(replace, diamond.undiamond())


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
            case Proof.Rule.Variable | Proof.Rule.Lexicon: return False
            case (Proof.Rule.BoxElimination | Proof.Rule.BoxIntroduction | Proof.Rule.DiamondElimination |
                  Proof.Rule.DiamondIntroduction): return not show_decorations and needs_par(_proof.body)
            case _: return True

    match proof.rule:
        case Proof.Rule.Lexicon:
            return f'{wp(proof.constant)}' if not show_types else f'{wp(proof.constant)}::{type(proof)}'
        case Proof.Rule.Variable:
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


def serialize_proof(proof: Proof) -> SerializedProof:
    name = proof.rule.name
    match proof.rule:
        case Proof.Rule.Lexicon: return name, (serialize_type(proof.type), proof.constant)
        case Proof.Rule.Variable: return name, (serialize_type(proof.type), proof.variable)
        case Proof.Rule.ArrowElimination: return name, (proof.function.serialize(), proof.argument.serialize())
        case Proof.Rule.ArrowIntroduction: return name, (proof.abstraction.serialize(), proof.body.serialize())
        case _: return name, (proof.decoration, proof.body.serialize())


def deserialize_proof(args) -> Proof:
    match args:
        case Proof.Rule.Lexicon.name, (wordtype, idx):
            return deserialize_type(wordtype).lex(idx)
        case Proof.Rule.Variable.name, (wordtype, idx):
            return deserialize_type(wordtype).var(idx)
        case Proof.Rule.ArrowElimination.name, (fn, arg):
            return deserialize_proof(fn).apply(deserialize_proof(arg))
        case Proof.Rule.ArrowIntroduction.name, (var, body):
            return deserialize_proof(body).abstract(deserialize_proof(var))
        case Proof.Rule.BoxElimination.name, (box, body):
            return deserialize_proof(body).unbox(box)
        case Proof.Rule.BoxIntroduction.name, (box, body):
            return deserialize_proof(body).box(box)
        case Proof.Rule.DiamondElimination.name, (diamond, body):
            return deserialize_proof(body).unbox(diamond)
        case Proof.Rule.DiamondIntroduction.name, (diamond, body):
            return deserialize_proof(body).box(diamond)
        case _:
            raise ValueError(f'Cannot deserialize {args}')


########################################################################################################################
# Examples / Tests
########################################################################################################################
A = Atom('A')

# <>[]A => A
p1 = (x := Box('', A).var(0)).unbox().dia_cut(x, Diamond('', Box('', A)).var(0))
assert p1.unbracketed()

# []A => []<>[]A
p3 = Box('', A).var(0).diamond('').box('')
assert p3.unbracketed()

# <>A => <>[]<>A
p4 = (x := A.var(0)).diamond('').box('').diamond('').dia_cut(x, Diamond('', A).var(0))
assert p4.unbracketed()

# <>[]<>A => <>A
p5 = (x := Box('', Diamond('', A)).var(0)).unbox().dia_cut(x, Diamond('', Box('', Diamond('', A))).var(0))
assert p5.unbracketed()

# []<>[]A => []A
p6 = (x := Box('', A).var(0)).unbox().dia_cut(x, Box('', Diamond('', Box('', A))).var(0).unbox()).box('')
assert p6.unbracketed()
