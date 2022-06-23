from typing import Type as TYPE
from .types import Type, Atom, Functor, Box, Diamond
from .terms import (TERM, Term, Variable, Constant, ArrowIntroduction, ArrowElimination,
                    DiamondIntroduction, DiamondElimination, BoxIntroduction, BoxElimination)
from .structures import Structure, Sequence, Unary
from .proofs import Judgement, Rule, Logical, Structural, Proof


SerializedType = tuple[TYPE,
                       tuple[str] |
                       tuple['SerializedType', 'SerializedType'] |
                       tuple[str, 'SerializedType']]
SerializedTerm = tuple[TYPE,
                       tuple[SerializedType, int] |
                       tuple['SerializedTerm', 'SerializedTerm'] |
                       tuple[str, 'SerializedTerm']]
SerializedStructure = tuple[TYPE,
                            tuple['SerializedStructure', ...] |
                            tuple['SerializedStructure', str] |
                            tuple[SerializedType, int]]
SerializedJudgement = tuple[SerializedStructure, SerializedTerm]
SerializedRule = str
SerializedProof = tuple[tuple['SerializedProof', ...], SerializedJudgement, SerializedRule, SerializedTerm | None]


def serialize_type(_type: Type) -> SerializedType:
    match _type:
        case Atom(sign): return Atom, (sign,)
        case Functor(argument, result): return Functor, (serialize_type(argument), serialize_type(result))
        case Box(decoration, content): return Box, (decoration, serialize_type(content))
        case Diamond(decoration, content): return Diamond, (decoration, serialize_type(content))
        case _: raise ValueError(f'Unknown type: {_type}')


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


def serialize_term(term: Term) -> SerializedTerm:
    match term:
        case Variable(_type, index): return Variable, (serialize_type(_type), index)
        case Constant(_type, index): return Constant, (serialize_type(_type), index)
        case ArrowElimination(function, argument):
            return ArrowElimination, (serialize_term(function), serialize_term(argument))
        case ArrowIntroduction(abstraction, body):
            return ArrowIntroduction, (serialize_term(abstraction), serialize_term(body))
        case DiamondIntroduction(diamond, body): return DiamondIntroduction, (diamond, serialize_term(body))
        case DiamondElimination(diamond, body): return DiamondElimination, (diamond, serialize_term(body))
        case BoxIntroduction(box, body): return BoxIntroduction, (box, serialize_term(body))
        case BoxElimination(box, body): return BoxElimination, (box, serialize_term(body))
        case _: raise ValueError(f'Unknown term: {term}')


def deserialize_term(serialized: SerializedTerm) -> TERM:
    constructor, args = serialized
    if constructor == Variable:
        (_type, index) = args
        return Variable(deserialize_type(_type), index)
    if constructor == Constant:
        (_type, index) = args
        return Constant(deserialize_type(_type), index)
    if constructor == ArrowElimination:
        (function, argument) = args
        return ArrowElimination(deserialize_term(function), deserialize_term(argument))
    if constructor == ArrowIntroduction:
        (abstraction, body) = args
        return ArrowIntroduction(deserialize_term(abstraction), deserialize_term(body))
    if constructor == DiamondIntroduction:
        (diamond, body) = args
        return DiamondIntroduction(diamond, deserialize_term(body))
    if constructor == DiamondElimination:
        (diamond, body) = args
        return DiamondElimination(diamond, deserialize_term(body))
    if constructor == BoxIntroduction:
        (box, body) = args
        return BoxIntroduction(box, deserialize_term(body))
    if constructor == BoxElimination:
        (box, body) = args
        return BoxElimination(box, deserialize_term(body))
    raise ValueError(f'Unknown term: {constructor}')


def serialize_structure(structure: Structure[Variable | Constant] | Variable | Constant) -> SerializedStructure:
    match structure:
        case Sequence(premises): return Sequence, tuple(map(serialize_structure, premises))
        case Unary(content, brackets): return Unary, (serialize_structure(content), brackets)
        case _: return serialize_term(structure)


def deserialize_structure(serialized: SerializedStructure) -> Structure[Variable | Constant] | Variable | Constant:
    cons, args = serialized
    if cons == Sequence:
        return Sequence(*tuple(map(deserialize_structure, args)))
    if cons == Unary:
        (content, brackets) = args
        return Unary(Sequence(deserialize_structure(content)), brackets)
    return deserialize_term(serialized)


def serialize_judgement(judgement: Judgement) -> SerializedJudgement:
    return serialize_structure(judgement.assumptions), serialize_term(judgement.term)


def deserialize_judgement(serialized: SerializedJudgement) -> Judgement:
    assumptions, term = serialized
    return Judgement(deserialize_structure(assumptions), deserialize_term(term))  # type: ignore


def serialize_rule(rule: Rule) -> SerializedRule:
    match rule:
        case Logical(): return f'Logical.{rule}'
        case Structural(): return f'Structural.{rule}'
        case _: raise NotImplementedError


def deserialize_rule(rule: SerializedRule) -> Rule:
    match rule.split('.'):
        case ('Logical', x): return Logical[x]
        case ('Structural', x): return Structural[x]
        case _: raise NotImplementedError


def serialize_proof(proof: Proof) -> SerializedProof:
    return (tuple(map(serialize_proof, proof.premises)),
            serialize_judgement(proof.conclusion),
            serialize_rule(proof.rule),
            serialize_term(proof.focus) if proof.focus is not None else None)


def deserialize_proof(serialized: SerializedProof) -> Proof:
    premises, conclusion, rule, focus = serialized
    return Proof(tuple(map(deserialize_proof, premises)),
                 deserialize_judgement(conclusion),
                 deserialize_rule(rule),
                 deserialize_term(focus) if focus is not None else None)
