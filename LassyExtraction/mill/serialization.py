from typing import Type as TYPE
from .types import Type, type_prefix, parse_prefix
from .terms import (TERM, Term, Variable, Constant, ArrowIntroduction, ArrowElimination,
                    DiamondIntroduction, DiamondElimination, BoxIntroduction, BoxElimination)
from .proofs import Rule, Logical, Structural, Proof


SerializedType = str
SerializedTerm = tuple[TYPE,
                       tuple[SerializedType, int]
                       | tuple['SerializedTerm', 'SerializedTerm']
                       | tuple[str, 'SerializedTerm']]
SerializedRule = str
SerializedProof = tuple[SerializedRule,
                        SerializedTerm
                        | tuple['SeralizedProof', SerializedTerm]
                        | tuple['SerializedProof', 'SerializedProof']
                        | tuple['SerializedProof', str]
                        | tuple['SerializedProof', SerializedTerm, 'SerializedProof']]


def serialize_type(_type: Type) -> SerializedType: return type_prefix(_type)


def deserialize_type(serialized: SerializedType) -> Type: return parse_prefix(serialized)


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


def serialize_maybe_term(term: Term | None) -> SerializedTerm | None:
    return serialize_term(term) if term is not None else None


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
    match proof.rule:
        case Logical.Variable | Logical.Constant:
            ret = serialize_term(proof.term)
        case Logical.ArrowIntroduction:
            (body,) = proof.premises
            ret = serialize_proof(body), serialize_term(proof.focus)
        case Logical.ArrowElimination:
            fn, arg = proof.premises
            ret = serialize_proof(fn), serialize_proof(arg)
        case Logical.DiamondIntroduction:
            (body,) = proof.premises
            (struct,) = proof.structure
            ret = (serialize_proof(body), struct.brackets)
        case Logical.DiamondElimination:
            original, becomes = proof.premises
            ret = serialize_proof(original), serialize_term(proof.focus), serialize_proof(becomes)
        case Logical.BoxIntroduction:
            (body,) = proof.premises
            (struct,) = body.structure
            ret = serialize_proof(body), struct.brackets
        case Logical.BoxElimination:
            (body,) = proof.premises
            (struct,) = proof.structure
            ret = serialize_proof(body), struct.brackets
        case Structural.Extract:
            (body,) = proof.premises
            ret = serialize_proof(body), serialize_term(proof.focus)
        case _:
            raise NotImplementedError
    return serialize_rule(proof.rule), ret


def deserialize_proof(serialized: SerializedProof) -> Proof:
    serial_rule, args = serialized
    rule = deserialize_rule(serial_rule)
    match rule:
        case Logical.Variable | Logical.Constant:
            return rule(deserialize_term(args))
        case Logical.ArrowIntroduction:
            body, var = args
            return rule(deserialize_term(var), deserialize_proof(body))
        case Logical.ArrowElimination:
            fn, arg = args
            return rule(deserialize_proof(fn), deserialize_proof(arg))
        case Logical.DiamondIntroduction:
            body, diamond = args
            return rule(deserialize_proof(body), diamond)
        case Logical.DiamondElimination:
            original, where, becomes = args
            return rule(deserialize_proof(original), deserialize_term(where), deserialize_proof(becomes))
        case Logical.BoxIntroduction:
            body, box = args
            return rule(deserialize_proof(body), box)
        case Logical.BoxElimination:
            body, box = args
            return rule(deserialize_proof(body), box)
        case Structural.Extract:
            body, focus = args
            return rule(deserialize_proof(body), deserialize_term(focus))
        case _:
            raise NotImplementedError

