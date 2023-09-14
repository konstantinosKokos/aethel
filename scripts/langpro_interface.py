"""
    Interface to LangPro's expected prolog format.
"""

from aethel.mill.proofs import decolor_proof, Proof
from aethel.mill.terms import Term, Constant, Variable, ArrowElimination, ArrowIntroduction
from aethel.mill.types import Type, Atom, Functor


def proof_to_natlog(proof: Proof) -> str:
    return term_to_natlog(decolor_proof(proof).term)


def term_to_natlog(term: Term) -> str:
    match term:
        case Variable(index):
            return f'v(X{index},{type_to_natlog(term.type)}'
        case Constant(index):
            return f't({index},{type_to_natlog(term.type)})'
        case ArrowElimination(function, argument):
            return f'(({term_to_natlog(function)}) @ ({term_to_natlog(argument)}))'
        case ArrowIntroduction(var, body):
            return f'(abst({term_to_natlog(var)},{term_to_natlog(body)}))'
        case _:
            raise ValueError(f'Unexpected term constructor: {type(term)}')


def type_to_natlog(_type: Type) -> str:
    match _type:
        case Atom(sign):
            return sign.lower()
        case Functor(argument, result):
            return f'({type_to_natlog(argument)}) ~> ({type_to_natlog(result)})'
        case _:
            raise ValueError(f'Unexpected type constructor: {type(_type)}')


