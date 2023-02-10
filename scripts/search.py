from __future__ import annotations
from aethel.frontend import Sample
from aethel.mill.proofs import Rule, Type
from aethel.mill.terms import (DiamondIntroduction, Term, Variable, Constant, ArrowIntroduction, ArrowElimination,
                               CaseOf)
from typing import Callable, Iterator, Iterable
from itertools import takewhile


def search(bank: Iterable[Sample], query: Callable[[Sample], bool], num_hits: int | None = None) -> Iterator[Sample]:
    f = filter(query, bank)
    return f if num_hits is None else map(lambda x: x[1], takewhile(lambda x: x[0] < num_hits, enumerate(f)))


class Query:
    def __init__(self, fn: Callable[[Sample], bool]):
        self.fn = fn

    def __and__(self, other: Query) -> Query:
        def f(sample: Sample) -> bool: return self.fn(sample) and other.fn(sample)
        return Query(f)

    def __or__(self, other) -> Query:
        def f(sample: Sample) -> bool: return self.fn(sample) or other.fn(sample)
        return Query(f)

    def __invert__(self) -> Query:
        def f(sample: Sample) -> bool: return not self.fn(sample)
        return Query(f)

    def __xor__(self, other) -> Query:
        def f(sample: Sample) -> bool: return self.fn(sample) ^ other.fn(sample)
        return Query(f)

    def __call__(self, sample: Sample) -> bool:
        return self.fn(sample)


def name_like(name: str) -> Query:
    def f(sample: Sample) -> bool:
        return name in sample.name
    return Query(f)


def contains_word(word: str) -> Query:
    def f(sample: Sample) -> bool:
        return any(word == item.word for phrase in sample.lexical_phrases for item in phrase.items)
    return Query(f)


def contains_lemma(lemma: str) -> Query:
    def f(sample: Sample) -> bool:
        return any(lemma == item.lemma for phrase in sample.lexical_phrases for item in phrase.items)
    return Query(f)


def of_type(_type: Type) -> Query:
    def f(sample: Sample) -> bool:
        return sample.proof.type == _type
    return Query(f)


def may_only_contain_rules(rules: set[Rule]) -> Query:
    def f(sample: Sample) -> bool:
        return all(proof.rule in rules for proof in sample.proof.subproofs())
    return Query(f)


def must_contain_rules(rules: set[Rule]) -> Query:
    def f(sample: Sample) -> bool:
        proof_rules = {proof.rule for proof in sample.proof.subproofs()}
        return all(rule in proof_rules for rule in rules)
    return Query(f)


def length_between(_min: int, _max: int) -> Query:
    def f(sample: Sample) -> bool:
        return _min <= len(sample.lexical_phrases) <= _max
    return Query(f)


def term_contains(string: str) -> Query:
    def f(sample: Sample) -> bool:
        return string in str(sample.proof.term)
    return Query(f)


def contains_type(_type: Type) -> Query:
    def f(sample: Sample) -> bool:
        return any(lp.type == _type for lp in sample.lexical_phrases)
    return Query(f)


def vc_chain(num: int) -> Query:
    def f(sample: Sample) -> bool:
        def g(term: Term, c: int) -> int:
            match term:
                case Variable(_) | Constant(_):
                    return c
                case ArrowElimination(l, r):
                    return max(g(l, c), g(r,c))
                case ArrowIntroduction(_, b):
                    return g(b, c)
                case CaseOf(_, _, original):
                    return g(original,c)
                case DiamondIntroduction(d, b):
                    c += d == 'vc'
                    return g(b, c)
                case _:
                    return g(term.body, c)
        return g(sample.proof.term, 0) >= num
    return Query(f)


def f(sample: Sample) -> bool:
    first_letters = [lp.string[0].lower() for lp in sample.lexical_phrases]
    return first_letters == sorted(first_letters)