from __future__ import annotations
from LassyExtraction.frontend import Sample
from LassyExtraction.mill.proofs import Rule, Type
from LassyExtraction.extraction import Atoms
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
        return any(proof.rule in rules for proof in sample.proof.subproofs())
    return Query(f)


def length_between(_min: int, _max: int) -> Query:
    def f(sample: Sample) -> bool:
        return _min <= len(sample.lexical_phrases) <= _max
    return Query(f)
