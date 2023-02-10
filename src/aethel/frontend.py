from __future__ import annotations

import sys

from .mill.proofs import Proof, term_repr, Type
from .mill.serialization import (SerializedType, serialize_type, deserialize_type,
                                 SerializedProof, serialize_proof, deserialize_proof)
from dataclasses import dataclass
import pathlib
import pickle


@dataclass(frozen=True)
class ProofBank:
    version: str
    samples: list[Sample]

    def __getitem__(self, item: int) -> Sample: return self.samples[item]
    def __len__(self) -> int: return len(self.samples)
    def find_by_name(self, name: str) -> list[Sample]: return [sample for sample in self.samples if name in sample.name]
    def __repr__(self) -> str: return f"æthel version {self.version} containing {len(self)} samples."
    def __post_init__(self): print(f'Loaded {self}')

    @staticmethod
    def load_data(path: str) -> ProofBank:
        print(f'Loading and verifying {pathlib.Path(path).name}...')
        with open(path, 'rb') as f:
            version, (train, dev, test) = pickle.load(f)
            return ProofBank(version=version, samples=[Sample.load(x) for x in train + dev + test])


@dataclass(frozen=True)
class Sample:
    lexical_phrases:    tuple[LexicalPhrase, ...]
    proof:              Proof
    name:               str
    subset:             str

    def __len__(self) -> int: return len(self.lexical_phrases)
    def __repr__(self) -> str: return self.name

    def show_term(self, show_types: bool = True, show_words: bool = True) -> str:
        return term_repr(term=self.proof.term,
                         show_type=show_types,
                         show_intermediate_types=False,
                         word_repr=lambda x: self.lexical_phrases[x].string if show_words else None)

    @property
    def sentence(self):
        return ' '.join(phrase.string for phrase in self.lexical_phrases)

    def save(self) -> SerializedSample: return serialize_sample(self)

    @staticmethod
    def load(serialized: SerializedSample) -> Sample: return deserialize_sample(serialized)


@dataclass
class LexicalPhrase:
    items:  tuple[LexicalItem, ...]
    type:   Type

    @property
    def string(self) -> str: return ' '.join(item.word for item in self.items)
    def __repr__(self) -> str: return f'LexicalPhrase(string={self.string}, type={self.type}, len={len(self)})'
    def __len__(self) -> int: return len(self.items)


@dataclass
class LexicalItem:
    word:   str
    pos:    str
    pt:     str
    lemma:  str


########################################################################################################################
# Serialization
########################################################################################################################

SerializedItem = tuple[str, str, str, str]
SerializedPhrase = tuple[tuple[SerializedItem, ...], SerializedType]
SerializedSample = tuple[tuple[SerializedPhrase, ...], SerializedProof, str, str]


def serialize_item(item: LexicalItem) -> SerializedItem:
    return item.word, item.pos, item.pt, item.lemma


def deserialize_item(item: SerializedItem) -> LexicalItem:
    return LexicalItem(*item)


def serialize_phrase(phrase: LexicalPhrase) -> SerializedPhrase:
    return tuple(serialize_item(item) for item in phrase.items), serialize_type(phrase.type)


def deserialize_phrase(phrase: SerializedPhrase) -> LexicalPhrase:
    items, _type = phrase
    return LexicalPhrase(items=tuple(deserialize_item(item) for item in items), type=deserialize_type(_type))


def serialize_sample(sample: Sample) -> SerializedSample:
    return (tuple(serialize_phrase(phrase) for phrase in sample.lexical_phrases),
            serialize_proof(sample.proof),
            sample.name,
            sample.subset)


def deserialize_sample(sample: SerializedSample) -> Sample:
    lexical_phrases, proof, name, subset = sample
    return Sample(lexical_phrases=tuple(deserialize_phrase(phrase) for phrase in lexical_phrases),
                  proof=deserialize_proof(proof),
                  name=name,
                  subset=subset)
