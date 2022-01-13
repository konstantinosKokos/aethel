from __future__ import annotations
from LassyExtraction.mill.types import (Proof, Type, show_term, deserialize_type, SerializedType, SerializedProof,
                                        deserialize_proof)
from dataclasses import dataclass
import pickle


@dataclass(frozen=True)
class aethel:
    version: str
    train: list[Sample]
    dev: list[Sample]
    test: list[Sample]

    def __repr__(self) -> str:
        return f"æthel dump version {self.version}," \
               f" containing {len(self.train)} training samples, " \
               f"{len(self.dev)} dev samples, " \
               f"and {len(self.test)} test samples."

    def __post_init__(self):
        print(f'Loaded {self}')

    @staticmethod
    def load_data(path: str) -> aethel:
        with open(path, 'rb') as f:
            version, (train, dev, test) = pickle.load(f)
            return aethel(version=version,
                          train=[Sample.load(*s) for s in train],
                          dev=[Sample.load(*s) for s in dev],
                          test=[Sample.load(*s) for s in test])


@dataclass(frozen=True)
class Sample:
    premises: list[Premise]
    proof: Proof
    name: str
    subset: 'str'

    def __repr__(self):
        return self.name

    def show_term(self, show_decorations: bool = True, show_types: bool = True, show_words: bool = True) -> str:
        return show_term(self.proof,
                         show_decorations=show_decorations,
                         show_types=show_types,
                         word_printer=(lambda x: self.premises[x].word) if show_words else str)

    def save(self) -> tuple[list[tuple[str, str, str, str, SerializedType]], SerializedProof, str, str]:
        return [premise.save() for premise in self.premises], self.proof.serialize(), self.name, self.subset

    @staticmethod
    def load(premises: list[tuple[str, str, str, str, SerializedType]],
             sproof: SerializedProof,
             name: str,
             subset: str) -> 'Sample':
        return Sample([Premise.load(*premise) for premise in premises], deserialize_proof(sproof), name, subset)

    def show_sentence(self) -> str:
        return ' '.join([premise.word for premise in self.premises])


@dataclass(frozen=True)
class Premise:
    word: str
    pos: str | list[str]
    pt: str | list[str]
    lemma: str | list[str]
    type: Type

    def save(self):
        return self.word, self.pos, self.pt, self.lemma, self.type.serialize_type()

    @staticmethod
    def load(word: str, pos: str, pt: str, lemma: str, stype: SerializedType) -> 'Premise':
        return Premise(word, pos, pt, lemma, deserialize_type(stype))

