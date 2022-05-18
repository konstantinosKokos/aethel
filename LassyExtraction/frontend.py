from __future__ import annotations
from LassyExtraction.mill.types import (Proof, Type, show_term, deserialize_type, SerializedType, SerializedProof,
                                        deserialize_proof)
from dataclasses import dataclass
import pickle

import pathlib


@dataclass(frozen=True)
class ProofBank:
    version: str
    samples: list[Sample]

    def __getitem__(self, item: int) -> Sample: return self.samples[item]
    def __len__(self) -> int: return len(self.samples)
    def find_by_name(self, name: str) -> list[Sample]: return [sample for sample in self.samples if name in sample.name]
    def __repr__(self) -> str: return f"æthel dump version {self.version} containing {len(self)} samples."
    def __post_init__(self): print(f'Loaded {self}')

    @staticmethod
    def load_data(path: str) -> ProofBank:
        print(f'Loading and verifying {pathlib.Path(path).name}...')
        with open(path, 'rb') as f:
            version, (train, dev, test) = pickle.load(f)
            return ProofBank(version=version, samples=[Sample.load(*x) for x in train + dev + test])


@dataclass(frozen=True)
class Sample:
    premises: list[Premise]
    proof: Proof
    name: str
    subset: str

    def __len__(self) -> int: return len(self.premises)
    def __repr__(self) -> str: return self.name

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
