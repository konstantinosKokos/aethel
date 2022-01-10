from LassyExtraction.mill.types import Proof, Type, show_term, deserialize_type, SerializedType, SerializedProof
from typing import NamedTuple
import pickle


class Premise(NamedTuple):
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


class Sample(NamedTuple):
    premises: list[Premise]
    proof: Proof
    name: str
    subset: 'str'

    def __repr__(self):
        return show_term(self.proof, word_printer=lambda x: self.premises[x].word)

    def save(self) -> tuple[list[tuple[str, str, str, str, SerializedType]], SerializedProof, str, str]:
        return [premise.save() for premise in self.premises], self.proof.serialize(), self.name, self.subset

    @staticmethod
    def load(premises: list[tuple[str, str, str, str, SerializedType]],
             sproof: SerializedProof,
             name: str,
             subset: str) -> 'Sample':
        return Sample([Premise.load(*premise) for premise in premises], Proof.deserialize_proof(sproof), name, subset)


def store_data(data: list[Sample], path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump([data.save() for data in data], f)


def load_data(path: str) -> list[Sample]:
    with open(path, 'rb') as f:
        return [Sample.load(*data) for data in pickle.load(f)]
