"""
    Interface to LangPro's expected prolog format.
"""

from src.aethel.alpino.extraction import prove_dag
from src.aethel.alpino.transformations import (prepare_for_extraction, DAG, sort_nodes, get_lex_nodes, is_ghost)
from xml.etree import ElementTree
from src.aethel.frontend import LexicalPhrase, LexicalItem
from src.aethel.mill.proofs import decolor_proof, Proof
from src.aethel.mill.terms import Term, Constant, Variable, ArrowElimination, ArrowIntroduction
from src.aethel.mill.types import Type, Atom, Functor


def get_lex_phrases(dag: DAG[str]) -> list[tuple[Proof, tuple[LexicalItem, ...]]]:
    def make_item(node: str) -> LexicalItem:
        word, pos, pt,  lemma = (dag.get(node, attr) for attr in ('word', 'pos', 'pt', 'lemma'))
        assert not any(x is None for x in (word, pos, pt, lemma))
        return LexicalItem(word=word, pos=pos, pt=pt, lemma=lemma)
    bottom_nodes = [(node,
                     sort_nodes(dag, {e.target for e in dag.outgoing_edges(node)
                                      if e.label == 'mwp' and not is_ghost(dag, e.target)}))
                    for node in get_lex_nodes(dag) if not is_ghost(dag, node)]
    return [(dag.get(bottom, 'proof'), tuple(make_item(c) for c in children) if children else (make_item(bottom),))
            for bottom, children in bottom_nodes]


def file_to_natlog(filename: str) -> ...:
    etree = ElementTree.parse(filename)
    return alpino_to_natlog(etree, filename)


def alpino_to_natlog(etree: ElementTree, filename: str) -> tuple[str, list[list[str]]] | None:
    dags = prepare_for_extraction(etree, filename)
    if len(dags) != 1:
        return
    [dag] = dags
    proof = prove_dag(dag)
    phrasal_proofs, lex_itemss = list(zip(*get_lex_phrases(dag)))
    lex_phrases = tuple(LexicalPhrase(type=sp.type, items=items) for sp, items in zip(phrasal_proofs, lex_itemss))
    proof = proof.translate_lex({sp.term.index: i for i, sp in enumerate(phrasal_proofs)})
    proof = decolor_proof(proof)
    proof_line = proof_to_natlog(proof)
    return proof_line, phrases_to_natlog(lex_phrases)


def phrases_to_natlog(phrases: tuple[LexicalPhrase, ...]) -> list[list[str]]:
    return [[item.word.replace("'", "\'") for item in lp.items] for lp in phrases]


def proof_to_natlog(proof: Proof) -> str:
    return term_to_natlog(decolor_proof(proof.eta_norm().standardize_vars()).term)


def term_to_natlog(term: Term) -> str:
    match term:
        case Variable(_type, index):
            return f'v(X{index},{type_to_natlog(_type)}'
        case Constant(_type, index):
            return f't({index},{type_to_natlog(_type)})'
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
