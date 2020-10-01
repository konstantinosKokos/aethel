from LassyExtraction.transformations import order_nodes
from LassyExtraction.graphutils import DAG
from LassyExtraction.transformations import get_sentence as get_words
from LassyExtraction.milltypes import (WordType, WordTypes, AtomicType, get_polarities_and_indices, PolarizedType,
                                       polarize_and_index_many)
from LassyExtraction.lambdas import make_graph as _make_graph, traverse, translate_id
from LassyExtraction.lassy import is_public as is_public_str

from typing import List, Tuple

from functools import reduce
from operator import add

Proof = List[Tuple[int, int]]


def get_types(dag: DAG) -> WordTypes:
    return list(map(lambda leaf: dag.attribs[leaf]['type'], list(order_nodes(dag, list(dag.get_leaves())))))


def get_context(dag: DAG) -> List[Tuple[str, WordType]]:
    words = get_words(dag)
    types = get_types(dag)
    return list(zip(words, types))


def get_conclusion(_atoms: List[Tuple[AtomicType, int]], _proof: Proof) -> Tuple[AtomicType, int]:
    if not _proof:
        assert len(_atoms) == 1
        return _atoms[0][0], 1
    premises = set(map(lambda x: x[1], _atoms))
    negatives = set(map(lambda p: p[1], _proof))
    positives = set(map(lambda p: p[0], _proof))
    conclusion_id = list((negatives.union(positives)).difference(premises))[0]
    match = [k for k, v in _proof if v == conclusion_id][0]
    conclusion_atom = [pair[0] for pair in _atoms if pair[1] == match][0]
    return conclusion_atom, conclusion_id


def get_lambda(dag: DAG, proof: Proof,
               show_word_names: bool = True, show_types: bool = True, show_decos: bool = True) -> str:
    if show_word_names:
        words = get_words(dag) + ['conc']
    else:
        words = list(map(lambda i: f'w{translate_id(i)}', range(len(get_words(dag) + ['conc']))))
    types = get_types(dag)
    if len(types) == 1:
        _, types = polarize_and_index_many(types, 0)
        proof = {(0, 1)}
    atoms = list(zip(*list(map(get_polarities_and_indices, types))))
    negative, positive = list(map(lambda x: reduce(add, x), atoms))
    conclusion, conclusion_id = get_conclusion(_atoms=negative + positive, _proof=proof)
    conclusion = PolarizedType(wordtype=conclusion.type, polarity=False, index=conclusion_id)
    graph = _make_graph(words, types, conclusion, show_types)
    the_lambda = traverse(graph, str(conclusion_id), {str(k): str(v) for k, v in proof},
                          {str(v): str(k) for k, v in proof}, True, 0, show_decos)[0]
    return the_lambda


def is_public(dag: DAG) -> bool:
    return is_public_str(get_name(dag))


def get_name(dag: DAG) -> str:
    name = dag.meta['src']
    if '_' not in name:
        return name
    else:
        prefix, suffix = name.split('_')[0:2]
        return prefix.split('.xml')[0] + f'_{suffix}' + '.xml'
