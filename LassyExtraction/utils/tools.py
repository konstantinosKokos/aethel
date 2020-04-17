from LassyExtraction.transformations import order_nodes
from LassyExtraction.graphutils import DAG
from LassyExtraction.transformations import get_sentence as get_words
from LassyExtraction.milltypes import WordTypes, AtomicType, get_polarities_and_indices, PolarizedType
from LassyExtraction.lambdas import make_graph as _make_graph, traverse

from typing import List, Tuple, Dict

from functools import reduce
from operator import add

Proof = Dict[int, int]


def get_types(dag: DAG) -> WordTypes:
    return list(map(lambda leaf: dag.attribs[leaf]['type'], list(order_nodes(dag, list(dag.get_leaves())))))


def get_conclusion(_atoms: List[Tuple[AtomicType, int]], _proof: Proof) -> Tuple[AtomicType, int]:
    antecedents = set(map(lambda x: x[1], _atoms))
    conclusion_id = list(set(map(lambda x: x[1],
                                 _proof)).
                         union(set(map(lambda x: x[0],
                                       _proof))).
                         difference(antecedents))[0]
    conclusion_pair = list(filter(lambda pair: pair[1] == conclusion_id, _proof))[0][0]
    conclusion_atom = list(filter(lambda a: a[1] == conclusion_pair, _atoms))[0][0]
    return conclusion_atom, conclusion_id


def make_graph(dag: DAG, proof: Dict[int, int]) -> str:
    words = get_words(dag) + ['conc']
    if len(words) == 2:
        return words[0]
    types = get_types(dag)
    atoms = list(zip(*list(map(get_polarities_and_indices, types))))
    negative, positive = list(map(lambda x: reduce(add, x), atoms))
    conclusion, conclusion_id = get_conclusion(_atoms=negative + positive, _proof=proof)
    conclusion = PolarizedType(wordtype=conclusion.type, polarity=False, index=conclusion_id)
    graph = _make_graph(words, types, conclusion)
    the_lambda = traverse(graph, str(conclusion_id), {str(k): str(v) for k, v in proof},
                          {str(v): str(k) for k, v in proof}, True, 0)[0]
    return the_lambda
