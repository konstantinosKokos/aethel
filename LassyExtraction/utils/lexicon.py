from LassyExtraction.milltypes import *
from LassyExtraction.graphutils import DAG, fst, snd
from LassyExtraction.transformations import order_nodes
from typing import *
from itertools import chain

Node = TypeVar('Node')
T = TypeVar('T')


def get_pos(dag: DAG[Node, Any], leaf: Node) -> str:
    return dag.attribs[leaf]['pos']


def get_type(dag: DAG[Node, Any], leaf: Node) -> str:
    return dag.attribs[leaf]['type'].depolarize()


def get_type_word(dag: DAG[Node, Any], leaf: Node) -> Tuple[WordType, str]:
    return dag.attribs[leaf]['type'].depolarize(), dag.attribs[leaf]['word']


def get_type_word_pos(dag: DAG[Node, Any], leaf: Node) -> Tuple[WordType, str, str]:
    return get_type_word(dag, leaf) + (dag.attribs[leaf]['word'],)


def get_type_word_pos_lemma(dag: DAG[Node, Any], leaf: Node) -> Tuple[WordType, str, str, str]:
    return get_type_word_pos(dag, leaf) + (dag.attribs[leaf]['lemma'] if 'lemma' in dag.attribs[leaf].keys()
                                           else dag.attribs[leaf]['word'],)


def project_dag(dag: DAG, fn: Callable[[DAG[Node, Any], Node], T]) -> List[T]:
    return list(map(lambda leaf: fn(dag, leaf), order_nodes(dag, list(dag.get_leaves()))))


def make_some_freqs(dags: List[DAG], fn: Callable[[DAG[Node, Any], Node], T]) -> Dict[T, int]:
    all_dags = list(chain.from_iterable(list(map(lambda dag: project_dag(dag, fn), dags))))
    return Counter(all_dags)


def make_type_lexicon(dags: List[DAG]) -> Dict[str, Counter]:
    all_dags = list(chain.from_iterable(list(map(lambda dag: project_dag(dag, get_type_word), dags))))
    c = {wordtype: Counter() for wordtype in set(map(fst, all_dags))}
    for pair in all_dags:
        c[pair[0]][pair[1]] += 1
    return c


def make_plain_word_lexicon(dags: List[DAG]) -> Dict[str, Counter]:
    all_dags = list(chain.from_iterable(list(map(lambda dag: project_dag(dag, get_type_word), dags))))
    c = {str(word): Counter() for word in set(map(snd, all_dags))}
    for pair in all_dags:
        c[pair[1]][pair[0]] += 1
    return c


def make_word_lexicon(dags: List[DAG]) -> Dict[Tuple[str, str], Counter]:
    all_dags = list(chain.from_iterable(list(map(lambda dag: project_dag(dag, get_type_word_pos), dags))))
    c = {wordpos: Counter() for wordpos in set(map(lambda x: (x[1], x[2]), all_dags))}
    for pair in all_dags:
        c[(pair[1], pair[2])][pair[0]] += 1
    return c


def print_type_lexicon(type_lexicon: Dict[str, Counter]) -> str:
    def print_one(blah) -> str:
        return str(blah[0]) + '\t' + ' | '.join(word + ' #= ' + str(count) for word, count in blah[1])

    type_lexicon = sorted(type_lexicon.items(), key=lambda x: sum(x[1].values()), reverse=True)
    type_lexicon = [(pair[0], sorted(pair[1].items(), key=lambda x: x[1], reverse=True)) for pair in type_lexicon]
    return '\n'.join(print_one(x) for x in type_lexicon)


def print_word_lexicon(word_lexicon: Dict[Tuple[str, str], Counter]) -> str:
    def print_one(blah) -> str:
        return blah[0][0] + '\t' + blah[0][1] + '\t' + ' | '.join(str(wordtype) + ' #= ' + str(count)
                                                                  for wordtype, count in blah[1])

    word_lexicon = sorted(word_lexicon.items(), key=lambda x: x[0])
    word_lexicon = [(pair[0], sorted(pair[1].items(), key=lambda x: x[1], reverse=True)) for pair in word_lexicon]
    return '\n'.join(print_one(x) for x in word_lexicon)


def print_counter(counter: Counter) -> str:
    return '\n'.join(str(k)+'\t'+str(v) for k, v in sorted(counter.items(), key=lambda x: x[1], reverse=True))


def make_pos_freqs(dags: List[DAG]) -> Dict[str, int]:
    return make_some_freqs(dags, get_pos)


def make_type_freqs(dags: List[DAG]) -> Dict[str, int]:
    return make_some_freqs(dags, get_type)


def make_word_map(dags: List[DAG]) -> Dict[Tuple[str, str, str], Counter]:
    all_dags = list(chain.from_iterable(list(map(lambda dag: project_dag(dag, get_type_word_pos_lemma), dags))))
    c = {wordposlemma: Counter() for wordposlemma in set(map(lambda x: (x[1], x[2], x[3]), all_dags))}
    for pair in all_dags:
        c[(pair[1], pair[2], pair[3])][pair[0]] += 1
    return c


def print_word_map(word_map: Dict[Tuple[str, str, str], Counter]) -> str:
    word_map = sorted(word_map.items(), key=lambda x: x[0])
    word_map = [(pair[0], sorted(pair[1].items(), key=lambda x: x[1], reverse=True)) for pair in word_map]
    ret = ''
    for pair in word_map:
        wordposlemma = pair[0]
        type_counter = pair[1]
        for wordtype, count in type_counter:
            ret += '\t'.join(wordposlemma) + '\t' + str(wordtype) + ' #= ' + str(count) + '\n'
    return ret
