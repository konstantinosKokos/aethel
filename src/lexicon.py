from src.graphutils import *
from src.transformations import order_nodes

from itertools import chain


def project_one_dag(dag: DAG) -> List[Sequence[str]]:
    leaves = set(filter(dag.is_leaf, dag.nodes))
    leaves = order_nodes(dag, leaves)
    return list(map(lambda leaf: project_leaf(dag, leaf)[0:4], leaves))


def project_leaf(dag: DAG, leaf: Node) -> Sequence[str]:

    def wrap_mwu(fun: Callable[[Node], str]) -> Callable[[Node], str]:
        return lambda leaf_: 'mwp' if ' ' in get_word(leaf_) else fun(leaf_)

    def get_word(leaf_: Node) -> str:
        return dag.attribs[leaf_]['word']

    def get_type(leaf_: Node) -> str:
        return dag.attribs[leaf_]['type'].depolarize().__str__()

    @wrap_mwu
    def get_pos(leaf_: Node) -> str:
        return dag.attribs[leaf_]['pos']

    @wrap_mwu
    def get_postag(leaf_: Node) -> str:
        return dag.attribs[leaf_]['postag']

    @wrap_mwu
    def get_pt(leaf_: Node) -> str:
        return dag.attribs[leaf_]['pt']

    @wrap_mwu
    def get_lemma(leaf_: Node) -> str:
        return dag.attribs[leaf_]['lemma']

    return tuple(map(lambda function: function(leaf), (get_word, get_type, get_pos, get_postag, get_lemma)))


WordTypePos = Tuple[str, str, str]


def get_wtp_tuples(dags: List[DAG]) -> List[WordTypePos]:
    return list(map(lambda seq: tuple(seq[0:3]),
                    list(chain.from_iterable(list(map(project_one_dag, dags))))))


def wp_to_t(wtps: List[WordTypePos]):
    def getkey(wtp: WordTypePos) -> Tuple[str, str]:
        return fst(wtp), last(wtp)

    def getvalue(wtp: WordTypePos) -> str:
        return snd(wtp)

    keys = set(map(getkey, wtps))
    values = set(map(getvalue, wtps))

    outer = {k: {v: 0 for v in values} for k in keys}

    for wtp in wtps:
        key = getkey(wtp)
        value = getvalue(wtp)
        outer[key][value] = outer[key][value] + 1
    return outer


def sort_wpt(outer: Dict[Tuple[str, str], Dict[str, int]]) -> List[Tuple[Tuple[str, str], Sequence[Tuple[str, int]]]]:
    outer = {outer_key:
                 sorted(filter(lambda pair: snd(pair) > 0, inner_dict.items()),
                        key=lambda pair: snd(pair),
                        reverse=True)
             for outer_key, inner_dict in outer.items()}
    return sorted(outer.items(), key=lambda pair: sum(list(map(snd, snd(pair)))))


def print_sorted_wpt(outer: List[Tuple[Tuple[str, str], Sequence[Tuple[str, int]]]]):
    def print_one(wpt: Tuple[Tuple[str, str], Sequence[Tuple[str, int]]]) -> str:
        def print_inner(inner: Sequence[Tuple[str, int]]) -> str:
            def print_pair(pair: Tuple[str, int]) -> str:
                return fst(pair) + ' #= ' + str(snd(pair))
            return ' | '.join(list(map(print_pair, inner)))
        return fst(fst(wpt))+'\t'+snd(fst(wpt)) + '\t' + print_inner(snd(wpt))
    return '\n'.join(list(map(print_one, outer)))
