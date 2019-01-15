from typing import Tuple, List, Sequence, TypeVar, Set, Optional, Dict, Any
from collections import Counter
from itertools import chain

"""
Post-processing of the extraction output
"""

T1 = TypeVar('T1')
T2 = TypeVar('T2')

# # # # # # # # # # # # # # # # Search and Exploration Tools # # # # # # # # # # # # # # # #


def deannotate(X: List[Tuple[int, Sequence[T1]]]) -> List[Sequence[T1]]:
    return [x[1] for x in X]


def count_occurrences(X: List[Sequence[T1]]) -> Dict[T1, int]:
    return Counter([x for x in chain(*X)])


def get_unique(X: List[Sequence[T1]]) -> Set[T1]:
    return set([x for x in chain.from_iterable(X)])


def matchings(X: List[Sequence[T1]], Y: List[Sequence[T2]], sX: Optional[Set[T1]]=None, sY: Optional[Set[T2]]=None) \
        -> Dict[T1, List[T2]]:
    if sX is None:
        sX = get_unique(X)
    if sY is None:
        sY = get_unique(Y)
    d = {sx: [] for sx in sX}
    for i, it in enumerate(X):
        for j, sx in enumerate(it):
            d[sx].append(Y[i][j])
    return d


def annotated_matchings(X: List[Tuple[int, Sequence[T1]]], Y: List[Tuple[int, Sequence[T1]]],
                        sX: Optional[Set[T1]]=None, sY: Optional[Set[T2]]=None) -> Dict[T1, List[Tuple[int, T2]]]:
    if sX is None:
        sX = get_unique(deannotate(X))
    if sY is None:
        sY = get_unique(deannotate(Y))
    d = {sx: [] for sx in sX}
    Y = deannotate(Y)
    for i, (index, it) in enumerate(X):
        for j, sx in enumerate(it):
            d[sx].append((index, Y[i][j]))
    return d


def indexize(X: Set[T1]) -> Dict[T1, int]:
    return {x: i for i, x in enumerate(X)}


def freqsort(counter: Dict[T1, int], indices: Dict[T1, int]):
    counter = sorted([(k, v) for k, v in counter.items()], key=lambda x: x[1], reverse=True)
    ret = [(indices[c], c, freq) for c, freq in counter]
    return ret


def search(t: T1, X: Dict[T1, Sequence[Tuple[int, T2]]]) -> Sequence[int]:
    return [x[0] for x in X[t]]


def search_by_id(i: int, inverted: Dict[int, T1], X: Dict[T1, Sequence[Tuple[int, T2]]]) -> Sequence[int]:
    return [x[0] for x in X[inverted[i]]]


# # # # # # # # # # # # # # # # Filtering Tools # # # # # # # # # # # # # # # #

def filter_by_occurrence(X: List[Sequence[T1]], Y: List[Sequence[T2]], mode: str, threshold: int,
                         counter: Dict[T2, int]=None) -> Tuple[List[Sequence[T1]], List[Sequence[T2]]]:
    if counter is None:
        counter = count_occurrences(Y)
    if mode == 'max':
        cmp = int.__lt__
        cond = any
    elif mode == 'min':
        cmp = int.__ge__
        cond = all
    else:
        raise ValueError('mode must be "min" or "max"')
    indices = [i for i, y in enumerate(Y) if cond(list(map(lambda x: cmp(counter[x], threshold), y)))]
    return [X[i] for i in indices], [Y[i] for i in indices]


def filter_by_length(X: List[Sequence[T1]], Y: List[Sequence[T2]], threshold: int) \
                     -> Tuple[List[Sequence[T1]], List[Sequence[T2]]]:
    indices = [i for i, x in enumerate(X) if len(x) < threshold]
    return [X[i] for i in indices], [Y[i] for i in indices]

