from typing import Tuple, List, Sequence, TypeVar, Set, Optional, Dict
from collections import Counter
from itertools import chain
try:
    from src.WordType import AtomicType
except ImportError:
    from LassyExtraction.src.WordType import AtomicType

"""
Post-processing of the extraction output
"""

T1 = TypeVar('T1')
T2 = TypeVar('T2')
UNK = AtomicType('UNK')


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


def annotated_matchings(X: List[Tuple[int, Sequence[T1]]], Y: List[Sequence[T1]],
                        sX: Optional[Set[T1]]=None, sY: Optional[Set[T2]]=None) -> Dict[T1, List[Tuple[int, T2]]]:
    if sX is None:
        sX = get_unique(deannotate(X))
    if sY is None:
        sY = get_unique(Y)
    d = {sx: [] for sx in sX}

    for i, (index, it) in enumerate(X):
        for j, sx in enumerate(it):
            d[sx].append((index, Y[i][j]))
    return d


def indexize(X: Set[T1], special: Sequence[T1]=tuple()) -> Dict[T1, int]:
    return {**{x: i+1 for i, x in enumerate(X)}, **{s: 0 for s in special}}


def freqsort(counter: Dict[T1, int], indices: Dict[T1, int]):
    counter = sorted([(k, v) for k, v in counter.items()], key=lambda x: x[1], reverse=True)
    ret = [(indices[c], c, freq) for c, freq in counter]
    return ret


def search(t: T1, X: Dict[T1, Sequence[Tuple[int, T2]]]) -> Sequence[int]:
    return [x[0] for x in X[t]]


def search_by_id(i: int, inverted: Dict[int, T1], X: Dict[T1, Sequence[Tuple[int, T2]]]) -> Sequence[int]:
    return [x[0] for x in X[inverted[i]]]


# # # # # # # # # # # # # # # # Filtering Tools # # # # # # # # # # # # # # # #


def filter_by_occurrence(Y: List[Sequence[T2]], mode: str, threshold: int, counter: Optional[Dict[T2, int]]=None) \
        -> List[int]:
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
    return [i for i, y in enumerate(Y) if cond(list(map(lambda x: cmp(counter[x], threshold), y)))]


def filter_by_length(X: List[Sequence[T1]], threshold: int) -> List[int]:
    return [i for i, x in enumerate(X) if len(x) < threshold]


def replace_by_occurrence(Y: List[Sequence[T2]], threshold: int, default: T2, counter: Optional[Dict[T2, int]]=None) \
        -> List[Sequence[T2]]:
    if counter is None:
        counter = count_occurrences(Y)
    return list(map(lambda x: replace_one(x, threshold, counter, default), Y))


def replace_one(y: Sequence[T2], threshold: int, counter: Dict[T2, int], default: T2) -> Sequence[T2]:
    return [default if counter[item] < threshold else item for item in y]

# # # # # # # # # # # # # # # Quickstart Scripts # # # # # # # # # # # # # # #


def create_dataset():
    from src import Extraction
    L0, L, tg = Extraction.main()
    aX, aY, aZ = Extraction.iterate(L, num_workers=8, batch_size=128)
    X, Y, Z = deannotate(aX), deannotate(aY), deannotate(aZ)
    print('Unique Types: {}'.format(len(get_unique(Y))))
    high_freq_indices = filter_by_occurrence(Y, 'min', 10, count_occurrences(Y))
    fX = [X[i] for i in high_freq_indices]
    fY = [Y[i] for i in high_freq_indices]
    fZ = [Z[i] for i in high_freq_indices]
    print('Frequent Types: {}'.format(len(get_unique(fY))))
    low_len_indices = filter_by_length(fX, 40)
    lfX = [fX[i] for i in low_len_indices]
    lfY = [fY[i] for i in low_len_indices]
    lfZ = [fZ[i] for i in low_len_indices]
    print('Type Coverage: {}'.format(len(fX)/len(X)))
    return lfX, lfY, lfZ, indexize(get_unique(lfY), special=[UNK])


def create_dataset_with_replace():
    from src import Extraction
    L0, L, tg = Extraction.main()
    aX, aY, aZ = Extraction.iterate(L, num_workers=8, batch_size=128)
    X, Y, Z = deannotate(aX), deannotate(aY), deannotate(aZ)
    print('Unique Types: {}'.format(len(get_unique(Y))))
    from src.WordType import AtomicType
    default = AtomicType('UNK')
    fX = X
    fY = replace_by_occurrence(Y, 10, default, count_occurrences(Y))
    fZ = Z
    print('Frequent Types: {}'.format(len(get_unique(fY))))
    low_len_indices = filter_by_length(fX, 25)
    lfX = [fX[i] for i in low_len_indices]
    lfY = [fY[i] for i in low_len_indices]
    lfZ = [fZ[i] for i in low_len_indices]
    return lfX, lfY, lfZ, indexize(get_unique(lfY), special=[UNK])


def create_stats():
    from src import Extraction

    L0, L, tg = Extraction.main()
    aX, aY, aZ = Extraction.iterate(L, num_workers=8, batch_size=128)
    X, Y, Z = deannotate(aX), deannotate(aY), deannotate(aZ)
    print(len(get_unique(Y)))
    m = annotated_matchings(aY, X)
    reverse_m = annotated_matchings(aX, Y)
    indices = indexize(get_unique(Y))
    type_occurrences = count_occurrences(Y)

    f = freqsort(type_occurrences, indices)
    return f, {v: k for k, v in indices.items()}, m, reverse_m, L0, L, tg
