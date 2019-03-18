from src.WordType import polish, WordType, CombinatorType
from src.utils.PostProcess import deannotate
from typing import Tuple, List, Sequence, TypeVar, Callable
from itertools import chain
from collections import Counter
from functools import reduce
import pickle
from src.utils.PostProcess import get_unique, indexize

T = TypeVar('T')


def is_non_binary_comb(x: WordType):
    return True if isinstance(x, CombinatorType) and len(x.types) != 2 else False


def index_non_binary_combs(Y: List[Sequence[WordType]]) -> List[int]:
    return [i for i, y in enumerate(Y) if not any(list(map(is_non_binary_comb, y)))]


def flatten(Y: List[Sequence[T]]) -> List[T]:
    return list(chain.from_iterable(Y))


def polish_many(y: List[WordType]) -> List[str]:
    return list(map(polish, y))


def pair(y: str) -> List[Tuple[str, str]]:
    y = y.split()
    return list(zip(y, y[1:]))


def pair_many(Y: List[str]) -> List[Tuple[str, str]]:
    return list(filter(lambda x: x, chain.from_iterable(map(pair, Y))))


def replace_one(y: str, merged: Tuple[str, str]) -> str:
    l = len(' '.join(merged))
    if ' '.join(merged) == y:
        return '+'.join(merged)
    if ' ' + ' '.join(merged) + ' ' in y:
        y = y.replace(' ' + ' '.join(merged) + ' ', ' ' + '+'.join(merged) + ' ')
    if ' ' + ' '.join(merged) == y[-(l+1):]:
        y = y[:len(y) - l - 1] + ' ' + '+'.join(merged)
    if ' '.join(merged) + ' ' == y[:l+1]:
        y = '+'.join(merged) + ' ' + y[l+1:]
    return y


def replace_many(Y: List[str], merged: Tuple[str, str]) -> List[str]:
    return list(map(lambda x: replace_one(x, merged), Y))


def BPE_step(Y: List[str], merges: List[Tuple[Tuple[str, str], int]]) \
        -> Tuple[List[str], List[Tuple[Tuple[str, str], int]]]:
    Y_paired = pair_many(Y)
    c = Counter(Y_paired)
    most_common_k, most_common_v = c.most_common(1)[0]
    new_Y = replace_many(Y, most_common_k)
    merges.append((most_common_k, most_common_v))
    return new_Y, merges


def BPE(Y: List[str], max_merges: int=5000) -> List[Tuple[Tuple[str, str], int]]:
    merges = []
    while len(merges) < max_merges:
        try:
            Y, merges = BPE_step(Y, merges)
            print('Merge {}: {} -- {}'.format(len(merges), merges[-1][0], merges[-1][1]))
        except IndexError:
            return merges
    return merges


def functionalize_merges(merges: List[Tuple[Tuple[str, str], int]]) -> Callable:
    fst = lambda x: x[0]
    merges = list(map(fst, merges))
    return lambda s: reduce(lambda x, y: replace_one(x, y), merges, s)


def encode(Y: List[Sequence[str]], merges: List[Tuple[Tuple[str, str], int]]) -> List[Sequence[str]]:
    merge_fn = functionalize_merges(merges)
    return list(map(lambda y: list(map(merge_fn, y)), Y))


def do_everything():
    with open('XYZ.p', 'rb') as f:
        X, Y, _ = pickle.load(f)
    Y = deannotate(Y)
    I = index_non_binary_combs(Y)
    X = [X[i] for i in I]
    Y = [Y[i] for i in I]
    Y_p = flatten(Y)
    Y_p = polish_many(Y_p)
    merges = BPE(Y_p)

    Y_enc = encode(list(map(polish_many, Y)), merges[:100])
    Y_enc_joined = list(map(lambda y: ' <TE> '.join(y), Y_enc))
    Y_enc_split = list(map(lambda y: y.split(), Y_enc_joined))
    unique = get_unique(Y_enc_joined)
    type_to_int = indexize(unique)
    int_to_type = {v: k for k, v in type_to_int.items()}

    Y_int = list(map(lambda y: list(map(lambda t: type_to_int[t], y)), Y_enc_split))
