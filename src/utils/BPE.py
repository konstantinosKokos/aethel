from src.WordType import polish, WordType, CombinatorType
from src.utils.PostProcess import deannotate
from typing import Tuple, List, Sequence, TypeVar, Set, Optional, Dict
from itertools import chain
from collections import Counter
import re
import pickle

T = TypeVar('T')


def flatten(Y: List[Sequence[T]]) -> List[T]:
    return list(chain.from_iterable(Y))


def polish_many(Y: List[WordType]) -> List[str]:
    return list(map(polish, Y))


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


def do_everything():
    with open('XYZ.p', 'rb') as f:
        _, Y, _ = pickle.load(f)
    Y = deannotate(Y)
    Y = flatten(Y)
    Y = list(filter(lambda x: not (isinstance(x, CombinatorType) and len(x.types)>2), Y))
    Y = polish_many(Y)
    return BPE(Y)
