from src.WordType import polish, WordType
from typing import Tuple, List, Sequence, TypeVar, Set, Optional, Dict
from itertools import chain
from collections import Counter
import re

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
        y = re.sub(' ' + ' '.join(merged) + ' ', ' ' + '+'.join(merged) + ' ', y)
    if ' ' + ' '.join(merged) == y[-(l+1):]:
        y = y[:len(y) - l - 1] + ' ' + '+'.join(merged)
    if ' '.join(merged) + ' ' == y[:l+1]:
        y = '+'.join(merged) + ' ' + y[l+1:]
    return y


def replace_many(Y: List[str], merged: Tuple[str, str]) -> List[str]:
    return list(map(lambda x: replace_one(x, merged), Y))


def BPE(Y: List[str], depth: int=1) -> None:
    Y_paired = pair_many(Y)
    c = Counter(Y_paired)
    most_common_k, most_common_v = c.most_common(1)[0]
    new_Y = replace_many(Y, most_common_k)
    print('Merge {} {} ({})'.format(depth, most_common_k, most_common_v))
    import pdb
    pdb.set_trace()
    BPE(new_Y, depth=depth+1)


def do_everything(Y: List[Sequence[WordType]]):
    Y = flatten(Y)
    Y = polish_many(Y)
    BPE(Y)
