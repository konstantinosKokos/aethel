from itertools import chain

from .extraction import extractor
from .lassy import Lassy
from .proofs import prover, AxiomLinks
from .transformations import transformer
from .graphutils import DAG

from typing import Union, Tuple, Iterable

_lassy = Lassy()


def compose(sample: Union[int, str]) -> Iterable[Tuple[AxiomLinks, DAG]]:
    dags = transformer(_lassy[sample][2], meta={'src': _lassy[sample][1]})
    return filter(lambda e: e is not None, map(prover, filter(lambda e: e is not None, map(extractor, dags))))


def exhaust() -> Iterable[Tuple[AxiomLinks, DAG]]:
    return chain.from_iterable(map(lambda i: compose(i), range(len(_lassy))))
