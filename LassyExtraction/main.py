from itertools import chain

from .extraction import extractor
from .lassy import Lassy
from .proofs import prover, AxiomLinks
from .transformations import transformer
from .graphutils import DAG
from .viz import ToGraphViz

from typing import Optional, Union, Tuple, List


_lassy = Lassy()
_viz = ToGraphViz()


def compose(sample: Union[int, str]) -> List[Optional[Tuple[AxiomLinks, DAG]]]:
    dags = transformer(_lassy[sample][2], meta={'src': _lassy[sample][1]})
    extracted = list(filter(lambda e: e is not None, list(map(extractor, dags))))
    proven = list(map(lambda e: prover(e, raise_errors=False), extracted))
    return proven


def exhaust():
    return list(chain.from_iterable(map(lambda i: compose(i), range(len(_lassy)))))
