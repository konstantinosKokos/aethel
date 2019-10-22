from itertools import chain

from src.extraction import Extraction, _cat_dict, _pt_dict, _head_deps, _mod_deps, ExtractionError
from src.lassy import Lassy
from src.proofs import Prove
from src.transformations import Transformation
from src.viz import ToGraphViz

from typing import Optional, Union, Tuple, List
from src.graphutils import DAG
from src.proofs import ProofNet

_lassy = Lassy()
_transform = Transformation()
_extract = Extraction(_cat_dict, _pt_dict, 'pt', _head_deps, _mod_deps)
_prove = Prove()
_viz = ToGraphViz()


def test(size, start=0):
    meta = [{'src': _lassy[i][1]} for i in range(start, start+size)]

    transformed = list(chain.from_iterable(list(map(_transform,
                                                    list(map(lambda i: _lassy[i][2], range(start, start+size))),
                                                    meta))))
    transformed = list(filter(lambda t: t is not None, transformed))

    extracted = list(map(_extract, transformed))
    extracted = list(filter(lambda e: e is not None, extracted))
    return extracted
    # prover = Prover(extracted, _prove)
    # return prover


def compose(sample: Union[int, str]) -> List[Optional[Tuple[ProofNet, DAG]]]:
    dags = _transform(_lassy[sample][2])
    extracted = list(filter(lambda e: e is not None, list(map(_extract, dags))))
    proven = list(map(_prove, extracted))
    return proven
