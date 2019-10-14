from itertools import chain

from src.extraction import Extraction, _cat_dict, _pt_dict, _head_deps, _mod_deps
from src.lassy import Lassy
from src.proofs import Prove
from src.transformations import Transformation
from src.viz import ToGraphViz

_lassy = Lassy()
_transform = Transformation()
_extract = Extraction(_cat_dict, _pt_dict, 'pt', _head_deps, _mod_deps)
_prove = Prove()
_viz = ToGraphViz()


def test(samples=100):
    meta = [{'src': _lassy[i][1]} for i in range(samples)]

    transformed = list(chain.from_iterable(list(map(_transform,
                                                    list(map(lambda i: _lassy[i][2], range(samples))), meta))))
    transformed = list(filter(lambda t: t is not None, transformed))
    extracted = list(map(_extract, transformed))
    extracted = list(filter(lambda e: e is not None, extracted))
    return extracted
    # prover = Prover(extracted, _prove)
    # return prover
