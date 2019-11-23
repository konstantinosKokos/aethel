from itertools import chain

from LassyExtraction.extraction import Extraction, _cat_dict, _pt_dict, _head_deps, _mod_deps, ExtractionError
from LassyExtraction.lassy import Lassy
from LassyExtraction.proofs import Prove, ProofError
from LassyExtraction.transformations import Transformation
from LassyExtraction.viz import ToGraphViz

from typing import Optional, Union, Tuple, List
from LassyExtraction.graphutils import DAG
from LassyExtraction.proofs import ProofNet

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


def count_errors():
    meta = [{'src': _lassy[i][1]} for i in range(len(_lassy))]

    transformed = list(chain.from_iterable(list(map(_transform,
                                                    list(map(lambda i: _lassy[i][2], range(len(_lassy)))),
                                                    meta))))

    extracted = []

    ex_errors = []
    for t in transformed:
        try:
            extracted.append(_extract(t, raise_errors=True))
        except ExtractionError as ee:
            ex_errors.append(ee)

    proven = []

    proof_errors = []
    for e in extracted:
        try:
            proven.append(_prove(e, raise_errors=True))
        except ProofError as pe:
            proof_errors.append(pe)
    return ex_errors, proof_errors


def compose(sample: Union[int, str]) -> List[Optional[Tuple[ProofNet, DAG]]]:
    dags = _transform(_lassy[sample][2], meta={'src': _lassy[sample][1]})
    extracted = list(filter(lambda e: e is not None, list(map(_extract, dags))))
    proven = list(map(lambda e: _prove(e, raise_errors=False), extracted))
    return proven
