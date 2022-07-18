"""
    An example interface to Alpino parse outputs.
"""

from LassyExtraction.transformations import prepare_for_extraction
from LassyExtraction.extraction import prove_dag, Proof, ExtractionError
from LassyExtraction.utils.lassy import parse


def parse_alpino_file(path: str) -> list[Proof]:
    """
        Parses an Alpino file containing a single sentence and returns a list of proofs.
    """
    etree = parse(path)
    name = etree.find('sentence').attrib['sentid']
    dags = prepare_for_extraction(etree, name)
    proofs = []
    for dag in dags:
        try:
            proofs.append(prove_dag(dag))
        except ExtractionError:
            continue
    return proofs
