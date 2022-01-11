"""
    A
    n example interface to Alpino parse outputs.
"""

from LassyExtraction.transformations import prepare_for_extraction
from LassyExtraction.extraction import prove, Proof
from LassyExtraction.utils.lassy import parse


def parse_alpino_file(path: str) -> list[Proof]:
    """
        Parses an Alpino file containing a single sentence and returns a list of proofs.
    """
    with open(path, 'r') as f:
        etree = parse(path)
    name = etree.find('sentence').attrib['sentid']
    trees = prepare_for_extraction(etree, name)
    return [prove(tree) for tree in trees]
