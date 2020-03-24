from LassyExtraction.transformations import order_nodes
from LassyExtraction.graphutils import DAG
from LassyExtraction.transformations import get_sentence as get_words
from LassyExtraction.milltypes import WordTypes


def get_types(dag: DAG) -> WordTypes:
    return list(map(lambda leaf: dag.attribs[leaf]['type'], list(order_nodes(dag, list(dag.get_leaves())))))
