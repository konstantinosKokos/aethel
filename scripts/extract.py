from LassyExtraction.utils.lassy import Lassy
from LassyExtraction.utils.graph import DAG
from LassyExtraction.transformations import prepare_many, get_lex_nodes, sort_nodes, is_ghost
from LassyExtraction.frontend import Sample, LexicalPhrase, LexicalItem
from LassyExtraction.extraction import prove_dag, ExtractionError, Proof
import os


name_to_subset = {name: subset for name, subset in
                  map(lambda x: x.split('\t'), open('../data/name_to_subset.tsv').read().splitlines())}


def get_lex_phrases(dag: DAG[str]) -> list[tuple[Proof, tuple[LexicalItem, ...]]]:
    def make_item(node: str) -> LexicalItem:
        word, pos, pt,  lemma = (dag.get(node, attr) for attr in ('word', 'pos', 'pt', 'lemma'))
        assert not any(x is None for x in (word, pos, pt, lemma))
        return LexicalItem(word=word, pos=pos, pt=pt, lemma=lemma)
    bottom_nodes = [(node,
                     sort_nodes(dag, {e.target for e in dag.outgoing_edges(node)
                                      if e.label == 'mwp' and not is_ghost(dag, e.target)}))
                    for node in get_lex_nodes(dag) if not is_ghost(dag, node)]
    return [(dag.get(bottom, 'proof'), tuple(make_item(c) for c in children) if children else (make_item(bottom),))
            for bottom, children in bottom_nodes]


def make_sample(dag: DAG[str]) -> Sample:
    proof = prove_dag(dag)
    phrasal_proofs, lex_itemss = list(zip(*get_lex_phrases(dag)))
    lex_phrases = tuple(LexicalPhrase(type=sp.type, items=items) for sp, items in zip(phrasal_proofs, lex_itemss))
    proof = proof.translate_lex({sp.term.index: i for i, sp in enumerate(phrasal_proofs)}).eta_norm().standardize_vars()
    return Sample(
        lexical_phrases=lex_phrases,
        proof=proof,
        name=(name := dag.meta['name']),
        subset=name_to_subset[name.split('(')[0]])


def store_aethel(version: str,
                 transform_path: str = '../data/transformed.pickle',
                 save_intermediate: bool = False,
                 output_path: str = f'../data/aethel.pickle') -> None:
    import pickle
    if save_intermediate or not os.path.exists(transform_path):
        lassy = Lassy()
        print('Transforming LASSY trees...')
        transformed = prepare_many(lassy)
        if save_intermediate:
            with open(transform_path, 'wb') as f:
                print('Saving transformed trees...')
                pickle.dump(transformed, f)
                print('Done.')
    else:
        with open(transform_path, 'rb') as f:
            print('Loading transformed trees...')
            transformed = pickle.load(f)
            print('Loaded.')

    print('Proving transformed trees...')
    train, dev, test = [], [], []
    for tree in transformed:
        try:
            sample = make_sample(tree)
        except ExtractionError:
            continue
        (train if sample.subset == 'train' else dev if sample.subset == 'dev' else test).append(sample)
    print(f'Proved {(l:=sum(map(len, (train, dev, test))))} samples. (coverage: {l / len(transformed)})')
    print('Saving samples...')
    with open(output_path, 'wb') as f:
        pickle.dump((version, [[sample.save() for sample in subset] for subset in (train, dev, test)]), f)
    print('Done.')


if __name__ == '__main__':
    store_aethel('1.0.0a3', save_intermediate=False)
