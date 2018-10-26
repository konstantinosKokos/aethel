from itertools import chain
import pickle
from utils import fastText


def get_all_unique(iterable_of_sequences):
    return set([x for x in chain.from_iterable(iterable_of_sequences)])


def make_oov_files(input_file='test-output/sequences/words-types.p', iv_vectors_file='wiki.nl/wiki.nl/vec',
                   oov_words_file='wiki.nl/oov.txt', oov_vectors_file='wiki.nl/oov.vec'):
    with open(input_file, 'rb') as f:
        ws, ts = pickle.load(f)

    all_words = get_all_unique(ws)

    model_vectors = fastText.load_vectors(iv_vectors_file)
    oov_words = fastText.find_oov(all_words, model_vectors)
    fastText.write_oov(oov_words, oov_words_file=oov_words_file)
    fastText.vectorize_oov(oov_words_file, oov_vectors_file)

def get_type_occurrences(type_sequences):
    pass