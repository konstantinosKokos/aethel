from itertools import chain
import pickle
from utils import fastText


def load():
    with open('test-output/sequences/words-types.p', 'rb') as f:
        wss, tss = pickle.load(f)
    return wss, tss


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
    all_types = get_all_unique(type_sequences)
    counts = {t: 0 for t in all_types}
    for ts in type_sequences:
        for t in ts:
            counts[t] = counts[t] + 1
    return counts


def get_high_occurrence_sequences(word_sequences, type_sequences, threshold):
    """
    Given an iterable of word sequences, an iterable of their corresponding type sequences, and a minimum occurrence
    threshold, returns a new pair of iterables by pruning type sequences containing low occurrence types.
    :param word_sequences:
    :param type_sequences:
    :param threshold:
    :return:
    """
    assert (len(word_sequences) == len(type_sequences))
    type_occurrences = get_type_occurrences(type_sequences)
    kept_indices = [i for i in range(len(word_sequences)) if not
                    list(filter(lambda x: type_occurrences[x] < threshold, type_sequences[i]))]
    return [word_sequences[i] for i in kept_indices], [type_sequences[i] for i in kept_indices]


# todo: vectorize word sequences

# todo: one-hot type sequences

