import pickle
from utils import FastText
from collections import defaultdict, Counter
from functools import reduce
from itertools import chain
from WordType import decolor, ColoredType

import torch
from torch.utils.data import Dataset

import numpy as np
from matplotlib import pyplot as plt


# # # # # # # # General stuff # # # # # # # #

def load(file='test-output/sequences/words-types.p'):
    with open(file, 'rb') as f:
        wss, tss = pickle.load(f)
    return wss, tss


def get_all_unique(iterable_of_sequences):
    return set([x for x in chain.from_iterable(iterable_of_sequences)])


def make_oov_files(input_file='test-output/sequences/words-types.p', iv_vectors_file='wiki.nl/wiki.nl.vec',
                   oov_words_file='wiki.nl/oov.txt', oov_vectors_file='wiki.nl/oov.vec'):
    with open(input_file, 'rb') as f:
        ws, ts = pickle.load(f)

    all_words = get_all_unique(ws)

    model_vectors = FastText.load_vectors(iv_vectors_file)
    oov_words = FastText.find_oov(all_words, model_vectors)
    FastText.write_oov(oov_words, oov_words_file=oov_words_file)
    FastText.vectorize_oov(oov_words_file, oov_vectors_file)


def get_type_occurrences(type_sequences):
<<<<<<< HEAD
    return Counter([x for x in chain(*type_sequences)])
=======
    return Counter([tuple(x) for x in chain(*type_sequences)])
>>>>>>> origin/master


def get_all_chars(word_sequences):
    return list(reduce(set.union, [set(''.join(w)) for w in word_sequences], set()))


def filter_by_occurrence(word_sequences, type_sequences, threshold, mode='min', return_indices=False):
    """
    Given an iterable of word sequences, an iterable of their corresponding type sequences, and a minimum occurrence
    threshold, returns a new pair of iterables by pruning type sequences containing low occurrence types.
    :param return_indices:
    :type return_indices:
    :param mode:
    :type mode:
    :param word_sequences:
    :param type_sequences:
    :param threshold:
    :return:
    """
    assert (len(word_sequences) == len(type_sequences))
    type_occurrences = get_type_occurrences(type_sequences)
<<<<<<< HEAD
    # kept indices are the ones where each type has an occurrence above the threshold
    kept_indices = [i for i in range(len(s.type_sequences)) if
                    all(list(map(lambda x: type_occurrences[x] > threshold, type_sequences[i])))]
=======
    kept_indices = [i for i in range(len(word_sequences)) if not
                        list(filter(lambda x: type_occurrences[tuple(x)] <= threshold, type_sequences[i]))]
>>>>>>> origin/master
    if mode != 'min':
        kept_indices = [i for i in range(len(word_sequences)) if i not in kept_indices]
    if return_indices:
        return kept_indices
    else:
        return [word_sequences[i] for i in kept_indices], [type_sequences[i] for i in kept_indices]


def get_low_len_sequences(word_sequences, type_sequences, max_len):
    """

    :param word_sequences:
    :param type_sequences:
    :param max_len:
    :return:
    """
    assert len(word_sequences) == len(type_sequences)
    kept_indices = [i for i in range(len(word_sequences)) if len(word_sequences[i]) <= max_len]
    return [word_sequences[i] for i in kept_indices], [type_sequences[i] for i in kept_indices]


def get_low_len_words(word_sequences, type_sequences, max_word_len):
    assert len(word_sequences) == len(type_sequences)
    kept_indices = [i for i in range(len(word_sequences)) if
                    all(map(lambda x: len(x) <= max_word_len, word_sequences[i]))]
    return [word_sequences[i] for i in kept_indices], [type_sequences[i] for i in kept_indices]


def map_ws_to_vs(word_sequence, vectors):
    return torch.Tensor(list(map(lambda x:
                                 vectors[x] if len(x.split()) == 1 else sum(list(map(lambda y: vectors[y], x.split()))),
                                 word_sequence)))


def map_cs_to_is(char_sequence, char_dict, max_len):
    # return torch.Tensor(list(map(lambda x: char_dict[x], char_sequence)))
    a = torch.zeros(max_len)
    b = torch.Tensor(list(map(lambda x: char_dict[x], char_sequence)))
    a[:len(b)] = b
    return a

# # # # # # # # Lassy Stuff # # # # # # # #


def all_atomic(type_sequences):
    atomic = set()
    for ts in type_sequences:
        atomic = atomic.union(reduce(set.union, [t.retrieve_atomic() for t in ts]))
    atomic_dict = {i+1: k for i, k in enumerate(sorted([',', '(', ')', '→', '&'] + list(atomic), key=lambda x: -len(x)))}
    atomic_dict[len(atomic_dict) + 1] = '<SOS>'
    atomic_dict[len(atomic_dict) + 1] = '<EOS>'
    atomic_dict[0] = '<PAD>'
    return atomic_dict


# # # # # # # # FR stuff # # # # # # # #


def load_data_fr(x_path='fr/X.txt', y_path='fr/Ypolish.txt', store_path='test-output/sequences/words-types_fr.p'):
    with open(x_path) as f:
        samples = [line.split() for line in f]
    with open(y_path) as f:
        targets = [line.split('|')[:-1] for line in f]

    def space(fr_type):
        fr_type = fr_type.replace('0\\', '0 \\').replace('1\\', '1 \\').replace('0/', '0 /').replace('1/', '1 /').\
            replace('0@', '0 @').replace('1@', '1 @').replace('1^', '1 ^').replace('0^', '0 ^').replace('0*', '0 *').\
            replace('1*', '1 *').replace(' txt', '')
        return fr_type

    targets = list(map(lambda x: list(map(space, x)), targets))

    assert len(samples) == len(targets)
    assert all(list(map(lambda x: len(x[0]) == len(x[1]), zip(samples, targets))))

    if store_path is not None:
        with open(store_path, 'wb') as f:
            pickle.dump([samples, targets], f)
    else:
        return samples, targets


def init_fr_dict():
    aux = tuple()
    connectives = ('/0', '\\0', '/1', '\\1', '^0', '^1', '@0', '@1', '*0', '*1')
    atoms = ('cl_r', 'cl_y', 'n', 'np', 's', 's_inf', 's_pass', 's_ppart', 's_ppres', 's_q', 's_whq', 'pp', 'pp_a',
             'pp_de', 'pp_par', 'let')
    union = aux + connectives + atoms
    atomic_dict = {i+1: k for i, k in enumerate(sorted(union, key=lambda x: -len(x)))}
    atomic_dict[len(atomic_dict) + 1] = '<SOS>'
    atomic_dict[len(atomic_dict) + 1] = '<EOS>'
    atomic_dict[0] = '<PAD>'
    return atomic_dict


def convert_type_to_vector(wt, int_to_atomic_type, atomic_type_to_int, process = lambda x: x):
    # wt = wt.__repr__()
    wt = process(wt)
    to_return = [atomic_type_to_int['<SOS>']]
    # to_return = []
    wt = wt.replace(',', ' ,').replace('(', '( ').replace(')', ' )')
    for subpart in wt.split():
        to_return.append(atomic_type_to_int[subpart])
    to_return.append(atomic_type_to_int['<EOS>'])
    return to_return


def convert_vector_sequence_to_type_sequence(ts, int_to_atomic_type, spaces=True):
    """
    Takes a sequence of type_vectors and gives back a tuple of strings. Go figure..
    :param ts:
    :param int_to_atomic_type:
    :return:
    """
    to_return = []
    for tv in ts:
        local = []
        for i, t in enumerate(tv):
            if i == 0 and int_to_atomic_type[t] == '<SOS>':
                continue
            if int_to_atomic_type[t] == '<EOS>':
                break
            if int_to_atomic_type[t] == '<PAD>':
                continue
            local.append(int_to_atomic_type[t])
        if spaces:
            to_return.append(' '.join(local).replace(' ,', ',').replace('( ', '(').replace(' )', ')'))
        else:
            to_return.append(' '.join(local))
    return tuple(to_return)


def convert_many_vector_sequences_to_type_sequences(many_vector_sequences, int_to_atomic_type):
    return list(map(lambda x: convert_vector_sequence_to_type_sequence(x, int_to_atomic_type), many_vector_sequences))


def map_tv_to_is(type_vector, max_len):
    # return torch.Tensor(type_vector).long()
    a = torch.zeros(max_len).long()
    b = torch.Tensor(type_vector).long()
    a[:len(b)] = b
    return a


def cdf(sequences):
    x = np.sort([len(x) for x in sequences])
    f = np.array(range(len(sequences)))/float(len(sequences))
    plt.plot(x, f)
    plt.show()


def decolor_sequences(type_sequences):
    return list(map(lambda x: tuple(map(decolor, x)), type_sequences))


def get_atomic_occurrences(type_vectors, int_to_atomic_type):
    int_to_occurrences = {x: 0 for x in int_to_atomic_type.keys()}
    for type_vector_sequence in type_vectors:
        for type_vector in type_vector_sequence:
            indices = list(filter(lambda x: int_to_atomic_type[x], type_vector))
            for index in indices:
                int_to_occurrences[index] = int_to_occurrences[index] + 1
    return [int_to_occurrences[i] for i in range(len(int_to_occurrences))]


class Sequencer(Dataset):
    def __init__(self, word_sequences, type_sequences, vectors, max_sentence_length=None,
                 minimum_type_occurrence=None, return_char_sequences=False, return_word_distributions=False,
                 max_word_len=20, decolor=False):
        """
        Necessary in the case of larger data sets that cannot be stored in memory.
        :param word_sequences:
        :param type_sequences:
        :param vectors:
        :param max_sentence_length:
        :param minimum_type_occurrence:
        """
        print('Received {} samples..'.format(len(word_sequences)))
        if decolor:
            type_sequences = decolor_sequences(type_sequences)

        if max_sentence_length:
            word_sequences, type_sequences = get_low_len_sequences(word_sequences, type_sequences, max_sentence_length)
            print(' .. of which {} are ≤ the maximum sentence length ({}).'.format(len(word_sequences),
                                                                                     max_sentence_length))
        if minimum_type_occurrence:
            word_sequences, type_sequences = filter_by_occurrence(word_sequences, type_sequences,
                                                                  minimum_type_occurrence)
            print(' .. of which {} are ≥ the minimum type occurrence ({}).'.format(len(word_sequences),
                                                                                       minimum_type_occurrence))
        if return_char_sequences:
            word_sequences, type_sequences = get_low_len_words(word_sequences, type_sequences, max_word_len)
            print(' .. of which {} are ≤ the minimum word length ({}).'.format(len(word_sequences), max_word_len))

        self.word_sequences = word_sequences
        self.type_sequences = type_sequences
        self.max_sentence_length = max_sentence_length
        self.types = {t: i+1 for i, t in enumerate(get_all_unique(type_sequences))}
        self.words = {w: torch.zeros(len(self.types)) for i, w in enumerate(get_all_unique(word_sequences))}
        self.types[None] = 0
        self.vectors = vectors
        assert len(word_sequences) == len(type_sequences)
        self.len = len(word_sequences)

        self.return_word_distributions = return_word_distributions
        if self.return_word_distributions:
            self.build_word_lexicon()
            self.word_lexicon_to_pdf()

        self.return_char_sequences = return_char_sequences
        if self.return_char_sequences:
            self.chars = {c: i+1 for i, c in enumerate(get_all_chars(self.word_sequences))}
            self.chars[None] = 0
            self.max_word_len = max_word_len

        print('Constructed dataset of {} word sequences with a total of {} unique types'.
              format(self.len, len(self.types) - 1))
        print('Average sentence length is {}'.format(np.mean(list(map(len, word_sequences)))))

    def __len__(self):
        return self.len

    def get_item_without_chars(self, index):
        try:
            return (map_ws_to_vs(self.word_sequences[index], self.vectors),
                    torch.Tensor(list(map(lambda x: self.types[x], self.type_sequences[index]))))
        except KeyError:
            return None

    def __getitem__(self, index):
        temp = self.get_item_without_chars(index)
        if not temp:
            return None
        vectors, types = temp
        if self.return_char_sequences:
            char_sequences = torch.stack(list(map(lambda x: map_cs_to_is(x, self.chars, self.max_word_len),
                                                  self.word_sequences[index])))
        if self.return_word_distributions:
            word_distributions = torch.stack(list(map(lambda word: self.tensorize_dict(word),
                                                      self.word_sequences[index])))

        if self.return_char_sequences:
            if self.return_word_distributions:
                return vectors, char_sequences, word_distributions, types
            else:
                return vectors, char_sequences, types
        elif self.return_word_distributions:
            return vectors, word_distributions, types
        else:
            return vectors, types

    def build_word_lexicon(self):
        self.words = {w: dict() for w in get_all_unique(self.word_sequences)}
        for ws, ts in zip(self.word_sequences, self.type_sequences):
            for w, t in zip(ws, ts):
                if self.types[t] in self.words[w].keys():
                    self.words[w][self.types[t]] = self.words[w][self.types[t]] + 1
                else:
                    self.words[w][self.types[t]] = 1

    def word_lexicon_to_pdf(self):
        self.words = {w: {k: v/sum(self.words[w].values())} for w in self.words for k, v in self.words[w].items()}

    def tensorize_dict(self, word):
        a = torch.zeros(len(self.types))
        for k, v in self.words[word].items():
            a[k] = v
        return a


class ConstructiveSequencer(Dataset):
    def __init__(self, word_sequences, type_sequences, vectors, max_sentence_length=None, max_word_len=20,
                 return_types=False, mini=False, language='nl'):
        """
        ...
        """
        if mini:
            word_sequences, type_sequences = word_sequences[:20], type_sequences[:20]
        print('Received {} samples..'.format(len(word_sequences)))
        if max_sentence_length:
            word_sequences, type_sequences = get_low_len_sequences(word_sequences, type_sequences, max_sentence_length)
            print(' .. of which {} are ≤ the maximum sentence length ({}).'.format(len(word_sequences),
                                                                                     max_sentence_length))
        word_sequences, type_sequences = get_low_len_words(word_sequences, type_sequences, max_word_len)
        print(' .. of which {} are ≤ the maximum word length ({}).'.format(len(word_sequences), max_word_len))
        if language == 'nl':
            self.atomic_dict = all_atomic(type_sequences)
            self.inverse_atomic_dict = {v: k for k, v in self.atomic_dict.items()}
        elif language == 'fr':
            self.atomic_dict = init_fr_dict()
            self.inverse_atomic_dict = {v: k for k, v in self.atomic_dict.items()}
        num_types = len(get_all_unique(type_sequences))
        self.type_vectors = list(map(lambda x: tuple(map(lambda y: convert_type_to_vector(y, self.atomic_dict,
                                                                                          self.inverse_atomic_dict),
                                                         x)), type_sequences))
        self.max_type_len = max(list(map(lambda x: max(list(map(len, x))), self.type_vectors))) + 1
        self.frequencies = get_atomic_occurrences(self.type_vectors, self.atomic_dict)
        self.frequencies = self.frequencies
        self.return_types = return_types
        self.word_sequences = word_sequences
        self.max_sentence_length = max_sentence_length
        self.vectors = vectors
        assert len(self.word_sequences) == len(self.type_vectors)
        self.len = len(word_sequences)
        self.chars = {c: i+1 for i, c in enumerate(get_all_chars(self.word_sequences))}
        self.chars[None] = 0
        self.max_word_len = max_word_len
        if self.return_types:
            self.types = {t: i + 1 for i, t in enumerate(get_all_unique(type_sequences))}
            self.types[None] = 0
            self.type_sequences = type_sequences

        print('Constructed dataset of {} word sequences with a total of {} unique types built from {} atomic symbols'.
              format(self.len, num_types, len(self.atomic_dict)))
        print('Average sentence length is {}'.format(np.mean(list(map(len, word_sequences)))))
        print('Maximum type length is {}'.format(self.max_type_len))
        self.assert_sanity(type_sequences)
        self.debug = False

    def assert_sanity(self, type_sequences):
        for i, sample in enumerate(self.type_vectors):
            recovered = convert_vector_sequence_to_type_sequence(sample, self.atomic_dict)
            original = tuple(map(str, type_sequences[i]))
            try:
                assert recovered == original
            except AssertionError:
                print(i)
                print(sample)
                print(recovered)
                print(original)
                raise AssertionError

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.debug:
            return self.word_sequences[index], self.type_sequences[index]
        char_vectors = list(map(lambda x: map_cs_to_is(x, self.chars, self.max_word_len), self.word_sequences[index]))
        word_vectors = map_ws_to_vs(self.word_sequences[index], self.vectors)
        type_vectors = list(map(lambda x: map_tv_to_is(x, self.max_type_len), self.type_vectors[index]))
        if self.return_types:
            types = torch.Tensor(list(map(lambda x: self.types[x], self.type_sequences[index])))
            return word_vectors, char_vectors, type_vectors, types
        return word_vectors, char_vectors, type_vectors


def fake_vectors():
    return defaultdict(lambda: np.random.random(300))


def check_uniqueness(word_sequences, type_sequences):
    u = dict()
    bad = 0
    for ws, ts in zip(word_sequences, type_sequences):
        ws = ' '.join([w for w in ws])
        if ws in u.keys():
            try:
                assert u[ws] == ts
            except AssertionError:
                bad += 1
        else:
            u[ws] = ts
    print('{} sequences are assigned non-singular types'.format(bad))


def __main__(sequence_file='test-output/sequences/words-types.p', return_char_sequences=False, constructive=False,
             return_word_distributions=False, fake=False, max_sentence_length=25, minimum_type_occurrence=10,
             return_types=False, mini=False, language='nl'):
    ws, ts = load(sequence_file)

    if language == 'nl':
        inv_file = 'vectors/wiki.nl/wiki.nl.vec'
        oov_file = 'vectors/wiki.nl/oov.vec'
    elif language == 'fr':
        inv_file = 'vectors/cc.fr/cc.fr.300.vec'
        oov_file = 'vectors/cc.fr/oov.vec'
    else:
        raise ValueError('Unknown language.')

    # check_uniqueness(ws, ts)
    if fake:
        vectors = fake_vectors()
    else:
        vectors = FastText.load_vectors([inv_file, oov_file])

    if constructive:
        sequencer = ConstructiveSequencer(ws, ts, vectors, max_sentence_length=max_sentence_length,
                                          return_types=return_types, mini=mini, language=language)
    else:
        sequencer = Sequencer(ws, ts, vectors, max_sentence_length=max_sentence_length,
                              minimum_type_occurrence=minimum_type_occurrence,
                              return_char_sequences=return_char_sequences,
                              return_word_distributions=return_word_distributions,)
    return sequencer
