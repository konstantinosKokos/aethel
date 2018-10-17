import numpy as np
import pickle
from matplotlib import pyplot as plt


def get_all_types(lexicon):
    return set([key for subdict in lexicon.values() for key in subdict])


def count_type_values(lexicon):
    """
    count the number of times each type occurs and sort them in descending order
    :param lexicon:
    :return: [(type_1, count_1), .. , (type_n, count_n)]
    """
    values = dict()
    for word in lexicon:
        for key in lexicon[word]:
            if key in values.keys():
                values[key] = values[key] + lexicon[word][key]
            else:
                values[key] = lexicon[word][key]
    return sorted(values.items(), key=lambda x: -x[1])


def words_to_types(lexicon):
    """
    count the number of types assigned toe each word and sort them in descending order
    :param lexicon:
    :return:
    """
    values = [[k, len(v)] for k, v in lexicon.items()]
    return sorted(values, key=lambda x: -x[1])


def types_to_words(lexicon, types):
    ttw = {word_type: 0 for word_type in types}
    for key in lexicon:
        for word_type in lexicon[key].keys():
            ttw[word_type] = ttw[word_type]+1
    return sorted(ttw.items(), key=lambda x: -x[1])


def fit_occurrences(values):
    values = np.array(values)
    average = np.average(values)
    std = np.std(values)


def __main__(lexicon=None):
    if lexicon is None:
        with open('test-output/some_dict.p', 'rb') as f:
            lexicon = pickle.load(f)

    all_types = get_all_types(lexicon)

    print('Words inspected: {}'.format(len(lexicon)))
    print('Types extracted: {}'.format(len(all_types)))
    word_ambiguity = words_to_types(lexicon)
    word_ambiguity_values = np.array([v[1] for v in word_ambiguity])
    print('Average types per word: {}'.format(np.average(word_ambiguity_values)))
    print('Standard deviation of types per word: {}'.format(np.std(word_ambiguity_values)))


    type_generality = types_to_words(lexicon, all_types)
    type_generality_values = np.array([v[1] for v in type_generality])
    print('Average words per type: {}'.format(np.average(type_generality_values)))
    print('Standard deviation of words per type: {}'.format(np.std(type_generality_values)))

    word_to_type_thresholds = [2, 3, 5, 10, 15]
    print('Words with just 1 type: {}'.format(len(word_ambiguity_values[word_ambiguity_values == 1])))
    for wttt in word_to_type_thresholds:
        print('Words with {} or more types: {}'.format(wttt,
                                                         len(word_ambiguity_values[word_ambiguity_values >= wttt])))

    type_to_word_thresholds = [5, 10, 20]
    print('Types with just 1 occurrence: {}'.format(len(type_generality_values[type_generality_values == 1])))
    for ttwt in type_to_word_thresholds:
        print('Types with at least {} occurrences: {}'.format(ttwt,
                                                       len(type_generality_values[type_generality_values >= ttwt])))
