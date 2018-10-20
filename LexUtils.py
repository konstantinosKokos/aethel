import numpy as np
import pickle
from DS import WordType
import spacy
from tqdm import tqdm
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


def count_type_occurrences(lexicon, types):
    ttw = {word_type: 0 for word_type in types}
    for key in lexicon:
        for word_type in lexicon[key].keys():
            ttw[word_type] = ttw[word_type] + lexicon[key][word_type]
    return sorted(ttw.items(), key=lambda x: -x[1])


def fit_occurrences(values):
    values = np.array(values)
    average = np.average(values)
    std = np.std(values)
    # todo


def remove_deps_from_type(wordtype):
    arglist = []
    for a in wordtype.arglist:
        if type(a) == WordType:
            arglist.append(remove_deps_from_type(a).__str__())
        else:
            arglist.append(a[0].__str__())
    if type(wordtype.result) == WordType:
        result = remove_deps_from_type(wordtype.result)
    else:
        result = wordtype.result
    return WordType(arglist, result)


def remove_deps(lexicon):
    for key in lexicon:
        new_assignments = {remove_deps_from_type(k): v for k, v in lexicon[key].items()}
        lexicon[key] = new_assignments
    return __main__(lexicon)


def reduce_lexicon(lexicon, threshold=2):
    """
    remove types with occurrence count below the threshold, and all words that end up with no type assignment
    :param lexicon:
    :param threshold:
    :return:
    """
    all_types = get_all_types(lexicon)
    type_generality = count_type_occurrences(lexicon, all_types)
    single_occurrence_types = [t[0] for t in type_generality if t[1] <= threshold]

    words_to_remove = []
    deleted = 0

    for k in lexicon:
        types_to_remove = []
        for kk in lexicon[k]:
            if kk in single_occurrence_types:
                deleted += 1
                types_to_remove.append(kk)
        for kk in types_to_remove:
            del lexicon[k][kk]
        if len(lexicon[k]) == 0:
            words_to_remove.append(k)
    for k in words_to_remove:
        del lexicon[k]

    print('Deleted {} types and {} words'.format(deleted, len(words_to_remove)))

    return __main__(lexicon)


def convert_to_pdf(lexicon):
    """
    converts the type assignment counts to a probability distribution over the entire domain
    :param lexicon:
    :return:
    """

    all_types = get_all_types(lexicon)
    for word in lexicon:
        prob_vector = np.zeros(len(all_types))
        prob_mass = sum([v for v in lexicon[word].values()])
        for i, word_type in enumerate(all_types):
            if word_type in lexicon[word].keys():
                prob_vector[i] = lexicon[word][word_type] / prob_mass
            else:
                prob_vector[i] = 0
        lexicon[word] = prob_vector
    return lexicon


def convert_to_vectors(lexicon):
    nl = spacy.load('nl_core_news_sm')
    all_words = list(lexicon.keys())
    for word in all_words:
        vector_len = nl(word).vector.shape[0]
        num_types = lexicon[word].shape[0]
        break
    print(vector_len)
    print(num_types)
    X = np.zeros([len(lexicon), vector_len])
    Y = np.zeros([len(lexicon), num_types])
    for i, word in enumerate(all_words):
        vector = nl(word).vector
        X[i] = vector
        Y[i] = lexicon[word]
    return X, Y


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


    type_generality = count_type_occurrences(lexicon, all_types)
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

    print('-----------------------------------------------------------')

    return lexicon