from typing import List
import numpy as np
import xml.etree.cElementTree as ET
from matplotlib import pyplot as plt
import seaborn as sns
from collections import counter
import pickle

from src.Extraction import Lassy
from src.utils.PostProcess import *
from src.WordType import get_colors, get_atomic

import math



def get_sentence_lengths(L0: Lassy) -> List[int]:
    return list(map(get_sample_length, L0))


def get_sample_length(sample: ET):
    return len(list(filter(lambda x: 'word' in x.attrib, list(sample.iter('node')))))


def plot_stuff(lens):
    mean = np.mean(lens)
    ax = sns.distplot(lens, color='orange')
    ax.set_xlabel('Sentence Length')
    ax.axvline(np.mean(lens), color='red', linestyle='dashed')
    ax.set_xticks(np.arange(0, 200, 20))

    ax.legend(['Mean', 'Sentence Length Distribution'])


def stuff():
    import pickle
    from src.utils.PostProcess import *
    from src.WordType import get_colors, get_atomic
    import seaborn as sns
    from matplotlib import pyplot as plt
    import numpy as np

    with open('XYZ.p', 'rb') as f:
        X, Y, _ = pickle.load(f)

    X = deannotate(X)
    Y = deannotate(Y)
    m_XY = matchings(X, Y)
    for k in m_XY:
        m_XY[k] = len(set(m_XY[k]))

    countY = count_occurrences(Y)
    values = sorted(countY.values())
    unique_values = sorted(set(countY.values()))
    sum_values = sum(countY.values())

    def at_least(i):
        return len([x for x in countY if countY[x]>=i])

    _at_least = list(map(at_least, unique_values))

    def plot_at_least():
        f, ax = plt.subplots()
        ax.semilogx(unique_values, list(map(lambda x: x/len(countY), _at_least)), color='#525252')
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax.grid(which='minor', linewidth=0.15, linestyle='-')
        ax.grid(which='major', linewidth=0.5)
        ylabels = [item.get_text() for item in ax.get_yticklabels(minor=False)]
        ylabels[1] = '0'
        ylabels[2] = str(int(0.2 * len(countY))) + ' (20)'
        ylabels[3] = str(int(0.4 * len(countY))) + ' (40)'
        ylabels[4] = str(int(0.8 * len(countY))) + ' (60)'
        ylabels[5] = str(int(0.8 * len(countY))) + ' (80)'
        ylabels[6] = str(len(countY)) + ' (100)'
        ax.set_yticklabels(ylabels, minor=False)
        ax.set_xlabel('# Occurrences (log)', family='serif')
        ax.set_ylabel('# of Unique Types (%)', family='serif')
        return f, ax

    def sentence_min(sentence):
        return min(list(map(lambda x: countY[x], sentence)))

    _sentence_mins = list(map(sentence_min, Y))

    def sentences_covered(i):
        return len(list(filter(lambda x: x >= i, _sentence_mins)))

    _sentences_covered = list(map(sentences_covered, unique_values))

    def plot_sentences_covered():
        f, ax = plt.subplots()
        ax.semilogx(unique_values, list(map(lambda y: y/len(Y), _sentences_covered)), color='#525252')
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax.grid(which='minor', linewidth=0.15)
        ax.grid(which='major', linewidth=0.5)
        ylabels = [item.get_text() for item in ax.get_yticklabels(minor=False)]
        ylabels[1] = '0'
        ylabels[2] = str(int(0.2 * len(Y))) + ' (20)'
        ylabels[3] = str(int(0.4 * len(Y))) + ' (40)'
        ylabels[4] = str(int(0.6 * len(Y))) + ' (60)'
        ylabels[5] = str(int(0.8 * len(Y))) + ' (80)'
        ylabels[6] = str(len(Y)) + ' (100)'
        ax.set_yticklabels(ylabels, minor=False)
        ax.set_xlabel('# Occurrences (log)', family='serif')
        ax.set_ylabel('# of Sentences (%)', family='serif')

    def types_covered(i):
        gec = sum([countY[x] for x in countY if countY[x] >= i])
        return gec

    _types_covered = list(map(types_covered, unique_values))

    def plot_types_covered():
        f, ax = plt.subplots()
        ax.semilogx(unique_values, list(map(lambda y: y/sum_values, _types_covered)), color='#525252')
        ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)
        ax.grid(which='minor', linewidth=0.15)
        ax.grid(which='major', linewidth=0.5)
        ylabels = [item.get_text() for item in ax.get_yticklabels(minor=False)]
        ylabels[1] = str(0.2 * sum_values) + ' (20)'
        ylabels[2] = str(int(0.4 * sum_values)) + ' (40)'
        ylabels[3] = str(int(0.6 * sum_values)) + ' (60)'
        ylabels[4] = str(int(0.8 * sum_values)) + ' (80)'
        ylabels[5] = str(sum_values) + ' (100)'
        ax.set_yticklabels(ylabels, minor=False)
        ax.set_xlabel('# Occurrences (log)', family='serif')
        ax.set_ylabel('# of Types (%)', family='serif')

    def plot_sentence_lens():
        f, ax = plt.subplots()
        ax = sns.kdeplot(list(map(len, Y)), cumulative=True, color='#525252')
        ax.set_xticks([10, 30, 50, 70, 90], minor=True)
        ax.grid(which='minor', linewidth=0.15)
        ax.grid(which='major', linewidth=0.5)
        ylabels = [item.get_text() for item in ax.get_yticklabels(minor=False)]
        ylabels[1] = str(0.2 * len(Y)) + ' (20)'
        ylabels[2] = str(int(0.4 * len(Y))) + ' (40)'
        ylabels[3] = str(int(0.6 * len(Y))) + ' (60)'
        ylabels[4] = str(int(0.8 * len(Y))) + ' (80)'
        ylabels[5] = str(len(Y)) + ' (100)'
        ax.set_yticklabels(ylabels, minor=False)
        ax.set_xlabel('Sentence Length', family='serif')
        ax.set_ylabel('# of Sentences (%)', family='serif')


    def count_arities():
        arities = dict()
        for k in countY:
            arity = k.get_arity()
            if arity in arities:
                arities[arity] = arities[arity] + countY[k]
            else:
                arities[arity] = countY[k]
        return arities

    def coverage(i):
        geq = [x for x in countY if countY[x] >= i]
        cumgeq = sum([countY[x] for x in geq])
        return cumgeq / sum_values * 100

    def sentence_coverage(v):
        return len(list(filter(lambda x: x >= v, sentence_mins))) / len(sentence_mins)

    def types_tossed(i):
        geq = [x for x in countY if countY[x] == i]
        return len(geq)

    def sentence_min(y):
        return min(list(map(lambda x: countY[x], y)))

    sentence_mins = list(map(sentence_min, Y))

    coverages = list(map(coverage, unique_values))
    types_c = np.array(list(map(types_tossed, unique_values))).cumsum()

    def plot_word_ambiguity():
        bins = [1, 2, 5, 10, 20, 50, 100, 200, 500]
        f, ax = plt.subplots()
        ax.hist(m_XY.values(), bins=bins, log=True,  color='#525252')
        ax.set_xscale("log")
        ax.grid(which='minor', linewidth=0.15, linestyle='-')
        ax.grid(which='major', linewidth=0.5)
        ax.set_xlabel('Type Ambiguity (log)', family='serif')
        ax.set_ylabel('# of Words', family='serif')

