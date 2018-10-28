import subprocess
import pickle
import io
import numpy as np
from tqdm import tqdm
from functools import reduce


def load_vectors(fname):
    if isinstance(fname, list):
        return reduce(lambda x, y: {**x, **load_vectors(y)}, fname, dict())
    else:
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        try:
            n, d = list(map(int, fin.readline().split()))
        except ValueError:  # not writing n, d in oov files
            pass
        vectors = {}
        for line in tqdm(fin):
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.array(list(map(float, tokens[1:])))
        return vectors


def find_oov(words, vectors):
    oov = []
    for word in words:
        if word not in vectors.keys():
            oov.append(word)
    return oov


def write_oov(oov, oov_words_file):
    with open(oov_words_file, 'w') as f:
        for word in oov:
            f.write(word + '\n')


def vectorize_oov(oov_words_file, oov_vectors_file,
                  fastText_path='/home/kokos/Documents/Projects/FastText/fastText-0.1.0/fasttext',
                  model_path='/home/kokos/Documents/Projects/FastText/fastText-0.1.0/models/wiki.nl.bin'):
    with open(oov_vectors_file, 'w') as f:
        ft = subprocess.Popen([fastText_path, 'print-word-vectors', model_path],
                              stdin=open(oov_words_file), stdout=f)
        ft.wait()
        f.flush()


def to_hdf5(vectors):
    # todo
    raise NotImplementedError