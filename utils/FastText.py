import subprocess
import pickle
import io
import numpy as np
from tqdm import tqdm
from functools import reduce


def __main__(vector_path='vectors/wiki.nl/wiki.nl.vec', oov_path='vectors/wiki.nl/oov.vec',
             oov_txt_path='vectors/wiki.nl/oov.txt',
             fastText_path='/home/kokos/Documents/Projects/FastText/fastText-0.1.0/fasttext',
             model_path='/home/kokos/Documents/Projects/FastText/fastText-0.1.0/models/wiki.nl.bin',
             sequence_path='test-output/sequences/words-types.p'):
    vectors = load_vectors(vector_path)
    from utils.SeqUtils import get_all_unique
    with open(sequence_path, 'rb') as f:
        X, _ = pickle.load(f)
    words = get_all_unique(X)
    oov = find_oov(words, vectors)
    write_oov(oov, oov_txt_path)
    vectorize_oov(oov_words_file=oov_txt_path, oov_vectors_file=oov_path,
                  fastText_path=fastText_path, model_path=model_path)
    vectors_prime = load_vectors([vectors, oov_path])
    oov = find_oov(words, vectors_prime)


def load_vectors(fname, superdict=dict()):
    if isinstance(fname, list):
        for file in fname:
            load_vectors(file, superdict)
    else:
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        for line in tqdm(fin.readlines()):
            if len(line.split()) == 2:
                continue
            tokens = line.rstrip().split(' ')
            superdict[tokens[0]] = np.array(list(map(float, tokens[1:])))
    return superdict


def find_oov(words, vectors):
    oov = []
    for word in words:
        for subword in word.split():  # dirty hack for collapsed mwu
            if subword not in vectors.keys():
                oov.append(subword)
    return oov


def write_oov(oov, oov_words_file):
    with open(oov_words_file, 'w') as f:
        for word in oov:
            for subword in word.split():
                f.write(subword + '\n')


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