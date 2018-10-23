import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from LexUtils import init_char_dict
import numpy as np

"""
Evaluation script of the extracted type's generality using word vectors to predict types
"""


class VectorsToTypes(nn.Module):
    """
    Simple network that transforms the vector space and predicts the probability distribution over types
    """
    def __init__(self, num_types, num_chars, vector_shape=384, embedding_dim=50, vector_transform_dim=100,
                 rnn_dim=50, device='cpu'):
        self.vector_shape = vector_shape
        self.embedding_dim = embedding_dim
        self.vector_transform_dim = vector_transform_dim
        self.rnn_dim=rnn_dim

        super(VectorsToTypes, self).__init__()

        self.char_embedding = nn.Sequential(
            nn.Linear(in_features=num_chars, out_features=self.embedding_dim),
            nn.ReLU()
        ).to(device)

        self.char_rnn = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.rnn_dim, bidirectional=True).to(device)

        self.vector_transformation = nn.Sequential(
            nn.Linear(in_features=vector_shape, out_features=self.vector_transform_dim),
            nn.ReLU()
        ).to(device)
        self.type_prediction = nn.Sequential(
            nn.Linear(in_features=self.vector_transform_dim+self.rnn_dim, out_features=num_types),
            nn.LogSoftmax(dim=1)
        ).to(device)
        self.device = device

    def forward(self, word_vector, character_indices):
        batch_shape = word_vector.shape[0]
        hc = self.init_hidden(batch_shape)
        char_embeddings = self.char_embedding(character_indices)
        # print(char_embeddings.shape) [32, 115, 50] -> [batch, seq_len, dim]
        _, (char_vector, _) = self.char_rnn(char_embeddings.view(-1, batch_shape, self.embedding_dim), hc)
        char_vector = char_vector[0] + char_vector[1]  # sum over the two directions
        transformed_vector = self.vector_transformation(word_vector)
        transformed_vector = torch.cat((transformed_vector, char_vector), dim=1)
        type_prediction = self.type_prediction(transformed_vector)
        return type_prediction

    def train_batch(self, batch_inputs_x, batch_inputs_c, batch_outputs, optimizer, criterion):
        optimizer.zero_grad()
        loss = 0.
        predictions = self.forward(batch_inputs_x, batch_inputs_c)
        loss += criterion(predictions, batch_outputs)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_batch(self, batch_inputs_x, batch_inputs_c, batch_outputs, criterion):
        predictions = self.forward(batch_inputs_x, batch_inputs_c)
        loss = criterion(predictions, batch_outputs)
        return loss.item()

    def init_hidden(self, batch_shape):
        return (torch.zeros(2, batch_shape, self.rnn_dim, device=self.device),
                torch.zeros(2, batch_shape, self.rnn_dim, device=self.device))


def encode_word(char_sequence, char_dict):
    try:
        return np.array(list(map(lambda x: char_dict[x], char_sequence)))
    except KeyError:
        for c in char_sequence:
            if c not in char_dict.keys():
                char_dict[c] = len(char_dict) + 1
        return encode_word(char_sequence, char_dict)


def to_categorical(y, num_classes=None):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def __main__(filename='test-output/XYW.p'):
    with open(filename, 'rb') as f:
        x, y, words = pickle.load(f)

    char_dict = dict()
    max_word_len = max(list(map(len, words)))
    print('Max word length: ', max_word_len)
    char_indices = np.zeros([len(words), max_word_len])
    for i, word in enumerate(words):
        char_indices[i][:len(word)] = encode_word(word, char_dict)
    print('Number of characters: ', len(char_dict))
    char_one_hots = to_categorical(char_indices)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}.'.format(device))

    x_train, x_val, y_train, y_val, c_train, c_val = train_test_split(x, y, char_one_hots)

    num_train_samples, num_types, num_chars = y_train.shape[0], y_train.shape[1], char_one_hots.shape[2]
    network = VectorsToTypes(num_types, num_chars, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-04, weight_decay=1e-04)
    criterion = lambda inp, outp: F.kl_div(inp, outp, reduction='sum')
    batch_size = 32
    num_epochs = 50

    x_val, y_val, c_val = torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device), torch.Tensor(c_val).to(device)
    print('C val shape: ', c_val.shape)

    val_loss = network.eval_batch(x_val, c_val, y_val, criterion)
    print('Epoch -1 validation loss: {}'.format(val_loss / c_val.shape[0]))

    for i in range(num_epochs):
        permutation = np.random.permutation(x_train.shape[0])
        epoch_loss = 0.
        batch_start = 0

        while batch_start < num_train_samples:
            batch_end = np.min([batch_start + batch_size, num_train_samples])
            batch_x = torch.Tensor(np.array([x_train[permutation[i]] for i in range(batch_start, batch_end)])).\
                to(device)
            batch_c = torch.Tensor(np.array([c_train[permutation[i]] for i in range(batch_start, batch_end)])).\
                to(device)
            batch_y = torch.Tensor(np.array([y_train[permutation[i]] for i in range(batch_start, batch_end)])).\
                to(device)
            # batch loss corresponds to the elementwise mean of the batch
            batch_loss = network.train_batch(batch_x, batch_c, batch_y, optimizer, criterion)
            # weight each batch by its relative size compared to the normal batch size
            epoch_loss += batch_loss
            batch_start = batch_start + batch_size
        # now divide the epoch loss by the total number of batches
        epoch_loss = epoch_loss / x_train.shape[0]
        val_loss = network.eval_batch(x_val, c_val, y_val, criterion) / c_val.shape[0]
        print('Epoch {} training loss: {}'.format(i, epoch_loss))
        print('Epoch {} validation loss: {}'.format(i, val_loss))
