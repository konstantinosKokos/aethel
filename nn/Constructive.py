import torch.nn as nn
import torch
from torch.nn import functional as F

from utils import SeqUtils

import numpy as np


def accuracy(predictions, ground_truth):
    seq_len = ground_truth.shape[0]
    batch_shape = ground_truth.shape[1]
    num_atomic = predictions.shape[-1]

    predictions = predictions.view(seq_len, batch_shape, -1, num_atomic)  # seq_len, batch_shape, timesteps, num_atomic
    predictions = torch.argmax(predictions, dim=-1)  # seq_len, batch_shape, timesteps

    correct_steps = torch.ones(predictions.size()).to('cuda')  # every timestep is correct
    correct_steps[predictions != ground_truth] = 0  # except the ones are that aren't
    correct_steps[ground_truth == 0] = 1  # .. except the ones we don't care about
    # for a word to be correct, it must be correct on all timesteps
    correct_words = correct_steps.prod(dim=-1)  # seq_len, batch_shape
    # for a sentence to be correct, it must be correct on all words
    correct_sentences = correct_words.prod(dim=0)  # batch_shape

    # mask all words whose timesteps sum to zero
    word_mask = ground_truth.sum(dim=-1).ne(0)  # seq_len, batch_shape
    non_masked_correct_words = torch.Tensor.masked_select(correct_words, word_mask)
    return (torch.sum(non_masked_correct_words).item(), torch.sum(word_mask).item()), \
           (torch.sum(correct_sentences).item(), batch_shape)


class DecodingSupertagger(nn.Module):
    def __init__(self, num_atomic, hidden_size, device, max_steps=50):
        super(DecodingSupertagger, self).__init__()
        self.device = device
        self.max_steps = max_steps
        self.body = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size,).to(device)
        self.predictor = nn.Linear(in_features=self.body.hidden_size, out_features=num_atomic).to(device)

    def forward(self, encoder_output):
        batch_shape = encoder_output.shape[1]

        h_t = torch.zeros(1, batch_shape, self.body.hidden_size).to(self.device)
        c_t = torch.zeros(1, batch_shape, self.body.hidden_size).to(self.device)
        o_t = encoder_output
        predicted = torch.zeros(self.max_steps, batch_shape, self.predictor.out_features).to(self.device)
        for t in range(self.max_steps):
            o_t, (h_t, c_t) = self.body.forward(o_t, (h_t, c_t))
            p_t = self.predictor(o_t)
            predicted[t, :, :] = p_t
        return predicted


class EncoderDecoderWithWordDistributions(nn.Module):
    def __init__(self, num_atomic, num_chars, char_embedding_dim, char_rnn_dim, max_steps, device):
        super(EncoderDecoderWithWordDistributions, self).__init__()
        self.device = device
        self.num_atomic = num_atomic
        self.num_chars = num_chars
        self.char_embedding_dim = char_embedding_dim
        self.char_rnn_dim = char_rnn_dim

        self.char_embedder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_chars, embedding_dim=self.char_embedding_dim),
            nn.ReLU()
        ).to(device)
        self.char_encoder = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.char_rnn_dim,
                                    bidirectional=True).to(device)
        self.word_encoder = nn.LSTM(input_size=300+self.char_rnn_dim, hidden_size=300+self.char_rnn_dim,
                                    bidirectional=True, num_layers=2, dropout=0.5).to(device)
        self.type_decoder = DecodingSupertagger(num_atomic=num_atomic, hidden_size=300 + self.char_rnn_dim,
                                                device=self.device, max_steps=max_steps).to(device)

    def forward(self, word_vectors, char_indices):
        seq_len = word_vectors.shape[0]
        batch_shape = word_vectors.shape[1]

        # reshape from (seq_len, batch_shape, max_word_len) ↦ (seq_len * batch_shape, max_word_len)
        char_embeddings = self.char_embedder(char_indices.view(seq_len*batch_shape, -1))
        # apply embedding layer and get (seq_len * batch_shape, max_word_len, e_c)
        char_embeddings = char_embeddings.view(-1, seq_len*batch_shape, self.char_embedding_dim)
        # reshape from (seq_len * batch_shape, max_word_len, e_c) ↦ (max_word_len, seq_len * batch_shape, e_c)
        _, (char_embeddings, _) = self.char_encoder(char_embeddings)
        # apply recurrence and get (at timestep max_word_len): (1, seq_len * batch_shape, e_c)
        char_embeddings = char_embeddings[0, :, :] + char_embeddings[1, :, :]
        # reshape from (1, seq_len * batch_shape, e_c) ↦ (seq_len, batch_shape, e_c)
        char_embeddings = char_embeddings.view(seq_len, batch_shape, self.char_rnn_dim)
        # concatenate with word vectors and get (seq_len, batch_shape, e_w + e_c)
        word_vectors = torch.cat([word_vectors, char_embeddings], dim=-1)

        encoder_o, _ = self.word_encoder(word_vectors)
        encoder_o = encoder_o.view(seq_len, batch_shape, 2, self.word_encoder.hidden_size)
        encoder_o = encoder_o[:, :, 0, :] + encoder_o[:, :, 1, :]
        encoder_o = encoder_o.view(1, seq_len*batch_shape, self.word_encoder.hidden_size)
        prediction = self.type_decoder(encoder_o)
        return prediction

    def train_epoch(self, dataset, batch_size, criterion, optimizer, train_indices=None):
        if train_indices is None:
            permutation = np.random.permutation(len(dataset))
        else:
            permutation = np.random.permutation(train_indices)

        loss = 0.
        batch_start = 0

        correct_predictions, total_predictions, correct_sentences, total_sentences = 0, 0, 0, 0

        while batch_start < len(permutation):
            batch_end = min([batch_start + batch_size, len(permutation)])
            batch_xcy = [dataset[permutation[i]] for i in range(batch_start, batch_end)]
            batch_x = torch.nn.utils.rnn.pad_sequence([xcy[0] for xcy in batch_xcy if xcy]).to(self.device)
            batch_c = torch.nn.utils.rnn.pad_sequence([xcy[1] for xcy in batch_xcy if xcy]).long().to(self.device)
            batch_y = torch.nn.utils.rnn.pad_sequence([xcy[2] for xcy in batch_xcy if xcy]).long().to(self.device)

            batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) = \
                self.train_batch(batch_x, batch_c, batch_y, criterion, optimizer)
            loss += batch_loss
            correct_predictions += batch_correct
            total_predictions += batch_total
            correct_sentences += sentence_correct
            total_sentences += sentence_total

            batch_start += batch_size
        return loss, correct_predictions/total_predictions, correct_sentences/total_sentences

    def eval_epoch(self, dataset, batch_size, criterion, val_indices=None):
        if val_indices is None:
            val_indices = [i for i in range(len(dataset))]
        loss = 0.
        batch_start = 0

        correct_predictions, total_predictions, correct_sentences, total_sentences = 0, 0, 0, 0

        while batch_start < len(val_indices):
            batch_end = min([batch_start + batch_size, len(val_indices)])
            batch_xcy = [dataset[val_indices[i]] for i in range(batch_start, batch_end)]
            batch_x = torch.nn.utils.rnn.pad_sequence([xcy[0] for xcy in batch_xcy if xcy]).to(self.device)
            batch_c = torch.nn.utils.rnn.pad_sequence([xcy[1] for xcy in batch_xcy if xcy]).long().to(self.device)
            batch_y = torch.nn.utils.rnn.pad_sequence([xcy[2] for xcy in batch_xcy if xcy]).long().to(self.device)

            batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) =\
                self.eval_batch(batch_x, batch_c, batch_y, criterion)
            loss += loss
            correct_predictions += batch_correct
            total_predictions += batch_total
            correct_sentences += sentence_correct
            total_sentences += sentence_total

            batch_start += batch_size
        return loss, correct_predictions/total_predictions, correct_sentences/total_sentences

    def train_batch(self, batch_x, batch_c, batch_y, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        prediction = self.forward(batch_x, batch_c).view(-1, self.num_atomic)
        loss = criterion(prediction, batch_y.view(-1))
        (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy(prediction, batch_y)
        loss.backward()
        optimizer.step()
        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def eval_batch(self, batch_x, batch_c, batch_y, criterion):
        self.eval()
        prediction = self.forward(batch_x, batch_c).view(-1, self.num_atomic)
        loss = criterion(prediction, batch_y.view(-1))
        (batch_correct, batch_total), (sentence_correct, sentence_total) = \
            accuracy(prediction, batch_y)
        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def predict(self, x, c):
        # todo
        seq_len = x.shape[0]
        return self.forward(x.view(seq_len, 1, 300), c.view(seq_len, 1, -1)).view(-1, self.num_types).argmax(-1)


def __main__(fake=False):
    s = SeqUtils.__main__(fake=fake, constructive=True, sequence_file='test-output/sequences/words-types.p')

    num_epochs = 100
    batch_size = 64
    val_split = 0.25

    indices = [i for i in range(len(s))]
    splitpoint = int(np.floor(val_split * len(s)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[splitpoint:], indices[:splitpoint]
    print('Training on {} and validating on {} samples.'.format(len(train_indices), len(val_indices)))

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    ecdc = EncoderDecoderWithWordDistributions(num_atomic=len(s.atomic_dict), num_chars=len(s.chars),
                                               char_embedding_dim=32, char_rnn_dim=64, device=device,
                                               max_steps=s.max_type_len)
    criterion = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    optimizer = torch.optim.Adam(ecdc.parameters(), weight_decay=1e-03)

    print('================== Epoch -1 ==================')
    l, a, b = ecdc.eval_epoch(s, batch_size, criterion, val_indices)
    print(' Validation Loss: {}'.format(l))
    print(' Validation Accuracy: {}'.format(a))
    print(' Validation Sentence Accuracy : {}'.format(b))
    for i in range(num_epochs):
        print('================== Epoch {} =================='.format(i))
        l, a, b = ecdc.train_epoch(s, batch_size, criterion, optimizer, train_indices)
        print(' Training Loss: {}'.format(l))
        print(' Training Accuracy: {}'.format(a))
        print(' Training Sentence Accuracy : {}'.format(b))
        print('- - - - - - - - - - - - - - - - - - - - - - -')
        l, a, b = ecdc.eval_epoch(s, batch_size, criterion, val_indices)
        print(' Validation Loss: {}'.format(l))
        print(' Validation Accuracy: {}'.format(a))
        print(' Validation Sentence Accuracy : {}'.format(b))