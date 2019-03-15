s[import torch.nn as nn
import torch
from torch.nn import functional as F

from src.utils import SeqUtils2

import numpy as np


def accuracy(predictions, ground_truth, ignore_index=0):
    predictions = torch.argmax(predictions, dim=-1)

    correct_sentences = torch.ones(predictions.size())
    correct_sentences[predictions != ground_truth] = 0
    correct_sentences[ground_truth == ignore_index] = 1
    correct_sentences = torch.prod(correct_sentences, dim=0)

    mask = ground_truth.ne(ignore_index)
    non_masked_predictions = torch.Tensor.masked_select(predictions, mask)
    non_masked_truths = torch.Tensor.masked_select(ground_truth, mask)
    return (len(non_masked_predictions[non_masked_predictions == non_masked_truths]), len(non_masked_truths)), \
           (len(correct_sentences[correct_sentences == 1]), predictions.size()[1])


class EncoderDecoderWithWordDistributions(nn.Module):
    def __init__(self, num_types, num_chars, char_embedding_dim, char_rnn_dim, device):
        super(EncoderDecoderWithWordDistributions, self).__init__()
        self.device = device
        self.num_types = num_types
        self.num_chars = num_chars
        self.char_embedding_dim = char_embedding_dim
        self.char_rnn_dim = char_rnn_dim

        self.word_transformation = nn.Sequential(
            nn.Linear(in_features=300+self.char_rnn_dim, out_features=300+self.char_rnn_dim),
            nn.ReLU(),
            nn.Linear(in_features=300+self.char_rnn_dim, out_features=self.num_types),
            nn.LogSoftmax(dim=-1)
        ).to(device)

        self.char_embedder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_chars, embedding_dim=self.char_embedding_dim),
            nn.ReLU()
        ).to(device)
        self.char_encoder = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.char_rnn_dim,
                                    bidirectional=True).to(device)
        self.word_encoder = nn.LSTM(input_size=300+self.char_rnn_dim, hidden_size=300+self.char_rnn_dim,
                                    bidirectional=True, num_layers=2, dropout=0.5).to(device)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=300+self.char_rnn_dim, out_features=self.num_types),
        ).to(device)

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
        # individual word predictions
        type_distribution = self.word_transformation(word_vectors)

        encoder_output, _ = self.word_encoder(word_vectors)
        encoder_output = encoder_output.view(seq_len, batch_shape, 2, self.word_encoder.hidden_size)
        encoder_output = encoder_output[:, :, 0, :] + encoder_output[:, :, 1, :]

        prediction = self.predictor(encoder_output)
        prediction = F.layer_norm(prediction, prediction.size()[1:])
        return prediction, type_distribution

    def train_epoch(self, dataset, batch_size, criteria, optimizer, train_indices=None):
        if train_indices is None:
            permutation = np.random.permutation(len(dataset))
        else:
            permutation = np.random.permutation(train_indices)

        losses = [0., 0.]
        batch_start = 0

        correct_predictions, total_predictions, correct_sentences, total_sentences = 0, 0, 0, 0

        while batch_start < len(permutation):
            batch_end = min([batch_start + batch_size, len(permutation)])
            batch_xcy = [dataset[permutation[i]] for i in range(batch_start, batch_end)]
            batch_x = torch.nn.utils.rnn.pad_sequence([xcy[0] for xcy in batch_xcy if xcy]).to(self.device)
            batch_c = torch.nn.utils.rnn.pad_sequence([xcy[1] for xcy in batch_xcy if xcy]).long().to(self.device)
            batch_d = torch.nn.utils.rnn.pad_sequence([xcy[2] for xcy in batch_xcy if xcy]).to(self.device)
            batch_y = torch.nn.utils.rnn.pad_sequence([xcy[3] for xcy in batch_xcy if xcy]).long().to(self.device)

            batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) = \
                self.train_batch(batch_x, batch_c, batch_d, batch_y, criteria, optimizer)
            losses[0] += batch_loss[0]
            losses[1] += batch_loss[1]
            correct_predictions += batch_correct
            total_predictions += batch_total
            correct_sentences += sentence_correct
            total_sentences += sentence_total

            batch_start += batch_size
        return losses, correct_predictions/total_predictions, correct_sentences/total_sentences

    def eval_epoch(self, dataset, batch_size, criteria, val_indices=None):
        if val_indices is None:
            val_indices = [i for i in range(len(dataset))]
        losses = [0., 0.]
        batch_start = 0

        correct_predictions, total_predictions, correct_sentences, total_sentences = 0, 0, 0, 0

        while batch_start < len(val_indices):
            batch_end = min([batch_start + batch_size, len(val_indices)])
            batch_xcy = [dataset[val_indices[i]] for i in range(batch_start, batch_end)]
            batch_x = torch.nn.utils.rnn.pad_sequence([xcy[0] for xcy in batch_xcy if xcy]).to(self.device)
            batch_c = torch.nn.utils.rnn.pad_sequence([xcy[1] for xcy in batch_xcy if xcy]).long().to(self.device)
            batch_d = torch.nn.utils.rnn.pad_sequence([xcy[2] for xcy in batch_xcy if xcy]).to(self.device)
            batch_y = torch.nn.utils.rnn.pad_sequence([xcy[3] for xcy in batch_xcy if xcy]).long().to(self.device)

            batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) =\
                self.eval_batch(batch_x, batch_c, batch_d, batch_y, criteria)
            losses[0] += batch_loss[0]
            losses[1] += batch_loss[1]
            correct_predictions += batch_correct
            total_predictions += batch_total
            correct_sentences += sentence_correct
            total_sentences += sentence_total

            batch_start += batch_size
        return losses, correct_predictions/total_predictions, correct_sentences/total_sentences

    def train_batch(self, batch_x, batch_c, batch_d, batch_y, criteria, optimizer):
        self.train()
        optimizer.zero_grad()
        prediction, distribution = self.forward(batch_x, batch_c)
        loss0 = criteria[0](prediction.view(-1, self.num_types), batch_y.view(-1))
        loss1 = criteria[1](distribution, batch_d)
        (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy(prediction, batch_y)
        loss = loss0 + loss1
        loss.backward()
        optimizer.step()
        return (loss0.item(), loss1.item()), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def eval_batch(self, batch_x, batch_c, batch_d, batch_y, criteria):
        self.eval()
        prediction, distribution = self.forward(batch_x, batch_c)
        loss0 = criteria[0](prediction.view(-1, self.num_types), batch_y.view(-1))
        loss1 = criteria[1](distribution, batch_d)
        (batch_correct, batch_total), (sentence_correct, sentence_total) = \
            accuracy(prediction, batch_y)
        return (loss0.item(), loss1.item()), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def predict(self, x, c):
        seq_len = x.shape[0]
        return self.forward(x.view(seq_len, 1, 300), c.view(seq_len, 1, -1)).view(-1, self.num_types).argmax(-1)


class EncoderDecoderWithCharRNN(nn.Module):
    def __init__(self, num_types, num_chars, char_embedding_dim, char_rnn_dim, device):
        super(EncoderDecoderWithCharRNN, self).__init__()
        self.device = device
        self.num_types = num_types
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
        self.predictor = nn.Sequential(
            nn.Linear(in_features=300+self.char_rnn_dim, out_features=self.num_types),
        ).to(device)

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

        encoder_output, _ = self.word_encoder(word_vectors)
        encoder_output = encoder_output.view(seq_len, batch_shape, 2, self.word_encoder.hidden_size)
        encoder_output = encoder_output[:, :, 0, :] + encoder_output[:, :, 1, :]

        prediction = self.predictor(encoder_output)
        prediction = F.layer_norm(prediction, prediction.size()[1:])
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

            batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) = self.eval_batch(batch_x, batch_c, batch_y, criterion)
            loss += batch_loss
            correct_predictions += batch_correct
            total_predictions += batch_total
            correct_sentences += sentence_correct
            total_sentences += sentence_total

            batch_start += batch_size
        return loss, correct_predictions/total_predictions, correct_sentences/total_sentences

    def train_batch(self, batch_x, batch_c, batch_y, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        prediction = self.forward(batch_x, batch_c)
        loss = criterion(prediction.view(-1, self.num_types), batch_y.view(-1))
        (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy(prediction, batch_y)
        loss.backward()
        optimizer.step()
        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def eval_batch(self, batch_x, batch_c, batch_y, criterion):
        self.eval()
        prediction = self.forward(batch_x, batch_c)
        loss = criterion(prediction.view(-1, self.num_types), batch_y.view(-1))
        (batch_correct, batch_total), (sentence_correct, sentence_total) = \
            accuracy(prediction, batch_y)
        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def predict(self, x, c):
        seq_len = x.shape[0]
        return self.forward(x.view(seq_len, 1, 300), c.view(seq_len, 1, -1)).view(-1, self.num_types).argmax(-1)


def __main__(fake=False, decolor=False):
    s = SeqUtils2.__main__(fake=fake, return_char_sequences=True, return_word_distributions=True, decolor=decolor)

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
    ecdc = EncoderDecoderWithWordDistributions(num_types=len(s.types), num_chars=len(s.chars), char_embedding_dim=32,
                                     char_rnn_dim=64, device=device)
    cel = nn.CrossEntropyLoss(ignore_index=0, reduction='sum')
    kld = nn.KLDivLoss(reduction='sum')
    criteria = [cel, kld]
    optimizer = torch.optim.Adam(ecdc.parameters(), weight_decay=1e-03)

    print('================== Epoch -1 ==================')
    l, a, b = ecdc.eval_epoch(s, batch_size, criteria, val_indices)
    print(' Validation Loss: {}'.format(l))
    print(' Validation Accuracy: {}'.format(a))
    print(' Validation Sentence Accuracy : {}'.format(b))
    for i in range(num_epochs):
        print('================== Epoch {} =================='.format(i))
        l, a, b = ecdc.train_epoch(s, batch_size, criteria, optimizer, train_indices)
        print(' Training Loss: {}'.format(l))
        print(' Training Accuracy: {}'.format(a))
        print(' Training Sentence Accuracy : {}'.format(b))
        print('- - - - - - - - - - - - - - - - - - - - - - -')
        l, a, b = ecdc.eval_epoch(s, batch_size, criteria, val_indices)
        print(' Validation Loss: {}'.format(l))
        print(' Validation Accuracy: {}'.format(a))
        print(' Validation Sentence Accuracy : {}'.format(b))


# # # # # # # # # # # # # SIMPLE MODEL BELOW # # # # # # # # # # # # #

class SimpleEncoderDecoder(nn.Module):
    def __init__(self, num_types, device='cuda'):
        super(SimpleEncoderDecoder, self).__init__()
        self.device = device
        self.num_types = num_types
        self.encoder = nn.LSTM(input_size=300, hidden_size=300, bidirectional=True,
                               num_layers=2, dropout=0.5).to(device)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=300, out_features=self.num_types),
        ).to(device)

    def forward(self, input):
        seq_len = input.shape[0]
        batch_shape = input.shape[1]

        encoder_output, _ = self.encoder(input)
        encoder_output = encoder_output.view(seq_len, batch_shape, 2, self.encoder.hidden_size)
        encoder_output = encoder_output[:, :, 0, :] + encoder_output[:, :, 1, :]
        prediction = self.predictor(encoder_output)
        return prediction.view(-1, self.num_types)  # collapse the time dimension

    def train_epoch(self, dataset, batch_size, criterion, optimizer, train_indices=None):
        if train_indices is None:
            permutation = np.random.permutation(len(dataset))
        else:
            permutation = np.random.permutation(train_indices)

        loss = 0.
        batch_start = 0

        correct_predictions, total_predictions = 0, 0

        while batch_start < len(permutation):
            batch_end = min([batch_start + batch_size, len(permutation)])
            batch_xy = [dataset[permutation[i]] for i in range(batch_start, batch_end)]
            batch_x = torch.nn.utils.rnn.pad_sequence([xy[0] for xy in batch_xy if xy]).to(self.device)
            batch_y = torch.nn.utils.rnn.pad_sequence([xy[1] for xy in batch_xy if xy]).long().to(self.device)

            batch_loss, (batch_correct, batch_total) = self.train_batch(batch_x, batch_y, criterion, optimizer)
            loss += batch_loss
            correct_predictions += batch_correct
            total_predictions += batch_total

            batch_start += batch_size
        return loss, correct_predictions / total_predictions

    def eval_epoch(self, dataset, batch_size, criterion, val_indices=None):
        if val_indices is None:
            val_indices = [i for i in range(len(dataset))]
        loss = 0.
        batch_start = 0

        correct_predictions, total_predictions = 0, 0

        while batch_start < len(val_indices):
            batch_end = min([batch_start + batch_size, len(val_indices)])
            batch_xy = [dataset[val_indices[i]] for i in range(batch_start, batch_end)]
            batch_x = torch.nn.utils.rnn.pad_sequence([xy[0] for xy in batch_xy if xy]).to(self.device)
            batch_y = torch.nn.utils.rnn.pad_sequence([xy[2] for xy in batch_xy if xy]).long().to(self.device)

            batch_loss, (batch_correct, batch_total) = self.eval_batch(batch_x, batch_y, criterion)
            loss += batch_loss
            correct_predictions += batch_correct
            total_predictions += batch_total

            batch_start += batch_size
        return loss, correct_predictions / total_predictions

    def train_batch(self, batch_x, batch_y, criterion, optimizer):
        self.train()
        optimizer.zero_grad()
        prediction = self.forward(batch_x)
        loss = criterion(prediction, batch_y.view(-1))
        batch_correct, batch_total = accuracy(prediction, batch_y.view(-1))
        loss.backward()
        optimizer.step()
        return loss.item(), (batch_correct, batch_total)

    def eval_batch(self, batch_x, batch_y, criterion):
        self.eval()
        prediction = self.forward(batch_x)
        loss = criterion(prediction, batch_y.view(-1))
        batch_correct, batch_total = accuracy(prediction, batch_y.view(-1))
        return loss.item(), (batch_correct, batch_total)
