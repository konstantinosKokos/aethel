import torch.nn as nn
import torch
from torch.nn import functional as F

from utils import SeqUtils

import numpy as np


def accuracy(predictions, ground_truth):
    seq_len = ground_truth.shape[0]
    batch_shape = ground_truth.shape[1]
    num_atomic = predictions.shape[1]

    predictions = torch.argmax(predictions, dim=1)  # seq_len, batch_shape, timesteps

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


def accuracy_old(predictions, ground_truth, ignore_index=0):
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


class Decoder(nn.Module):
    def __init__(self, encoder_output_size, num_atomic, hidden_size, device, sos, max_steps=50, embedding_size=50):
        super(Decoder, self).__init__()
        self.sos = sos
        self.device = device
        self.max_steps = max_steps
        self.num_atomic = num_atomic
        self.hidden_size = hidden_size
        self.encoder_output_size = encoder_output_size
        self.embedding_size = embedding_size

        self.body = nn.GRUCell(input_size=embedding_size, hidden_size=hidden_size,).to(device)
        self.predictor = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=num_atomic),
            nn.LogSoftmax(dim=-1)).to(device)
        self.embedding = nn.Sequential(
            nn.Embedding(num_embeddings=num_atomic, embedding_dim=embedding_size)).to(device)
        self.encoder_to_h0 = nn.Sequential(
            nn.Linear(in_features=encoder_output_size, out_features=hidden_size),
            nn.Tanh()
        ).to(device)
        self.encoder_to_input = nn.Sequential(
            nn.Linear(in_features=encoder_output_size, out_features=embedding_size),
        ).to(device)
        # self.encoder_to_c0 = nn.Sequential(
        #     nn.Linear(in_features=encoder_output_size, out_features=hidden_size),
        #     nn.Tanh()
        # ).to(device)

    def forward(self, encoder_output, batch_y=None):
        batch_shape = encoder_output.shape[0]  # batch_shape * seq_len, encoder_hidden_size

        h_t = F.tanh(self.encoder_to_h0(encoder_output))  # first decoder hidden
        encoder_to_input = self.encoder_to_input(encoder_output)

        e_t = torch.zeros(batch_shape, self.embedding_size).to(self.device)

        predicted = []  # nothing predicted
        if batch_y is not None:
            # from (seq_len, batch_shape, max_timesteps) ↦ (seq_len * batch_shape, max_timesteps)
            batch_y = batch_y.view(-1, self.max_steps)

        for t in range(self.max_steps):
            h_t = self.body.forward(e_t, h_t)  # at t=0, predicted : (b, a)
            y_t = self.predictor(h_t)  # y_t ~ probability distribution over atomic types
            predicted.append(y_t)
            if batch_y is not None:
                e_t = self.embedding(batch_y[:, t].long())
            else:
                e_t = self.embedding(torch.argmax(y_t, dim=-1))
            e_t = F.tanh(e_t + encoder_to_input)
        return torch.stack(predicted)


class Encoder(nn.Module):
    def __init__(self, num_atomic, num_chars, char_embedding_dim, char_rnn_dim, max_steps, device,
                 sos=None, num_types=None):
        super(Encoder, self).__init__()
        self.device = device
        self.num_atomic = num_atomic
        self.num_chars = num_chars
        self.char_embedding_dim = char_embedding_dim
        self.char_rnn_dim = char_rnn_dim
        self.num_types = num_types
        self.mode = None

        self.char_embedder = nn.Sequential(
            nn.Embedding(num_embeddings=self.num_chars, embedding_dim=self.char_embedding_dim),
            nn.ReLU(),
        ).to(device)
        self.char_encoder = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.char_rnn_dim,
                                    bidirectional=True).to(device)

        self.word_encoder = nn.LSTM(input_size=300+self.char_rnn_dim, hidden_size=300+self.char_rnn_dim,
                                    bidirectional=True, num_layers=2, dropout=0.5).to(device)
        self.utility_predictor = nn.Sequential(
            nn.Linear(in_features=300+self.char_rnn_dim, out_features=num_types),
            nn.LogSoftmax(dim=-1)).to(device)

        self.type_decoder = Decoder(encoder_output_size=300 + self.char_rnn_dim, num_atomic=num_atomic,
                                    hidden_size=32, device=self.device, max_steps=max_steps,
                                    sos=sos).to(device)

    def forward_core(self, word_vectors, char_indices):
        seq_len = word_vectors.shape[0]
        batch_shape = word_vectors.shape[1]

        # reshape from (seq_len, batch_shape, max_word_len) ↦ (seq_len * batch_shape, max_word_len)
        char_embeddings = self.char_embedder(char_indices.view(seq_len*batch_shape, -1))
        # apply embedding layer and get (seq_len * batch_shape, max_word_len, e_c)
        char_embeddings = char_embeddings.view(-1, seq_len*batch_shape, self.char_embedding_dim)
        # reshape from (seq_len * batch_shape, max_word_len, e_c) ↦ (max_word_len, seq_len * batch_shape, e_c)
        _, (char_embeddings, _) = self.char_encoder(char_embeddings)
        # apply recurrence and get (at timestep max_word_len): (1, seq_len * batch_shape, e_c)
        char_embeddings = char_embeddings.sum(dim=0)
        # reshape from (1, seq_len * batch_shape, e_c) ↦ (seq_len, batch_shape, e_c)
        char_embeddings = char_embeddings.view(seq_len, batch_shape, self.char_rnn_dim)
        # concatenate with word vectors and get (seq_len, batch_shape, e_w + e_c)
        word_vectors = torch.cat([word_vectors, char_embeddings], dim=-1)  # seq_len, batch_shape, 300 + char_rnn_dim

        encoder_o, _ = self.word_encoder(word_vectors)  # seq_len, batch_shape, 2 * hidden_size
        encoder_o = encoder_o.view(seq_len, batch_shape, 2 if self.word_encoder.bidirectional else 1,
                                   self.word_encoder.hidden_size)  # s, b, 2, h

        if self.word_encoder.bidirectional:
            encoder_o = encoder_o.sum(dim=2)
        else:
            encoder_o = encoder_o.view(seq_len, batch_shape, self.word_encoder.hidden_size)

        return encoder_o, seq_len, batch_shape

    def forward_constructive(self, encoder_o, seq_len, batch_shape, batch_y):
        # print('encoder_o in fc:', encoder_o.shape)
        construction = self.type_decoder(encoder_o.view(seq_len * batch_shape, -1), batch_y)
        construction = construction.view(seq_len, batch_shape, self.num_atomic, self.type_decoder.max_steps)
        construction = construction.permute(0, 2, 1, 3)
        return construction

    def forward_predictive(self, encoder_o, seq_len, batch_shape):
        encoder_o = encoder_o.view(seq_len * batch_shape, self.word_encoder.hidden_size)
        return self.utility_predictor(encoder_o).view(seq_len, batch_shape, -1)

    def forward(self, word_vectors, char_indices, batch_y=None):
        encoder_output, s, b = self.forward_core(word_vectors, char_indices)
        if self.mode == 'predictive':
            return self.forward_predictive(encoder_output, s, b)
        elif self.mode == 'constructive':
            return self.forward_constructive(encoder_output, s, b, batch_y)
        else:
            raise ValueError('Mode not set.')

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
            batch_t = torch.nn.utils.rnn.pad_sequence([xcy[3] for xcy in batch_xcy if xcy]).long().to(self.device)

            if self.mode == 'predictive':
                batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) = \
                    self.train_batch(batch_x, batch_c, batch_t, criterion, optimizer)
            elif self.mode == 'constructive':
                batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) = \
                    self.train_batch(batch_x, batch_c, batch_y, criterion, optimizer)
            else:
                raise ValueError('Mode not set.')

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
            batch_t = torch.nn.utils.rnn.pad_sequence([xcy[3] for xcy in batch_xcy if xcy]).long().to(self.device)

            if self.mode == 'predictive':
                batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) =\
                    self.eval_batch(batch_x, batch_c, batch_t, criterion)
            elif self.mode == 'constructive':
                batch_loss, (batch_correct, batch_total), (sentence_correct, sentence_total) = \
                    self.eval_batch(batch_x, batch_c, batch_y, criterion)
            else:
                raise ValueError('Mode not set.')

            loss += batch_loss
            correct_predictions += batch_correct
            total_predictions += batch_total
            correct_sentences += sentence_correct
            total_sentences += sentence_total

            batch_start += batch_size
        return loss, correct_predictions/total_predictions, correct_sentences/total_sentences

    def train_batch(self, batch_x, batch_c, batch_y, criterion, optimizer, opt_local=None):
        self.train()
        optimizer.zero_grad()

        if self.mode == 'predictive':
            prediction = self.forward(batch_x, batch_c, batch_y)
            loss = criterion(prediction.view(prediction.shape[0] * prediction.shape[1], -1), batch_y.view(-1))
            (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy_old(prediction, batch_y)
        elif self.mode == 'constructive':
            prediction = self.forward(batch_x, batch_c, batch_y)
            loss = criterion(prediction, batch_y)  # sequence_length, batch_shape, timesteps
            torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
            (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy(prediction, batch_y)
        else:
            raise ValueError('Mode not set.')

        loss.backward()
        optimizer.step()
        return torch.sum(loss).item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def eval_batch(self, batch_x, batch_c, batch_y, criterion):
        self.eval()

        if self.mode == 'predictive':
            prediction = self.forward(batch_x, batch_c, batch_y)
            loss = criterion(prediction.view(prediction.shape[0] * prediction.shape[1], -1), batch_y.view(-1))
            (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy_old(prediction, batch_y)
        elif self.mode == 'constructive':
            prediction = self.forward(batch_x, batch_c, batch_y)
            loss = criterion(prediction, batch_y)  # sequence_length, batch_shape, timesteps
            torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
            (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy(prediction, batch_y)
        else:
            raise ValueError('Mode not set.')

        return torch.sum(loss).item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def predict(self, x, c):
        raise NotImplementedError


def __main__(fake=False):
    s = SeqUtils.__main__(fake=fake, constructive=True, sequence_file='test-output/sequences/words-types.p',
                          return_types=True)
    print(s.atomic_dict)

    if fake:
        print('Warning! You are using fake data!')

    num_epochs = 1000
    batch_size = 64
    val_split = 0.25

    indices = [i for i in range(len(s))]
    splitpoint = int(np.floor(val_split * len(s)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[splitpoint:], indices[:splitpoint]
    print('Training on {} and validating on {} samples.'.format(len(train_indices), len(val_indices)))

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    ecdc = Encoder(num_atomic=len(s.atomic_dict), num_chars=len(s.chars),
                   char_embedding_dim=32, char_rnn_dim=64, device=device,
                   max_steps=s.max_type_len,
                   num_types=len(s.types), )
    ecdc.mode = 'predictive'

    criterion = nn.NLLLoss(reduction='elementwise_mean', ignore_index=0)
    # if utility_output:
    #     criterion = (criterion, nn.CrossEntropyLoss(reduction='sum', ignore_index=0))
    optimizer = torch.optim.Adam([{'params': ecdc.word_encoder.parameters()},
                                  {'params': ecdc.char_embedder.parameters()},
                                  {'params': ecdc.char_encoder.parameters()},
                                  {'params': ecdc.utility_predictor.parameters()}])

    # print('================== Epoch -1 ==================')
    # l, a, b = ecdc.eval_epoch(s, batch_size, criterion, val_indices)
    # print(' Validation Loss: {}'.format(l))
    # print(' Validation Accuracy: {}'.format(a))
    # print(' Validation Sentence Accuracy : {}'.format(b))
    for i in range(num_epochs):
        if i == 0:
            optimizer = torch.optim.RMSprop([
                {'params': ecdc.word_encoder.parameters(), 'lr': 1e-05},
                {'params': ecdc.char_encoder.parameters(), 'lr': 1e-05},
                {'params': ecdc.char_embedder.parameters(), 'lr': 1e-05},
                {'params': ecdc.type_decoder.parameters(), 'lr': 1e-02}],
                centered=True
            )
            ecdc.mode = 'constructive'
            batch_size = 32
            print('\nSwitching to constructive.\n')

        print('================== Epoch {} =================='.format(i))
        l, a, b = ecdc.train_epoch(s, batch_size, criterion, optimizer, train_indices)
        print(' Training Loss: {}'.format(l))
        print(' Training Accuracy: {}'.format(a))
        print(' Training Sentence Accuracy : {}'.format(b))
        print('- - - - - - - - - - - - - - - - - - - - - - -')
        l, a, b = ecdc.eval_epoch(s, 256, criterion, val_indices)
        print(' Validation Loss: {}'.format(l))
        print(' Validation Accuracy: {}'.format(a))
        print(' Validation Sentence Accuracy : {}'.format(b))