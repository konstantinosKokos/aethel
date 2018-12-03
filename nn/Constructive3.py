import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import *
from utils import SeqUtils
import numpy as np


def accuracy_new(predictions, truth):
    """

    :param predictions: max_seq_len, num_atomic, num_words
    :param truth: max_seq_len, num_words
    :return:
    """

    pred = predictions.argmax(dim=1)  # max_seq_len, num_words
    correct_types = torch.ones(pred.size()).to('cuda')  # max_seq_len, num_words: everything is correct
    correct_types[pred != truth] = 0  # .. except everything that isn't
    correct_types[truth == 0] = 1  # .. double except the pads
    # multiplying across the sequence_length dimension results in a num_words long vector, with 1s when no mistakes
    # were make across the entire axis, and 0 otherwise
    # summing the elements of this vector we obtain the number of correctly predicted words
    correct_words = correct_types.prod(dim=0).sum()
    false_truths = truth.sum(dim=0)
    false_truths = len(false_truths[false_truths == 0])
    return (correct_words - false_truths, truth.shape[-1] - false_truths), \
           (correct_types.sum() - truth[truth == 0].shape[0], truth[truth != 0].shape[0])


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

        self.body = nn.GRU(input_size=embedding_size, hidden_size=hidden_size, num_layers=2).to(device)
        self.embedder = nn.Embedding(num_embeddings=self.num_atomic, embedding_dim=self.embedding_size, padding_idx=0)
        self.hidden_to_output = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=num_atomic)).to(device)
        self.encoder_to_h0 = nn.Sequential(
            nn.Linear(in_features=encoder_output_size, out_features=self.body.num_layers*hidden_size),
            nn.Tanh()
        ).to(device)

    def forward(self, encoder_output, batch_y=None):
        if batch_y is not None:
            embeddings = self.embedder(batch_y)
            msl, bs, mtl, = embeddings.shape[0:3]
            embeddings = embeddings.view(msl*bs, mtl, self.embedding_size).permute(1, 0, 2)
            h_0 = self.encoder_to_h0(encoder_output).view(msl*bs, 2, self.hidden_size).permute(1, 0, 2).contiguous()
            h_t, _ = self.body.forward(embeddings, h_0)
            y_t = self.hidden_to_output(h_t)
            y_t = F.log_softmax(y_t, dim=-1)
            y_t = y_t[:-1, :, :]
            return y_t


class Model(nn.Module):
    def __init__(self, num_atomic, max_steps, device, sos=None, num_types=None):
        super(Model, self).__init__()
        self.device = device
        self.num_atomic = num_atomic
        self.num_types = num_types
        self.mode = None

        self.word_encoder = nn.GRU(input_size=300, hidden_size=300,
                                   bidirectional=True, num_layers=2, dropout=0.5).to(device)
        self.type_decoder = Decoder(encoder_output_size=300, num_atomic=num_atomic,
                                    hidden_size=256, device=self.device, max_steps=max_steps,
                                    sos=sos).to(device)

    def forward_core(self, word_vectors):
        seq_len, batch_size = word_vectors.shape[0:2]
        encoder_o, _ = self.word_encoder(word_vectors)  # num_words, 2 * h
        encoder_o = encoder_o.view(seq_len, batch_size, 2, self.word_encoder.hidden_size)  # msl, b, 2, h
        encoder_o = encoder_o.sum(dim=2)  # msl, b, h
        return encoder_o

    def forward_constructive(self, encoder_o, batch_y):
        construction = self.type_decoder(encoder_o, batch_y)
        return construction

    def forward(self, word_vectors, batch_y=None):
        encoder_output = self.forward_core(word_vectors)
        return self.forward_constructive(encoder_output, batch_y)

    def train_epoch(self, dataset, batch_size, criterion, optimizer, train_indices=None):
        if train_indices is None:
            permutation = np.random.permutation(len(dataset))
        else:
            permutation = np.random.permutation(train_indices)

        loss = 0.
        batch_start = 0

        correct_words, total_words, correct_atomic, total_atomic = 0, 0, 0, 0

        while batch_start < len(permutation):
            batch_end = min([batch_start + batch_size, len(permutation)])

            # perform bucketing on the batch (-> mini-batching)
            batch_all = sorted([dataset[train_indices[i]] for i in range(batch_start, batch_end)],
                               key=lambda x: x[0].shape[0], reverse=True)

            batch_x = pad_sequence([x[0] for x in batch_all]).to(self.device)
            batch_y = pad_sequence([torch.stack(x[2]) for x in batch_all]).to(self.device)

            batch_loss, (batch_correct_words, batch_total_words), (batch_correct_atomic, batch_total_atomic) = \
                self.train_batch(batch_x, batch_y, criterion, optimizer)

            loss += batch_loss
            correct_words += batch_correct_words
            total_words += batch_total_words
            correct_atomic += batch_correct_atomic
            total_atomic += batch_total_atomic

            batch_start += batch_size
        return loss, correct_words/total_words, correct_atomic/total_atomic

    def train_batch(self, batch_x, batch_y, criterion, optimizer):
        self.train()
        optimizer.zero_grad()

        prediction = self.forward(batch_x, batch_y).permute(0, 2, 1)
        mtl = prediction.shape[0]
        batch_y = batch_y[:, :, 1:].permute(2, 0, 1).view(mtl, -1)
        loss = criterion(prediction, batch_y)
        loss = loss.sum(dim=0)
        loss = loss[loss != 0.].mean()
        loss.backward()
        (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy_new(prediction, batch_y)
        optimizer.step()
        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)


def __main__(fake=False, mini=False):
    s = SeqUtils.__main__(fake=fake, constructive=True, sequence_file='test-output/sequences/words-types.p',
                          return_types=True, mini=mini)
    print(s.atomic_dict)

    if fake:
        print('Warning! You are using fake data!')

    num_epochs = 1000
    batch_size = 128
    val_split = 0.25

    indices = [i for i in range(len(s))]
    splitpoint = int(np.floor(val_split * len(s)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[splitpoint:], indices[:splitpoint]
    print('Training on {} and validating on {} samples.'.format(len(train_indices), len(val_indices)))

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))

    class_weights = 1 / torch.Tensor(s.frequencies).to('cuda')
    print(class_weights)

    ecdc = Model(num_atomic=len(s.atomic_dict), device=device, max_steps=s.max_type_len, num_types=len(s.types),)
    criterion = nn.NLLLoss(reduction='none', ignore_index=0)
    optimizer = torch.optim.RMSprop(ecdc.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=0.001,
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-08, eps=1e-08)
    history = []
    for i in range(num_epochs):
        print('================== Epoch {} =================='.format(i))
        l, a, b = ecdc.train_epoch(s, batch_size, criterion, optimizer, train_indices)
        print(' Training Loss: {}'.format(l))
        print(' Training Type Accuracy: {}'.format(a))
        print(' Training Atomic Accuracy : {}'.format(b))
        scheduler.step(l)
