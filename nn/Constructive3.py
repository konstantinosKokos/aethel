import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import *
from utils import SeqUtils
import numpy as np
from itertools import chain


def accuracy_new(predictions, truth, phrase_lens):
    """

    :param predictions: mtl, a, sl, bs
    :param truth: mtl, sl, bs
    :return:
    """
    phrase_lens = phrase_lens.to('cpu').numpy().tolist()
    predictions = predictions.argmax(dim=1)
    correct_subtypes = torch.ones(predictions.size()).to('cuda')
    correct_subtypes[predictions.ne(truth)] = 0
    correct_subtypes[truth.ne(0)] = 1
    correct_words = correct_subtypes.prod(dim=1)
    phrases = torch.split(correct_words, split_size_or_sections=phrase_lens)
    correct_phrases = list(map(lambda x: torch.sum(x).item(), phrases))
    correct_phrases = sum(list(map(lambda x: 1 if x[0] == x[1] else 0, zip(correct_phrases, phrase_lens))))
    return (sum(correct_words), correct_words.shape[0]), (correct_phrases, len(phrase_lens))


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
        # training -- fast mode
        if batch_y is not None:
            unsorted_batch_y, sequence_lengths = pad_packed_sequence(batch_y)
            unsorted_embeddings = self.embedder(unsorted_batch_y)
            unsorted_embeddings = pack_padded_sequence(unsorted_embeddings, sequence_lengths)
            indices = reindex(unsorted_embeddings.batch_sizes)
            sorted_embeddings = torch.index_select(unsorted_embeddings.data, 0, indices).permute(1, 0, 2)

            h_0 = self.encoder_to_h0(encoder_output).reshape(-1, 2, self.hidden_size).permute(1, 0, 2).contiguous()

            h_t, _ = self.body.forward(sorted_embeddings, h_0)  # mts, msl*bs, h

            y_t = self.hidden_to_output(h_t)
            y_t = F.log_softmax(y_t, dim=-1)
            y_t = y_t[:-1, :, :]  # mtl-1, nw, atomic
            return y_t

        # validation -- slow mode
        h_t = self.encoder_to_h0(encoder_output).reshape(-1, 2, self.hidden_size).permute(1, 0, 2).contiguous()
        sos = (torch.ones(1, h_t.shape[1]) * self.sos).to(self.device).long()
        e_t = self.embedder(sos)

        Y = []
        for t in range(self.max_steps):
            _, h_t = self.body.forward(e_t, h_t)
            y_t = self.hidden_to_output(h_t[1])
            y_t = F.log_softmax(y_t, dim=-1)
            Y.append(y_t)
            p_t = y_t.argmax(dim=-1)
            e_t = self.embedder(p_t).unsqueeze(0)
        return torch.stack(Y)[:-1]


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
        encoder_o, _ = self.word_encoder(word_vectors)  # num_words, 2 * h
        indices = reindex(encoder_o.batch_sizes)
        ordered_encoder_output = torch.index_select(encoder_o.data, 0, indices)
        ordered_encoder_output = ordered_encoder_output[:, :300] + ordered_encoder_output[:, 300:]
        return ordered_encoder_output

    def forward_constructive(self, encoder_o, batch_y):
        construction = self.type_decoder(encoder_o, batch_y)
        return construction

    def forward(self, word_vectors, batch_y=None):
        encoder_output = self.forward_core(word_vectors)
        return self.forward_constructive(encoder_output, batch_y)

    def iter_epoch(self, dataset, batch_size, criterion, optimizer, iter_indices=None, mode='train'):
        if iter_indices is None:
            permutation = np.random.permutation(len(dataset))
        else:
            permutation = np.random.permutation(iter_indices)

        loss = 0.
        batch_start = 0

        correct_words, total_words, correct_phrases, total_phrases = 0, 0, 0, 0

        while batch_start < len(permutation):
            batch_end = min([batch_start + batch_size, len(permutation)])

            # perform bucketing on the batch (-> mini-batching)
            batch_all = sorted([dataset[iter_indices[i]] for i in range(batch_start, batch_end)],
                               key=lambda x: x[0].shape[0], reverse=True)

            batch_x = pack_sequence([x[0] for x in batch_all]).to(self.device)
            batch_y = pack_sequence([torch.stack(x[2]) for x in batch_all]).to(self.device)

            if mode == 'train':
                batch_loss, (batch_correct_words, batch_total_words), (batch_correct_phrases, batch_total_phrases) = \
                    self.train_batch(batch_x, batch_y, criterion, optimizer)
            elif mode == 'eval':
                batch_loss, (batch_correct_words, batch_total_words), (batch_correct_phrases, batch_total_phrases) = \
                    self.eval_batch(batch_x, batch_y, criterion)
            else:
                raise ValueError('Unknown mode.')

            loss += batch_loss
            correct_words += batch_correct_words
            total_words += batch_total_words
            correct_phrases += batch_correct_phrases
            total_phrases += batch_total_phrases

            batch_start += batch_size
        return loss, correct_words/total_words, correct_phrases/total_phrases

    # def truncated_train_batch(self, batch_x, batch_y, criterion, optimizer):
    #     self.train()
    #     optimizer.zero_grad()
    #
    #     loss = 0.
    #     encoder_output = self.forward_core(batch_x)
    #     partial_y, h_n, partial_embeddings, msl, bs = \
    #         self.type_decoder.truncated_forward_first_step(encoder_output, batch_y)
    #     partial_loss =

    def train_batch(self, batch_x, batch_y, criterion, optimizer):
        self.train()
        optimizer.zero_grad()

        prediction = self.forward(batch_x, batch_y).permute(1, 2, 0)  # NW, A, TS

        indices = reindex(batch_y.batch_sizes)
        _, phrase_lens = pad_packed_sequence(batch_y)
        batch_y = batch_y.data[indices][:, 1:]  # NW, TS

        loss = criterion(prediction, batch_y)
        loss = loss[loss != 0.].sum() / batch_y.shape[0]
        loss.backward()

        (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy_new(prediction,
                                                                                        batch_y, phrase_lens)
        optimizer.step()
        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)

    def eval_batch(self, batch_x, batch_y, criterion):
        self.eval()

        prediction = self.forward(batch_x).permute(1, 2, 0)

        indices = reindex(batch_y.batch_sizes)
        _, phrase_lens = pad_packed_sequence(batch_y)
        batch_y = torch.index_select(batch_y.data, 0, indices)[:, 1:]

        loss = criterion(prediction, batch_y)
        loss = loss[loss != 0.].sum() / batch_y.shape[0]

        (batch_correct, batch_total), (sentence_correct, sentence_total) = accuracy_new(prediction,
                                                                                        batch_y, phrase_lens)

        return loss.item(), (batch_correct, batch_total), (sentence_correct, sentence_total)


def __main__(fake=False, mini=False, language='nl'):
    if language == 'nl':
        s = SeqUtils.__main__(fake=fake, constructive=True, sequence_file='test-output/sequences/words-types.p',
                              return_types=True, mini=mini)
    elif language == 'fr':
        s = SeqUtils.__main__(fake=fake, constructive=True, sequence_file='test-output/sequences/words-types_fr.p',
                              return_types=True, mini=mini, language='fr')
    print(s.atomic_dict)

    if fake:
        print('Warning! You are using fake data!')

    num_epochs = 1000
    batch_size = 160
    val_split = 0.25

    indices = [i for i in range(len(s))]
    splitpoint = int(np.floor(val_split * len(s)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[splitpoint:], indices[:splitpoint]
    print('Training on {} and validating on {} samples.'.format(len(train_indices), len(val_indices)))

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))

    ecdc = Model(num_atomic=len(s.atomic_dict), device=device, max_steps=s.max_type_len, num_types=len(s.types),
                 sos=s.inverse_atomic_dict['<SOS>'],)
    criterion = nn.NLLLoss(reduction='none', ignore_index=s.inverse_atomic_dict['<PAD>'])
    optimizer = torch.optim.RMSprop(ecdc.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, threshold=0.001,
                                                           factor=0.33, threshold_mode='rel', cooldown=0, min_lr=1e-08,
                                                           eps=1e-08)

    val_history = []
    for i in range(num_epochs):
        print('================== Epoch {} =================='.format(i))
        l, a, b = ecdc.iter_epoch(s, batch_size, criterion, optimizer, train_indices)
        print(' Training Loss: {}'.format(l))
        print(' Training Word Accuracy: {}'.format(a))
        print(' Training Phrase Accuracy : {}'.format(b))

        if i % 5 == 0 and i != 0:
            l, a, b = ecdc.iter_epoch(s, 256, criterion, optimizer, val_indices, mode='eval')
            print('- - - - - - - - - - - - - - - - - - - - - - - - - ')
            print(' Validation Loss: {}'.format(l))
            print(' Validation Word Accuracy: {}'.format(a))
            print(' Validation Phrase Accuracy : {}'.format(b))
            val_history.append(l)
            if i % 10 == 0:
                store_samples(ecdc, s, val_indices)

        scheduler.step(l)


def store_samples(network, dataset, indices, device='cuda', batch_size=256, log_file='nn/val_log.tsv'):

    texts = []
    t_types = []
    p_types = []

    batch_start = 0
    while batch_start < len(indices):

        batch_end = min([batch_start + batch_size, len(indices)])

        batch_indices = [indices[i] for i in range(batch_start, batch_end)]

        batch_all = sorted([dataset[i] for i in batch_indices],
                           key=lambda x: x[0].shape[0], reverse=True)
        batch_x = pack_sequence([x[0] for x in batch_all]).to(device)

        batch_y = pack_sequence([torch.stack(x[2]) for x in batch_all]).to(device)
        _, phrase_lens = pad_packed_sequence(batch_y)
        phrase_lens = phrase_lens.cpu().numpy().tolist()
        indices = reindex(batch_y.batch_sizes)
        batch_y = torch.index_select(batch_y.data, 0, indices)[:, 1:]
        batch_y = torch.split(batch_y, phrase_lens)
        batch_y = list(map(lambda x: x.cpu().numpy().tolist(), batch_y))

        prediction = network.forward(batch_x).argmax(dim=-1).permute(1, 0)
        prediction = torch.split(prediction, phrase_lens)
        prediction = list(map(lambda x: x.cpu().numpy().tolist(), prediction))

        texts.extend([dataset.word_sequences[i] for i in batch_indices])

        batch_t_types = SeqUtils.convert_many_vector_sequences_to_type_sequences(batch_y, dataset.atomic_dict)
        batch_t_types = [[t for t in batch_t_types[i] if t] for i in range(len(batch_t_types))]
        t_types.extend(batch_t_types)

        batch_p_types = SeqUtils.convert_many_vector_sequences_to_type_sequences(prediction, dataset.atomic_dict)
        p_types.extend([[p for p in batch_p_types[i] if p] for i in range(len(batch_t_types))])

        batch_start += batch_size

    with open(log_file, 'w') as f:
        for i in range(len(texts)):
            f.write('\t'.join(texts[i]) + '\n')
            f.write('\t'.join(t_types[i]) + '\n')
            f.write('\t'.join(p_types[i]) + '\n')
            f.write('\n')


def reindex(batch_sizes):
    current = 0
    indices = []
    while current < batch_sizes[0]:
        for i in range(len(batch_sizes[batch_sizes > current])):
            index = torch.tensor(current) + sum(batch_sizes[:i])
            indices.append(index)
        current += 1
    return torch.Tensor(indices).to('cuda').long()
