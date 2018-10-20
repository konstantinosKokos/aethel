import torch.nn as nn
import torch
import torch.nn.functional as F
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np

"""
Evaluation script of the extracted type's generality using word vectors to predict types
"""


class VectorsToTypes(nn.Module):
    """
    Simple network that transforms the vector space and predicts the probability distribution over types
    """
    def __init__(self, num_types, vector_shape=384, device='cpu'):
        super(VectorsToTypes, self).__init__()
        self.vector_transformation = nn.Sequential(
            nn.Linear(in_features=vector_shape, out_features=100),
            nn.ReLU()
        ).to(device)
        self.type_prediction = nn.Sequential(
            nn.Linear(in_features=100, out_features=num_types),
            nn.LogSoftmax(dim=1)
        ).to(device)
        self.device = device

    def forward(self, input_vector):
        transformed_vector = self.vector_transformation(input_vector)
        type_prediction = self.type_prediction(transformed_vector)
        return type_prediction

    def train_batch(self, batch_inputs, batch_outputs, optimizer, criterion):
        optimizer.zero_grad()
        loss = 0.
        predictions = self.forward(batch_inputs)
        loss += criterion(predictions, batch_outputs)
        loss.backward()
        optimizer.step()
        return loss.item()

    def eval_batch(self, batch_inputs, batch_outputs, criterion):
        predictions = self.forward(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        return loss.item()


def __main__(xy=None):
    if xy is None:
        with open('test-output/XY.p', 'rb') as f:
            x, y = pickle.load(f)
    else:
        x, y = xy[0], xy[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}.'.format(device))

    x_train, x_val, y_train, y_val = train_test_split(x, y)

    num_train_samples, num_types = y_train.shape[0], y_train.shape[1]
    network = VectorsToTypes(num_types, device=device)
    optimizer = torch.optim.Adam(network.parameters(), lr=5e-05, weight_decay=1e-04)
    criterion = lambda inp, outp: F.kl_div(inp, outp, reduction='elementwise_mean')
    batch_size = 32
    num_epochs = 50

    x_val, y_val = torch.Tensor(x_val).to(device), torch.Tensor(y_val).to(device)
    val_loss = network.eval_batch(x_val, y_val, criterion)
    print('Epoch -1 validation loss: {}'.format(val_loss))

    for i in range(num_epochs):
        permutation = np.random.permutation(x_train.shape[0])
        epoch_loss = 0.
        batch_start = 0

        while batch_start < num_train_samples:
            batch_end = np.min([batch_start + batch_size, num_train_samples])
            batch_x = torch.Tensor(np.array([x_train[permutation[i]] for i in range(batch_start, batch_end)])).\
                to(device)
            batch_y = torch.Tensor(np.array([y_train[permutation[i]] for i in range(batch_start, batch_end)])).\
                to(device)
            # batch loss corresponds to the elementwise mean of the batch
            batch_loss = network.train_batch(batch_x, batch_y, optimizer, criterion)
            # weight each batch by its relative size compared to the normal batch size
            epoch_loss += batch_loss * (batch_end - batch_start) / batch_size
            batch_start = batch_start + batch_size
        # now divide the epoch loss by the total number of batches
        epoch_loss = epoch_loss / np.ceil(x_train.shape[0] / batch_size)
        val_loss = network.eval_batch(x_val, y_val, criterion)
        print('Epoch {} training loss: {}'.format(i, epoch_loss))
        print('Epoch {} validation loss: {}'.format(i, val_loss))








