import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim

import random
import numpy as np

import config

class LSTM(nn.Module):
    def __init__(self, nb_lstm_units, embedding_dim, batch_size, nb_classes):
        super(LSTM, self).__init__()
        self.nb_lstm_units = nb_lstm_units
        self.nb_classes = nb_classes
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.nb_lstm_layers = 1
    
        self.__build_model()

    def __build_model(self):

        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            batch_first=True,
        )
        self.hidden_to_stance = nn.Linear(self.nb_lstm_units, self.nb_classes)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        hidden_a = Variable(hidden_a).to(config.device)
        hidden_b = Variable(hidden_b).to(config.device)

        return (hidden_a, hidden_b)

    def forward(self, X, X_lengths):
        self.hidden = self.init_hidden()
        X = torch.nn.utils.rnn.pack_padded_sequence(X, X_lengths, batch_first=True, enforce_sorted=False).to(config.device)
        _, self.hidden = self.lstm(X, self.hidden)
        X = self.hidden[0].contiguous()
        X = self.hidden_to_stance(X.view(-1, self.nb_lstm_units))
        return X

class LSTMRelatedDetector():
    def __init__(self, phase, model_path=None):
        self.phase = phase
        if phase=='train':
            self.lstm = LSTM(nb_lstm_units=config.LSTM.HIDDEN_STATE, embedding_dim=config.W2V_SIZE,
                             batch_size=config.BATCH_SIZE, nb_classes=config.LSTM.classes)
        elif phase=='eval':
            self.lstm = torch.load(model_path)
            self.lstm.eval()
        self.device = config.device
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.SGD(self.lstm.parameters(), lr=config.SGD.LR, weight_decay=config.SGD.WEIGHT_DECAY)
        self.batch_size = config.BATCH_SIZE
        self.lstm.to(self.device)

    def get_padded_batch(self, text, padding_value=5):
        text = [torch.from_numpy(t) for t in text]
        text = pad_sequence(text, batch_first=True, padding_value=padding_value)
        text = text.to(device=config.device)
        return text

    def feed_data(self, data_loader):
        X, S, X_lengths = data_loader
        batches = list(zip(X, S, X_lengths))
        random.shuffle(batches)

        n_data = len(X)
        n_epochs = np.ceil(n_data/self.batch_size)
        n_epochs = int(n_epochs)

        total_loss,correct = 0, 0
        predictions = []
        for epoch in range(n_epochs):
            start = self.batch_size * epoch
            end = self.batch_size * (epoch+1)
            text, stance, lengths = zip(*batches[start:end])

            text = self.get_padded_batch(text).to(self.device)
            lens = torch.tensor(lengths).to(self.device)
            stance = torch.tensor(stance).to(self.device)

            pred = self.lstm(text, lens)
            loss = self.loss_fn(pred, stance)
            total_loss += loss

            if self.phase == 'train':   
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            elif self.phase == 'eval':
                correct += (pred.argmax(1) == stance).sum().item()
                predictions += pred.tolist()

        total_loss /= n_epochs
        correct /= n_data
        accuracy = 100*correct
        return total_loss.item(), accuracy, predictions

    def update_phase(self, phase):
        self.phase = phase