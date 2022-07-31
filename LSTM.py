from curses import start_color
from requests import head
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import torch.nn.functional as F

import random
import numpy as np

import config

class LSTM(nn.Module):
    def __init__(self, nb_lstm_units=config.LSTM.HIDDEN_STATE, embedding_dim=config.W2V_SIZE,
                       batch_size=config.BATCH_SIZE, nb_classes=config.LSTM.classes):
        super(LSTM, self).__init__()
        self.nb_lstm_units = nb_lstm_units
        self.nb_classes = nb_classes
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self.nb_lstm_layers = 1
        self.device = config.LSTM.DEVICE
    
        self.__build_model()

    def __build_model(self):

        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            batch_first=True,
        )
        self.lstm1 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.nb_lstm_units,
            batch_first=True,
        )
        self.hidden_to_stance = nn.Linear(self.nb_lstm_units*2, self.nb_classes)

    def init_hidden(self):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, self.batch_size, self.nb_lstm_units)

        hidden_a = Variable(hidden_a).to(self.device)
        hidden_b = Variable(hidden_b).to(self.device)

        return (hidden_a, hidden_b)

    def forward(self, headline, article, h_lengths, a_lengths):
        self.hidden1 = self.init_hidden()
        headline = torch.nn.utils.rnn.pack_padded_sequence(headline, h_lengths, batch_first=True, 
                                enforce_sorted=False).to(self.device)
        _, self.hidden1 = self.lstm1(headline, self.hidden1)

        self.hidden2 = self.init_hidden()
        article = torch.nn.utils.rnn.pack_padded_sequence(article, a_lengths, batch_first=True, 
                                enforce_sorted=False).to(self.device)
        _, self.hidden2 = self.lstm2(article, self.hidden2)

        X1 = self.hidden1[0].contiguous()
        X2 = self.hidden1[0].contiguous()
        X = torch.cat([X1, X2], dim=2)
        X = self.hidden_to_stance(X.view(-1, self.nb_lstm_units*2))
        return X

class LSTMRelatedDetector(LSTM):
    def __init__(self, phase, model_path=None):
        super(LSTMRelatedDetector, self).__init__()
        self.phase = phase
        self.lstm = LSTM().to(self.device)
        self.optimizer = optim.SGD(self.lstm.parameters(), 
        lr=config.LSTM.SGD.LR, weight_decay=config.LSTM.SGD.WEIGHT_DECAY)
        if phase=='eval':
            model_weights = torch.load(model_path)
            model_weights = {k[5:]: v for k, v in model_weights.items() if k.startswith('lstm.')}
            self.lstm.load_state_dict(model_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.batch_size = config.BATCH_SIZE

    def get_padded_batch(self, text, padding_value=5):
        text = [torch.from_numpy(t) for t in text]
        text = pad_sequence(text, batch_first=True, padding_value=padding_value)
        text = text.to(device=self.device)
        return text

    def feed_data(self, data_loader):
        h, a, S, h_lengths, a_lengths = data_loader
        batches = list(zip(h, a, S, h_lengths, a_lengths))
        if self.phase == 'train':
            random.shuffle(batches)

        n_data = len(h)
        n_epochs = np.ceil(n_data/self.batch_size)
        n_epochs = int(n_epochs)

        total_loss,correct = 0, 0
        predictions = []
        for epoch in range(n_epochs):
            start = self.batch_size * epoch
            end = self.batch_size * (epoch+1)
            headline, article, stance, h_lengths, a_lengths = zip(*batches[start:end])

            headline = self.get_padded_batch(headline).to(self.device)
            article = self.get_padded_batch(article).to(self.device)
            h_lengths = torch.tensor(h_lengths).to(self.device)
            a_lengths = torch.tensor(a_lengths).to(self.device)
            stance = torch.tensor(stance).to(self.device)

            pred = self.lstm(headline, article, h_lengths, a_lengths)
            loss = self.loss_fn(pred, stance)
            total_loss += loss

            if self.phase == 'train':   
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            elif self.phase == 'eval':
                correct += (pred.argmax(1) == stance).sum().item()
                predictions += pred.argmax(1).tolist()

        total_loss /= n_epochs
        correct /= n_data
        accuracy = 100*correct
        return total_loss.item(), accuracy, predictions

    def update_phase(self, phase):
        self.phase = phase