
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
    def __init__(self, headline_embedding_dim = config.BERT_DIM,
                       article_embedding_dim=config.BERT_DIM,
                       nb_lstm_units_h=config.LSTM.HIDDEN_STATE_h, 
                       nb_lstm_units_a=config.LSTM.HIDDEN_STATE_a,
                       nb_linear = config.LSTM.LINEAR,
                       batch_size=config.BATCH_SIZE, 
                       nb_classes=config.LSTM.classes,
                       nb_lstm_layers=config.LSTM.LAYERS):
        super(LSTM, self).__init__()
        self.article_embedding_dim = article_embedding_dim
        self.headline_embedding_dim = headline_embedding_dim

        self.nb_lstm_units_h = nb_lstm_units_h
        self.nb_lstm_units_a = nb_lstm_units_a
        self.nb_lstm_layers = nb_lstm_layers

        self.linear_input_dim = self.nb_lstm_units_h + self.nb_lstm_units_a
        self.nb_linear = nb_linear
        self.nb_classes = nb_classes
        self.batch_size = batch_size
        self.device = config.LSTM.DEVICE
        self.use_cosines = config.use_cosines
        if self.use_cosines:
            self.linear_input_dim += config.ARTICLE_MAX_SENTENCES
    
        self.__build_model()

    def __build_model(self):

        self.lstm1 = nn.LSTM(
            input_size=self.headline_embedding_dim,
            hidden_size=self.nb_lstm_units_h,
            batch_first=True,
            num_layers = self.nb_lstm_layers
        )
        self.lstm2 = nn.LSTM(
            input_size=self.article_embedding_dim,
            hidden_size=self.nb_lstm_units_a,
            batch_first=True,
            num_layers = self.nb_lstm_layers
        )
        # self.linear = nn.Linear(self.linear_input_dim, self.nb_linear)
        self.hidden_to_stance = nn.Linear(self.linear_input_dim, self.nb_classes)

    def init_hidden(self, nb_lstm_units, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        hidden_a = torch.randn(self.nb_lstm_layers, batch_size, nb_lstm_units)
        hidden_b = torch.randn(self.nb_lstm_layers, batch_size, nb_lstm_units)

        hidden_a = Variable(hidden_a).to(self.device)
        hidden_b = Variable(hidden_b).to(self.device)

        return (hidden_a, hidden_b)

    def forward(self, headline, article, a_lengths, cosines=None):
        batch_size = headline.shape[0]
        self.hidden1 = self.init_hidden(self.nb_lstm_units_h, batch_size)
        _, self.hidden1 = self.lstm1(headline, self.hidden1)

        self.hidden2 = self.init_hidden(self.nb_lstm_units_a, batch_size)
        article = torch.nn.utils.rnn.pack_padded_sequence(article, a_lengths, batch_first=True, 
                                enforce_sorted=False).to(self.device)
        _, self.hidden2 = self.lstm2(article, self.hidden2)

        X1 = self.hidden1[0][-1].contiguous()
        X2 = self.hidden2[0][-1].contiguous()

        if not self.use_cosines:
            X = torch.cat([X1, X2], dim=1)
        else: 
            X1 = X1.squeeze(1)
            X2 = X2.squeeze(1)
            X = torch.cat([X1, X2, cosines], dim=1)

        # X = self.linear(X)
        # X = F.relu(X)
        X = self.hidden_to_stance(X)
        return X

class LSTMRelatedDetector(LSTM):
    def __init__(self, phase, model_path=None):
        super(LSTMRelatedDetector, self).__init__()
        self.phase = phase
        self.lstm = LSTM().to(self.device)
        self.optimizer = optim.SGD(self.lstm.parameters(), 
        lr=config.LSTM.SGD.LR, weight_decay=config.LSTM.SGD.WEIGHT_DECAY)
        if model_path:
            model_weights = torch.load(model_path)
            model_weights = {k[5:]: v for k, v in model_weights.items() if k.startswith('lstm.')}
            self.lstm.load_state_dict(model_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.batch_size = config.BATCH_SIZE
        self.use_cosine = config.use_cosines

    def get_padded_batch(self, text, padding_value=5):
        text = [torch.from_numpy(t) for t in text]
        text = pad_sequence(text, batch_first=True, padding_value=padding_value)
        text = text.to(device=self.device)
        return text

    def feed_data(self, data_loader):
        if not self.use_cosine:
            h, a, S, a_lengths = data_loader
            batches = list(zip(h, a, S, a_lengths))
        else:
            h, a, S, a_lengths, cosines = data_loader
            batches = list(zip(h, a, S, a_lengths, cosines))

        if self.phase == 'train':
            random.shuffle(batches)

        n_data = h.shape[0]
        n_epochs = np.ceil(n_data/self.batch_size)
        n_epochs = int(n_epochs)

        total_loss,correct = 0, 0
        predictions = []
        for epoch in range(n_epochs):
            start = self.batch_size * epoch
            end = self.batch_size * (epoch+1)
            if not self.use_cosine:
                headline, article, stance, a_lengths = zip(*batches[start:end])
            else:
                headline, article, stance, a_lengths, cosines = zip(*batches[start:end])
            headline = torch.tensor(headline).unsqueeze(1).to(self.device)
            article = torch.tensor(article).type(torch.float32).to(self.device)
            a_lengths = torch.tensor(a_lengths)#.to(self.device)
            stance = torch.tensor(stance).to(self.device)
            if self.use_cosine:
                cosines = torch.tensor(cosines).type(torch.float32).to(self.device)

            if not self.use_cosine:
                pred = self.lstm(headline, article, a_lengths)
            else:
                pred = self.lstm(headline, article, a_lengths, cosines)

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