import torch
import torch.nn as nn

import config


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(config.W2V_SIZE*2, config.MLP.SIZE),
                                          nn.ReLU(),
                                          nn.Linear(config.MLP.SIZE, config.MLP.classes),
                                          nn.Softmax())

    def forward(self, x):
        prediction = self.feed_forward(x)
        return prediction


class LSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, n_classes)

    def forward(self, sentence):
        x = sentence.view(sentence.shape[1], 1, -1)
        _, (h, c) = self.lstm(x)
        x = self.linear(h)
        x = x.squeeze(1)
        return x

