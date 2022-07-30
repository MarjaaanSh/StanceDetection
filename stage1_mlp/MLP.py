from tokenize import Double
from turtle import stamp
import torch.nn as nn
import torch
from torch import optim

import random
import config
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_targets):
        super(MLP, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(input_size, hidden_size),
                                          nn.ReLU(),
                                          nn.Linear(hidden_size, num_targets),
                                          nn.Softmax())

    def forward(self, x):
        prediction = self.feed_forward(x)
        return prediction


class UnRelatedDetector():
    def __init__(self, phase, model_path=None):
        self.phase = phase

        if phase=='train':
            self.mlp = MLP(input_size=config.W2V_SIZE*2, hidden_size=config.MLP.SIZE,
                            num_targets=config.MLP.classes)
        elif phase=='eval':
            self.mlp = torch.load(model_path)
            self.mlp.eval()

        self.device = config.device
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.optimizer = optim.SGD(self.mlp.parameters(), lr=config.SGD.LR, weight_decay=config.SGD.WEIGHT_DECAY)
        self.batch_size = config.BATCH_SIZE
        self.mlp.to(self.device)

    def feed_data(self, data_loader):
        X, S = data_loader
        batches = list(zip(X, S))
        random.shuffle(batches)

        n_data = len(X)
        n_epochs = np.ceil(n_data/self.batch_size)
        n_epochs = int(n_epochs)

        total_loss, correct = 0, 0
        predictions = []
        for epoch in range(n_epochs):
            start = self.batch_size * epoch
            end = self.batch_size * (epoch+1)
            text, stance = zip(*batches[start:end])

            text = np.vstack(text)
            stance = np.vstack(stance)
            text = torch.from_numpy(text).type(torch.FloatTensor).to(self.device)
            stance = torch.from_numpy(stance).squeeze(1).to(self.device)

            pred = self.mlp(text)
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