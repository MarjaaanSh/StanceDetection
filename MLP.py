import torch.nn as nn
import torch
from torch import optim
import torch.nn.functional as F

import random
import config
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size=config.W2V_SIZE*2, hidden_size=config.MLP.SIZE, num_targets=config.MLP.classes):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_targets)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.softmax(x)

        return x


class UnRelatedDetector(MLP):
    def __init__(self, phase, model_path=None):
        super(UnRelatedDetector, self).__init__()
        self.phase = phase
        self.device = config.MLP.DEVICE
        self.mlp = MLP()
        self.optimizer = optim.SGD(self.mlp.parameters(), 
        lr=config.MLP.SGD.LR, weight_decay=config.MLP.SGD.WEIGHT_DECAY)

        if phase=='eval':
            model_weights = torch.load(model_path)
            model_weights = {k[4:]: v for k, v in model_weights.items() if k.startswith('mlp')}
            self.mlp.load_state_dict(model_weights)

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        self.batch_size = config.BATCH_SIZE
        self.mlp = self.mlp.to(self.device)

    def feed_data(self, data_loader):
        X, S = data_loader
        batches = list(zip(X, S))
        if self.phase != 'eval':
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
                predictions += pred.argmax(1).tolist()

        total_loss /= n_epochs
        correct /= n_data
        accuracy = 100*correct
        return total_loss.item(), accuracy, predictions


    def update_phase(self, phase):
        self.phase = phase