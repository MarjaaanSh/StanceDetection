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

