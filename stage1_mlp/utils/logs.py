import os
import pandas as pd


import torch

import config


class logger():
    def __init__(self, optimizer='SGD'):
        if optimizer == 'SGD':
            self.lr = config.SGD.LR
            self.weight_decay = config.SGD.WEIGHT_DECAY
        self.mlp_size = config.MLP.SIZE
        self.w2v_size = config.W2V_SIZE
        self.epochs = config.MLP.EPOCHS

    def make_path(self, d_name):
        name = '{}_lr={}_weightDecay={}_mlpSize={}_w2vSize={}'
        name = name.format(d_name, self.lr, self.weight_decay, self.mlp_size, self.w2v_size)
        path = os.path.join(config.LOGS_PATH, name)
        return path

    def log(self, log_name, log_data):
        path = self.make_path(log_name)
        df = pd.DataFrame(log_data)
        df.to_pickle(path)

    def save_model(self, model_name, model):
        path = self.make_path(model_name)
        torch.save(model, path)

    def print_log(self, epoch, accuracy, train_loss, validation_loss):
        print(f"Epoch {epoch} | train loss: {train_loss:>7f} | Val Accuracy: {accuracy:>0.1f}% | Val loss: {validation_loss:>8f}")

