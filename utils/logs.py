import os
import pandas as pd


import torch

import config


class logger():
    def __init__(self, name, stage=None):
        self.name = name if name=='mlp' else '{}_stage_{}'.format(name, stage)
        self.hidden_size = [config.LSTM.HIDDEN_STATE_h, config.LSTM.HIDDEN_STATE_a, config.LSTM.LINEAR] if name == 'lstm' else config.MLP.SIZE
        if name == 'mlp':
            self.lr = config.MLP.SGD.LR
            self.weight_decay = config.MLP.SGD.WEIGHT_DECAY
        else:
            self.lr = config.LSTM.SGD.LR
            self.weight_decay = config.LSTM.SGD.WEIGHT_DECAY
        self.bert = config.use_transformers
        self.make_path()

    def make_path(self):
        folder_name = '{}_lr={}_weightDecay={}_netSize={}_bert={}'
        folder_name = folder_name.format(self.name, self.lr, self.weight_decay, self.hidden_size, self.bert)
        self.log_path = os.path.join(config.LOGS_PATH, folder_name)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
    
    def get_path(self):
        return self.log_path
    
    def log(self, log_name, log_data):
        path = os.path.join(self.log_path, log_name)
        df = pd.DataFrame(log_data)
        df.to_pickle(path)

    def save_model(self, model, epoch):
        path = self.log_path
        path = os.path.join(path, 'model, epoch={}'.format(epoch))
        torch.save(model.state_dict(), path)

    def print_log(self, epoch, accuracy, train_loss, validation_loss):
        print(f"Epoch {epoch} | train loss: {train_loss:>7f} | Val Accuracy: {accuracy:>0.1f}% | Val loss: {validation_loss:>8f}")

