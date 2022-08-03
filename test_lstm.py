from distutils.command.config import config
import warnings
import os
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
import torch

from logs import logger
from MLP import UnRelatedDetector
from LSTM import LSTMRelatedDetector
from feature_engineering import DataSet
from score import report_score

# from feature_engineering import DataSet
# import config

# from utils.score import report_score


warnings.simplefilter("ignore")
name = 'competition_test'
e = 149

lstm_log = logger('lstm', '3targets_one_mlp')
lstm_path = os.path.join(lstm_log.get_path(), 'model, epoch={}'.format(e))
lstm = LSTMRelatedDetector('eval', lstm_path)

dataset = DataSet(name, 'lstm')
_, _, X_val, s_val = dataset.load_features()

data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)
h, a, s, l1, l2 = data_loader
data_loader = list(zip(h, a, s, l1, l2))
editted = []
for h, a, s, l1, l2 in data_loader:
    if l1==0:
        h = np.zeros((1, 300))
        l1 = len(h)
    editted.append((h, a, s, l1, l2))
data_loader = list(zip(*editted))

with torch.no_grad():
    _, acc, pred = lstm.feed_data(data_loader)

stance_dict = {0: 'agree', 1: 'disagree', 2: 'discuss',  3: 'unrelated'}
stances = [stance_dict[s] for s in data_loader[2]]
pred_stance = [stance_dict[s] for s in pred]
print(acc)
report_score(stances, pred_stance, 'final', ['agree', 'disagree', 'discuss', 'unrelated'])
