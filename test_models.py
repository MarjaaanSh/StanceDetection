import warnings
import os
import numpy as np
import random

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

mlp_log = logger('mlp')
mlp_path = os.path.join(mlp_log.get_path(), 'model')
mlp = UnRelatedDetector('eval', mlp_path)

dataset = DataSet(name, 'mlp')
data = dataset.load_features()
X, s = data[-2], data[-1]
data_loader = dataset.make_data_loader(X, s, ommit_unrelateds=False)

with torch.no_grad():
    _, acc, pred = mlp.feed_data(data_loader)

stances = ['related' if s == 1 else 'unrelated' for s in data_loader[1]]
pred = ['related' if s == 1 else 'unrelated' for s in pred]
print(acc)
report_score(stances, pred, 'mlp', ['unrelated', 'related'])

pred = np.array(pred)
predicted_related = (pred == 'related')
predicted_unrelated = (pred == 'unrelated')
true_unrelated = (np.array(stances) == 'unrelated')
lstm_data = predicted_related * ~true_unrelated

lstm_log = logger('lstm', 0)
lstm_path = os.path.join(lstm_log.get_path(), 'model')
lstm = LSTMRelatedDetector('eval', lstm_path)

dataset = DataSet(name, 'lstm')
data = dataset.load_features()
X, s = data[-2], data[-1]
X = X[lstm_data]
s = s[lstm_data]
data_loader = dataset.make_data_loader(X, s, ommit_unrelateds=False)
with torch.no_grad():
    _, acc, pred = lstm.feed_data(data_loader)

stance_dict = {0: 'agree', 1: 'disagree', 2: 'discuss'}
stances = [stance_dict[s] for s in data_loader[1]]
pred_stance = [stance_dict[s] for s in pred]
print(acc)
report_score(stances, pred_stance, 'lstm', ['agree', 'disagree', 'discuss'])

dataset = DataSet(name, 'mlp')
data = dataset.load_features()
X, all_stances = data[-2], data[-1]
stance_dict = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
all_stances = [stance_dict[s] for s in all_stances]
final_pred = np.array([-1]*lstm_data.shape[0])
final_pred[predicted_unrelated] = 3
final_pred[predicted_related * true_unrelated] = np.random.choice(range(3), 
               (predicted_related * true_unrelated).sum())
final_pred[predicted_related * ~true_unrelated] = pred
final_pred = [stance_dict[f] for f in final_pred]
report_score(all_stances, final_pred, 'final', ['agree', 'disagree', 'discuss', 'unrelated'])

# np.put(final_pred, )
# print(predicted_unrelated.shape, len(final_pred))
# final_pred[predicted_unrelated] = 'unrelated'
# final_pred[lstm_data] = pred
# print(set(final_pred))
# dataset = DataSet('competition_test', 'mlp')
# X_comp, s_comp = dataset.load_features()
# comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)


    # _, _, pred_comp = test_model(comp_data_loader, model)

# dataset = DataSet('competition_test', 'lstm')
# X_comp, s_comp = dataset.load_features()
# comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)



# report_score(s_val, pred_val)
# report_score(s_comp, pred_comp)

