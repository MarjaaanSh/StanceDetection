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
name = 'train'

mlp_log = logger('mlp')
mlp_path = os.path.join(mlp_log.get_path(), 'model')
mlp = UnRelatedDetector('eval', mlp_path)

dataset = DataSet(name, 'mlp')
X_train, _, X_val, s_val = dataset.load_features()

if name == 'competition_test':
    dataset = DataSet('train', 'mlp')
    X_train, _, _, _ = dataset.load_features()

scaler = MinMaxScaler()
X_train = scaler.fit_transform(np.vstack(X_train))
X_val = scaler.transform(np.vstack(X_val))

data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)
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
_, _, X_val, s_val = dataset.load_features()

X_val = X_val[lstm_data]
s_val = s_val[lstm_data]

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

stance_dict = {0: 'agree', 1: 'disagree', 2: 'discuss'}
stances = [stance_dict[s] for s in data_loader[2]]
pred_stance = [stance_dict[s] for s in pred]
print(acc)
report_score(stances, pred_stance, 'lstm', ['agree', 'disagree', 'discuss'])

dataset = DataSet(name, 'mlp')
_, _, X_val, s_val = dataset.load_features()
stance_dict = {0: 'agree', 1: 'disagree', 2: 'discuss', 3: 'unrelated'}
all_stances = [stance_dict[s] for s in s_val]

final_pred = np.array([-1]*lstm_data.shape[0])
final_pred[predicted_unrelated] = 3
final_pred[predicted_related * true_unrelated] = np.random.choice(range(3), 
               (predicted_related * true_unrelated).sum())

final_pred[predicted_related * ~true_unrelated] = pred
final_pred = [stance_dict[p] for p in final_pred]
report_score(all_stances, final_pred, 'final', ['agree', 'disagree', 'discuss', 'unrelated'])

# # np.put(final_pred, )
# # print(predicted_unrelated.shape, len(final_pred))
# # final_pred[predicted_unrelated] = 'unrelated'
# # final_pred[lstm_data] = pred
# # print(set(final_pred))
# # dataset = DataSet('competition_test', 'mlp')
# # X_comp, s_comp = dataset.load_features()
# # comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)


#     # _, _, pred_comp = test_model(comp_data_loader, model)

# # dataset = DataSet('competition_test', 'lstm')
# # X_comp, s_comp = dataset.load_features()
# # comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)



# # report_score(s_val, pred_val)
# # report_score(s_comp, pred_comp)

