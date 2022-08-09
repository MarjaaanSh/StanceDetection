import warnings
import os
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
import torch

from utils.logs import logger
from networks.MLP import UnRelatedDetector
from networks.LSTM import LSTMRelatedDetector
from feature_engineering import DataSet
from utils.score import report_score
import config

from sklearn.metrics import precision_recall_fscore_support

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

def compute_metrics(pred, actual, network):
    p, r, f1, _ = precision_recall_fscore_support(actual, pred, average="macro")
    p, r, f1 = np.round(p, 2), np.round(r, 2), np.round(f1, 4)
    print('\n')
    print('{} RESULT'.format(network))
    print("precision: {}, recall: {}, f1: {}".format(p, r, f1))

    if len(set(actual))>2:
        labels = set(actual)
        for l in labels:
            actual_binary = [1 if x==l else 0 for x in actual]
            pred_binary = [1 if x==l else 0 for x in pred]
            _, _, f1, _ = precision_recall_fscore_support(actual_binary, pred_binary, average="binary")
            f1 = np.round(f1*100, 1)
            print("{}, f1: {}".format(l, f1))

def get_mlp_result(name):
    mlp_log = logger('mlp')
    mlp_path = os.path.join(mlp_log.get_path(), 'model, epoch={}'.format(config.MLP.last_it))
    mlp = UnRelatedDetector('eval', mlp_path)

    dataset = DataSet(name, 'mlp')
    X_train, _, X_val, s_val = dataset.load_features()

    data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)
    with torch.no_grad():
        _, acc, pred = mlp.feed_data(data_loader)

    print('accuracy: {}'.format(np.round(acc, 2)))
    compute_metrics(pred, data_loader[1], 'MLP')

    all_stances = dataset.convert_label_to_stance(s_val)
    pred_stances = dataset.convert_label_to_stance(pred)

    conf = report_score(all_stances, pred_stances, 'mlp', list(set(pred_stances)))
    return pred_stances, s_val, conf

def get_lstm_result(name, stage, labels, lstm_input_idx=None):
    lstm_log = logger('lstm', stage)
    lstm_path = os.path.join(lstm_log.get_path(), 'model, epoch={}'.format(last_trained_it))
    lstm = LSTMRelatedDetector('eval', lstm_path)

    dataset = DataSet(name, 'lstm')
    _, _, X_val, s_val = dataset.load_features()

    ommit_unrelateds = False
    if lstm_input_idx is not None:
        X_val = X_val[lstm_input_idx]
        s_val = s_val[lstm_input_idx]
    else: 
        ommit_unrelateds = True

    data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=ommit_unrelateds)

    with torch.no_grad():
        _, acc, pred = lstm.feed_data(data_loader)

    print('accuracy: ', np.round(acc, 2))
    compute_metrics(pred, data_loader[2], 'LSTM')

    stances = dataset.convert_label_to_stance(data_loader[2])
    pred_stance = dataset.convert_label_to_stance(pred)

    conf = report_score(stances, pred_stance, 'lstm', labels=labels)
    return pred, conf


warnings.simplefilter("ignore")
name = 'competition_test'
# name = 'train'
last_trained_it = config.last_it
stage = config.stage

pred, all_stances, conf = get_mlp_result(name)
all_stances = [config.LABEL_MAP[l] for l in all_stances]

ax= plt.subplot()
sn.heatmap(conf, annot=True, fmt='g', cmap='BuGn',
           linewidths=4, square=True, ax=ax) # font size
ax.set(xlabel='predicted', ylabel='actual')
ax.xaxis.set_ticklabels(['unrelated', 'related'])
ax.yaxis.set_ticklabels(['unrelated', 'related'])
plt.savefig(name+'_mlp')


plt.clf()
_, conf = get_lstm_result(name, stage, ['agree', 'disagree', 'discuss'])
ax = plt.subplot()
sn.heatmap(conf, annot=True, fmt='g', cmap='BuGn',
           linewidths=4, square=True, ax=ax)
ax.set(xlabel='predicted', ylabel='actual')
ax.xaxis.set_ticklabels(['agree', 'disagree', 'discuss'])
ax.yaxis.set_ticklabels(['agree', 'disagree', 'discuss'])
plt.savefig(name+'_lstm')

pred = np.array(pred)
predicted_related = (pred == 'related')
predicted_unrelated = (pred == 'unrelated')
true_unrelated = (np.array(all_stances) == 'unrelated')
lstm_input_idx = predicted_related * ~true_unrelated

lstm_pred, _ = get_lstm_result(name, stage, lstm_input_idx=lstm_input_idx, labels=['agree', 'disagree', 'discuss'])

n_data = len(all_stances)
final_pred_labels = np.array([-1]*n_data)
final_pred_labels[predicted_unrelated] = 3

false_predicted_related = predicted_related * true_unrelated
final_pred_labels[false_predicted_related] = np.random.choice(range(3), false_predicted_related.sum())

final_pred_labels[lstm_input_idx] = lstm_pred
final_pred_stance = [config.LABEL_MAP[l] for l in final_pred_labels]

print((np.array(final_pred_stance) == np.array(all_stances)).sum() / len(final_pred_stance))
compute_metrics(final_pred_stance, all_stances, 'Overall')
conf = report_score(all_stances, final_pred_stance, 'final', ['agree', 'disagree', 'discuss', 'unrelated'])

plt.clf()
ax = plt.subplot()
sn.heatmap(conf, annot=True, fmt='g', cmap='BuGn',
           linewidths=4, square=True, ax=ax)
ax.set(xlabel='predicted', ylabel='actual')
ax.xaxis.set_ticklabels(['agree', 'disagree', 'discuss', 'unrelated'])
ax.yaxis.set_ticklabels(['agree', 'disagree', 'discuss', 'unrelated'])
plt.savefig(name+'_overal')


name = 'competition_test'
df = DataSet(name, 'mlp').data
df = df[['Headline', 'Body ID']]
df['Stance'] = final_pred_stance
df.to_csv('answers.csv', index=False, encoding='utf-8')

