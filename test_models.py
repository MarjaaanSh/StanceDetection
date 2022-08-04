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
import config

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def compute_metrics(pred, actual, network):
    p, r, f1, _ = precision_recall_fscore_support(actual, pred, average="macro")
    p, r, f1 = np.round(p, 2), np.round(r, 2), np.round(f1, 2)
    print('\n')
    print('{} RESULT'.format(network))
    print("precision: {}, recall: {}, f1: {}".format(p, r, f1))

def get_mlp_result(name):
    mlp_log = logger('mlp')
    mlp_path = os.path.join(mlp_log.get_path(), 'model, epoch=999')
    mlp = UnRelatedDetector('eval', mlp_path)

    dataset = DataSet(name, 'mlp', False)
    X_train, _, X_val, s_val = dataset.load_features()

    if name == 'competition_test':
        dataset = DataSet('train', 'mlp', False)
        X_train, _, _, _ = dataset.load_features()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(np.vstack(X_train.reshape(-1)))
    X_val = scaler.transform(np.vstack(X_val.reshape(-1)))

    data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)
    with torch.no_grad():
        _, acc, pred = mlp.feed_data(data_loader)

    print('accuracy: {}'.format(np.round(acc, 2)))
    compute_metrics(pred, data_loader[1], 'MLP')

    all_stances = dataset.convert_label_to_stance(s_val)
    pred_stances = dataset.convert_label_to_stance(pred)

    report_score(all_stances, pred_stances, 'mlp', list(set(pred_stances)))
    return pred_stances, s_val

def get_lstm_result(name, stage, lstm_input_idx):
    lstm_log = logger('lstm', stage)
    lstm_path = os.path.join(lstm_log.get_path(), 'model, epoch={}'.format(last_trained_it))
    lstm = LSTMRelatedDetector('eval', lstm_path)

    dataset = DataSet(name, 'lstm', config.use_transformers)
    _, _, X_val, s_val = dataset.load_features()

    X_val = X_val[lstm_input_idx]
    s_val = s_val[lstm_input_idx]

    data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)
    h, a, s, l1, l2, cosines = data_loader
    data_loader = list(zip(h, a, s, l1, l2, cosines))
    editted = []
    for h, a, s, l1, l2, cs in data_loader:
        if l1==0:
            h = np.random.rand(1, config.W2V_SIZE).astype(np.float32)
            l1 = len(h)
        editted.append((h, a, s, l1, l2, cs))
    data_loader = list(zip(*editted))

    with torch.no_grad():
        _, acc, pred = lstm.feed_data(data_loader)

    print('accuracy: ', np.round(acc, 2))
    compute_metrics(pred, data_loader[2], 'LSTM')

    stances = dataset.convert_label_to_stance(data_loader[2])
    pred_stance = dataset.convert_label_to_stance(pred)

    report_score(stances, pred_stance, 'lstm', list(set(pred_stance)))
    return pred


warnings.simplefilter("ignore")
name = 'competition_test'
# name = 'train'
last_trained_it = config.last_it
stage = config.stage

pred, all_stances = get_mlp_result(name)
all_stances = [config.LABEL_MAP[l] for l in all_stances]

pred = np.array(pred)
predicted_related = (pred == 'related')
predicted_unrelated = (pred == 'unrelated')
true_unrelated = (np.array(all_stances) == 'unrelated')
lstm_input_idx = predicted_related * ~true_unrelated

lstm_pred = get_lstm_result(name, stage, lstm_input_idx)

n_data = len(all_stances)
final_pred_labels = np.array([-1]*n_data)
final_pred_labels[predicted_unrelated] = 3

false_predicted_related = predicted_related * true_unrelated
final_pred_labels[false_predicted_related] = np.random.choice(range(3), false_predicted_related.sum())

final_pred_labels[lstm_input_idx] = lstm_pred
final_pred_stance = [config.LABEL_MAP[l] for l in final_pred_labels]

compute_metrics(final_pred_stance, all_stances, 'Overall')
report_score(all_stances, final_pred_stance, 'final', ['agree', 'disagree', 'discuss', 'unrelated'])

