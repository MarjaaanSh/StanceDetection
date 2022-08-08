import os
from tkinter.messagebox import NO
from easydict import EasyDict as edict

#ROOT_DIR = os.getcwd()
ROOT_DIR = '/home/marjan/StanceDetection/'
DATA_PATH = os.path.join(ROOT_DIR, 'data')
MODEL_PATH = os.path.join(ROOT_DIR, 'models')
LOGS_PATH = os.path.join(ROOT_DIR, 'logs')

W2V_SIZE = 300
BERT_DIM = 1024

TEST_SIZE = 0.1
BATCH_SIZE = 64

ARTICLE_MAX_SENTENCES = 28


MLP = edict()
MLP.SIZE = 512
MLP.classes = 2
MLP.ITERATIONS = 1000
MLP.OPTIM = 'SGD'
MLP.SGD = edict()
MLP.SGD.LR = 1e-2
MLP.SGD.WEIGHT_DECAY = 5e-5
MLP.DEVICE = 'cuda:0'

LSTM = edict()
LSTM.HIDDEN_STATE_h = 32
LSTM.HIDDEN_STATE_a = 128
LSTM.LINEAR = 128
LSTM.LAYERS = 1
LSTM.classes = 3
LSTM.OPTIM = 'SGD'
LSTM.SGD = edict()
LSTM.SGD.WEIGHT_DECAY = 5e-3

LSTM.DEVICE = 'cuda:1'
stage = 'bert_one_mlp_oneLs_cosines'
last_it = 399
use_transformers = True
use_cosines = True
LSTM.SGD.LR = 1e-2
LSTM.ITERATIONS = 200
MLP.last_it = 899

#599, 0.82, 77-88 | 699, 0.82, 76-89 | 799, 0.82, 78-88 | 899, 0.82, 77-89 | 99, 0.82, 75-89


STANCE_MAP = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
LABEL_MAP ={0: 'agree', 1:'disagree', 2: 'discuss', 3: 'unrelated'}

#32, 128->cntnu

# one mlp, one lstm, [128, 128]:  7739|7597, score: 66.4%|66.1%, f1: 0.47|0.52, acc: 68.8|67.8, 51%-5%-85%|58-17-79 | whole f1: 0.44 | 0.45
# one mlp, one lstm, [64, 32]:  7709|7736, score: 66.2%|66.4, f1: 0.46|0.5, acc: 68.8|68.7, 49%-6%-84%|55-10-82 | whole f1: 0.43|0.44
# one mlp, one lstm, [32, 32]:  7654|7669, score: 65.7%|65.8, f1: 0.49|0.5, acc: 66.8%|67.2, 49%-15%-81%| 57%-14%-79%| whole f1: 0.44
# one mlp, one lstm, [64, 64]:  7655|7673, score: 65.7%|65.9, f1: 0.48|0.48, acc: 66.8|67.25, 46%-14%-83%|56-8-80 | whole f1: 0.44 | 0.44
# one mlp, one lstm, [32, 64]:  7611|7684, score: 65.8%|66%, f1: 0.51|0.5, acc: 68.8, 61%-20%-74%|52%-14%-81% | whole f1: 0.45|0.44
# one mlp, one lstm, [32, 128]:  7659|7699, score: 65.7%|66.1, f1: 0.49|0.52, acc: 66.9|67.9, 52%-12%-81%|58-17-79 | whole f1: 0.44|0.45
# one mlp, one lstm, [64, 128]:  7653|7672, score: 65.7|65.8%, f1: 0.43|0.5, acc: 66.8|67.2, 43-15-84|56%-13%-79% | whole f1: 0.48|0.44
# one mlp, one lstm, [128, 32]:  7672|7676, score: 65.8|65.9%, f1: 0.49|0.5, acc: 67.2|67.3, 53-11-81|58%-13%-78% | whole f1: 0.44|0.45
# one mlp, one lstm, [128, 64]:  7750|7712, score: 66.5|68.2%, f1: 0.5|0.49, acc: 69.3|67.3, 48-16-85|45%-13%-85% | whole f1: 0.45|0.44

# one mlp, one lstm, [256, 256]:  7551, score: 64.8, f1: 0.45, acc: 64.5, 75-3-68 | whole f1: 0.44
# two mlps, one lstm, [128, 128, 128]:  7657, score: 65.7%, f1: 0.45, acc: 66.9, 51%-4%-82% | whole f1: 0.43
# one mlp, one lstm, [16, 16]:  7689, score: 66%, f1: 0.47, acc: 67.6%, 48%-8%-84% | whole f1: 0.43
# two mlps, two lstm, [128, 128, 128]:  7637, score: 65.5%, f1: 0.39, acc: 66.9, 26%-2%-92% | whole f1: 0.4
# two mlps, one lstm, [64, 64, 128]:  7634, score: 65.5%, f1: 0.45, acc: 66.4, 50%-4%-82% | whole f1: 0.43
# one mlp, two lstms, [128, 128]:  7630, score: 65.5%, f1: 0.39, acc: 66.3%, 26%-3%-92% | whole f1: 0.40
# one mlp, three lstms, [128, 128]:  7354, score: 63.2%, f1: 0.4, acc: 59.9%, 62%-0%-67% | whole f1: 0.41



