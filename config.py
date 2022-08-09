import os
from tkinter.messagebox import NO
from tkinter.tix import Tree
from easydict import EasyDict as edict

#ROOT_DIR = os.getcwd()
ROOT_DIR = '/home/marjan/StanceDetection/'
DATA_PATH = os.path.join(ROOT_DIR, 'data')
W2V_MODEL_PATH = os.path.join(ROOT_DIR, 'w2v_models')
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
MLP.last_it = 899
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
LSTM.SGD.LR = 1e-2
LSTM.ITERATIONS = 200
LSTM.DEVICE = 'cuda:1'

stage = 'bert_one_mlp_oneLs_cosines'
last_it = 399
use_transformers = True
use_cosines = True


STANCE_MAP = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
LABEL_MAP ={0: 'agree', 1:'disagree', 2: 'discuss', 3: 'unrelated'}

