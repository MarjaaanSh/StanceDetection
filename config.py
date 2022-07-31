import os
from easydict import EasyDict as edict

#ROOT_DIR = os.getcwd()
ROOT_DIR = '/home/marjan/StanceDetection/'
DATA_PATH = os.path.join(ROOT_DIR, 'data')
MODEL_PATH = os.path.join(ROOT_DIR, 'models')
LOGS_PATH = os.path.join(ROOT_DIR, 'logs')

W2V_SIZE = 300

TEST_SIZE = 0.1
BATCH_SIZE = 64

MLP = edict()
MLP.SIZE = 512
MLP.classes = 2
MLP.ITERATIONS = 500
MLP.OPTIM = 'SGD'
MLP.SGD = edict()
MLP.SGD.LR = 1e-3
MLP.SGD.WEIGHT_DECAY = 1e-4

LSTM = edict()
LSTM.HIDDEN_STATE = 128
LSTM.classes = 3
LSTM.ITERATIONS = 500
LSTM.OPTIM = 'SGD'
LSTM.SGD = edict()
LSTM.SGD.LR = 1e-2
LSTM.SGD.WEIGHT_DECAY = 5e-5

device = 'cuda:2'

