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
MLP.ITERATIONS = 5

LSTM = edict()
LSTM.HIDDEN_STATE = 128
LSTM.classes = 3



OPTIM = 'SGD'
SGD = edict()
SGD.LR = 1e-2
SGD.WEIGHT_DECAY = 5e-5

device = 'cuda:3'

