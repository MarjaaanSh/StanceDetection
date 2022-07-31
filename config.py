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
MLP.ITERATIONS = 1000
MLP.OPTIM = 'SGD'
MLP.SGD = edict()
MLP.SGD.LR = 1e-2
MLP.SGD.WEIGHT_DECAY = 5e-5
MLP.DEVICE = 'cuda:0'

LSTM = edict()
LSTM.HIDDEN_STATE = 128
LSTM.classes = 3
LSTM.ITERATIONS = 500
LSTM.OPTIM = 'SGD'
LSTM.SGD = edict()
LSTM.SGD.LR = 1e-2
LSTM.SGD.WEIGHT_DECAY = 5e-4
LSTM.DEVICE = 'cuda:1'


# 1. 2010, 89.09
# 2. 2000, 88.62
# 3. 2007, 88.96
# 4. 2010, 89.08
# best: 64, 5e-3
