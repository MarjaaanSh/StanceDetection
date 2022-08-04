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
LSTM.HIDDEN_STATE_h = 32
LSTM.HIDDEN_STATE_a = 64
LSTM.COSINE_DIM = 257
LSTM.LINEAR = 128
LSTM.classes = 3
LSTM.ITERATIONS = 200
LSTM.OPTIM = 'SGD'
LSTM.SGD = edict()
LSTM.SGD.LR = 1e-2
LSTM.SGD.WEIGHT_DECAY = 5e-3
LSTM.DEVICE = 'cuda:2'

stage = 'cosine_one_mlp'
last_it = 199
use_transformers = True

STANCE_MAP = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
LABEL_MAP ={0: 'agree', 1:'disagree', 2: 'discuss', 3: 'unrelated'}

# 3targets, one mlp, [32, 64]:  7692, 66.0%, 0.42
# 3targets, one mlp, [32, 256]:  7608, 65.3%, 0.43
# 3targets, one mlp, [32, 128]:  7601, 65.2%, 0.43
# 3targets, two mlps, [32, 64, 128]:  7594, 65.2%, 0.42
# 3targets, one mlp, [32, 32]:  7572, 65.0%, 0.42
# 3targets, two mlps, [32, 256, 128]:  7570, 65.0%, 0.42
# 3targets, two mlps, [32, 64, 64]:  7557, 64.9%, 0.42
# 3targets, two mlps, [32, 128, 256]:  7557, 64.9%, 0.42
# 3targets, two mlps, [64, 64, 128]:  7549, 64.8%, 0.42
# 3targets, two mlps, [32, 64, 256]:  7546, 64.8%, 0.42
# 3targets, two mlps, [16, 64, 128]:  7535, 64.7%, 0.42
# 3targets, one mlp, [64, 64]:  7514, 64.5%, 0.42
# 3targets, two mlps, [16, 64, 128]:  7514, 64.5%, 0.42




# 1. 2162, 95.83/40% - 7705, 66.13 / 0%
# 2. 2173, 96.30/58% - 7458, 64.01 / 1%

# best: 64, 5e-3

# 3targets, one mlp, [32, 64, 128], 0.01, 5e-3:  7572, 65.8
# 3targets, one mlp, [32, 128, 128], 0.01, 5e-3:  7602, 65.2
# 3targets, one mlp, [32, 256, 128], 0.01, 5e-3:  7596, 65.2


# 3targets, two mlps, [32, 128, 128], 0.01, 5e-3:  7650, 65.7%*** replaced :(
# 3targets, two mlps, [32, 128, 256], 0.01, 5e-3:  7578, 65.0%
# 3targets, two mlps, [32, 256, 128], 0.01, 5e-3:  7570``, 65.0%
# 3targets, two mlps, [32, 64, 64], 0.01, 5e-3:  7568, 65.0%
# 3targets, two mlps, [32, 64, 256], 0.01, 5e-3:  7555, 64.8%
# 3targets, two mlps, [16, 64, 128], 0.01, 5e-3:  7533, 64.6%
# 3targets, two mlps, [64, 64, 128], 0.01, 5e-3:  7527, 64.6%
# 3targets, two mlps, just bn1, [32, 64, 128], 0.01, 5e-3:  7514, 64.5%
# 3targets, two mlps, bn, [32, 64, 128], 0.01, 5e-3:  7466, 64.1%
# 3targets, two mlps, just bn2, [32, 64, 128], 0.01, 5e-3:  7461, 64.03%
