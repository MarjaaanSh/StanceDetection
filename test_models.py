import warnings
import os

from logs import logger
from MLP import UnRelatedDetector
from LSTM import LSTMRelatedDetector
from feature_engineering import DataSet

import torch

# from feature_engineering import DataSet
# import config

# from utils.score import report_score


warnings.simplefilter("ignore")

mlp_log = logger('mlp')
mlp_path = os.path.join(mlp_log.get_path(), 'model')
mlp = UnRelatedDetector('eval', mlp_path)


lstm_log = logger('lstm', 0)
lstm_path = os.path.join(lstm_log.get_path(), 'model')
lstm = LSTMRelatedDetector('eval', lstm_path)

dataset = DataSet('train', 'mlp')
_, _, X_val, s_val = dataset.load_features()
val_data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)

dataset = DataSet('competition_test', 'mlp')
X_comp, s_comp = dataset.load_features()
comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)

# with torch.no_grad():
_, _, pred_val = mlp.feed_data(val_data_loader)
#     print(type(pred_val))
    # _, _, pred_comp = test_model(comp_data_loader, model)
# dataset = DataSet('train', 'lstm')
# _, _, X_val, s_val = dataset.load_features()
# val_data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=False)

# dataset = DataSet('competition_test', 'lstm')
# X_comp, s_comp = dataset.load_features()
# comp_data_loader = dataset.make_data_loader(X_comp, s_comp, ommit_unrelateds=False)



# report_score(s_val, pred_val)
# report_score(s_comp, pred_comp)

