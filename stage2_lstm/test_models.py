from readline import set_completer_delims
import numpy as np
import warnings

import config

warnings.simplefilter("ignore")

from data_processing import *
from feature_engineering import DataSet
from training import *

from utils.score import report_score
from utils.logs import logger

mlp_log = logger('mlp')
lstm_log = logger('lstm', 0)

mlp_path = os.path.join(mlp_log.get_path(), 'model')
lstm_path = os.path.join(lstm_log.get_path(), 'model')

mlp = torch.load(mlp_path)
lstm = torch.load(lstm_path)

dataset = DataSet('train')
_, _, X_val, s_val = dataset.load_features()
val_data_loader = make_data_loader(X_val, s_val, ommit_unrelateds=False)

dataset = DataSet('competition_test')
X_comp, s_comp = dataset.load_features()
comp_data_loader = make_data_loader(X_comp, s_comp, ommit_unrelateds=False)

with torch.no_grad():
    _, _, pred_val = test_model(val_data_loader, mlp)
    print(type(pred_val))
    # _, _, pred_comp = test_model(comp_data_loader, model)

# report_score(s_val, pred_val)
# report_score(s_comp, pred_comp)

