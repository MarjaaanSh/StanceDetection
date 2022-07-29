import numpy as np
import warnings

import config

warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_processing import *
from feature_engineering import extract_features
from training import *

from utils.score import report_score


name = 'classifier'
model_name = '{}_lr={}_weightDecay={}_mlpSize={}_w2vSize={}'
lr = 1e-2
weight_decay = 5e-5
mlp_size = 512
w2v_size = 300
model_name = model_name.format(name, lr, weight_decay, mlp_size, w2v_size)
model_path = os.path.join(config.LOGS_PATH, model_name)

X_train, _, y_train = extract_features(name='train')
X_val, _, y_val = extract_features(name='val')

df = get_data('competition_test')
X_comp, _, y_comp = extract_features(df, 'competition_test')

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_comp = scaler.transform(X_comp)

val_data_loader = make_data_loader(X_val, y_val)
comp_data_loader = make_data_loader(X_comp, y_comp)

model = torch.load(model_path)

with torch.no_grad():
    pred_val = model(torch.tensor(X_val.astype(np.float32)))
    pred_val = pred_val.argmax(1).numpy()

    pred_comp = model(torch.tensor(X_comp.astype(np.float32)))
    pred_comp = pred_comp.argmax(1).numpy()


report_score(y_val, pred_val)
report_score(y_comp, pred_comp)

