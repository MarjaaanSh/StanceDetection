import os.path

import numpy as np
import warnings
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch import optim

from utils.system import parse_params, check_version
from data_processing import get_data, make_data_loader
from feature_engineering import DataSet
# from StanceDetection.stage1_mlp.old_feature_engineering import extract_features, clean_data
from training import *
from neural_networks import MLP
from utils.logs import logger
import config
from utils.score import report_score

if __name__ == "__main__":
    dataset = DataSet('train', 'mlp')
    X_train, y_train, X_val, y_val = dataset.load_features()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(np.vstack(X_train))
    X_val = scaler.transform(np.vstack(X_val))

    train_data_loader = dataset.make_data_loader(X_train, y_train, ommit_unrelateds=False)
    val_data_loader = dataset.make_data_loader(X_val, y_val, ommit_unrelateds=False)

    # clf = MLP()
    # loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = optim.SGD(clf.parameters(), lr=config.SGD.LR, weight_decay=config.SGD.WEIGHT_DECAY)

    # logger = logger(optimizer='SGD')

    # EPOCHS = config.MLP.EPOCHS
    # validation_loss_history = []
    # train_loss_history = []
    # validation_acc_history = []

    # for e in range(EPOCHS):
    #     model, train_loss = train_clf(train_data_loader, clf, loss_fn, optimizer)
    #     train_loss_history.append({'epoch': e, 'loss': train_loss})
    #     if e % 10 == 9:
    #         accuracy, validation_loss = test_model(val_data_loader, model, loss_fn)
    #         logger.print_log(e, accuracy, train_loss, validation_loss)
    #         validation_loss_history.append({'epoch': e, 'loss': validation_loss})
    #         validation_acc_history.append({'epoch': e, 'accuracy': accuracy})

    # logger.log('train_loss', train_loss_history)
    # logger.log('val_loss', validation_loss_history)
    # logger.log('val_acc', validation_acc_history)
    # logger.save_model('classifier', model)

    # with torch.no_grad():
    #     pred = model(torch.tensor(X_val.astype(np.float32)))
    #     pred = pred.argmax(1).numpy()

    # report_score(y_val, pred)

    # X_comp, stances_comp, y_comp = extract_features('competition_test')
    # with torch.no_grad():
    #     pred_comp = model(torch.tensor(X_comp.astype(np.float32)))
    #     pred_comp = pred_comp.argmax(1).numpy()
    #
    # report_score(y_comp, pred_comp)

    # headline = [stance['Headline'] for stance in stances_comp]
    # body_id = [stance['Body ID'] for stance in stances_comp]
    #
    # answers = pd.DataFrame()
    # answers['Headline'] = headline
    # answers['Body ID'] = body_id
    # answers['Stance'] = predicted
    # answers.to_csv('answers.csv', index=False, encoding='utf-8')
    #


