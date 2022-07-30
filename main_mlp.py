import os.path

import numpy as np
import warnings
warnings.simplefilter("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from torch import optim

from feature_engineering import DataSet
from training import *
from MLP import UnRelatedDetector
from logs import logger
import config
# from StanceDetection.stage1_mlp.score import report_score

if __name__ == "__main__":
    dataset = DataSet('train', 'mlp')
    X_train, y_train, X_val, y_val = dataset.load_features()

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(np.vstack(X_train))
    X_val = scaler.transform(np.vstack(X_val))

    train_data_loader = dataset.make_data_loader(X_train, y_train, ommit_unrelateds=False)
    val_data_loader = dataset.make_data_loader(X_val, y_val, ommit_unrelateds=False)

    mlp = UnRelatedDetector('train')
    # loss_fn = nn.CrossEntropyLoss(reduction='mean')
    # optimizer = optim.SGD(clf.parameters(), lr=config.SGD.LR, weight_decay=config.SGD.WEIGHT_DECAY)

    mlp_logger = logger('mlp')

    iterations = config.MLP.ITERATIONS
    validation_loss_history = []
    train_loss_history = []
    validation_acc_history = []

    for e in range(iterations):
        train_loss, _, _ = mlp.feed_data(train_data_loader)
        train_loss_history.append({'epoch': e, 'loss': train_loss})
        if e % 10 == 9:
            mlp.update_phase('eval')
            validation_loss, accuracy, _ = mlp.feed_data(val_data_loader)
            mlp.update_phase('train')
            mlp_logger.print_log(e, accuracy, train_loss, validation_loss)
            validation_loss_history.append({'epoch': e, 'loss': validation_loss})
            validation_acc_history.append({'epoch': e, 'accuracy': accuracy})

    mlp_logger.log('train_loss', train_loss_history)
    mlp_logger.log('val_loss', validation_loss_history)
    mlp_logger.log('val_acc', validation_acc_history)
    mlp_logger.save_model(mlp)

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


