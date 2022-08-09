import warnings
import numpy as np
import random

from sklearn.preprocessing import MinMaxScaler
import torch

from feature_engineering import DataSet
# from training import *
from networks.MLP import UnRelatedDetector
from utils.logs import logger
import config

warnings.simplefilter("ignore")
torch.manual_seed(0)
random.seed(0)
if __name__ == "__main__":
    dataset = DataSet('train', 'mlp')
    X_train, y_train, X_val, y_val = dataset.load_features()

    train_data_loader = dataset.make_data_loader(X_train, y_train, ommit_unrelateds=False)
    val_data_loader = dataset.make_data_loader(X_val, y_val, ommit_unrelateds=False)

    mlp = UnRelatedDetector('train')
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
            mlp_logger.save_model(mlp, e)

    mlp_logger.log('train_loss', train_loss_history)
    mlp_logger.log('val_loss', validation_loss_history)
    mlp_logger.log('val_acc', validation_acc_history)
    mlp_logger.save_model(mlp, e)

    # headline = [stance['Headline'] for stance in stances_comp]
    # body_id = [stance['Body ID'] for stance in stances_comp]
    #
    # answers = pd.DataFrame()
    # answers['Headline'] = headline
    # answers['Body ID'] = body_id
    # answers['Stance'] = predicted
    # answers.to_csv('answers.csv', index=False, encoding='utf-8')
    #


