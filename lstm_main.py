import warnings

import torch
import random

from feature_engineering import DataSet
from LSTM import LSTMRelatedDetector
from logs import logger
import config


# from utils.score import report_score


warnings.simplefilter("ignore")
torch.manual_seed(0)
random.seed(0)

if __name__ == "__main__":
    dataset = DataSet('train', 'lstm')
    X_train, s_train, X_val, s_val = dataset.load_features()

    train_data_loader = dataset.make_data_loader(X_train, s_train, ommit_unrelateds=True)
    val_data_loader = dataset.make_data_loader(X_val, s_val, ommit_unrelateds=True)

    lstm = LSTMRelatedDetector(phase='train')

    # lstm_logger = logger('lstm', 0)
    
    iterations = config.LSTM.ITERATIONS
    validation_loss_history = []
    train_loss_history = []
    validation_acc_history = []
    
    for e in range(iterations):
        train_loss, _, _ = lstm.feed_data(train_data_loader)
        train_loss_history.append({'epoch': e, 'loss': train_loss})
    #     if e % 10 == 9:
    #         lstm.update_phase('eval')
    #         validation_loss, accuracy, _ = lstm.feed_data(val_data_loader)
    #         lstm.update_phase('train')
    #         lstm_logger.print_log(e, accuracy, train_loss, validation_loss)
    #         validation_loss_history.append({'epoch': e, 'loss': validation_loss})
    #         validation_acc_history.append({'epoch': e, 'accuracy': accuracy})

    # lstm_logger.log('train_loss', train_loss_history)
    # lstm_logger.log('val_loss', validation_loss_history)
    # lstm_logger.log('val_acc', validation_acc_history)
    # lstm_logger.save_model(lstm)
    
    # headline = [stance['Headline'] for stance in stances_comp]
    # body_id = [stance['Body ID'] for stance in stances_comp]
    
    # answers = pd.DataFrame()
    # answers['Headline'] = headline
    # answers['Body ID'] = body_id
    # answers['Stance'] = predicted
    # answers.to_csv('answers.csv', index=False, encoding='utf-8')
    


