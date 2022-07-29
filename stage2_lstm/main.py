import warnings
warnings.simplefilter("ignore")


from data_processing import make_data_loader
from feature_engineering import DataSet
from neural_networks import LSTM
from LSTM import multiStageStanceDetector
from training import *

from utils.score import report_score

torch.cuda.empty_cache()

import gc
gc.collect()

torch.manual_seed(0)
random.seed(0)

if __name__ == "__main__":
    dataset = DataSet('train')
    X_train, s_train, X_val, s_val = dataset.load_features()

    train_data_loader = make_data_loader(X_train, s_train, ommit_unrelateds=True)
    val_data_loader = make_data_loader(X_val, s_val, ommit_unrelateds=True)

    # lstm = BieberLSTM(nb_lstm_units=config.LSTM.HIDDEN_STATE, embedding_dim=config.W2V_SIZE,
    #                   batch_size=config.BATCH_SIZE, nb_classes=config.LSTM.classes)
    # lstm.to(config.device)

    # loss_fn = nn.CrossEntropyLoss(reduction='mean')
    model = multiStageStanceDetector()
    
    lstm_logger = logger('lstm', 0)
    
    EPOCHS = config.EPOCHS
    validation_loss_history = []
    train_loss_history = []
    validation_acc_history = []
    
    for e in range(EPOCHS):
        lstm, train_loss = train_loop(train_data_loader, lstm, loss_fn, optimizer)
        train_loss_history.append({'epoch': e, 'loss': train_loss})
        if e % 9 == 0:
            accuracy, validation_loss, _ = test_model(val_data_loader, lstm, loss_fn)
            lstm_logger.print_log(e, accuracy, train_loss, validation_loss)
            validation_loss_history.append({'epoch': e, 'loss': validation_loss})
            validation_acc_history.append({'epoch': e, 'accuracy': accuracy})

    # lstm_logger.log('train_loss', train_loss_history)
    # lstm_logger.log('val_loss', validation_loss_history)
    # lstm_logger.log('val_acc', validation_acc_history)
    # lstm_logger.save_model(lstm)
    
    # with torch.no_grad():
    #     pred = lstm(torch.tensor(X_val.astype(np.float32)))
    #     pred = pred.argmax(1).numpy()
    
    # report_score(s_val, pred)

    # headline = [stance['Headline'] for stance in stances_comp]
    # body_id = [stance['Body ID'] for stance in stances_comp]
    #
    # answers = pd.DataFrame()
    # answers['Headline'] = headline
    # answers['Body ID'] = body_id
    # answers['Stance'] = predicted
    # answers.to_csv('answers.csv', index=False, encoding='utf-8')
    #


