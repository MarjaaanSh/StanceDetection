import sys

from scipy import rand
sys.path.insert(0, '/Users/marjanshahi/PycharmProjects/StanceDetection/stage2_lstm/utils')


import random
import numpy as np
import os.path

from gensim.models.word2vec import Word2Vec
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence

import config
# from StanceDetection.stage1_mlp.callback import *
# from StanceDetection.stage1_mlp.MLP import MLP
from utils.logs import logger


def train_word2vec(sentences):
    model_name = "word2vec.model"
    path = os.path.join(config.MODEL_PATH, model_name)
    if os.path.exists(path):
        model = Word2Vec.load(path)
    else:
        print('Training Word2Vec model')
        model = Word2Vec(sentences, window=8, min_count=5, size=config.W2V_SIZE,
                                    sg=1, hs=0, alpha=0.025, min_alpha=1e-4, negative=5,
                                    ns_exponent=0.75, compute_loss=True, callbacks=[callback()],
                                    seed=1234, iter=20, workers=4)
        model.save(path)
    return model.wv



# def train_clf(train_data_loader, val_data_loader):
#     clf = MLP()

#     loss_fn = nn.CrossEntropyLoss(reduction='mean')
#     optimizer = optim.SGD(clf.parameters(), lr=config.SGD.LR, weight_decay=config.SGD.WEIGHT_DECAY)

#     clf_logger = logger(optimizer='SGD')

#     EPOCHS = config.MLP.EPOCHS
#     validation_loss_history = []
#     train_loss_history = []
#     validation_acc_history = []

    # for e in range(EPOCHS):
    #     clf, train_loss = train_loop(train_data_loader, clf, loss_fn, optimizer)
    #     train_loss_history.append({'epoch': e, 'loss': train_loss})
    #     if e % 10 == 9:
    #         accuracy, validation_loss = test_model(val_data_loader, clf, loss_fn)
    #         clf_logger.print_log(e, accuracy, train_loss, validation_loss)
    #         validation_loss_history.append({'epoch': e, 'loss': validation_loss})
    #         validation_acc_history.append({'epoch': e, 'accuracy': accuracy})

    # clf_logger.log('train_loss', train_loss_history)
    # clf_logger.log('val_loss', validation_loss_history)
    # clf_logger.log('val_acc', validation_acc_history)
    # clf_logger.save_model('classifier', clf)
