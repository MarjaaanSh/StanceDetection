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
from utils.callback import *
from neural_networks import MLP
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


def train_loop(data_loader, model, loss_fn, optimizer):
    
    X, S, X_lengths = data_loader
    batches = list(zip(X, S, X_lengths))
    random.shuffle(batches)

    n_data = len(X)
    n_epochs = np.ceil(n_data/config.BATCH_SIZE)
    n_epochs = int(n_epochs)

    total_loss = 0
    for epoch in range(n_epochs):
        start = config.BATCH_SIZE * epoch
        end = config.BATCH_SIZE * (epoch+1)

        text, stance, lengths = zip(*batches[start:end])
        text = [torch.from_numpy(t) for t in text]
        text = pad_sequence(text, batch_first=True, padding_value=5)
        text = text.to(device=config.device)

        lens = torch.tensor(lengths).to(config.device)

        pred = model(text, lens)
        y = torch.tensor(stance).to(config.device)

        loss = loss_fn(pred, y)
        total_loss += loss    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    
    total_loss /= n_epochs
    return model, total_loss
    #print(f"train loss: {total_loss:>7f}  ")

def test_model(data_loader, model, loss_fn=None):
    X, S, X_lengths= data_loader
    batches = list(zip(X, S, X_lengths))

    n_data = len(X)
    n_epochs = np.ceil(n_data/config.BATCH_SIZE)
    n_epochs = int(n_epochs)

    test_loss, correct = 0, 0
    predictions = []
    with torch.no_grad():
        for epoch in range(n_epochs):
            start = config.BATCH_SIZE * epoch
            end = config.BATCH_SIZE * (epoch+1)

            text, stance, lengths = zip(*batches[start:end])
            text = [torch.from_numpy(t) for t in text]
            text = pad_sequence(text, batch_first=True, padding_value=5)
            text = text.to(device=config.device)

            lens = torch.tensor(lengths).to(config.device)

            pred = model(text, lens)
            if not loss_fn:
                predictions += pred.tolist()
            y = torch.tensor(stance).to(config.device)

            if loss_fn:
                test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= n_epochs
    correct /= n_data
    accuracy = 100*correct
    return accuracy, test_loss, predictions

def train_clf(train_data_loader, val_data_loader):
    clf = MLP()

    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(clf.parameters(), lr=config.SGD.LR, weight_decay=config.SGD.WEIGHT_DECAY)

    clf_logger = logger(optimizer='SGD')

    EPOCHS = config.MLP.EPOCHS
    validation_loss_history = []
    train_loss_history = []
    validation_acc_history = []

    for e in range(EPOCHS):
        clf, train_loss = train_loop(train_data_loader, clf, loss_fn, optimizer)
        train_loss_history.append({'epoch': e, 'loss': train_loss})
        if e % 10 == 9:
            accuracy, validation_loss = test_model(val_data_loader, clf, loss_fn)
            clf_logger.print_log(e, accuracy, train_loss, validation_loss)
            validation_loss_history.append({'epoch': e, 'loss': validation_loss})
            validation_acc_history.append({'epoch': e, 'accuracy': accuracy})

    clf_logger.log('train_loss', train_loss_history)
    clf_logger.log('val_loss', validation_loss_history)
    clf_logger.log('val_acc', validation_acc_history)
    clf_logger.save_model('classifier', clf)
