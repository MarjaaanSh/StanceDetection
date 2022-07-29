from multiprocessing.spawn import old_main_modules
import random
import pandas as pd
import os
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import config


def make_data_loader(X, S, ommit_unrelateds):
    if ommit_unrelateds:
        unrelateds = (S==3)
        X = X[~unrelateds]
        S = S[~unrelateds]

    X_lengths = [len(x) for x in X]
    data_loader = [X, S, X_lengths]
    return data_loader
