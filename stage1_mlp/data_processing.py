from csv import DictReader
import pandas as pd
import os

import torch
from torch.utils.data import DataLoader

import config


def read(filename, path):
    rows = []
    with open(path + "/" + filename, "r", encoding='utf-8') as table:
        r = DictReader(table)
        for line in r:
            rows.append(line)
    return rows

def load_data(name):
    path = config.DATA_PATH
    stances_path = name + "_stances.csv"
    bodies_path = name + "_bodies.csv"

    print("Reading {} dataset".format(name))
    data = read(stances_path, path)  # stances: list of dicts, dicts: Headline, Body ID, Stance
    articles = read(bodies_path, path)  # articles: list of dicts, dicts: {'Body ID', 'articleBody'}

    data = pd.DataFrame(data)
    articles = pd.DataFrame(articles)
    df = data.merge(articles, on='Body ID')

    print("Total distinct HeadLines: ", df['Headline'].nunique())
    print("Total distinct BodyIDs: ", df['Body ID'].nunique())

    df['relevancy'] = df['Stance'].apply(lambda x: 0 if x == 'unrelated' else 1)

    # data_name = 'raw_data.pkl'
    # path = os.path.join(path, data_name)
    # df.to_pickle(path)

    return df

def get_data(name):
    data_name = 'cleaned_{}_data.pkl'.format(name)
    path = os.path.join(config.DATA_PATH, data_name)
    if not os.path.exists(path):
        df = load_data(name=name)
    else:
        df = pd.read_pickle(path)
    return df

def make_data_loader(X, y, batch_size=config.BATCH_SIZE):
    X = torch.Tensor(X)
    y = torch.Tensor(y).type(torch.LongTensor)

    data = []
    for i in range(X.shape[0]):
        data.append([X[i], y[i]])

    data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)
    return data_loader
