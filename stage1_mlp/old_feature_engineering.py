import re
import nltk
from sklearn import feature_extraction
import os
import pandas as pd
import numpy as np

import config
from data_processing import load_data
from training import train_word2vec

from gensim.models.word2vec import Word2Vec

W2V_SIZE = 300
_wnl = nltk.WordNetLemmatizer()

def normalize_word(w):
    return _wnl.lemmatize(w).lower()

def get_tokenized_lemmas(s):
    return [normalize_word(w) for w in nltk.word_tokenize(s)]

def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

def clean_data(df):
    cols = ['Headline', 'articleBody']
    print('Cleaning Data...')
    for col in cols:
        col_name = 'tokenized_'+col
        df[col_name] = df[col].apply(lambda x: clean(x))
        df[col_name] = df[col_name].apply(lambda x: get_tokenized_lemmas(x))
        df[col_name] = df[col_name].apply(lambda x: remove_stopwords(x))
    return df

def buildSentenceVector(w2v_model, sentence):
    size = config.W2V_SIZE
    vec = np.zeros((1, size))
    vec_size = 0
    for token in sentence:
        try:
            vec += w2v_model[token].reshape(1, size)
            vec_size += 1
        except KeyError:
            continue
    if vec_size != 0:
        vec /= vec_size
    return vec

def convert_to_vector(w2v_model, headlines, articles, name):
    path = os.path.join(config.DATA_PATH, '{}_numerical_feats.pkl'.format(name))
    if os.path.exists(path):
        w2v_feats = pd.read_pickle(path)
    else:
        vects = []
        for headline, article in zip(headlines, articles):
            headline_vec = buildSentenceVector(w2v_model, headline)
            article_vec = buildSentenceVector(w2v_model, article)
            headline_vec = list(headline_vec)
            article_vec = list(article_vec)
            vect = {'headline': headline, 'article': article,
                    'vector': np.concatenate([headline_vec, article_vec], axis=1)}
            vects.append(vect)
        w2v_feats = pd.DataFrame(vects)
        w2v_feats.to_pickle(path)
    return w2v_feats

def extract_features(df=None, name='train'):
    file_name = 'cleaned_{}_data.pkl'.format(name)
    path = os.path.join(config.DATA_PATH, file_name)
    if os.path.exists(path):
        df = pd.read_pickle(path)
    else:
        df = clean_data(df)
        df.to_pickle(path)

    stances = df['Stance'].values
    relevancy = df['relevancy'].values
    headlines, articles = df['tokenized_Headline'].values.tolist(), df['tokenized_articleBody'].values.tolist()
    sentences = headlines + articles

    if name == 'train':
        wv = train_word2vec(sentences)
    elif name in ['val', 'competition_test']:
        model_name = "word2vec.model"
        path = os.path.join(config.MODEL_PATH, model_name)
        wv = Word2Vec.load(path).wv

    df = convert_to_vector(wv, headlines, articles, name)
    del sentences

    text_feats = np.vstack(df['vector'].values)

    return text_feats, stances, relevancy

