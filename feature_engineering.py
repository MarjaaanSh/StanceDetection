import re

import nltk
from sklearn import feature_extraction
import os
import pandas as pd
import numpy as np
from csv import DictReader

import config
from training import train_word2vec

from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split

_wnl = nltk.WordNetLemmatizer()

class DataSet():
    def __init__(self, phase, network):
        self.W2V_SIZE = config.W2V_SIZE
        self.phase = phase
        self.network = network
        self.batch_size = config.BATCH_SIZE
    
    def make_path(self, phase, type):
        name = phase+'_'+type
        return os.path.join(config.DATA_PATH, name)

    def read(self, filename, path):
        rows = []
        with open(path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)
        return rows

    def load_data(self, name):
        path = config.DATA_PATH
        stances_path = name + "_stances.csv"
        bodies_path = name + "_bodies.csv"

        print("Reading {} dataset".format(name))
        data = self.read(stances_path, path)  # stances: list of dicts, dicts: Headline, Body ID, Stance
        articles = self.read(bodies_path, path)  # articles: list of dicts, dicts: {'Body ID', 'articleBody'}

        data = pd.DataFrame(data)
        articles = pd.DataFrame(articles)
        df = data.merge(articles, on='Body ID')

        print("Total distinct HeadLines: ", df['Headline'].nunique())
        print("Total distinct BodyIDs: ", df['Body ID'].nunique())

        df['relevancy'] = df['Stance'].apply(lambda x: 0 if x == 'unrelated' else 1)

        return df

    def extract_features(self, df, phase):
        stance_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
        feature_path = self.make_path(phase, self.network+'_feature_matrix')
        stance_path = self.make_path(phase, 'stance')

        if os.path.exists(feature_path) and os.path.exists(stance_path):
            text_feats = pd.read_pickle(feature_path)
            stances = pd.read_pickle(stance_path)
        else:
            df = self.clean_data(df)

            headlines, articles = df['tokenized_Headline'].values.tolist(), df['tokenized_articleBody'].values.tolist()
            text_feats = self.get_numerical_feats(headlines, articles, phase)
            text_feats = text_feats['sentence_feature']
            text_feats.to_pickle(feature_path)

            df['Stance'] = df['Stance'].apply(lambda x: stance_map[x])
            stances = df['Stance']
            stances.to_pickle(stance_path)

        return text_feats.values, stances.values

    def load_features(self):
        df = self.load_data(self.phase)
        if self.phase=='train':
            train_df_path = self.make_path('train', 'df')
            val_df_path = self.make_path('val', 'df')
            if os.path.exists(train_df_path) & os.path.exists(val_df_path):
                train_df = pd.read_pickle(train_df_path)
                val_df = pd.read_pickle(val_df_path)
            else:
                train_df, val_df = train_test_split(df, test_size=config.TEST_SIZE, random_state=42, 
                                                    stratify=df['Stance'])
                train_df.to_pickle(train_df_path)
                val_df.to_pickle(val_df_path)

            train_features, train_stances = self.extract_features(train_df, 'train')
            val_features, val_stances = self.extract_features(val_df, 'val')
            result = [train_features, train_stances, val_features, val_stances]
        
        elif self.phase=='competition_test':
            comp_features, comp_stances = self.extract_features(df, 'competition_test')
            result = [comp_features, comp_stances]

        return result

    def make_data_loader(self, X, S, ommit_unrelateds):
        if ommit_unrelateds:
            unrelateds = (S==3)
            X = X[~unrelateds]
            S = S[~unrelateds]

        if self.network == 'lstm':
            X_lengths = [len(x) for x in X]
            data_loader = [X, S, X_lengths]
        elif self.network == 'mlp':
            data_loader = []
            Y = [0 if s == 3 else 1 for s in S]
            data_loader = [X, Y]
        return data_loader

    def normalize_word(self, w):
        return _wnl.lemmatize(w).lower()

    def get_tokenized_lemmas(self, s):
        return [self.normalize_word(w) for w in nltk.word_tokenize(s)]

    def clean(self, s):
        # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

    def remove_stopwords(self, l):
        # Removes stopwords from a list of tokens
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

    def clean_data(self, df):
        cols = ['Headline', 'articleBody']
        print('Cleaning Data...')
        for col in cols:
            col_name = 'tokenized_'+col
            df[col_name] = df[col].apply(lambda x: self.clean(x))
            df[col_name] = df[col_name].apply(lambda x: self.get_tokenized_lemmas(x))
            df[col_name] = df[col_name].apply(lambda x: self.remove_stopwords(x))
        return df

    def buildSentenceMat(self, w2v_model, sentence):
        embeding_size = config.W2V_SIZE
        vocabs = 0
        sentence_matrix = []
        for token in sentence:
            try:
                vocab = w2v_model[token].reshape(1, embeding_size)
                sentence_matrix.append(vocab)
                vocabs += 1
            except KeyError:
                continue
        sentence_matrix = np.array(sentence_matrix).reshape(vocabs, embeding_size)
        return sentence_matrix

    def buildSentenceVector(self, w2v_model, sentence):
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

    def get_numerical_feats(self, headlines, articles, name):
        if name == 'train':
            sentences = headlines + articles
            w2v_model = train_word2vec(sentences)
        elif name in ['val', 'competition_test']:
            model_name = "word2vec.model"
            path = os.path.join(config.MODEL_PATH, model_name)
            w2v_model = Word2Vec.load(path).wv

        feats = []
        for headline, article in zip(headlines, articles):
            if self.network == 'mlp':
                headline_vec = self.buildSentenceVector(w2v_model, headline)
                article_vec = self.buildSentenceVector(w2v_model, article)
                headline_vec = list(headline_vec)
                article_vec = list(article_vec)
                sentence_feat = np.concatenate([headline_vec, article_vec], axis=1)
                
            elif self.network == 'lstm':          
                head_ar = headline + article
                sentence_feat = self.buildSentenceMat(w2v_model, head_ar)
            
            mat = {'headline': headline, 'article': article, 'sentence_feature': sentence_feat}
            feats.append(mat)
        
        w2v_feats = pd.DataFrame(feats)
        return w2v_feats
