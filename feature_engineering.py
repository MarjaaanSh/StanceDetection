import imp


import re
from matplotlib.pyplot import axis

from requests import head
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
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm

_wnl = nltk.WordNetLemmatizer()

class DataSet():
    def __init__(self, phase, network, use_transformers):
        # self.W2V_SIZE = config.W2V_SIZE
        self.phase = phase
        self.network = network
        self.batch_size = config.BATCH_SIZE
        self.use_transformers = use_transformers
    
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

        # print("Total distinct HeadLines: ", df['Headline'].nunique())
        # print("Total distinct BodyIDs: ", df['Body ID'].nunique())

        #df['relevancy'] = df['Stance'].apply(lambda x: 0 if x == 'unrelated' else 1)

        return df

    def extract_features(self, df, phase):
        stance_map = {'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3}
        stance_path = self.make_path(phase, 'stance')

        if self.use_transformers:
            text_feats = self.extract_bert_features(df, phase)
        if self.network == 'mlp':
            if os.path.exists(stance_path):
        if os.path.exists(stance_path):
            stances = pd.read_pickle(stance_path)
        else:
            df = self.clean_data(df)
            df['label'] = df['Stance'].apply(lambda x: stance_map[x])
            stances = df['label']
            stances.to_pickle(stance_path)
        
        text_feats = np.array(text_feats)
        stances = np.array(stances)
        return text_feats, stances
    


    def extract_bert_features(self, df, phase):
        bert_feature_path = self.make_path(phase, self.network+'_bert_feats')
        if os.path.exists(bert_feature_path):
            df = pd.read_pickle(bert_feature_path)
            bert_feats = df[['Headline_sentences', 'article_sentences']]
        else:
            model = SentenceTransformer('bert-large-uncased').to(config.LSTM.DEVICE)
            df['Headline_sentences'] = df['Headline'].apply(lambda x: re.split("[.\n]+", x))
            df['article_sentences'] = df['articleBody'].apply(lambda x: re.split("[.\n]+", x))

            def encode_row(row):
                return model.encode(row)

            df['Headline_sentences'] = df['Headline_sentences'].apply(lambda row: encode_row(row))
            df['article_sentences'] = df['article_sentences'].apply(lambda row: encode_row(row))

            df[['Headline_sentences', 'article_sentences']].to_pickle(bert_feature_path)
            bert_feats = df[['Headline_sentences', 'article_sentences']]
        
        bert_feats = np.array(bert_feats)
        return bert_feats



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
            result = [1, 1, comp_features, comp_stances]

        return result

    def make_data_loader(self, X, S, ommit_unrelateds):
        if self.network == 'lstm':
            if ommit_unrelateds:
                unrelateds = (S==3)
                X = X[~unrelateds, :]
                S = S[~unrelateds]

            headlines = [h.mean(axis=0) for h in X[:, 0]]
            headlines = np.array(headlines).reshape(-1, config.BERT_DIM)

            article_len = [x.shape[0] for x in X[:, 1]]
            article_len = [l if l<=config.ARTICLE_MAX_SENTENCES else config.ARTICLE_MAX_SENTENCES for l in article_len]
            padded_articles = np.zeros((X.shape[0], config.ARTICLE_MAX_SENTENCES, config.BERT_DIM))
            for i, l in enumerate(article_len):
                padded_articles[i, :l, :] = X[i, 1][:l]

            data_loader = [headlines, padded_articles, S, article_len]

        elif self.network == 'mlp':
            S = [0 if s == 3 else 1 for s in S]
            data_loader = [X, S]
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
        return list(vec)

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
                sentence_feat = np.concatenate([headline_vec, article_vec], axis=1)
                mat = {'sentence_feature': sentence_feat}

            elif self.network == 'lstm':          
                # head_ar = headline + article
                # sentence_feat = self.buildSentenceMat(w2v_model, head_ar)
                headline_feat = self.buildSentenceMat(w2v_model, headline)
                article_feat = self.buildSentenceMat(w2v_model, article)
                mat = {'headline_feature': headline_feat, 'article_feature': article_feat}

            feats.append(mat)
        
        w2v_feats = pd.DataFrame(feats)
        return w2v_feats

    def convert_label_to_stance(self, labels):
        if self.network == 'mlp':
            num_s = len(set(labels))
            if num_s == 4:
                stances = ['unrelated' if l == 3 else 'related' for l in labels]
            elif num_s == 2:
                stances = ['unrelated' if l == 0 else 'related' for l in labels]
        elif self.network == 'lstm':
            stances = [config.LABEL_MAP[l] for l in labels]
        return stances

