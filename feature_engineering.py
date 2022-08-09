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

from gensim.models.word2vec import Word2Vec
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from numpy.linalg import norm
from gensim.models.word2vec import Word2Vec

from utils.callback import callback
_wnl = nltk.WordNetLemmatizer()

class DataSet():
    def __init__(self, phase, network):
        # self.W2V_SIZE = config.W2V_SIZE
        self.phase = phase
        self.network = network
        self.batch_size = config.BATCH_SIZE
        self.use_transformers = config.use_transformers
        self.w2vect_model_path = config.W2V_MODEL_PATH
        self.data = self.load_data(self.phase)
    
    def load_data(self, name):
        path = config.DATA_PATH
        stances_path = name + "_stances.csv"
        bodies_path = name + "_bodies.csv"

        print("Reading {} dataset".format(name))
        data = self.read(stances_path, path)  
        articles = self.read(bodies_path, path) 

        data = pd.DataFrame(data)
        articles = pd.DataFrame(articles)
        df = data.merge(articles, on='Body ID')
        return df

    def read(self, filename, path):
        rows = []
        with open(path + "/" + filename, "r", encoding='utf-8') as table:
            r = DictReader(table)
            for line in r:
                rows.append(line)
        return rows

    def make_path(self, phase, type):
        name = phase+'_'+type
        return os.path.join(config.DATA_PATH, name)

    def load_features(self):
        if self.phase=='train':
            train_df_path = self.make_path('train', 'df')
            val_df_path = self.make_path('val', 'df')
            if os.path.exists(train_df_path) & os.path.exists(val_df_path):
                train_df = pd.read_pickle(train_df_path)
                val_df = pd.read_pickle(val_df_path)
            else:
                train_df, val_df = train_test_split(self.data, test_size=config.TEST_SIZE, random_state=42, 
                                                    stratify=self.data['Stance'])
                train_df.to_pickle(train_df_path)
                val_df.to_pickle(val_df_path)
            
            train_features, train_stances = self.extract_features(train_df, 'train')
            val_features, val_stances = self.extract_features(val_df, 'val')
            result = [train_features, train_stances, val_features, val_stances]

        elif self.phase=='competition_test':
            comp_features, comp_stances = self.extract_features(self.data, 'competition_test')
            result = [1, 1, comp_features, comp_stances]
        return result

    def extract_features(self, df, name):
        if self.use_transformers:
            text_feats = self.extract_bert_features(df, name)
            if self.network == 'mlp':
                headlines = [x.mean(axis=0) for x in text_feats[:, 0]]
                articles = [x.mean(axis=0) for x in text_feats[:, 1]]
                text_feats = np.concatenate([headlines, articles], axis=1)
        else:
            text_feats = self.extract_word2vect_features(df, name)
        if config.use_cosines and self.network!='mlp':
            cosine_feats = self.extract_cosine_features(df, name)
            text_feats = np.concatenate([text_feats, cosine_feats[:, None]], axis=1)

        stance_path = self.make_path(name, 'stance')
        if os.path.exists(stance_path):
            stances = pd.read_pickle(stance_path)
        else:
            df['label'] = df['Stance'].apply(lambda x: config.STANCE_MAP[x])
            stances = df['label']
            stances.to_pickle(stance_path)
        
        text_feats = np.array(text_feats)
        stances = np.array(stances)
        return text_feats, stances

    def extract_bert_features(self, df, phase):
        bert_feature_path = self.make_path(phase, 'lstm_bert_feats')
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

    def extract_word2vect_features(self, df, name):
        feature_path = self.make_path(name, self.network+'_w2v_feature_matrix')
        if not os.path.exists(feature_path):
            df = self.clean_data(df)
            headlines = df['tokenized_Headline'].values.tolist()
            articles = df['tokenized_articleBody'].values.tolist()
            w2v_model = self.load_w2vec_model(name, headlines, articles)
            feats = []
            for headline, article in zip(headlines, articles):
                if self.network == 'mlp':
                    headline_feat = self.buildSentenceVector(w2v_model, headline)
                    article_feat = self.buildSentenceVector(w2v_model, article)
                elif self.network == 'lstm':
                    headline_feat = self.buildSentenceMat(w2v_model, headline)
                    article_feat = self.buildSentenceMat(w2v_model, article)
                    print(headline_feat.shape)
                mat = {'headline_feature': headline_feat, 'article_feature': article_feat}
                feats.append(mat)
            w2v_feats = pd.DataFrame(feats)
            w2v_feats.to_pickle(feature_path)
        else:
            w2v_feats = pd.read_pickle(feature_path)
        w2v_feats = np.array(w2v_feats)
        return w2v_feats

    def clean_data(self, df):
        cols = ['Headline', 'articleBody']
        print('Cleaning Data...')
        for col in cols:
            col_name = 'tokenized_'+col
            df[col_name] = df[col].apply(lambda x: self.clean(x))
            df[col_name] = df[col_name].apply(lambda x: self.get_tokenized_lemmas(x))
            df[col_name] = df[col_name].apply(lambda x: self.remove_stopwords(x))
        return df

    def clean(self, s):
        # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
        return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()

    def get_tokenized_lemmas(self, s):
        return [self.normalize_word(w) for w in nltk.word_tokenize(s)]

    def normalize_word(self, w):
        return _wnl.lemmatize(w).lower()

    def remove_stopwords(self, l):
        # Removes stopwords from a list of tokens
        return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]

    def load_w2vec_model(self, name, headlines, articles):
        if name == 'train':
            sentences = headlines + articles
            w2v_model = self.train_word2vec(sentences)
        elif name in ['val', 'competition_test']:
            model_name = "word2vec.model"
            path = os.path.join(self.w2vect_model_path, model_name)
            w2v_model = Word2Vec.load(path).wv
        return w2v_model

    def train_word2vec(self, sentences):
        model_name = "word2vec.model"
        path = os.path.join(self.w2vect_model_path, model_name)
        if not os.path.exists(path):
            print('Training Word2Vec model')
            model = Word2Vec(sentences, window=8, min_count=5, size=config.W2V_SIZE,
                                        sg=1, hs=0, alpha=0.025, min_alpha=1e-4, negative=5,
                                        ns_exponent=0.75, compute_loss=True, callbacks=[callback()],
                                        seed=1234, iter=20, workers=4)
            model.save(path)
        else:
            model = Word2Vec.load(path)
        return model.wv

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
        return sentence_matrix

    def extract_cosine_features(self, df, phase):
        if self.use_transformers:
            cosine_feature_name = 'cosine_bert_feats'
            feature_name = '_bert_feats'
        else:
            cosine_feature_name = 'cosine_w2v_feature_matrix'
            feature_name = '_w2v_feature_matrix'
        
        cosine_path = self.make_path(phase, self.network+cosine_feature_name)
        feature_path = self.make_path(phase, self.network+feature_name)
        if os.path.exists(cosine_path):
            cosine_feats = pd.read_pickle(cosine_path)
        else:
            df = pd.read_pickle(feature_path)
            def sim(A, B):
                A = np.array(A).reshape(-1, 1024)
                B = np.array(B).reshape(1024)
                return np.dot(A,B)/(norm(A, axis=1)*norm(B))

            df['avg_head'] = df['Headline_sentences'].apply(lambda x: np.mean(x, axis=0) if len(x)>1 else x)
            df['cosine'] = df.apply(lambda x: sim(x.article_sentences, x.avg_head), axis=1)
            df['cosine'].to_pickle(cosine_path)
            cosine_feats = df['cosine'].values.tolist()
        
        cosine_feats = [x / x.sum() for x in cosine_feats]
        cosine_feats = np.array(cosine_feats)
        return cosine_feats

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

            if config.use_cosines:
                padded_cosines = np.zeros((X.shape[0], config.ARTICLE_MAX_SENTENCES))
                for i, l in enumerate(article_len):
                    padded_cosines[i, :l] = X[i, 2][:l]
                data_loader.append(padded_cosines)

        elif self.network == 'mlp':
            S = [0 if s == 3 else 1 for s in S]
            if not self.use_transformers:
                headline = np.vstack(X[:, 0][:][:])
                article = np.vstack(X[:, 1][:][:])
                X = np.concatenate([headline, article], axis=1)
            data_loader = [X, S]
        return data_loader

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


