import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_pickle('/home/marjan/StanceDetection/data/val_lstm_bert_feats')
df['len_head'] = df['Headline_sentences'].apply(lambda x: x.shape[0])
df['len_art'] = df['article_sentences'].apply(lambda x: x.shape[0])

print(df['len_head'].quantile(q=0.5), df['len_head'].quantile(q=0.75), df['len_head'].quantile(q=0.95), df['len_head'].max())
print(df['len_art'].quantile(q=0.5), df['len_art'].quantile(q=0.75), df['len_art'].quantile(q=0.95), df['len_art'].max())
