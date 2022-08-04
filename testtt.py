import pandas as pd

df = pd.read_pickle('/home/marjan/StanceDetection/data/train_lstm_bert_feats')
print(df.head(1))