import pandas as pd
import numpy as np

df = pd.read_pickle('/home/marjan/StanceDetection/data/val_lstm_bert_feats')
df['avg_head'] = df['Headline_sentences'].apply(lambda x: np.mean(x, axis=0) if len(x)>1 else x)
from numpy.linalg import norm
 
# # define two lists or array
A = np.array([[2,1],[3,2], [-1,-3]])
print(A.shape)
B = np.array([3,2])
print(B.shape)
# # print("A:\n", A)
# # print("B:\n", B)
 
# # # compute cosine similarity
# cosine = np.dot(A,B)/(norm(A, axis=1)*norm(B))
# print(cosine)

def sim(A, B):
    A = np.array(A).reshape(-1, 1024)
    B = np.array(B).reshape(1024)
    return np.dot(A,B)/(norm(A, axis=1)*norm(B))

print(df.columns)
df['res'] = df.apply(lambda x: sim(x.article_sentences, x.avg_head), axis=1)
print(df['res'].head(1))