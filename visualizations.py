from feature_engineering import DataSet

import pandas as pd
import matplotlib.pyplot as plt
from score import report_score
import seaborn as sn

from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# dataset = DataSet('competition_test', 'mlp', True)
# df = dataset.load_data('competition_test')

#print(df['Headline'].nunique(), df['articleBody'].nunique(), df.shape, df.drop_duplicates().shape)

# rs = df.groupby('Stance').count()
# rs /= df.shape[0]
# print(rs)

# rs = df[df['Stance']!='unrelated']
# x = rs.shape[0]
# rs = rs.groupby('Stance').count()['Headline']
# rs /= x
# print(rs)



# df = pd.read_pickle('/home/marjan/StanceDetection/data/train_lstm_bert_feats')
# df['len_head'] = df['Headline_sentences'].apply(lambda x: x.shape[0])
# df['len_art'] = df['article_sentences'].apply(lambda x: x.shape[0])

# print(df['len_head'].quantile(q=0.5), df['len_head'].quantile(q=0.75), df['len_head'].quantile(q=0.95), df['len_head'].max())
# print(df['len_art'].quantile(q=0.5), df['len_art'].quantile(q=0.75), df['len_art'].quantile(q=0.95), df['len_art'].max())

df = pd.read_csv('res.csv')
print(df[df['actual']==df['predicted']].shape[0]/df.shape[0])
conf = report_score(df['actual'], df['predicted'], stage='final', labels=['agree', 'disagree', 'discuss', 'unrelated'])
p, r, f1, _ = precision_recall_fscore_support(df['actual'], df['predicted'], average="macro")
print(f1)

all_stances = df['actual'].values.tolist()
final_pred_stance = df['predicted'].values.tolist()

actual_agree = [1 if x=='agree' else 0 for x in all_stances]
pred_agree = [1 if x=='agree' else 0 for x in final_pred_stance]
_, _, f1, _ = precision_recall_fscore_support(actual_agree, pred_agree, average="binary")
print('agree', f1)

actual_agree = [1 if x=='disagree' else 0 for x in all_stances]
pred_agree = [1 if x=='disagree' else 0 for x in final_pred_stance]
_, _, f1, _ = precision_recall_fscore_support(actual_agree, pred_agree, average="binary")
print('disagree', f1)

actual_agree = [1 if x=='discuss' else 0 for x in all_stances]
pred_agree = [1 if x=='discuss' else 0 for x in final_pred_stance]
_, _, f1, _ = precision_recall_fscore_support(actual_agree, pred_agree, average="binary")
print('discuss', f1)

actual_agree = [1 if x=='unrelated' else 0 for x in all_stances]
pred_agree = [1 if x=='unrelated' else 0 for x in final_pred_stance]
_, _, f1, _ = precision_recall_fscore_support(actual_agree, pred_agree, average="binary")
print('unrelated', f1)

plt.clf()
ax = plt.subplot()
sn.heatmap(conf, annot=True, fmt='g', cmap='BuGn',
           linewidths=4, square=True, ax=ax)
ax.set(xlabel='predicted', ylabel='actual')
ax.xaxis.set_ticklabels(['agree', 'disagree', 'discuss', 'unrelated'])
ax.yaxis.set_ticklabels(['agree', 'disagree', 'discuss', 'unrelated'])
plt.savefig('baseline_conf.png')