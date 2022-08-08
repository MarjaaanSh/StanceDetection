from sentence_transformers import SentenceTransformer, models, InputExample, losses, evaluation
from torch.utils.data import DataLoader

import config
import re
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from feature_engineering import DataSet
import pandas as pd
from datasets import Dataset, DatasetDict

def remove_cot(row):
    row = [e for e in row if e!='']
    return row

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

dataset = DataSet('train', 'lstm', True)

train_df_path = dataset.make_path('train', 'df')
train_df = pd.read_pickle(train_df_path)
train_df['Headline'] = train_df['Headline'].apply(lambda x: re.split("[.\n]+", x))
train_df['articleBody'] = train_df['articleBody'].apply(lambda x: re.split("[.\n]+", x))
train_df['Headline'] = train_df['Headline'].apply(lambda x: remove_cot(x))
train_df['articleBody'] = train_df['articleBody'].apply(lambda x: remove_cot(x))
train_df = train_df.explode('Headline')
train_df = train_df.explode('articleBody')
train_df['label'] = train_df['Stance'].apply(lambda x: config.STANCE_MAP[x])
train_df = train_df[train_df['label']!=3]
train_df['feature'] = train_df['Headline'] + train_df['articleBody']
features = train_df['feature'].values.tolist()
labels = train_df['label'].values.tolist()


# train_examples = [dict(texts=text, label=label) for text, label in zip(features, labels)]
train_examples = Dataset.from_dict({'texts': features, 'label': labels})
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

val_df_path = dataset.make_path('train', 'df')
val_df = pd.read_pickle(val_df_path)
val_df['Headline'] = val_df['Headline'].apply(lambda x: re.split("[.\n]+", x))
val_df['articleBody'] = val_df['articleBody'].apply(lambda x: re.split("[.\n]+", x))
val_df['Headline'] = val_df['Headline'].apply(lambda x: remove_cot(x))
val_df['articleBody'] = val_df['articleBody'].apply(lambda x: remove_cot(x))
val_df = val_df.explode('Headline')
val_df = val_df.explode('articleBody')
val_df['label'] = val_df['Stance'].apply(lambda x: config.STANCE_MAP[x])
val_df = val_df[val_df['label']!=3]
h = val_df['Headline'].values.tolist()
a = val_df['articleBody'].values.tolist()
l = val_df['label'].astype(float).values.tolist()

evaluator = evaluation.EmbeddingSimilarityEvaluator(h, a, l)
word_embedding_model = models.Transformer('bert-large-uncased', max_seq_length=512).to('cuda:0')
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(device='cuda:0', cache_folder='./')
train_loss = losses.CosineSimilarityLoss(model).to('cuda:0')

model.fit(train_objectives=[(train_dataloader, train_loss)], 
          epochs=1, 
          warmup_steps=100,
          evaluator=evaluator, 
          evaluation_steps=500,
          weight_decay=0.01,
          output_path='/',
          save_best_model=True,
          show_progress_bar=True)

              #     "checkpoints",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=1e-5,
    #     warmup_ratio=0.1,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     num_train_epochs=5,
    #     metric_for_best_model="accuracy",
    #     per_device_eval_batch_size=64,
    #     per_device_train_batch_size=32,
    # )








# def main():
#     data_dict = get_dataset()

#     tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)

    # def preprocess(example):
    #     return tokenizer(example["text"], max_length=50, truncation=True)

    # encoded_dataset = data_dict.map(preprocess, batched=True)

    # backbone = AutoModelForSequenceClassification.from_pretrained(
    #     "bert-large-uncased", num_labels=4
    # ).to(device)
    # # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments
    # training_args = TrainingArguments(
    #     "checkpoints",
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     learning_rate=1e-5,
    #     warmup_ratio=0.1,
    #     weight_decay=0.01,
    #     load_best_model_at_end=True,
    #     num_train_epochs=5,
    #     metric_for_best_model="accuracy",
    #     per_device_eval_batch_size=64,
    #     per_device_train_batch_size=32,
    # )

    # # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Trainer
    # trainer = Trainer(
    #     backbone,
    #     training_args,
    #     train_dataset=encoded_dataset["train"],
    #     eval_dataset=encoded_dataset["val"],
    #     tokenizer=tokenizer,
    #     compute_metrics=compute_metrics
    # )

    # trainer.train()

    # print(
    #     trainer.evaluate(eval_dataset=encoded_dataset["val"], metric_key_prefix="val")
    # )


# if __name__ == "__main__":
#     main()