import os, numpy as np
import config
import re
import pandas as pd

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset, DatasetDict
from feature_engineering import DataSet

def remove_cot(row):
    row = [e for e in row if e!='']
    return row

def get_dataset(data_dir):
    dataset = DataSet('train', 'lstm', True)

    train_df_path = dataset.make_path('train', 'df')
    train_df = pd.read_pickle(train_df_path)
    train_df = train_df.sample(frac=0.01)
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
    val_df['feature'] = val_df['Headline'] + val_df['articleBody']
    val_feat = val_df['feature'].values.tolist()
    val_labels = val_df['label'].values.tolist()

    return DatasetDict(
        train=Dataset.from_dict({"text": features, "label": labels}),
        val=Dataset.from_dict(
            {
                "text": val_feat,
                "label": val_labels,
            }
        )
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    data_dict = get_dataset("data")

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)

    def preprocess(example):
        return tokenizer(example["text"], max_length=50, truncation=True)

    encoded_dataset = data_dict.map(preprocess, batched=True)

    backbone = AutoModelForSequenceClassification.from_pretrained(
        "bert-large-uncased", num_labels=3
    ).to('cuda:0')
    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        "checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        num_train_epochs=1,
        metric_for_best_model="accuracy",
        per_device_eval_batch_size=4,
        per_device_train_batch_size=4,
    )

    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Trainer
    trainer = Trainer(
        backbone,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    print(
        trainer.evaluate(eval_dataset=encoded_dataset["test"], metric_key_prefix="test")
    )


if __name__ == "__main__":
    main()