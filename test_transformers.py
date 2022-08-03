import os, numpy as np
import torch

print(torch.cuda.current_device())

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from transformers import logging
logging.set_verbosity_error()

from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset, DatasetDict
from feature_engineering import DataSet



def get_dataset():
    dataset = DataSet('train', 'lstm', True)
    df_train, df_val = dataset.load_features()
    df_train = df_train.sample(frac=0.01)
    df_val = df_val.sample(frac=0.01)

    return DatasetDict(
        train=Dataset.from_dict({"text": df_train['Headline'], 
                                 "label": df_train['label']}),
        val=Dataset.from_dict(
            {
                "text": df_val['Headline'],
                "label": df_val['label'],
            }
        )
    )


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    device = torch.device('cuda')
    data_dict = get_dataset()

    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased", use_fast=True)

    def preprocess(example):
        return tokenizer(example["text"], max_length=50, truncation=True)

    encoded_dataset = data_dict.map(preprocess, batched=True)

    backbone = AutoModelForSequenceClassification.from_pretrained(
        "bert-large-uncased", num_labels=4
    ).to(device)
    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.TrainingArguments
    training_args = TrainingArguments(
        "checkpoints",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        load_best_model_at_end=True,
        num_train_epochs=5,
        metric_for_best_model="accuracy",
        per_device_eval_batch_size=64,
        per_device_train_batch_size=32,
    )

    # https://huggingface.co/docs/transformers/v4.20.1/en/main_classes/trainer#transformers.Trainer
    trainer = Trainer(
        backbone,
        training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print(
        trainer.evaluate(eval_dataset=encoded_dataset["val"], metric_key_prefix="val")
    )


if __name__ == "__main__":
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print(torch.cuda.current_device())
    main()