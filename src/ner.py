import re
import os
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

conll = load_dataset("text", data_files={
    "train": "../data/cleaned/train.conll",
    "validation": "../data/cleaned/valid.conll",
    "test": "../data/cleaned/test.conll"
})

# read conll & parsing label
def parse_conll(text):
    sentence = text["text"].strip().split("\n\n")
    tokens = []
    labels = []

    for s in sentence:
        toks = []
        labs = []
        for line in s.split("\n"):
            if line.strip():
                tok, lab = line.rsplit(maxsplit=1)
                toks.append(tok)
                labs.append(lab)
        tokens.append(toks)
        labels.append(labs)

    return {"tokens": tokens, "ner_tags": labels}

parsed = conll.map(parse_conll, batched=False)

label_set = set()

for example_labels in parsed["train"]["ner_tags"]:
    for sentence_labels in example_labels:
        for label in sentence_labels:
            label_set.add(label)

label_list = sorted(list(label_set))

label2id = {}
for i, l in enumerate(label_list):
    label2id[l] = i

id2label = {}
for i, l in enumerate(label_list):
    id2label[i] = l

print(label2id)

def flatten_example(example):
    all_tokens = []
    all_labels = []

    for sentence_tokens, sentence_labels in zip(example["tokens"], example["ner_tags"]):
        all_tokens.extend(sentence_tokens)
        all_labels.extend(sentence_labels)

    example["tokens"] = all_tokens
    example["ner_tags"] = all_labels
    return example

parsed = parsed.map(flatten_example, batched=False)

# tokenization
model_pretrained = "cahya/bert-base-indonesian-NER"
tokenizer = AutoTokenizer.from_pretrained(model_pretrained)

def tokenize(example):
    tokenized = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=128
    )

    labels = []
    previous_word_idx = None
    word_ids = tokenized.word_ids()

    for word_idx in word_ids:
        if word_idx is None:
            labels.append(-100)
        elif word_idx != previous_word_idx:
            labels.append(label2id[example["ner_tags"][word_idx]])
        else:
            labels.append(-100)
        previous_word_idx = word_idx
    
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = parsed.map(tokenize, batched=False)

# load model
model = AutoModelForTokenClassification.from_pretrained(
    model_pretrained,
    num_labels=len(label2id),
    ignore_mismatched_sizes=True
)

model.config.label2id = label2id
model.config.id2label = id2label

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# model.to(device)

data_collator = DataCollatorForTokenClassification(tokenizer)

training_args = TrainingArguments(
    output_dir = "../model/ner_model",
    logging_dir="../model/logs",
    evaluation_strategy = "epoch",
    save_strategy = "epoch",
    learning_rate = 5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=20,
    load_best_model_at_end=True,
)

def compute_metrics(predic):
    preds = np.argmax(predic.predictions, axis=-1)
    labels = predic.label_ids

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):

        seq_preds = []
        seq_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                seq_preds.append(id2label[pred_id])
                seq_labels.append(id2label[label_id])

        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# eval
results = trainer.evaluate()
print(results)

test_results = trainer.predict(tokenized_dataset["test"])
print(test_results.metrics)

predictions, labels, _ = test_results
preds = np.argmax(predictions, axis=-1)

# confusion matrix
def convert_predictions(preds, labels, id2label):
    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):

        seq_preds = []
        seq_labels = []

        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id != -100:
                seq_preds.append(id2label[pred_id])
                seq_labels.append(id2label[label_id])

        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return true_labels, true_preds

def flatten_sequences(true_labels, true_preds):
    flat_true = []
    flat_pred = []

    for seq_true, seq_pred in zip(true_labels, true_preds):
        for t, p in zip(seq_true, seq_pred):
            flat_true.append(t)
            flat_pred.append(p)

    return flat_true, flat_pred

def plot_confusion_matrix(flat_true, flat_pred, idtolabel):
    labels_list = list(idtolabel.values())
    cm = confusion_matrix(flat_true, flat_pred, labels=labels_list)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues",
                xticklabels=labels_list,
                yticklabels=labels_list)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("NER Confusion Matrix")
    plt.savefig("../model/confusion_matrix.png")
    plt.show()

def evaluate_testset(trainer, tokenized_dataset, idtolabel):
    test_results = trainer.predict(tokenized_dataset["test"])
    predictions, labels, _ = test_results

    preds = np.argmax(predictions, axis=-1)

    true_labels, true_preds = convert_predictions(preds, labels, idtolabel)

    print("\n=== Classification Report ===\n")
    print(classification_report(true_labels, true_preds))

    flat_true, flat_pred = flatten_sequences(true_labels, true_preds)

    print("\n=== Confusion Matrix ===\n")
    plot_confusion_matrix(flat_true, flat_pred, idtolabel)

    return true_labels, true_preds

if __name__ == "__main__":
    results = trainer.evaluate()
    with open("../model/validation_metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    print(results)

    test_results = trainer.predict(tokenized_dataset["test"])
    with open("../model/test_metrics.json", "w") as f:
        json.dump(test_results.metrics, f, indent=4)

    print(test_results.metrics)

    evaluate_testset(trainer, tokenized_dataset, id2label)