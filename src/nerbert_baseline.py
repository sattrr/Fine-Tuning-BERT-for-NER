import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification, DataCollatorForTokenClassification, Trainer
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("Using device:", device)

# Config
DATA_FILE = "../data/gizi-klinis.conll"
OUTPUT_DIR = "../model/baseline_model"
MODEL_NAME = "syafiqfaray/indobert-model-ner"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_conll_file(path):
    tokens_list, labels_list = [], []
    tokens, labels = [], []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("-DOCSTART-") or line == "":
                continue

            parts = line.split()
            tokens.append(parts[0])
            labels.append(parts[-1])

            if len(tokens) >= 128:
                tokens_list.append(tokens)
                labels_list.append(labels)
                tokens, labels = [], []

    if tokens:
        tokens_list.append(tokens)
        labels_list.append(labels)

    return tokens_list, labels_list

all_tokens, all_labels = read_conll_file(DATA_FILE)

train_tokens, test_tokens, train_labels, test_labels = train_test_split(
    all_tokens, all_labels, test_size=0.20, random_state=42
)

train_tokens, val_tokens, train_labels, val_labels = train_test_split(
    train_tokens, train_labels, test_size=0.20, random_state=42
)

dataset = DatasetDict({
    "train": Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels}),
    "validation": Dataset.from_dict({"tokens": val_tokens, "ner_tags": val_labels}),
    "test": Dataset.from_dict({"tokens": test_tokens, "ner_tags": test_labels}),
})

label_set = set()

for seq in all_labels:
    for tag in seq:
        label_set.add(tag)

label_list = sorted(label_set)

label2id = {}
for i, l in enumerate(label_list):
    label2id[l] = i

id2label = {}
for i, l in enumerate(label_list):
    id2label[i] = l

print("NER Labels:", label_list)

def convert_labels(example):
    example["ner_tags"] = [label2id[tag] for tag in example["ner_tags"]]
    return example

dataset = dataset.map(convert_labels)

# Tokenisasi
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_and_align_labels(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous = None
        label_ids = []

        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != previous:
                label_ids.append(label[w])
            else:
                label_ids.append(-100)
            previous = w

        labels.append(label_ids)

    tokenized["labels"] = labels
    return tokenized

tokenized = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=["tokens", "ner_tags"]
)

if "token_type_ids" in tokenized.column_names:
    tokenized = tokenized.remove_columns(["token_type_ids"])

num_labels = len(label_list)

# Load model
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# model.to(device)
model.eval()

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids

    true_preds, true_labels = [], []

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds, seq_labels = [], []
        for p_id, l_id in zip(pred_seq, label_seq):
            if l_id != -100:
                seq_preds.append(id2label[p_id])
                seq_labels.append(id2label[l_id])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return {
        "precision": precision_score(true_labels, true_preds),
        "recall": recall_score(true_labels, true_preds),
        "f1": f1_score(true_labels, true_preds),
    }

data_collator = DataCollatorForTokenClassification(tokenizer)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Eval
def convert_predictions(preds, labels, id2label):
    true_preds, true_labels = [], []

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds, seq_labels = [], []
        for p, l in zip(pred_seq, label_seq):
            if l != -100:
                seq_preds.append(id2label[p])
                seq_labels.append(id2label[l])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    return true_labels, true_preds

def flatten_sequences(true_labels, true_preds):
    flat_true, flat_pred = [], []

    for t, p in zip(true_labels, true_preds):
        flat_true.extend(t)
        flat_pred.extend(p)

    return flat_true, flat_pred

def plot_confusion_matrix(flat_true, flat_pred, id2label, save_path):
    labels_list = list(id2label.values())
    cm = confusion_matrix(flat_true, flat_pred, labels=labels_list)

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        cmap="Blues",
        xticklabels=labels_list,
        yticklabels=labels_list
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Baseline NER Confusion Matrix")
    plt.savefig(save_path)
    plt.show()

if __name__ == "__main__":
    val_results = trainer.evaluate(tokenized["validation"])
    with open(os.path.join(OUTPUT_DIR, "validation_metrics.json"), "w") as f:
        json.dump(val_results, f, indent=4)

    print("Validation metrics:", val_results)

    test_results = trainer.predict(tokenized["test"])
    preds = np.argmax(test_results.predictions, axis=-1)

    true_labels, true_preds = convert_predictions(
        preds,
        test_results.label_ids,
        id2label
    )

    flat_true, flat_pred = flatten_sequences(true_labels, true_preds)

    print("\n=== BASELINE CLASSIFICATION REPORT ===\n")
    print(classification_report(true_labels, true_preds))

    plot_confusion_matrix(
        flat_true,
        flat_pred,
        id2label,
        save_path=os.path.join(OUTPUT_DIR, "confusion_matrix_baseline.png")
    )

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_results.metrics, f, indent=4)

    print("Test metrics:", test_results.metrics)