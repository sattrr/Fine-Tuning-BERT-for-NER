import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from collections import Counter
from datasets import Dataset, DatasetDict, Features, Sequence, Value, ClassLabel, load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification, DataCollatorForTokenClassification, TrainingArguments, Trainer, AdamW
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

device = torch.device("cuda")
print("Using device:", device)

# Config
DATA_FILE = "../data/gizi-klinis.conll"
OUTPUT_DIR = "../model/ner_model"

def read_conll_file(path: str):
    tokens_list = []
    labels_list = []
    tokens = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("-DOCSTART-") or line == "":
                continue
            parts = line.split()
            token = parts[0]
            tag = parts[-1]
            tokens.append(token)
            labels.append(tag)

            if len(tokens) >= 128:
                tokens_list.append(tokens)
                labels_list.append(labels)
                tokens, labels = [], []
    if tokens:
        tokens_list.append(tokens)
        labels_list.append(labels)

    return tokens_list, labels_list

all_tokens, all_labels = read_conll_file(DATA_FILE)

total_tokens_count = sum(len(chunk) for chunk in all_tokens)
print("Total raw tokens in dataset:", total_tokens_count)
print("Total sentences (chunks):", len(all_tokens))

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
MODEL_NAME = "cahya/bert-base-indonesian-NER"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=128
    )

    all_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        previous = None
        label_ids = []
        for w in word_ids:
            if w is None:
                label_ids.append(-100)
            elif w != previous:
                label_ids.append(labels[w])
            else:
                label_ids.append(-100)
            previous = w
        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    return tokenized

tokenized = dataset.map(tokenize, batched=True, remove_columns=["tokens", "ner_tags"])

if "token_type_ids" in tokenized.column_names:
    tokenized = tokenized.remove_columns(["token_type_ids"])

print(tokenized["train"].column_names)

o_id = label2id.get("O", None)

def has_entity(example):
    return any(l != -100 and l != o_id for l in example["labels"])

if o_id is not None:
    tokenized["train"] = tokenized["train"].filter(has_entity)

num_labels = len(label_list)
model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

model.config.use_cache = False

def compute_class_weights(dataset, num_labels):
    counts = np.zeros(num_labels, dtype=np.int64)

    for labels in dataset["labels"]:
        for lab in labels:
            if lab != -100:
                counts[lab] += 1

    counts = np.where(counts == 0, 1, counts)

    weights = 1.0 / counts
    weights = weights * (2.7 / np.max(weights))

    return torch.tensor(weights, dtype=torch.float)

class_weights = compute_class_weights(tokenized["train"], num_labels)
print("Class weights:", class_weights)

class WeightedNERTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = outputs.logits
        
        loss_fct = nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device),
            ignore_index=-100
        )
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,
    num_train_epochs=4,
    weight_decay=0.01,
    logging_steps=500,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_checkpointing=False,
    warmup_ratio=0.1,
    fp16=torch.cuda.is_available()
)

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    labels = p.label_ids

    true_preds = []
    true_labels = []

    for pred_seq, label_seq in zip(preds, labels):
        seq_preds = []
        seq_labels = []
        for pred_id, lab_id in zip(pred_seq, label_seq):
            if lab_id != -100:
                seq_preds.append(id2label[pred_id])
                seq_labels.append(id2label[lab_id])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    precision = precision_score(true_labels, true_preds)
    recall = recall_score(true_labels, true_preds)
    f1 = f1_score(true_labels, true_preds)

    return {"precision": precision, "recall": recall, "f1": f1}

optimizer = AdamW(model.parameters(), lr=5e-5)
trainer = WeightedNERTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    optimizers=(optimizer, None)
)

# Train eval
train_result = trainer.train()
trainer.save_model(OUTPUT_DIR)

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
        flat_true.extend(seq_true)
        flat_pred.extend(seq_pred)

    return flat_true, flat_pred

def plot_confusion_matrix(flat_true, flat_pred, id2label, save_path):
    labels_list = list(id2label.values())
    cm = confusion_matrix(flat_true, flat_pred, labels=labels_list)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=labels_list, yticklabels=labels_list)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("NER Confusion Matrix")
    plt.savefig(save_path)
    plt.show()

def evaluate_testset(trainer, tokenized_dataset, id2label):
    test_results = trainer.predict(tokenized_dataset["test"])
    predictions, labels, _ = test_results
    preds = np.argmax(predictions, axis=-1)

    true_labels, true_preds = convert_predictions(preds, labels, id2label)

    print("\n=== Classification Report ===\n")
    print(classification_report(true_labels, true_preds))

    flat_true, flat_pred = flatten_sequences(true_labels, true_preds)

    print("\n=== Confusion Matrix ===\n")
    plot_confusion_matrix(flat_true, flat_pred, id2label, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    return true_labels, true_preds

if __name__ == "__main__":
    val_results = trainer.evaluate()
    with open(os.path.join(OUTPUT_DIR, "validation_metrics.json"), "w") as f:
        json.dump(val_results, f, indent=4)
    print("Validation metrics:", val_results)

    test_results = trainer.predict(tokenized["test"])
    preds = np.argmax(test_results.predictions, axis=-1)

    true_labels, true_preds = convert_predictions(preds, test_results.label_ids, id2label)
    flat_true, flat_pred = flatten_sequences(true_labels, true_preds)

    print("\n=== Classification Report ===\n")
    print(classification_report(true_labels, true_preds))

    plot_confusion_matrix(flat_true, flat_pred, id2label, save_path=os.path.join(OUTPUT_DIR, "confusion_matrix.png"))

    with open(os.path.join(OUTPUT_DIR, "test_metrics.json"), "w") as f:
        json.dump(test_results.metrics, f, indent=4)
    print("Test metrics:", test_results.metrics)