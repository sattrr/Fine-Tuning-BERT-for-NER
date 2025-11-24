from sklearn.model_selection import train_test_split

def read_conll_docs(path):
    docs = []
    tokens = []
    labels = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    docs.append({"tokens": tokens, "ner_tags": labels})
                    tokens = []
                    labels = []
            else:
                parts = line.split()
                tok = parts[0]
                lab = parts[-1]
                tokens.append(tok)
                labels.append(lab)

    if tokens:
        docs.append({"tokens": tokens, "ner_tags": labels})

    return docs

data = read_conll_docs("../data/gizi-klinis.clean.conll")

def get_patient_id(doc):
    for tok, lab in zip(doc["tokens"], doc["ner_tags"]):
        if lab == "B-ID":
            return tok
    return f"NOID-{id(doc)}" 

grouped = {}
for doc in data:
    pid = get_patient_id(doc)
    if pid not in grouped:
        grouped[pid] = []
    grouped[pid].append(doc)

pids = list(grouped.keys())
print("Jumlah pasien unik:", len(pids))

train_id, temp_id = train_test_split(pids, test_size=0.3, random_state=42)
val_id, test_id = train_test_split(temp_id, test_size=0.5, random_state=42)

train = [s for pid in train_id for s in grouped[pid]]
val = [s for pid in val_id for s in grouped[pid]]
test = [s for pid in test_id for s in grouped[pid]]

print(len(train), len(val), len(test))

def write_conll(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for doc in data:
            for tok, lbl in zip(doc["tokens"], doc["ner_tags"]):
                f.write(f"{tok} _ _ {lbl}\n")
            f.write("\n")

write_conll(train, "../data/train.conll")
write_conll(val, "../data/val.conll")
write_conll(test, "../data/test.conll")