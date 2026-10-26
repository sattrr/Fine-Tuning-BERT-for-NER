import os
import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

MAX_LENGTH = 128

def get_setting(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

MODEL_ID = get_setting("MODEL_ID", "model/")
HF_TOKEN = get_setting("HF_TOKEN")


@st.cache_resource(show_spinner="Loading model, please wait.")
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()
id2label = model.config.id2label

def predict_ner(text):
    n_tokens = len(tokenizer(text, add_special_tokens=True)["input_ids"])
    truncated = n_tokens > MAX_LENGTH

    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        return_offsets_mapping=True,
    )
    offsets = enc.pop("offset_mapping")[0].tolist()

    with torch.inference_mode():
        logits = model(**enc).logits

    preds = logits.argmax(dim=-1)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])

    results = []
    for token, pred, (start, end) in zip(tokens, preds, offsets):
        if start == end:
            continue
        results.append(
            {"token": token, "label": id2label[pred], "start": start, "end": end}
        )
    return results, truncated

def merge_entities(text, results):
    entities = []
    cur = None

    def close():
        nonlocal cur
        if cur:
            entities.append(
                {
                    "Entitas": text[cur["start"] : cur["end"]],
                    "Label": cur["label"],
                    "IOB Seq": " ".join(cur["iob"]),
                }
            )
            cur = None

    for r in results:
        label, token = r["label"], r["token"]

        if token.startswith("##") and cur:
            cur["end"] = r["end"]
            if label != "O":
                cur["iob"].append(label)
        elif label.startswith("B-"):
            close()
            cur = {"start": r["start"], "end": r["end"], "label": label[2:], "iob": [label]}
        elif label.startswith("I-"):
            if cur and cur["label"] == label[2:]:
                cur["end"] = r["end"]
                cur["iob"].append(label)
            else:
                close()
                cur = {"start": r["start"], "end": r["end"], "label": label[2:], "iob": [label]}
        else:
            close()

    close()
    return entities

st.title("Named Entity Recognition with BERT")

CONTOH_ENTITAS = [
    {"Entity example": "An. A", "Label": "NAME"},
    {"Entity example": "20-12-2025", "Label": "DATE"},
    {"Entity example": "Jl. Merdeka Barat No. 17", "Label": "ADDR"},
    {"Entity example": "11581687", "Label": "ID"},
    {"Entity example": "Islam", "Label": "PII"},
    {"Entity example": "Tuberkulosis", "Label": "COND"},
    {"Entity example": "Vitamin C", "Label": "NUTR"},
    {"Entity example": "Tempe", "Label": "FOOD"},
]

with st.expander("Entity example and label list"):
    st.table(CONTOH_ENTITAS)

st.write("Input text for labelling.")
st.caption("ex: an. A adalah seorang pasien yang mengidap penyakit tuberkulosis")
st.warning("This model supports Indonesian text only")

text = st.text_area("Input Text", height=200)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please input text first.")
    else:
        results, truncated = predict_ner(text)
        entities = merge_entities(text, results)

        if truncated:
            st.warning(
                f"The text exceeds {MAX_LENGTH} tokens."
            )

        st.subheader("Detected Entities")
        if entities:
            st.dataframe(entities, use_container_width=True, hide_index=True)
        else:
            st.info("No entity detected.")

        with st.expander("See labels per token"):
            st.dataframe(
                [{"Token": r["token"], "Label": r["label"]} for r in results],
                use_container_width=True,
                hide_index=True,
            )