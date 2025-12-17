import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForTokenClassification

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("model/")
    model = AutoModelForTokenClassification.from_pretrained("model/")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# with open('label_map.json', 'r') as f:
#     label_map = json.load(f)

id2label = model.config.id2label

def predict_ner(text):
    inputs = tokenizer(
        text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128,
        is_split_into_words=False
    )

    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    results = []
    for token, pred in zip(tokens, predictions):
        if token not in ["[CLS]", "[SEP]"]:
            label = id2label[pred]
            results.append((token, label))
    
    return results

# def merge_entities(results):
#     entities = []
#     current_entity = ""
#     current_label = ""

#     for token, label in results:
#         if label.startswith("B-"):
#             if current_entity:
#                 entities.append((current_entity.strip(), current_label))
#             current_entity = token
#             current_label = label[2:]

#         elif label.startswith("I-") and current_label == label[2:]:
#             if token.startswith("##"):
#                 current_entity += token.replace("##", "")
#             else:
#                 current_entity += " " + token

#         else:
#             if current_entity:
#                 entities.append((current_entity.strip(), current_label))
#                 current_entity = ""
#                 current_label = ""

#     if current_entity:
#         entities.append((current_entity.strip(), current_label))

#     return entities

def merge_entities_iob(results):
    entities = []
    current_entity = ""
    current_label = ""
    iob_seq = []

    for token, label in results:
        if label.startswith("B-"):
            if current_entity:
                entities.append((current_entity.strip(), current_label, " ".join(iob_seq)))
            current_entity = token.replace("##", "")
            current_label = label[2:]
            iob_seq = [label]

        elif label.startswith("I-") and current_label == label[2:]:
            if token.startswith("##"):
                current_entity += token.replace("##", "")
            else:
                current_entity += " " + token
            iob_seq.append(label)

        else:
            if current_entity:
                entities.append((current_entity.strip(), current_label, " ".join(iob_seq)))
                current_entity = ""
                current_label = ""
                iob_seq = []

    if current_entity:
        entities.append((current_entity.strip(), current_label, " ".join(iob_seq)))

    return entities

st.title("Named Entity Recognition with BERT")
st.write("Enter text to recognize named entities.")

text = st.text_area("Input Text", height=200)

if st.button("Predict"):
    if text.strip():
        results = predict_ner(text)
        entities = merge_entities_iob(results)

        st.subheader("Detected Entities")

        if entities:
            for ent, label, iob in entities:
                st.markdown(f"""
                            **Entitas** : {ent}  
                            **Label** : `{label}`  
                            **IOB Seq** : `{iob}`
                            """)
        else:
            st.info("No named entities detected.")
    else:
        st.warning("Please enter some text.")

# if st.button("Predict"):
#     if text.strip():
#         results = predict_ner(text)
#         entities = merge_entities(results)

#         st.subheader("Detected Entities")

#         if entities:
#             for ent, label in entities:
#                 st.write(f"{ent} → {label}")
#         else:
#             st.info("No named entities detected.")
#     else:
#         st.warning("Please enter some text.")