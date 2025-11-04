import re
import os
import pandas as pd
import json
from tqdm import tqdm

INPUT_PATH = "../data/cleaned/csv/text_data.csv"
OUTPUT_PATH = "../data/cleaned/csv/text_data_tagged.csv"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

PATTERNS = {
    "NAME": r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,3})\b",
    "DATE": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\b\d{4}\b)\b",
    "ADDR": r"\b(Jl\.?|Jalan|Gang|RT|RW|No\.?|Nomor)\s[\w\s\.,-]*",
    "ID": r"\b\d{6,}\b",
    "PII": r"\b(\+62|08\d{8,})\b",
    "NUTR": r"\b(vitamin\s?[A-Z]?|zat besi|protein|karbohidrat|lemak|kalsium|natrium|kalium|serat|vitamin C|vitamin D)\b",
    "FOOD": r"\b(nasi|roti|ikan|sayur|buah|ayam|telur|susu|tempe|tahu|daging)\b",
    "COND": r"\b(anemia|obesitas|malnutrisi|gizi buruk|stunting|hipertensi|diabetes)\b"
}

def auto_tag(sentence):
    entities = []
    for label, pattern in PATTERNS.items():
        for match in re.finditer(pattern, sentence, flags=re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "label": label
            })
    return json.dumps(entities, ensure_ascii=False)

def main():
    if not os.path.exists(INPUT_PATH):
        print(f"File not found: {INPUT_PATH}")
        return

    df = pd.read_csv(INPUT_PATH)

    if "sentence" not in df.columns:
        print("'sentence' not found.")
        return

    tqdm.pandas(desc="Auto-tagging NER")
    df["entities"] = df["sentence"].progress_apply(auto_tag)

    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"NER saved at: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()