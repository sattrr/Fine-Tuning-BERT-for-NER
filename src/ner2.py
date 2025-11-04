import os, json, re
from tqdm import tqdm

INPUT_FOLDER = "../data/cleaned/json/"
OUTPUT_FILE = "../data/cleaned/text_tagged.json"
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

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

def auto_tag(text):
    entities = []
    for label, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "label": label
            })
    return entities

def main():
    all_tagged = []

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".json")]
    if not files:
        print("JSON file not found.")
        return

    for file in tqdm(files, desc="JSON tagging in process"):
        path = os.path.join(INPUT_FOLDER, file)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                for page in data:
                    content = page.get("content", "").strip()
                    if not content:
                        continue
                    tagged = {
                        "file": file,
                        "page": page.get("page", None),
                        "text": content,
                        "entities": auto_tag(content)
                    }
                    all_tagged.append(tagged)

            # Kalau JSON-nya dict tunggal
            elif isinstance(data, dict):
                text = data.get("text", "") or data.get("content", "")
                if text.strip():
                    all_tagged.append({
                        "file": file,
                        "page": data.get("page", None),
                        "text": text,
                        "entities": auto_tag(text)
                    })

        except Exception as e:
            print(f"Process failed {file}: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_tagged, f, ensure_ascii=False, indent=2)

    print(f"\nTagging saved to: {OUTPUT_FILE}")
    print(f"Total data: {len(all_tagged)}")

if __name__ == "__main__":
    main()