import re
from pathlib import Path

TXT_DIR = Path("../data/cleaned/txt/")
OUTPUT_DIR = Path("../data/cleaned/ground_truth/norm/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for file in TXT_DIR.glob("*.txt"):
    output_file = OUTPUT_DIR / file.name

    with file.open("r", encoding="utf-8") as f:
        text = f.read()

    text = text.replace("\xa0", " ")
    text = re.sub(r"[|#]", " ", text)
    text = re.sub(r"(?<=\b\w)(\s)(?=\w\s\w)", "", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"(?m)^[ \t]+|[ \t]+$", "", text)
    text = text.lower()
    text = text.strip()

    with output_file.open("w", encoding="utf-8") as f:
        f.write(text)

    print(f"File saved: {output_file}")