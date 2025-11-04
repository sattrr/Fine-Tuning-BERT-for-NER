import os
import fitz
import csv
import re
from tqdm import tqdm

RAW_PATH = "../data/raw/"
OUTPUT_PATH = "../data/cleaned/csv/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_PATH, "text_data.csv")

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                blocks = page.get_text("blocks")
                for block in blocks:
                    text += block[4].strip() + "\n"
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")

    if not text.strip():
        try:
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text") + "\n"
        except:
            pass

    return text.strip()

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

if __name__ == "__main__":
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "sentence"])

        for file in tqdm(os.listdir(RAW_PATH), desc="Convert PDF to CSV"):
            if not file.lower().endswith('.pdf'):
                continue

            pdf_path = os.path.join(RAW_PATH, file)
            text = extract_text_from_pdf(pdf_path)
            sentences = split_into_sentences(text)

            for sentence in sentences:
                writer.writerow([file, sentence])

    print(f"PDF converted into CSV file: {OUTPUT_FILE}")
