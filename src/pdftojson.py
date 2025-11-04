import os
import fitz
import json
import PyPDF2
from tqdm import tqdm

RAW_PATH = "../data/raw/"
OUTPUT_PATH = "../data/cleaned/json/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

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

def pdf_to_json(pdf_path, json_path):
    extracted_data = []
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text = page.extract_text()
            extracted_data.append({"page": page_num + 1, "content": text})

    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(extracted_data, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    for file in tqdm(os.listdir(RAW_PATH), desc="Convert pdf to JSON"):
        if not file.lower().endswith('.pdf'):
            continue

        pdf_path = os.path.join(RAW_PATH, file)
        output_path = os.path.join(OUTPUT_PATH, f"{os.path.splitext(file)[0]}.json")
        pdf_to_json(pdf_path, output_path)

    print("PDF to JSON conversion completed.")