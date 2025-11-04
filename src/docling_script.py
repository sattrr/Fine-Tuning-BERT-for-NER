import os
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
import gc

RAW_DIR = Path("../data/raw/")
OUTPUT_DIR = Path("../data/cleaned/txt/")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

artifacts_path = str(Path.home() / ".cache" / "docling" / "models")

pipeline_options = PdfPipelineOptions(
    artifacts_path=artifacts_path,
    always_use_ocr=True
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

pdf_files = [f for f in RAW_DIR.glob("*.pdf")]

if not pdf_files:
    print(f"No files found in {RAW_DIR}")
else:
    for pdf_path in pdf_files:
        output_file = OUTPUT_DIR / f"{pdf_path.stem}.txt"

        if output_file.exists():
            print(f"Already processed: {pdf_path.name}")
            continue

        try:
            print(f"Processing: {pdf_path.name}")
            doc = doc_converter.convert(str(pdf_path))
            text = doc.document.export_to_text()

            output_file.write_text(text, encoding="utf-8")
            print(f"Saved: {output_file}")

        except Exception as e:
            print(f"Failed to process {pdf_path.name}: {e}")
        finally:
            gc.collect()

print("\nConversion completed")
