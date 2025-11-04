from docling.utils import model_downloader
import os

target_dir = "C:/Users/Satriock/.cache/docling/models"

os.makedirs(target_dir, exist_ok=True)
model_downloader.download_models()

print("Model downloaded to: " + target_dir)
