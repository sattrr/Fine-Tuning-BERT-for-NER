import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ID = "sattr/clinical-ner-modelling"
MODEL_DIR = Path("../NER_gizi_klinis/model/checkpoint-1352")
TOKENIZER_DIR = Path("../NER_gizi_klinis/model")

TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
]

if not HF_TOKEN:
    raise SystemExit("HF_TOKEN tidak ditemukan. Isi di file .env: HF_TOKEN=hf_xxx")
 
for name in ("config.json", "model.safetensors"):
    if not (MODEL_DIR / name).exists():
        raise SystemExit(f"{name} tidak ada di {MODEL_DIR.resolve()}")
 
api = HfApi(token=HF_TOKEN)
api.create_repo(REPO_ID, private=True, exist_ok=True)
 
api.upload_folder(
    folder_path=str(MODEL_DIR),
    repo_id=REPO_ID,
    ignore_patterns=[
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "training_args.bin",
        "trainer_state.json",
        "checkpoint-*",
    ],
)