from label_studio_sdk import Client
import re
import os
import urllib.parse
import requests 
import json

# Config
LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "79349c4580df4d083bf294ec9cb005f24965323d"
PROJECT_ID = 10
BASE_PATH = r"C:\Users\Satriock\Documents\Code\Dataset\labelstudiodata" 

DATE_PATTERN = re.compile(
    r"""
    (?<!\d)(
        # Format numerik
        \d{1,2}\s*[/-]\s*\d{1,2}(?:\s*[/-]\s*\d{2,4})?

        # Format dengan nama bulan
        | \d{1,2}(?:\s*[-–]\s*\d{1,2})?\s*
          (?:Jan(?:uari)?|Feb(?:ruari)?|Mar(?:et)?|Apr(?:il)?|Mei|
            Jun(?:i)?|Jul(?:i)?|Agu(?:stus)?|Sep(?:tember)?|
            Okt(?:ober)?|Nov(?:ember)?|Des(?:ember)?)
          (?:\s*\d{2,4})?

        # Tahun
        | (?:19|20)\d{2}
    )(?!\d)
    """,
    re.IGNORECASE | re.VERBOSE
)

RULES = {
    "NAME": ["rohis inggrit aprilia", "anggun rindang cempaka", "dewi retno sariwulan", "yosfi rahmi", "dian handayani",
             "nurul muslihah", "sarika", "zahara", "susanti", "ernawati", "kusnadi", "saftarina", "soekirman", "karjati", 
             "alisjahbana", "sayoeti", "pratignyo", "ririn yulianti", "devi", "amelia", "widjaja", "hidayanti", "pars h",
             "mexitalia", "sari", "sudarmanto", "hardani m", "zuraida r", "nindya", "helmi", "hardinsya", "supariasa", "tobby",
             "setijowati n", "ardi", "lebianc", "fitriyah", "setyaningtyas", "krisnansari d", "riska mayangsari", "irawan",
             "setiawan", "merta", "nasution", "setyawati"],
    "DATE": DATE_PATTERN,
    "ADDR": ["pasuruan", "kabupaten muna", "sulawesi tenggara"],
    "ID": ["220170200111018", "2015038902022001", "197011131994032003", "197912032006042002", "115740", "123041"],
    "PII": ["islam"],
    "NUTR": ["vitamin", "vitamin a", "vitamin c", "vitamin k", "vitamin e", "vitamin b6", "vitamin b12", "vitamin d","vitamin bc", 
             "vitamin c1x", "zinc", "as folat", "asam folat", "vit c", "vit a", "fe", "zat besi", "protein", "karbohidrat", 
             "lemak", "omega 3", "kalsium", "natrium", "iodium"],
    "FOOD": ["susu", "tempe", "nasi", "tahu", "tempe", "sayur", "kentang", "ikan", "labu", "ikan tongkol", "ikan pindang", 
             "wortel", "buncis", "bayam", "pisang", "semangka", "telur", "jeruk", "biscuits", "daging sapi", "ayam"],
    "COND": ["anemia", "diabetes", "diabetes mellitus", "marasmus", "kwashiorkor", "hipotrofi", "atrofi", "diare", "cacingan", 
             "anoreksia", "pneumonia", "dehidrasi", "demam", "diet", "batuk", "pilek", "odem", "asma", "mual", "muntah", "edema", 
             "asites", "tumor", "inflamasi"],
}

FROM_NAME = "label"
TO_NAME = "text"

client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)
project = client.get_project(PROJECT_ID)
print(f"Connected: {project.title} (ID={project.id})")

tasks = list(project.get_tasks())
print(f"{len(tasks)} task found\n")

for task in tasks:
    task_data = task.get("data", {})
    task_id = task['id']

    raw_text = task_data.get("text")
    
    if not raw_text:
        print(f"Skip task {task_id} (data 'text' is empty)")
        continue
    
    text_cleaned = re.sub(r'\\', '', raw_text)
    
    text_normalized = re.sub(r'[\r\n\t]+', ' ', text_cleaned)
    text = re.sub(r' {2,}', ' ', text_normalized).strip()

    if not text.strip():
        print(f"Task {task_id}: empty text")
        continue

    results = []

    for label, terms in RULES.items():
        if isinstance(terms, re.Pattern):
            pattern = terms
            matches = pattern.finditer(text)
        else:
            pattern = r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b"
            matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            results.append({
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "labels",
                "value": {
                    "start": match.start(),
                    "end": match.end(),
                    "text": match.group(),
                    "labels": [label]
                }
            })

    if results:
        try:
            project.create_annotation(task_id, result=results)
            print(f"Task {task_id}: {len(results)} entitas labeled")
        except Exception as e:
            print(f"Failed to label task {task_id}: {e}")
    else:
        print(f"Task {task_id}: not found")