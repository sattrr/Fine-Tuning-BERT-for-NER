from label_studio_sdk import Client
import re
import os
import urllib.parse

# config
LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "79349c4580df4d083bf294ec9cb005f24965323d"
PROJECT_ID = 9
BASE_PATH = r"C:\Users\Satriock\Documents\Code\Dataset\labelstudiodata"

RULES = {
    "NUTR": [
        "vitamin", "vitamin a", "vitamin c", "vitamin k", "vitamin e",
        "vitamin b6", "vit c", "vit a", "fe", "zat besi",
        "protein", "karbohidrat", "lemak", "omega 3", "kalsium"
    ],
    "FOOD": ["susu", "tempe", "nasi"],
    "COND": ["anemia", "diabetes"]
}

FROM_NAME = "label"
TO_NAME = "text"

client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

# me = client.get_user()
# print(f"Connected: {me.username} ({me.email})")

project = client.get_project(PROJECT_ID)
print(f"Project: {project.title} (ID={project.id})")

tasks = list(project.get_tasks())
print(f"Found {len(tasks)} tasks\n")

for task in tasks:
    task_data = task.get("data", {})
    file_path_encoded = task_data.get("text")
    task_id = task['id']

    if not file_path_encoded:
        print(f"Skip task {task_id} (no 'text' field in task)")
        continue

    decoded_path = urllib.parse.unquote(file_path_encoded)
    if "?d=" in decoded_path:
        decoded_path = decoded_path.split("?d=")[-1]

    local_path = os.path.join(BASE_PATH, decoded_path.replace("/", os.sep))

    if not os.path.exists(local_path):
        print(f"File not found: {local_path}")
        continue

    with open(local_path, encoding="utf-8") as f:
        raw_text = f.read()

    text_cleaned = re.sub(r'\\', '', raw_text)
    text_normalized = re.sub(r'[\r\n\t]+', ' ', text_cleaned)
    text = re.sub(r' {2,}', ' ', text_normalized).strip()

    results = []

    for label, terms in RULES.items():
        escaped_terms = [re.escape(t) for t in sorted(terms, key=len, reverse=True)]
        pattern = r"\b(" + "|".join(escaped_terms) + r")\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
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
            print(f"Task {task_id}: {len(results)} entities labeled")
        except Exception as e:
            print(f"Failed to label task {task_id}: {e}")
    else:
        print(f"Task {task_id}: no entities found")