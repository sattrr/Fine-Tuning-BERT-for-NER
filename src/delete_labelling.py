from label_studio_sdk import Client
import urllib.parse

# config
LABEL_STUDIO_URL = "http://localhost:8080"
LABEL_STUDIO_API_KEY = "79349c4580df4d083bf294ec9cb005f24965323d"
PROJECT_ID = 10

client = Client(url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

project = client.get_project(PROJECT_ID)
print(f"Project: {project.title} (ID={project.id})")

tasks = list(project.get_tasks())
print(f"Found {len(tasks)} tasks\n")

for task in tasks:
    annotation_ids = [anno['id'] for anno in task.get('annotations', [])]

    if annotation_ids:
        for anno_id in annotation_ids:
            project.delete_annotation(anno_id)
        print(f"Task {task['id']}: {len(annotation_ids)} annotations deleted")
    else:
        print(f"Task {task['id']}: no annotations to delete")

print(f"\nAnnotations deleted from project {PROJECT_ID}.")