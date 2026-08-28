"""Deploy Mini-GPT to Hugging Face Spaces."""
import os
from huggingface_hub import HfApi, login

HF_TOKEN = os.environ.get("HF_TOKEN")
SPACE_NAME = "mau-0305/mini-gpt"
LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

login(token=HF_TOKEN)
api = HfApi()

# Create Space
print("Creating HF Space...")
try:
    api.create_repo(
        repo_id=SPACE_NAME,
        repo_type="space",
        space_sdk="gradio",
        space_hardware="cpu-basic",
        exist_ok=True,
    )
    print(f"Space created: https://huggingface.co/spaces/{SPACE_NAME}")
except Exception as e:
    print(f"Space exists or error: {e}")

# Upload files
print("\nUploading files...")
files_to_upload = ["app.py", "requirements.txt", "README.md"]
for f in files_to_upload:
    filepath = os.path.join(LOCAL_DIR, f)
    if os.path.exists(filepath):
        print(f"  Uploading {f}...")
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=f,
            repo_id=SPACE_NAME,
            repo_type="space",
        )
        print(f"  {f} uploaded!")

print(f"\nDone! Space: https://huggingface.co/spaces/{SPACE_NAME}")
