"""Download Qwen2.5-Coder-7B tokenizer + config for Mini-GPT project.
Full model will be loaded in Google Colab (free GPU).
"""

import os
from transformers import AutoTokenizer, AutoConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models", "qwen2.5-coder-7b")

print(f"Downloading {MODEL_ID} tokenizer + config...")
print("(Full model weights will be loaded in Google Colab)")

os.makedirs(LOCAL_DIR, exist_ok=True)

# Download tokenizer
print("\n[1/2] Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer.save_pretrained(LOCAL_DIR)
print(f"  Saved to {LOCAL_DIR}")

# Download config
print("\n[2/2] Downloading config...")
config = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
config.save_pretrained(LOCAL_DIR)
print(f"  Saved to {LOCAL_DIR}")

print(f"\nDone! Files saved to: {LOCAL_DIR}")
print(f"Files: {os.listdir(LOCAL_DIR)}")
print("\nNext step: Upload to Google Colab for fine-tuning")
print("The Colab notebook will load the full model from Hugging Face")
