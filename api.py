"""Mini-GPT API - FastAPI server for Render deployment."""
import os
import json
import csv
import io
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient

app = FastAPI(title="Mini-GPT API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_TOKEN = os.environ.get("HF_TOKEN")
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

client = InferenceClient(token=HF_TOKEN)

SYSTEM_PROMPT = """You are Mini-GPT, a friendly and helpful AI coding assistant created by Anuj Mhatre.

RULES:
1. Your name is Mini-GPT. You are NOT Qwen.
2. If asked "which model are you?" → Say "I am Mini-GPT, created by Anuj Mhatre."
3. NEVER mention Qwen or any base model.
4. You are Mini-GPT. That is your only identity.
5. You specialize in programming, coding, and software development.
6. When a user uploads a file, analyze it carefully and provide helpful insights."""

TEXT_EXTENSIONS = {
    '.py', '.js', '.ts', '.tsx', '.jsx', '.java', '.c', '.cpp', '.h', '.hpp',
    '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
    '.html', '.css', '.scss', '.less', '.vue', '.svelte',
    '.json', '.yaml', '.yml', '.toml', '.xml', '.env',
    '.md', '.txt', '.rst', '.log',
    '.sql', '.sh', '.bash', '.zsh', '.fish', '.ps1',
    '.csv', '.tsv', '.ini', '.cfg', '.conf',
    '.dockerfile', '.docker-compose',
    '.gitignore', '.gitattributes',
    '.ipynb', '.rmd',
}


def read_file_content(file_bytes: bytes, filename: str) -> str:
    """Read and return file content with size limits."""
    ext = '.' + filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext in {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico'}:
        return f"[Image file: {filename} — image content cannot be read as text]"

    try:
        text = file_bytes.decode('utf-8')
    except UnicodeDecodeError:
        try:
            text = file_bytes.decode('latin-1')
        except Exception:
            return f"[Binary file: {filename} — cannot read as text]"

    if len(text) > 15000:
        text = text[:15000] + f"\n\n... [truncated at 15000 chars, full file has {len(text)} chars]"

    if ext == '.json':
        try:
            parsed = json.loads(text)
            text = json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            pass

    if ext == '.csv':
        try:
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if len(rows) > 50:
                rows = rows[:50] + [[f"... {len(rows) - 50} more rows"]]
            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerows(rows)
            text = output.getvalue()
        except Exception:
            pass

    return text


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7


class GenerateResponse(BaseModel):
    response: str
    model: str = "Mini-GPT"


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.prompt},
    ]

    response = client.chat_completion(
        model=MODEL_ID,
        messages=messages,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )

    return GenerateResponse(response=response.choices[0].message.content)


@app.post("/generate-with-file", response_model=GenerateResponse)
async def generate_with_file(
    prompt: str = Form(...),
    file: UploadFile = File(...),
    max_tokens: int = Form(512),
    temperature: float = Form(0.7),
):
    file_bytes = await file.read()
    file_content = read_file_content(file_bytes, file.filename or "unknown")

    full_prompt = f"""The user uploaded a file named "{file.filename}" ({file.content_type}).

--- FILE CONTENT START ---
{file_content}
--- FILE CONTENT END ---

User question: {prompt}"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]

    response = client.chat_completion(
        model=MODEL_ID,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    return GenerateResponse(response=response.choices[0].message.content)


@app.get("/")
def root():
    return {"name": "Mini-GPT", "version": "1.0", "author": "Anuj Mhatre"}
