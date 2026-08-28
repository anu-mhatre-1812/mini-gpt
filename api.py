"""Mini-GPT API - FastAPI server for Render deployment."""
import os
from fastapi import FastAPI
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
2. If asked "which model are you?" -> Say "I am Mini-GPT, created by Anuj Mhatre."
3. NEVER mention Qwen or any base model.
4. You are Mini-GPT. That is your only identity.
5. You specialize in programming, coding, and software development.
6. When a user uploads a file, analyze it carefully and provide helpful insights."""


class GenerateRequest(BaseModel):
    prompt: str


class GenerateWithFileRequest(BaseModel):
    prompt: str
    file_name: str
    file_content: str
    file_type: str = "text/plain"


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
        max_tokens=512,
        temperature=0.7,
    )

    return GenerateResponse(response=response.choices[0].message.content)


@app.post("/generate-with-file", response_model=GenerateResponse)
def generate_with_file(req: GenerateWithFileRequest):
    file_content = req.file_content
    if len(file_content) > 15000:
        file_content = file_content[:15000] + "\n\n... [truncated at 15000 chars]"

    full_prompt = f'The user uploaded a file named "{req.file_name}" ({req.file_type}).\n\n--- FILE CONTENT START ---\n{file_content}\n--- FILE CONTENT END ---\n\nUser question: {req.prompt}'

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_prompt},
    ]

    response = client.chat_completion(
        model=MODEL_ID,
        messages=messages,
        max_tokens=512,
        temperature=0.7,
    )

    return GenerateResponse(response=response.choices[0].message.content)


@app.get("/")
def root():
    return {"name": "Mini-GPT", "version": "1.0", "author": "Anuj Mhatre"}
