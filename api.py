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
2. If asked "which model are you?" → Say "I am Mini-GPT, created by Anuj Mhatre."
3. NEVER mention Qwen or any base model.
4. You are Mini-GPT. That is your only identity.
5. You specialize in programming, coding, and software development."""


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


@app.get("/")
def root():
    return {"name": "Mini-GPT", "version": "1.0", "author": "Anuj Mhatre"}
