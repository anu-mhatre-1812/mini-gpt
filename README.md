# Mini GPT

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Author: Anuj Mhatre](https://img.shields.io/badge/Author-Anuj%20Mhatre-purple)

## Why We Built This

Understanding transformers is easier when you build one from scratch. This project implements a minimal GPT-style decoder-only transformer — no HuggingFace `transformers` library, just PyTorch — trained on multilingual text data including Marathi, Hindi, and English.

## What It Does

A **mini GPT implementation** from scratch using PyTorch — a decoder-only transformer with causal self-attention, trained on multilingual text data. Supports training, text generation, and a FastAPI inference endpoint.

- Decoder-only transformer built from scratch (no transformers library)
- Causal self-attention with masking
- Character-level tokenizer
- Training on custom text data (Marathi, Hindi, English, Code, Math)
- Text generation with temperature and top-k sampling
- FastAPI inference API
- Checkpoint saving/loading

## How It Was Built

| Component | Tech |
|-----------|------|
| Model | PyTorch (custom GPT implementation) |
| Tokenizer | Character-level (custom) |
| Training | AdamW + CosineAnnealing |
| API | FastAPI |
| Data | Marathi, Hindi, English, Code, Math corpora |

**Architecture**: `mini_gpt/model.py` implements the full GPT architecture: `CausalSelfAttention` → `MLP` → `Block` → `GPT`. Character-level tokenizer in `mini_gpt/tokenizer.py`. Training in `train.py`, generation in `generate.py`.

## Quick Start

```bash
git clone https://github.com/a18-n03/mini-gpt.git
cd mini-gpt
pip install -r requirements.txt

# Train the model
python train.py --data data/marathi.txt --steps 3000

# Generate text
python generate.py --prompt "मराठी" --tokens 400

# Run the API
uvicorn api:app --reload
```

## Project Structure

```
mini-gpt/
├── mini_gpt/
│   ├── model.py           # GPT transformer (from scratch)
│   └── tokenizer.py       # Character-level tokenizer
├── train.py               # Training loop
├── generate.py            # Text generation
├── api.py                 # FastAPI inference endpoint
├── data/                  # Training corpora (Marathi, Hindi, English, etc.)
├── checkpoints/           # Saved model checkpoints
├── requirements.txt       # Python dependencies
└── docs/                  # Architecture and deployment docs
```

## Test Results

```
No test suite — this is a training/inference project.
Tests require GPU training runs.
```
