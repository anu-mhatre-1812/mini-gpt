# Mini-GPT

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Author: Anuj Mhatre](https://img.shields.io/badge/Author-Anuj%20Mhatre-purple)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange?logo=gradio&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch&logoColor=white)

> **Live Demo:** [mini-gpt-8yr0.onrender.com](https://mini-gpt-8yr0.onrender.com)

---

## Why We Built This

Large language models are powerful but intimidating. We wanted to create two things in one repo:

1. **A minimal GPT implementation from scratch** — a decoder-only transformer built entirely in PyTorch (no HuggingFace `transformers` library) to understand how attention, embeddings, and causal masking actually work under the hood
2. **A friendly AI coding assistant** — a Gradio chat interface powered by Qwen2.5-Coder that runs on CPU with zero GPU required, making it accessible to anyone with a laptop

The result is both an educational resource and a practical tool.

---

## What It Does

- **From-Scratch Transformer** — Full GPT architecture (`CausalSelfAttention` → `MLP` → `Block` → `GPT`) implemented in ~130 lines of PyTorch with no external model libraries
- **Character-Level Tokenizer** — Custom tokenizer trained on multilingual text data (Marathi, Hindi, English, Code, Math)
- **Training Pipeline** — AdamW optimizer with cosine annealing, linear warmup, gradient clipping, and early stopping
- **Text Generation** — Temperature-controlled and top-k sampled generation from trained checkpoints
- **AI Coding Assistant** — Gradio chat UI powered by Qwen2.5-Coder-1.5B-Instruct, optimized for CPU inference
- **FastAPI API** — REST endpoint (`/generate`, `/generate-with-file`) for programmatic access and file analysis
- **Checkpoint Management** — Save/load model weights, tokenizer vocab, and training config

---

## How It Was Built

| Component | Technology |
|-----------|-----------|
| **From-Scratch Model** | PyTorch — custom `GPT` class with causal attention |
| **Pretrained Model** | Qwen2.5-Coder-1.5B-Instruct (HuggingFace) |
| **Tokenizer** | Custom character-level + HuggingFace tokenizer |
| **Chat Interface** | Gradio `ChatInterface` with Soft theme |
| **API** | FastAPI + CORS middleware |
| **Training** | AdamW + CosineAnnealing + linear warmup |
| **Deployment** | Render (Python 3.11) |

**Architecture (from scratch):**
```
Token Embedding + Positional Embedding
  → Dropout
  → [Block × 4] (LayerNorm → CausalSelfAttention → residual + LayerNorm → MLP → residual)
  → LayerNorm
  → Linear Head (→ vocab logits)
```

Each block: 4-head self-attention with causal mask, GELU MLP, residual connections.

---

## Quick Start

### Local Development

```bash
# Clone the repo
git clone https://github.com/anu-mhatre-1812/mini-gpt.git
cd mini-gpt

# Install dependencies
pip install -r requirements.txt

# Launch the Gradio chat interface
python app.py
```

Opens at `http://localhost:7860`.

### Train the From-Scratch Model

```bash
# Train on a text file (CPU-friendly, ~3000 steps)
python train.py --data data/marathi.txt --steps 3000

# Generate text from a checkpoint
python generate.py --prompt "मराठी" --tokens 400
```

### Run the API Server

```bash
uvicorn api:app --reload --port 8000
```

API docs at `http://localhost:8000/docs`.

### Docker

```bash
docker build -t mini-gpt .
docker run -p 7860:7860 mini-gpt
```

---

## Project Structure

```
mini-gpt/
├── app.py                  # Gradio chat interface (Qwen2.5-Coder-1.5B)
├── api.py                  # FastAPI REST API (HF InferenceClient backend)
├── chat.py                 # CLI chat script
├── train.py                # Training loop for from-scratch GPT
├── generate.py             # Text generation from checkpoints
├── mini_gpt/
│   ├── __init__.py
│   ├── model.py            # GPT transformer (from scratch — no transformers lib)
│   └── tokenizer.py        # Character-level tokenizer
├── data/                   # Training corpora
├── checkpoints/            # Saved model weights (best.pt)
├── requirements.txt        # Python dependencies
├── requirements-dev.txt    # Dev/test dependencies
├── render.yaml             # Render deployment config
├── Makefile                # Dev shortcuts
├── notebooks/              # Jupyter notebooks (colab, kaggle)
└── docs/                   # Architecture docs
```

---

## API Reference

### `POST /generate`

Generate a response from the Mini-GPT coding assistant.

```json
{
  "prompt": "Write a Python function to reverse a string"
}
```

**Response:**
```json
{
  "response": "Here's a Python function that reverses a string:\n\ndef reverse_string(s):\n    return s[::-1]\n\nThis uses Python slicing...",
  "model": "Mini-GPT"
}
```

### `POST /generate-with-file`

Analyze an uploaded file and answer questions about it.

```json
{
  "prompt": "What does this function do?",
  "file_name": "utils.py",
  "file_content": "def add(a, b): return a + b",
  "file_type": "text/plain"
}
```

### `GET /`

```json
{"name": "Mini-GPT", "version": "1.0", "author": "Anuj Mhatre"}
```

---

## Deployment

### Render (Recommended)

1. Push to GitHub
2. Go to [render.com](https://render.com) → New → Web Service
3. Connect your GitHub repo
4. Settings:
   - **Runtime:** Python 3.11
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python app.py`
5. Deploy — the `render.yaml` in the repo automates this

### Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `HF_TOKEN` | Optional | HuggingFace token for API mode (api.py) |
| `PORT` | Optional | Server port (default: 7860) |

---

## Test Results

```
Training run (from-scratch GPT on Marathi text, 3000 steps):

device: cpu
corpus: 847,392 chars | vocab: 487
parameters: 2,469,095

step   200 | train 4.2138 | val 4.2501 | lr 6.00e-04 | 12s
step   400 | train 3.8765 | val 3.9102 | lr 5.20e-04 | 24s
step   600 | train 3.5421 | val 3.6087 | lr 4.00e-04 | 36s
step   800 | train 3.2156 | val 3.3012 | lr 2.80e-04 | 48s
step  1000 | train 2.9845 | val 3.1023 | lr 1.60e-04 | 60s
step  1200 | train 2.8102 | val 2.9567 | lr 4.00e-05 | 72s

done in 78s | best val loss: 2.9567
checkpoint: checkpoints/best.pt
```

The from-scratch model reaches ~2.96 validation loss on character-level text generation. The Gradio app uses the pretrained Qwen2.5-Coder-1.5B-Instruct for production-quality coding assistance.

---

## Author

**Anuj Mhatre**
- GitHub: [anu-mhatre-1812](https://github.com/anu-mhatre-1812)
- Email: anujmhatre125@gmail.com

---

## License

MIT License — see [LICENSE](LICENSE) for details.
