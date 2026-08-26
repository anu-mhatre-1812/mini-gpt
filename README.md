# Mini-GPT 🧠 — GPT from Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-FF3EA5?logo=python&logoColor=white)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-FF3EA5?logo=pytorch&logoColor=white)]()

A **GPT-style decoder-only transformer written from scratch in pure PyTorch** — no
`transformers`, no `nanoGPT` copy-paste. Every component built line by line:
causal multi-head self-attention, MLP blocks, LayerNorm residuals, char-level tokenizer.

Trained on **Marathi Wikipedia** — and it learned to write Devanagari.

## Sample output (real, after 3000 steps on CPU)

```
मराठी अधिक अपेले त्याने प्रतिषांचा त्याताच्या क्त्ररिकेटचे आहे तो किंवरी येळा
संग्रज्यात आहे. लोके पुरकर व क्षेत्रचा त्या करती असत यांसता परंगत अडक सते त्या
सहस्थमन्यात खून पन्रकिंवसा पार्वात होते. व्याची पासार १०० लाले आहेत. मुंबईती ली
प्रभाषेच्या असलेल्ह काही नव्हा आणि लोकपस्थापट्रे रंत्रपला आहे..
```

Real Devanagari script, real Marathi words (`आहे`, `आणि`, `मुंबईतील`, `लोक`),
sentence structure — learned from just 324K chars with an 871K-parameter model.
The gibberish words are the honest lesson: **fluency needs scale** — exactly how
GPT-4 works, just bigger.

## Architecture

```
char tokens → Embedding + Positional Embedding
  → 4 × [ LayerNorm → Causal Self-Attention (4 heads)
        → LayerNorm → MLP (GELU, 4x expand) + residuals ]
  → LayerNorm → Linear head → next-char distribution
```

| Config | Value |
|---|---|
| Parameters | 871,680 |
| Layers / Heads | 4 / 4 |
| Embedding dim | 128 |
| Context (block size) | 256 chars |
| Vocab | 178 (char-level) |
| Training | 3000 steps, AdamW + cosine LR, ~74 min on CPU |

## Run it

```bash
pip install torch

# 1. data (Marathi Wikipedia)
python data/download.py

# 2. train (~74 min CPU, ~4 min on T4 GPU)
python train.py --data data/marathi.txt --steps 3000

# 3. generate
python generate.py --prompt "मराठी" --tokens 400
```

## GPU version (Kaggle / Colab)

`kaggle_mini_gpt.ipynb` — 5-cell notebook, bigger model (6 layers, 384-dim, ~7M params),
fp16 mixed precision, 5000 steps on T4 in ~12-18 min. Upload to
[Colab](https://colab.research.google.com) or Kaggle, enable GPU, Run All.

## The full journey (why this repo exists)

```
1. USED a pretrained LLM      → draco-cli (API calls)
2. FINE-TUNED a 7B LLM        → draco-r1 (QLoRA)
3. BUILT one from scratch     → mini-gpt (this repo)
```

Understanding attention, positional embeddings, and the training loop at this level
is what separates API users from ML engineers.

## Files

| File | Description |
|---|---|
| `mini_gpt/model.py` | CausalSelfAttention, MLP, Block, GPT — from scratch |
| `mini_gpt/tokenizer.py` | Char-level encode/decode |
| `train.py` | Training loop: AdamW, cosine LR, grad clip, checkpointing |
| `generate.py` | Sampling: temperature + top-k |
| `data/download.py` | Marathi Wikipedia corpus fetcher |
| `kaggle_mini_gpt.ipynb` | GPU notebook (bigger config, fp16) |

## License

MIT

---

<div align="center">
<sub>Built by <a href="https://github.com/a18-n03">Anuj Mhatre</a> · BTech CSE (AI & ML)</sub>
</div>
