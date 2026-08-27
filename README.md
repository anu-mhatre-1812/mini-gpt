# Mini-GPT — GPT from Scratch

[![Python](https://img.shields.io/badge/Python-3.10%2B-FF3EA5?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-FF3EA5?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Stars](https://img.shields.io/github/stars/a18-n03/mini-gpt?style=flat&color=FFD700)](https://github.com/a18-n03/mini-gpt/stargazers)

A **GPT-style decoder-only transformer written from scratch in pure PyTorch** — no
`transformers`, no `nanoGPT` copy-paste. Every component built line by line:
causal multi-head self-attention, MLP blocks, LayerNorm residuals, char-level tokenizer.

Trained on **Marathi Wikipedia** — and it learned to write Devanagari.

## Quick Start

```bash
# Clone the repo
git clone https://github.com/a18-n03/mini-gpt.git
cd mini-gpt

# Install dependencies
pip install -r requirements.txt

# Download training data
python data/download.py

# Train (74 min CPU, ~4 min T4 GPU)
python train.py --data data/marathi.txt --steps 3000

# Generate text
python generate.py --prompt "मराठी" --tokens 400
```

## Multi-domain upgrade

v2 trained on **5 domains** (teacher-distilled data + Wikipedia):
Marathi + Hindi + English + Maths patterns + Code.

```
prompt: "मराठी"      → मराठी दिव जीन जार्ण महाराष्ट्रेच्या लागत...
prompt: "भारत"       → भारत, इ.स. साम्राज्य... मुंबईत... पुणे...
prompt: "2 + 2 ="    → 2 + 2 = ...  (format learned ✓)
prompt: "def add("   → def add(a, b): ...  (structure learned ✓)
```

**The data-volume lesson:** Marathi (850K chars) generates fluently; the other
domains (under 1K chars each) show the model *recognized* all languages but
fluency follows data volume — the same imbalance that shapes real LLMs.

## GPU rerun result (teacher data 8x + T4)

Same architecture, trained on T4 with teacher data repeated 8x — **massive jump:**

| Domain | CPU run (no teacher) | GPU run (teacher 8x) |
|---|---|---|
| Marathi | broken words | 🟢 fluent sentences with dates & places |
| Hindi | Marathi drift | 🟢 Hindi-Marathi mixed flow |
| English | letter soup | 🟢 **verbatim recall** of teacher text |
| Maths | format only | 🟢 **15/16 equations correct** |
| Code | gibberish | 🟢 **working recursion** (factorial) |

```
===== ENGLISH (actual output) =====
The sun in the west. Trees give us oxygen, fruits and shade.
Children go to school to learn reading and writing. Knowledge is
the greatest treasure a person can own. Practice makes a person
perfect. Hard work always pays off in the end.

===== CODE (actual output) =====
def add(a, b):
    return a + b

def factorial(n):
    if n <= 1:
       return 1
    return n * factorial(n - 1)
```

**This is synthetic-data distillation in action** — the teacher's (ox-alpha)
text became the student's knowledge. Same technique used to train the phi model family.

Build the corpus yourself: `python data/build_corpus.py`
(teacher data lives in `data/*.txt` — add more text per domain to improve it)

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

## How to Improve

- Add more training data per domain (especially Hindi/English)
- Train for more steps (5000+ on GPU)
- Use a larger model (6 layers, 384-dim)
- Fine-tune with teacher data at higher repetition (16x, 32x)
- Add more code examples to the corpus

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
