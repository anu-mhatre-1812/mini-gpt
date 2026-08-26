"""Build kaggle_mini_gpt.ipynb from cell sources."""

import json

CELLS = [
    ("markdown", "# Mini-GPT on Kaggle GPU 🚀\n\nMarathi Wikipedia char-level GPT, trained from scratch.\n\n"
     "**Before running:** Settings (right panel) → Accelerator → **GPU T4 x2** → Internet **ON**\n\n"
     "Then: Run All"),

    ("code", """import urllib.parse
import urllib.request
import json
import time

print("downloading Marathi data...")
titles = [
    "महात्मा_गांधी", "मराठी_भाषा", "महाराष्ट्र", "पुणे", "मुंबई",
    "संगणक", "क्रिकेट", "शिवाजी_महाराज", "भारत", "विज्ञान",
    "गणित", "संगीत", "आरोग्य", "शिक्षण", "इतिहास",
    "तंत्रज्ञान", "प्राणी", "वृक्ष", "आकाशगंगा", "सूर्य",
    "नदी", "पर्वत", "खेळ", "चित्रपट", "साहित्य", "कला", "रसोई",
]
out = []
for t in titles:
    try:
        url = ("https://mr.wikipedia.org/w/api.php?action=query&prop=extracts"
               f"&explaintext=1&format=json&titles={urllib.parse.quote(t)}")
        req = urllib.request.Request(url, headers={"User-Agent": "MiniGPT/0.1"})
        data = json.loads(urllib.request.urlopen(req, timeout=20).read())
        for page in data["query"]["pages"].values():
            ext = page.get("extract", "")
            if len(ext) > 500:
                out.append(ext)
    except Exception as e:
        print("  skip", t, str(e)[:40])
    time.sleep(1.5)

text = "\\n\\n".join(out)
print(f"corpus: {len(text):,} chars from {len(out)} articles")"""),

    ("code", """import torch
import torch.nn as nn
import torch.nn.functional as F
import math

chars = sorted(set(text))
vocab_size = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]

N_LAYER, N_HEAD, N_EMBD, BLOCK, DROPOUT, BATCH = 6, 6, 384, 512, 0.1, 64


class CausalSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.attn_drop = DROPOUT
        self.resid_drop = nn.Dropout(DROPOUT)
        mask = torch.tril(torch.ones(BLOCK, BLOCK)).view(1, 1, BLOCK, BLOCK)
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(N_EMBD, dim=2)
        q = q.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        k = k.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        v = v.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = F.dropout(att, self.attn_drop, self.training)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(N_EMBD, 4 * N_EMBD)
        self.proj = nn.Linear(4 * N_EMBD, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = CausalSelfAttention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = MLP()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=40):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCK:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, 1)), dim=1)
        return idx


device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else "")
model = GPT().to(device)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")"""),

    ("code", """opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5000)
scaler = torch.amp.GradScaler("cuda")


def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK - 1, (BATCH,))
    x = torch.stack([d[i:i + BLOCK].clone() for i in ix]).to(device)
    y = torch.stack([d[i + 1:i + BLOCK + 1].clone() for i in ix]).to(device)
    return x, y


STEPS = 5000
t0 = time.time()
for step in range(1, STEPS + 1):
    model.train()
    x, y = get_batch("train")
    with torch.amp.autocast("cuda", dtype=torch.float16):
        _, loss = model(x, y)
    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    sched.step()

    if step % 250 == 0 or step == STEPS:
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
            vx, vy = get_batch("val")
            _, vloss = model(vx, vy)
        print(f"step {step:5d} | train {loss.item():.4f} | val {vloss.item():.4f} "
              f"| {time.time() - t0:.0f}s", flush=True)
        model.train()

torch.save(model.state_dict(), "mini_gpt_marathi.pt")
print("checkpoint saved: mini_gpt_marathi.pt")"""),

    ("code", """model.eval()
start = "मराठी"
idx = torch.tensor([[stoi.get(c, 0) for c in start]], device=device)
out = model.generate(idx, 600, temperature=0.8, top_k=40)
print("===== SAMPLE OUTPUT =====")
print("".join(itos[i] for i in out[0].tolist()))"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "accelerator": "GPU",
    },
    "cells": [
        {
            "cell_type": ct,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if ct == "code" else {}),
        }
        for ct, src in CELLS
    ],
}

with open(r"C:\Users\ADMIN\Projects\mini-gpt\kaggle_mini_gpt.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("kaggle_mini_gpt.ipynb created:", len(CELLS), "cells")
