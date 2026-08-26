"""Build the multi-domain Colab notebook (v2)."""

import json

TEACHER_DATA_CELL = '''# Teacher data (written by ox-alpha) + multi-domain corpus
teacher = {}

teacher["marathi_clean"] = """मराठी भाषा भारतातील महाराष्ट्र राज्याची अधिकृत भाषा आहे. ही भाषा देवनागरी लिपीत लिहिली जाते. मराठी भाषेचा इतिहास अतिशय जुना आहे. संत ज्ञानेश्वरांनी ज्ञानेश्वरी ग्रंथ लिहिला. त्यानंतर संत तुकाराम आणि संत नामदेव यांनी अभंग रचले. छत्रपती शिवाजी महाराजांनी स्वराज्य स्थापन केले. पुणे शहर म्हणजे मराठी साहित्याचे माहेरघर. लोकमान्य टिळक म्हणाले, स्वराज्य हा माझा जन्मसिद्ध हक्क आहे.

महाराष्ट्रात मुंबई ही राजधानी आहे. मुंबई हे भारताचे आर्थिक महानगर आहे. नाशिक शहर द्राक्षांसाठी प्रसिद्ध आहे. नागपूर हे संत्र्यांसाठी प्रसिद्ध आहे. कोकणचा किनारा अतिशय सुंदर आहे. सह्याद्री पर्वतात जंगले आहेत. गोदावरी आणि कृष्णा या मोठ्या नद्या आहेत. शेतकरी भात, ऊस आणि ज्वारी पिकवतात. गणेशोत्सव हा सर्वात मोठा सण आहे.

विज्ञानामुळे जग बदलले आहे. सूर्य पृथ्वीभोवती फिरत नाही, तर पृथ्वी सूर्याभोवती फिरते. वनस्पती प्रकाशापासून अन्न बनवतात. या प्रक्रियेला प्रकाशसंश्लेषण म्हणतात. वृक्ष ऑक्सिजन देतात. पाऊस पडल्यास शेती भरभराटते."""

teacher["hindi"] = """भारत एक विशाल देश है। यहाँ कई भाषाएँ बोली जाती हैं। हिंदी भारत की राष्ट्रभाषा है। दिल्ली भारत की राजधानी है। ताजमहल आगरा में स्थित है। यह विश्व का सबसे सुंदर स्मारक है। गंगा नदी हिमालय से निकलती है। किसान खेत में काम करते हैं। वे गेहूँ, चावल और मक्का उगाते हैं। बच्चे स्कूल जाते हैं। शिक्षा जीवन की नींव है। मेहनत का फल मीठा होता है।

विज्ञान ने जीवन बदल दिया है। बिजली से घर रोशन होते हैं। ट्रेन, बस और हवाई जहाज यात्रा आसान करते हैं। इंटरनेट से दुनिया की हर जानकारी मिल जाती है। हमें प्रकृति की रक्षा करनी चाहिए। पेड़ लगाओ, जीवन बचाओ।"""

teacher["english"] = """The sun rises in the east and sets in the west. Trees give us oxygen, fruits and shade. Children go to school to learn reading and writing. Knowledge is the greatest treasure a person can own. Practice makes a person perfect. Hard work always pays off in the end.

Machine learning is a branch of artificial intelligence. A model learns patterns from data instead of following fixed rules. Training means adjusting weights to reduce error. A neural network has layers of connected neurons. Attention lets a model focus on important parts of the input. Large language models predict the next token in a sequence. The transformer architecture changed everything in modern AI. Data quality matters more than model size."""

teacher["maths"] = """1 + 1 = 2
2 + 2 = 4
5 + 5 = 10
10 + 10 = 20
25 + 25 = 50
2 x 2 = 4
3 x 3 = 9
5 x 5 = 25
7 x 7 = 49
10 x 10 = 100
12 x 12 = 144
100 / 4 = 25
the square root of 16 is 4
the square root of 81 is 9
pi is approximately 3.14
2, 3, 5, 7, 11 are prime numbers
1, 1, 2, 3, 5, 8, 13 is the fibonacci sequence
x + 5 = 10 therefore x = 5"""

teacher["code"] = \'\'\'# python basics
x = 5
y = 10
print(x + y)
for i in range(10):
    print(i)

def add(a, b):
    return a + b

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

numbers = [1, 2, 3, 4, 5]
print(sum(numbers))

// c binary search
int low = 0;
int high = n - 1;
while (low <= high) {
    int mid = (low + high) / 2;
    if (arr[mid] == target) return mid;
    if (arr[mid] < target) low = mid + 1;
    else high = mid - 1;
}\'\'\'

# repeat teacher data 8x to balance against the big Wikipedia corpus
TEACHER_REPEAT = 8
'''

DATA_CELL = """# Download Marathi Wikipedia + build multi-domain corpus
import urllib.parse
import urllib.request

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
        pass
    time.sleep(1.5)

print(f"wikipedia: {len(out)} articles")

parts = []
for name, txt in teacher.items():
    parts.extend([txt.strip()] * TEACHER_REPEAT)
    print(f"teacher {name}: {len(txt):,} chars x{TEACHER_REPEAT}")
parts.append("\\n\\n".join(out))
text = "\\n\\n".join(parts)
print(f"TOTAL corpus: {len(text):,} chars")"""

MODEL_CELL = """import torch
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
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")"""

TRAIN_CELL = """opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
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

torch.save(model.state_dict(), "mini_gpt_multi.pt")
print("checkpoint saved: mini_gpt_multi.pt")"""

GEN_CELL = """# Test ALL 5 domains
model.eval()
tests = [
    ("marathi", "मराठी"),
    ("hindi", "भारत"),
    ("english", "The sun"),
    ("maths", "2 + 2 ="),
    ("code", "def add(a, b):"),
]
for name, prompt in tests:
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    out = model.generate(idx, 250, temperature=0.8, top_k=40)
    print(f"===== {name.upper()} =====")
    print("".join(itos[i] for i in out[0].tolist()))
    print()"""

CELLS = [
    ("markdown", "# Mini-GPT Multi-Domain 🌐 — Marathi · Hindi · English · Maths · Code\n\n"
     "GPT built from scratch (pure PyTorch), trained on teacher-distilled data + Marathi Wikipedia.\n\n"
     "**Setup:** Runtime → Change runtime type → **T4 GPU** → Save. Then Run All."),
    ("code", TEACHER_DATA_CELL),
    ("code", DATA_CELL),
    ("code", MODEL_CELL),
    ("code", TRAIN_CELL),
    ("code", GEN_CELL),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
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

with open(r"C:\Users\ADMIN\Projects\mini-gpt\mini_gpt_multi.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("mini_gpt_multi.ipynb created:", len(CELLS), "cells")
