"""Generate text from a trained mini-gpt checkpoint.

Usage: python generate.py --prompt "मराठी" --tokens 400
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mini_gpt.model import GPT
from mini_gpt.tokenizer import CharTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="मराठी")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--checkpoint", default="checkpoints/best.pt")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    stoi = ckpt["stoi"]
    itos = ckpt["itos"]
    tok = CharTokenizer("")
    tok.stoi, tok.itos, tok.vocab_size = stoi, itos, len(stoi)

    model = GPT(vocab_size=tok.vocab_size).to(device)
    model.load_state_dict(ckpt["model"])

    start = args.prompt or " "
    idx = torch.tensor([tok.encode(start)], dtype=torch.long, device=device)
    out = model.generate(idx, args.tokens, temperature=args.temperature)
    print(tok.decode(out[0].tolist()))


if __name__ == "__main__":
    main()
