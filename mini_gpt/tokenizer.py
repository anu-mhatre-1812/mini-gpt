"""Character-level tokenizer."""

import json
from pathlib import Path
from typing import List, Optional


class CharTokenizer:
    def __init__(self, text: str = ""):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> List[int]:
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids)

    def save(self, path: str) -> None:
        data = {"stoi": self.stoi, "itos": {int(k): v for k, v in self.itos.items()}}
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls("")
        tok.stoi = data["stoi"]
        tok.itos = {int(k): v for k, v in data["itos"].items()}
        tok.vocab_size = len(tok.stoi)
        return tok

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
