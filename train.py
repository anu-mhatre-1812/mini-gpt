"""Train mini-gpt on a text file (char-level). CPU-friendly.

Usage: python train.py --data data/marathi.txt --steps 3000
"""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from mini_gpt.model import GPT
from mini_gpt.tokenizer import CharTokenizer


def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size].clone() for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1].clone() for i in ix])
    return x.to(device), y.to(device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/marathi.txt")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=500, help="Early stopping patience (steps)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    text = Path(args.data).read_text(encoding="utf-8")
    tok = CharTokenizer(text)
    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    print(f"corpus: {len(data):,} chars | vocab: {tok.vocab_size}")

    model = GPT(vocab_size=tok.vocab_size).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = min(200, args.steps // 10)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps - warmup_steps)

    best_val = float("inf")
    patience = 500
    patience_counter = 0
    t0 = time.time()
    for step in range(1, args.steps + 1):
        # Linear warmup to prevent early training instability
        if step <= warmup_steps:
            lr_scale = step / warmup_steps
            for pg in opt.param_groups:
                pg["lr"] = args.lr * lr_scale
        model.train()
        x, y = get_batch(train_data, model.block_size, args.batch, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                vx, vy = get_batch(val_data, model.block_size, args.batch, device)
                _, vloss = model(vx, vy)
            elapsed = time.time() - t0
            print(f"step {step:5d} | train {loss.item():.4f} | val {vloss.item():.4f} "
                  f"| lr {sched.get_last_lr()[0]:.2e} | {elapsed:.0f}s")
            if vloss.item() < best_val:
                best_val = vloss.item()
                patience_counter = 0
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save({"model": model.state_dict(), "stoi": tok.stoi, "itos": tok.itos,
                            "config": {"vocab_size": tok.vocab_size}},
                           "checkpoints/best.pt")
            else:
                patience_counter += 200
                if patience_counter >= patience:
                    print(f"\nearly stopping at step {step} (no improvement for {patience} steps)")
                    break

    print(f"\ndone in {time.time() - t0:.0f}s | best val loss: {best_val:.4f}")
    print("checkpoint: checkpoints/best.pt")


if __name__ == "__main__":
    main()
