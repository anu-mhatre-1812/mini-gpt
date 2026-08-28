# mini-gpt

GPT-style transformer from scratch in pure PyTorch - char-level, trained on Marathi Wikipedia. No transformers library, every component hand-built.

## Features
- Character-level GPT transformer built entirely from scratch
- Multi-head self-attention mechanism with causal masking
- Trained on Marathi Wikipedia corpus
- Positional encoding, layer normalization, and feed-forward networks
- No HuggingFace transformers library - every component hand-built
- Training pipeline with gradient clipping and learning rate scheduling

## Tech Stack
- **Language:** Python
- **Topics:** attention, from-scratch, gpt, language-model, marathi, pytorch, transformer

## Quick Start
```bash
git clone https://github.com/a18-n03/mini-gpt.git
cd mini-gpt
pip install torch numpy
python train.py
```

## Author
**Anuj Mhatre** - [GitHub](https://github.com/a18-n03) | [Portfolio](https://anujmhatre.me)

## License
MIT
