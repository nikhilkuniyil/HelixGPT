# HelixGPT

A tiny GPT-style language model for learning and experimentation on CPU/M1.

## What’s Inside

- Minimal GPT with RoPE positional encoding and KV cache
- Char-level tokenizer + data prep script
- Simple training loop with checkpoints
- Standalone generation script

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare Data

Place your text at `data/input.txt`, then run:

```bash
python prepare_data.py
```

This writes:

- `data/tokens.pt`
- `data/vocab.json`

## Train

```bash
python train.py
```

Checkpoints are written to `checkpoints/ckpt.pt`.

## Generate

```bash
python generate.py --ckpt checkpoints/ckpt.pt
```

Options:

- `--max_new_tokens` (default: 200)
- `--temperature` (default: 0.9)
- `--top_k` (default: 40)

## Project Layout

- `model.py` — GPT model, RoPE, KV cache
- `tokenizer.py` — char-level vocab + encode/decode
- `prepare_data.py` — build vocab + token ids
- `train.py` — training loop + checkpointing
- `generate.py` — sampling script

## Notes

The defaults are intentionally small so the model can train on CPU/M1. Edit
the `TrainConfig` in `train.py` for bigger models once you’re ready.
