# mini-gpt-playground

A small GPT-style language model for learning and experimentation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prepare data

Place your text at `data/input.txt`, then run:

```bash
python prepare_data.py
```

## Train

```bash
python train.py
```

## Generate

```bash
python generate.py --ckpt checkpoints/ckpt.pt
```

## Notes

- RoPE positional encoding and KV cache are included.
- The model is intentionally small for CPU/M1 training.
