import argparse
from pathlib import Path

import torch

from model import GPTConfig, GPTLanguageModel
from tokenizer import decode, load_vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/ckpt.pt")
    parser.add_argument("--tokens", default="data/vocab.json")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.ckpt, map_location=device)
    model_cfg = GPTConfig(**ckpt["model_config"])
    model = GPTLanguageModel(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    _, itos = load_vocab(Path(args.tokens))
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(
        context,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(decode(out[0].tolist(), itos))


if __name__ == "__main__":
    main()
