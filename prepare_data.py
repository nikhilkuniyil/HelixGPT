from pathlib import Path

import torch

from tokenizer import build_vocab, encode, save_vocab


def main():
    input_path = Path("data") / "input.txt"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing {input_path}")

    text = input_path.read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    ids = torch.tensor(encode(text, stoi), dtype=torch.long)

    Path("data").mkdir(parents=True, exist_ok=True)
    torch.save(ids, Path("data") / "tokens.pt")
    save_vocab(Path("data") / "vocab.json", stoi, itos)
    print(f"Saved {ids.numel()} tokens to data/tokens.pt")


if __name__ == "__main__":
    main()
