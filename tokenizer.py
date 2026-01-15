import json
from pathlib import Path


def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def encode(text, stoi):
    return [stoi[c] for c in text]


def decode(ids, itos):
    return "".join(itos[i] for i in ids)


def save_vocab(path, stoi, itos):
    payload = {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}
    Path(path).write_text(json.dumps(payload), encoding="utf-8")


def load_vocab(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    stoi = payload["stoi"]
    itos = {int(k): v for k, v in payload["itos"].items()}
    return stoi, itos
