import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from model import GPTConfig, GPTLanguageModel
from tokenizer import build_vocab, decode, encode, load_vocab, save_vocab


@dataclass
class TrainConfig:
    batch_size: int = 16
    block_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 300
    learning_rate: float = 1e-3
    eval_iters: int = 200
    n_embed: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    seed: int = 1337
    ckpt_path: str = "checkpoints/ckpt.pt"
    tokens_path: str = "data/tokens.pt"
    vocab_path: str = "data/vocab.json"
    input_path: str = "data/input.txt"


cfg = TrainConfig()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(cfg.seed)


def load_or_build_data():
    # load preprocessed data if available; otherwise build from input.txt
    tokens_path = Path(cfg.tokens_path)
    vocab_path = Path(cfg.vocab_path)
    if tokens_path.exists() and vocab_path.exists():
        ids = torch.load(tokens_path)
        stoi, itos = load_vocab(vocab_path)
        return ids, stoi, itos

    text = Path(cfg.input_path).read_text(encoding="utf-8")
    stoi, itos = build_vocab(text)
    ids = torch.tensor(encode(text, stoi), dtype=torch.long)
    save_vocab(vocab_path, stoi, itos)
    tokens_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ids, tokens_path)
    return ids, stoi, itos


data, stoi, itos = load_or_build_data()
vocab_size = len(stoi)

# split datasets into train/val sets
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # sample random sequences of length block_size from train/val
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - cfg.block_size, (cfg.batch_size,))
    x = torch.stack([data[i : i + cfg.block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + cfg.block_size + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    # run a few forward passes and average loss for a stable estimate
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(split)
            _, loss, _ = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_checkpoint(model, optimizer, step, model_cfg):
    # save model/optimizer for resume or fine-tuning later
    os.makedirs(Path(cfg.ckpt_path).parent, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "train_config": asdict(cfg),
            "model_config": asdict(model_cfg),
        },
        cfg.ckpt_path,
    )


def load_checkpoint(model, optimizer):
    # resume if a checkpoint exists
    if not Path(cfg.ckpt_path).exists():
        return 0
    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt.get("step", 0)


model_cfg = GPTConfig(
    vocab_size=vocab_size,
    block_size=cfg.block_size,
    n_embed=cfg.n_embed,
    n_head=cfg.n_head,
    n_layer=cfg.n_layer,
    dropout=cfg.dropout,
)
model = GPTLanguageModel(model_cfg).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
start_iter = load_checkpoint(model, optimizer)

for iter in range(start_iter, cfg.max_iters):
    if iter % cfg.eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        save_checkpoint(model, optimizer, iter, model_cfg)

    xb, yb = get_batch("train")
    _, loss, _ = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# sample a short continuation from a single-token context
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(model.generate(context, max_new_tokens=200, temperature=0.9, top_k=40)[0].tolist(), itos))
