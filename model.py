from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int
    n_embed: int = 64
    n_head: int = 4
    n_layer: int = 2
    dropout: float = 0.1
    lora_enabled: bool = False
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_freeze_base: bool = True


class LoRALinear(nn.Module):
    """Linear layer with optional LoRA adapter for low-rank fine-tuning."""

    def __init__(
        self,
        in_features,
        out_features,
        r,
        lora_alpha=1.0,
        lora_dropout=0.0,
        bias=True,
        freeze_base=False,
    ):
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=bias)
        self.r = r
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

        if r > 0:
            self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    def forward(self, x):
        out = self.base(x)
        if self.r > 0:
            lora = self.lora_dropout(x) @ self.lora_A.t()
            lora = lora @ self.lora_B.t()
            out = out + lora * self.scaling
        return out


def make_linear(in_features, out_features, cfg: GPTConfig, bias=True):
    # use LoRA adapters when enabled for low-rank fine-tuning
    if cfg.lora_enabled:
        return LoRALinear(
            in_features,
            out_features,
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=bias,
            freeze_base=cfg.lora_freeze_base,
        )
    return nn.Linear(in_features, out_features, bias=bias)


class Head(nn.Module):
    """One head of self-attention with RoPE and optional KV cache."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        head_size = cfg.n_embed // cfg.n_head
        if head_size % 2 != 0:
            raise ValueError("RoPE requires an even head_size")
        self.head_size = head_size
        # LoRA can be injected into the attention projections for efficient fine-tuning
        self.key = make_linear(cfg.n_embed, head_size, cfg, bias=False)
        self.query = make_linear(cfg.n_embed, head_size, cfg, bias=False)
        self.value = make_linear(cfg.n_embed, head_size, cfg, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size)))
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_size, 2) / head_size))
        self.register_buffer("inv_freq", inv_freq)
        self.dropout = nn.Dropout(cfg.dropout)
        self.block_size = cfg.block_size

    def _rope_cos_sin(self, t, device, offset=0):
        # precompute rotary frequencies for sequence length t
        # offset lets us continue positions when using a KV cache
        positions = torch.arange(offset, offset + t, device=device)
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq)
        return torch.cos(freqs), torch.sin(freqs)

    def _apply_rope(self, x, cos, sin):
        # x: (B, T, head_size), cos/sin: (T, head_size/2)
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.zeros_like(x)
        out[..., ::2] = out_even
        out[..., 1::2] = out_odd
        return out

    def forward(self, x, past_kv=None):
        # compute causal self-attention for a single head
        # past_kv: optional cached (k, v) with shape (B, T_past, head_size)
        _, t, _ = x.shape
        k = self.key(x)  # (B, T_new, head_size)
        q = self.query(x)  # (B, T_new, head_size)
        past_len = 0 if past_kv is None else past_kv[0].size(1)
        cos, sin = self._rope_cos_sin(t, x.device, offset=past_len)
        k = self._apply_rope(k, cos, sin)
        q = self._apply_rope(q, cos, sin)
        v = self.value(x)  # (B, T_new, head_size)

        # append cached keys/values for fast autoregressive generation
        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat((past_k, k), dim=1)  # (B, T_total, head_size)
            v = torch.cat((past_v, v), dim=1)
            # keep cache within the context window
            if k.size(1) > self.block_size:
                k = k[:, -self.block_size :, :]
                v = v[:, -self.block_size :, :]

        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)  # (B, T_new, T_total)
        t_total = k.size(1)
        # only apply a causal mask when we have more than one query position
        if t > 1:
            wei = wei.masked_fill(self.tril[:t, :t_total] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # (B, T_new, head_size)
        return out, (k, v)


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.n_head)])
        # LoRA on the output projection helps adaptation with minimal params
        self.proj = make_linear(cfg.n_embed, cfg.n_embed, cfg)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, past_kv=None):
        # concatenate heads then project back to embedding size
        # past_kv: list of (k, v) per head, or None
        new_kv = []
        head_outs = []
        for i, h in enumerate(self.heads):
            head_past = None if past_kv is None else past_kv[i]
            out, head_kv = h(x, past_kv=head_past)
            head_outs.append(out)
            new_kv.append(head_kv)
        out = torch.cat(head_outs, dim=-1)
        out = self.proj(out)
        return self.dropout(out), new_kv


class FeedFoward(nn.Module):
    """A simple linear layer followed by a non-linearity."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.net = nn.Sequential(
            make_linear(cfg.n_embed, 4 * cfg.n_embed, cfg),
            nn.ReLU(),
            make_linear(4 * cfg.n_embed, cfg.n_embed, cfg),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MultiHeadAttention(cfg)
        self.ffwd = FeedFoward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x, past_kv=None):
        # pre-norm residual connections
        # past_kv: list of per-head caches for this block
        sa_out, new_kv = self.sa(self.ln1(x), past_kv=past_kv)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_kv


class GPTLanguageModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)

        # weight tying for better quality and fewer params
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets=None, past_kv=None, use_cache=False):
        b, t = idx.shape
        # token embeddings; RoPE supplies position information in attention
        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        x = tok_emb

        new_kv = []
        if use_cache:
            # propagate cache through each block during generation
            for i, block in enumerate(self.blocks):
                block_past = None if past_kv is None else past_kv[i]
                x, block_kv = block(x, past_kv=block_past)
                new_kv.append(block_kv)
        else:
            # training path (no cache)
            for block in self.blocks:
                x, _ = block(x, past_kv=None)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            # flatten batch/time for cross-entropy
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss, new_kv if use_cache else None

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        # autoregressive sampling with optional temperature/top-k
        self.eval()
        # per-layer KV cache to avoid recomputing past states
        past_kv = None
        for _ in range(max_new_tokens):
            # only feed the full prompt once, then a single token each step
            if past_kv is None:
                idx_cond = idx[:, -self.cfg.block_size :]
            else:
                idx_cond = idx[:, -1:]

            logits, _, past_kv = self(idx_cond, past_kv=past_kv, use_cache=True)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        self.train()
        return idx
