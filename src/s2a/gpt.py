"""A tiny, trainable GPT in PyTorch.

The same modern architecture as `s2a.transformer` (pre-norm RMSNorm,
grouped-query attention with RoPE, SwiGLU), but in PyTorch so it can be
trained: PyTorch works out the gradients automatically.

Needs PyTorch: `pip install torch` (preinstalled on Colab).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from s2a.transformer import TransformerConfig


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def rope(x: torch.Tensor, base: float = 10_000.0) -> torch.Tensor:
    """Rotate dimension pairs (0,1), (2,3), ... by position. x: (..., T, head_dim)."""
    T, hd = x.shape[-2], x.shape[-1]
    freqs = base ** (-torch.arange(0, hd, 2, device=x.device, dtype=x.dtype) / hd)
    angles = torch.outer(torch.arange(T, device=x.device, dtype=x.dtype), freqs)
    cos, sin = angles.cos(), angles.sin()
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        d, kv = cfg.d_model, cfg.n_kv_heads * cfg.head_dim
        self.w_q = nn.Linear(d, d, bias=False)
        self.w_k = nn.Linear(d, kv, bias=False)
        self.w_v = nn.Linear(d, kv, bias=False)
        self.w_o = nn.Linear(d, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        c, hd = self.cfg, self.cfg.head_dim
        q = self.w_q(x).view(B, T, c.n_heads, hd).transpose(1, 2)        # (B, heads, T, hd)
        k = self.w_k(x).view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        v = self.w_v(x).view(B, T, c.n_kv_heads, hd).transpose(1, 2)
        group = c.n_heads // c.n_kv_heads                                # query heads per KV head
        k, v = k.repeat_interleave(group, dim=1), v.repeat_interleave(group, dim=1)
        q, k = rope(q), rope(k)
        scores = q @ k.transpose(-2, -1) / math.sqrt(hd)                 # (B, heads, T, T)
        future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        weights = scores.masked_fill(future, float("-inf")).softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(B, T, c.d_model)
        return self.w_o(out)


class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w_down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.norm1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.norm2, self.mlp = RMSNorm(cfg.d_model), MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: TransformerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model)
        self.output_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.output_head.weight = self.embedding.weight
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, ids: torch.Tensor, targets: torch.Tensor | None = None):
        """ids: (B, T) token ids. Returns (logits, loss); loss is None without targets."""
        x = self.embedding(ids)
        for blk in self.blocks:
            x = blk(x)
        logits = self.output_head(self.final_norm(x))                    # (B, T, vocab)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_batch(data: torch.Tensor, batch_size: int, context: int, device: str = "cpu"):
    """Random windows of `context` tokens, and the same windows shifted by one as targets."""
    starts = torch.randint(len(data) - context - 1, (batch_size,))
    x = torch.stack([data[s : s + context] for s in starts])
    y = torch.stack([data[s + 1 : s + context + 1] for s in starts])
    return x.to(device), y.to(device)


def train(model: GPT, data: torch.Tensor, steps: int, batch_size: int = 32, context: int = 64,
          lr: float = 1e-3, optimizer: torch.optim.Optimizer | None = None) -> list[float]:
    """The training loop. Returns the loss at every step."""
    device = next(model.parameters()).device
    optimizer = optimizer or torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for _ in range(steps):
        x, y = get_batch(data, batch_size, context, device)
        _, loss = model(x, y)                    # 1. forward: predict and measure the loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()                          # 2. backward: gradient of the loss for every weight
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()                         # 3. update: nudge each weight downhill
        losses.append(loss.item())
    return losses


@torch.no_grad()
def estimate_loss(model: GPT, data: torch.Tensor, batches: int = 20, batch_size: int = 32,
                  context: int = 64) -> float:
    """Average loss over a few random batches, without training."""
    device = next(model.parameters()).device
    model.eval()
    total = sum(model(*get_batch(data, batch_size, context, device))[1].item() for _ in range(batches))
    return total / batches


@torch.no_grad()
def generate(model: GPT, ids: list[int], n: int, context: int = 64, temperature: float = 1.0) -> list[int]:
    """Sample n new tokens. Recomputes everything each step: no KV cache, for simplicity."""
    device = next(model.parameters()).device
    model.eval()
    out = list(ids)
    for _ in range(n):
        x = torch.tensor([out[-context:]], device=device)
        logits, _ = model(x)
        probs = (logits[0, -1] / temperature).softmax(dim=-1)
        out.append(int(torch.multinomial(probs, 1)))
    return out


def to_numpy_params(model: GPT) -> dict:
    """Export weights in the layout `s2a.transformer.forward` expects (for cross-checking)."""
    a = lambda t: t.detach().cpu().double().numpy()
    return {
        "embedding": a(model.embedding.weight),
        "layers": [
            {
                "norm1": a(b.norm1.weight), "w_q": a(b.attn.w_q.weight).T, "w_k": a(b.attn.w_k.weight).T,
                "w_v": a(b.attn.w_v.weight).T, "w_o": a(b.attn.w_o.weight).T,
                "norm2": a(b.norm2.weight), "w_gate": a(b.mlp.w_gate.weight).T,
                "w_up": a(b.mlp.w_up.weight).T, "w_down": a(b.mlp.w_down.weight).T,
            }
            for b in model.blocks
        ],
        "final_norm": a(model.final_norm.weight),
        "output_head": a(model.output_head.weight),
    }
