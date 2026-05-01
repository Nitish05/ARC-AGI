"""Decoder-only transformer with LoRA-friendly module names.

Submodules are named exactly so peft.inject_adapter_in_model with
target_modules=["q_proj","v_proj","up_proj","down_proj"] finds nn.Linear
modules and adds adapters. Verify with model.named_modules() — naming drift
will silently produce a no-op adapter.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.o_proj(out)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(d_model, d_ff)
        self.down_proj = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x)))


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attn_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class CausalDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(d_model, n_heads, d_ff, dropout=dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.grad_checkpoint = grad_checkpoint

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x: [B, L, d_model]. pad_mask: [B, L] bool, True = real token."""
        B, L, _ = x.shape
        attn_mask = _build_attn_mask(L, pad_mask, device=x.device, dtype=x.dtype)
        for layer in self.layers:
            if self.grad_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, attn_mask, use_reentrant=False)
            else:
                x = layer(x, attn_mask)
        return self.norm(x)


def _build_attn_mask(
    L: int,
    pad_mask: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Combined causal + key-padding additive mask broadcastable to [B, n_heads, L, L]."""
    causal = torch.triu(torch.ones(L, L, device=device, dtype=torch.bool), diagonal=1)  # True = blocked
    add = torch.zeros(L, L, device=device, dtype=dtype)
    add = add.masked_fill(causal, float("-inf"))
    add = add.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
    if pad_mask is not None:
        # block attention TO padded positions; broadcast to [B, 1, 1, L]
        key_pad = (~pad_mask).unsqueeze(1).unsqueeze(2)
        add = add.expand(pad_mask.size(0), 1, L, L).clone()
        add.masked_fill_(key_pad, float("-inf"))
    return add
