"""Decoder-only transformer with LoRA-friendly module names.

Submodules are named exactly so peft.inject_adapter_in_model with
target_modules=["q_proj","v_proj","up_proj","down_proj"] finds nn.Linear
modules and adds adapters. Verify with model.named_modules() — naming drift
will silently produce a no-op adapter.

Supports an optional KV cache for fast greedy decoding. When use_cache=True
the forward call returns (output, new_kv) and accepts an optional past_kv
to concat against. Training never sets use_cache; this lives entirely on
the inference path.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

KVPair = tuple[torch.Tensor, torch.Tensor]
KVList = list[Optional[KVPair]]


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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        past_kv: Optional[KVPair] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVPair]]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)  # [B, H, L_past + L, D]
            v = torch.cat([past_v, v], dim=2)

        # When using a cache for incremental decoding (q has length 1, k/v have
        # full history), there's no causal mask needed inside SDPA — the new
        # query is by construction the rightmost position. The caller passes
        # attn_mask=None for that case. For prefill / no-cache, the caller
        # passes the standard causal + key-padding additive mask.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)
        new_kv = (k, v) if use_cache else None
        return self.o_proj(out), new_kv


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

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        *,
        past_kv: Optional[KVPair] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVPair]]:
        attn_out, new_kv = self.attn(self.norm1(x), attn_mask, past_kv=past_kv, use_cache=use_cache)
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x, new_kv


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

    def forward(
        self,
        x: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
        *,
        past_kv_list: Optional[KVList] = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, Optional[KVList]]:
        """x: [B, L, d_model]. pad_mask: [B, L] bool, True = real token.

        When use_cache=True (eval-time only):
          - past_kv_list is either None (prefill) or a list of (K, V) tuples
            from a previous call. Returns (h, new_kv_list) where new_kv_list
            is a list of length n_layers of cached (K, V) for next call.
          - For incremental decode (L==1, past_kv_list is not None), the
            attn_mask is None — the single new query has no need to be
            causally masked, and the cached prefix has no padding once the
            initial prefill has filtered it out.
          - For prefill (past_kv_list is None), the standard causal+pad mask
            is built as before.
        """
        B, L, _ = x.shape

        is_decode_step = use_cache and past_kv_list is not None and L == 1
        if is_decode_step:
            attn_mask = None
        else:
            attn_mask = _build_attn_mask(L, pad_mask, device=x.device, dtype=x.dtype)

        new_kv_list: KVList = [None] * len(self.layers) if use_cache else []

        for i, layer in enumerate(self.layers):
            past = past_kv_list[i] if past_kv_list is not None else None
            if self.grad_checkpoint and self.training:
                # Training never uses cache; route through a tensor-only helper
                # so torch.utils.checkpoint sees a clean (Tensor)->Tensor signature.
                x = torch.utils.checkpoint.checkpoint(
                    _layer_no_cache, layer, x, attn_mask, use_reentrant=False,
                )
            else:
                x, new_kv = layer(x, attn_mask, past_kv=past, use_cache=use_cache)
                if use_cache:
                    new_kv_list[i] = new_kv
        return self.norm(x), (new_kv_list if use_cache else None)


def _layer_no_cache(layer, x, attn_mask):
    """Tensor-in / tensor-out wrapper around Block.forward for grad-checkpointing."""
    out, _ = layer(x, attn_mask, past_kv=None, use_cache=False)
    return out


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
