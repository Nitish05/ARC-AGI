"""Hybrid CNN + Transformer.

forward consumes a packed batch (see arc_hybrid.data.tokenize.collate_batch)
and produces logits at every sequence position. The hybrid step:
  1. Run CNNEncoder over each grid in the batch -> per-cell feature map.
  2. Build the embedding sequence (color/row/col/role/grid-index embeds).
  3. Gather CNN features for color-cell positions, project, add to sequence.
  4. Run causal decoder -> LM head.

KV-cache mode (eval/inference only):
  When `use_cache=True`, returns `(logits, new_kv_list)`. On the prefill step
  (`past_kv_list=None`) the call is identical to the no-cache path but also
  emits per-layer K/V caches. On subsequent decode steps (`past_kv_list` set,
  L=1), we skip the CNN+gather entirely — the new generated token always has
  `grid_in_sample=-1` (it's a CNN-blind test-output cell), so the gathered
  CNN contribution at that position is zero by construction. Only embeddings
  matter for the new token.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ..data.tokenize import VOCAB_SIZE
from .cnn_encoder import CNNEncoder
from .embeddings import TokenEmbeddings
from .transformer import CausalDecoder, KVList


class HybridModel(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 12,
        d_ff: int = 2048,
        cnn_channels: int = 96,
        cnn_blocks: int = 4,
        max_grid: int = 30,
        max_grids_per_task: int = 32,
        vocab_size: int = VOCAB_SIZE,
        dropout: float = 0.0,
        grad_checkpoint: bool = False,
    ) -> None:
        super().__init__()
        self.max_grid = max_grid
        self.cnn = CNNEncoder(channels=cnn_channels, n_blocks=cnn_blocks, vocab_size=vocab_size)
        self.cnn_to_d = nn.Linear(cnn_channels, d_model)
        self.embeddings = TokenEmbeddings(
            d_model=d_model,
            max_grid=max_grid,
            max_grids_per_task=max_grids_per_task,
            vocab_size=vocab_size,
        )
        self.transformer = CausalDecoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            grad_checkpoint=grad_checkpoint,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,        # [B, L]
        row_ids: torch.Tensor,          # [B, L]
        col_ids: torch.Tensor,          # [B, L]
        role_ids: torch.Tensor,         # [B, L]
        grid_in_sample: torch.Tensor,   # [B, L], -1 for non-cells
        cell_mask: torch.Tensor,        # [B, L]
        pad_mask: torch.Tensor,         # [B, L]
        grids: torch.Tensor,            # [B, G, max_grid, max_grid]
        grid_sizes: torch.Tensor | None = None,  # [B, G, 2]; unused (kept for API symmetry)
        *,
        use_cache: bool = False,
        past_kv_list: Optional[KVList] = None,
    ):
        del grid_sizes  # padded grid is fine; CNN over PAD is masked by cell_mask later

        is_decode_step = use_cache and past_kv_list is not None and token_ids.size(1) == 1

        if is_decode_step:
            # New generated tokens are CNN-blind by construction (grid_in_sample=-1),
            # so we skip the CNN+gather pass entirely. Embeddings only.
            x = self.embeddings(token_ids, row_ids, col_ids, role_ids, grid_in_sample)
        else:
            B, G, H, W = grids.shape
            flat = grids.reshape(B * G, H, W)
            cnn_feats = self.cnn(flat)                                # [B*G, C, H, W]
            cnn_feats = cnn_feats.permute(0, 2, 3, 1).contiguous()     # [B*G, H, W, C]
            cnn_feats = cnn_feats.view(B, G, H, W, -1)                 # [B, G, H, W, C]

            b_idx = torch.arange(B, device=grids.device).view(B, 1).expand_as(grid_in_sample)
            gi = grid_in_sample.clamp(min=0)
            ri = row_ids.clamp(max=H - 1)
            ci = col_ids.clamp(max=W - 1)
            gathered = cnn_feats[b_idx, gi, ri, ci]                    # [B, L, C]
            cnn_lookup = cell_mask & (grid_in_sample >= 0)
            gathered = gathered * cnn_lookup.unsqueeze(-1).to(gathered.dtype)

            x = self.embeddings(token_ids, row_ids, col_ids, role_ids, grid_in_sample)
            x = x + self.cnn_to_d(gathered)

        h, new_kv_list = self.transformer(
            x, pad_mask=pad_mask, past_kv_list=past_kv_list, use_cache=use_cache
        )
        logits = self.lm_head(h)
        if use_cache:
            return logits, new_kv_list
        return logits


def build_hybrid_from_config(cfg) -> HybridModel:
    m = cfg.model
    return HybridModel(
        d_model=m.d_model,
        n_heads=m.n_heads,
        n_layers=m.n_layers,
        d_ff=m.d_ff,
        cnn_channels=m.cnn_channels,
        cnn_blocks=m.cnn_blocks,
        max_grid=m.max_grid_size,
        vocab_size=m.vocab_size,
        grad_checkpoint=getattr(cfg.train, "grad_checkpoint", False),
    )
