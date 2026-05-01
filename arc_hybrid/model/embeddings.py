"""Token + 2D position + role + grid-index embeddings.

Per-cell tokens get color, row, col, role, grid-index embeds. Special tokens
(EOR/EOG/SEP/IN/OUT/PAD) get a sentinel row/col index `max_grid` so the row/col
embedding tables have one extra "not-a-cell" slot.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..data.tokenize import N_ROLES, VOCAB_SIZE


class TokenEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_grid: int,
        max_grids_per_task: int = 32,
        vocab_size: int = VOCAB_SIZE,
        n_roles: int = N_ROLES,
    ) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model)
        self.row = nn.Embedding(max_grid + 1, d_model)
        self.col = nn.Embedding(max_grid + 1, d_model)
        self.role = nn.Embedding(n_roles, d_model)
        # +1 for "not-in-any-grid" slot at index 0; real grids start at 1
        self.grid_idx = nn.Embedding(max_grids_per_task + 1, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        token_ids: torch.Tensor,    # [B, L]
        row_ids: torch.Tensor,      # [B, L]
        col_ids: torch.Tensor,      # [B, L]
        role_ids: torch.Tensor,     # [B, L]
        grid_in_sample: torch.Tensor,  # [B, L], -1 for non-cells
    ) -> torch.Tensor:
        gi = grid_in_sample.clamp(min=-1) + 1  # 0 reserved for "no grid"
        x = (
            self.token(token_ids)
            + self.row(row_ids)
            + self.col(col_ids)
            + self.role(role_ids)
            + self.grid_idx(gi)
        )
        return self.norm(x)
