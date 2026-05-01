"""ResNet-style CNN encoder over a single grid.

Input  : [N, 1, max_grid, max_grid] long-cast-to-float color values (0..PAD).
         Internally embedded to channels via a small color embedding so PAD
         (10) does not corrupt the conv arithmetic.
Output : [N, d_cnn, max_grid, max_grid] feature map; gathered into per-cell
         features by the hybrid model.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data.tokenize import VOCAB_SIZE


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = self.gn2(self.conv2(h))
        return F.gelu(x + h)


class CNNEncoder(nn.Module):
    def __init__(self, channels: int = 64, n_blocks: int = 3, vocab_size: int = VOCAB_SIZE) -> None:
        super().__init__()
        self.color_embed = nn.Embedding(vocab_size, channels)
        self.blocks = nn.Sequential(*[_ResBlock(channels) for _ in range(n_blocks)])
        self.out_channels = channels

    def forward(self, grids: torch.Tensor) -> torch.Tensor:
        """grids: [N, H, W] long. Returns [N, channels, H, W]."""
        if grids.dim() != 3:
            raise ValueError(f"expected [N,H,W], got {grids.shape}")
        x = self.color_embed(grids).permute(0, 3, 1, 2).contiguous()  # [N, C, H, W]
        return self.blocks(x)
