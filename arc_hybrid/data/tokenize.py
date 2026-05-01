"""Grid-to-token serialization.

Token vocabulary (size 16):
    0..9  : grid colors
    10    : PAD       (also used as fill for padded grid tensors)
    11    : EOR       (end of row, emitted between rows of a grid)
    12    : EOG       (end of grid, terminates a grid)
    13    : SEP       (between demonstration pairs)
    14    : IN        (start of an input grid)
    15    : OUT       (start of an output grid)

For each token the collator also emits 2D position (row, col), grid index
within the task, and a role id. CNN features are computed in the model from
the per-grid color tensor and gathered into cell positions of the sequence.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .arc_loader import Task

PAD, EOR, EOG, SEP, IN, OUT = 10, 11, 12, 13, 14, 15
VOCAB_SIZE = 16
N_COLORS = 10

ROLE_DEMO_IN = 0
ROLE_DEMO_OUT = 1
ROLE_TEST_IN = 2
ROLE_TEST_OUT = 3
ROLE_SPECIAL = 4
N_ROLES = 5


@dataclass
class PackedTask:
    token_ids: torch.Tensor      # [L]   long
    row_ids: torch.Tensor        # [L]   long; max_grid means "not a cell"
    col_ids: torch.Tensor        # [L]   long
    role_ids: torch.Tensor       # [L]   long
    grid_in_sample: torch.Tensor # [L]   long; -1 for non-cells
    cell_mask: torch.Tensor      # [L]   bool; True iff token is a color cell
    target_mask: torch.Tensor    # [L]   bool; True iff position contributes to loss
    grids: torch.Tensor          # [G, max_grid, max_grid] long, padded with PAD
    grid_sizes: torch.Tensor     # [G, 2] long, real (H, W)


def pack_task(
    task: Task,
    *,
    test_idx: int = 0,
    include_test_output: bool = True,
    max_grid: int = 30,
) -> PackedTask:
    """Pack a task into a single token sequence.

    `include_test_output=True` produces a training example (target_mask covers
    the test output cells + EOR/EOG). `False` produces an inference prefix
    ending right after the OUT special — call generate() from there.
    """
    tok: list[int] = []
    row: list[int] = []
    col: list[int] = []
    role: list[int] = []
    grid_in_sample: list[int] = []
    cell_mask: list[bool] = []
    grids: list[np.ndarray] = []
    grid_sizes: list[tuple[int, int]] = []

    SENT = max_grid  # row/col sentinel for non-cell tokens

    def add_special(t: int) -> None:
        tok.append(t)
        row.append(SENT)
        col.append(SENT)
        role.append(ROLE_SPECIAL)
        grid_in_sample.append(-1)
        cell_mask.append(False)

    def add_grid(g: np.ndarray, role_id: int, *, cnn_visible: bool) -> tuple[int, int]:
        """Append a grid as a sequence of cell tokens + EOR/EOG specials.

        cnn_visible=True  : grid is stored in `grids` for CNN encoding (demos, test input).
        cnn_visible=False : grid cells appear in the sequence but the CNN is not given
                            access — used for the test output during training so the
                            model cannot peek at the answer via its CNN branch.
        """
        H, W = g.shape
        if H > max_grid or W > max_grid:
            raise ValueError(f"grid {H}x{W} exceeds max_grid={max_grid}")
        if cnn_visible:
            gi = len(grids)
            grids.append(g)
            grid_sizes.append((H, W))
        else:
            gi = -1
        start = len(tok)
        for r in range(H):
            for c in range(W):
                tok.append(int(g[r, c]))
                row.append(r)
                col.append(c)
                role.append(role_id)
                grid_in_sample.append(gi)
                cell_mask.append(True)
            tok.append(EOR)
            row.append(r)
            col.append(W)
            role.append(ROLE_SPECIAL)
            grid_in_sample.append(-1)
            cell_mask.append(False)
        tok.append(EOG)
        row.append(H)
        col.append(0)
        role.append(ROLE_SPECIAL)
        grid_in_sample.append(-1)
        cell_mask.append(False)
        return start, len(tok)

    for p in task.train:
        add_special(IN)
        add_grid(p.input, ROLE_DEMO_IN, cnn_visible=True)
        add_special(OUT)
        add_grid(p.output, ROLE_DEMO_OUT, cnn_visible=True)
        add_special(SEP)

    test_pair = task.test[test_idx]
    add_special(IN)
    add_grid(test_pair.input, ROLE_TEST_IN, cnn_visible=True)
    add_special(OUT)

    target_lo = len(tok)
    if include_test_output:
        if test_pair.output is None:
            raise ValueError("include_test_output=True but test pair has no output")
        add_grid(test_pair.output, ROLE_TEST_OUT, cnn_visible=False)
    target_hi = len(tok)

    L = len(tok)
    G = len(grids)
    grid_tensor = torch.full((G, max_grid, max_grid), PAD, dtype=torch.long)
    for i, g in enumerate(grids):
        H, W = g.shape
        grid_tensor[i, :H, :W] = torch.from_numpy(g.astype(np.int64))

    target_mask = torch.zeros(L, dtype=torch.bool)
    if include_test_output:
        target_mask[target_lo:target_hi] = True

    return PackedTask(
        token_ids=torch.tensor(tok, dtype=torch.long),
        row_ids=torch.tensor(row, dtype=torch.long),
        col_ids=torch.tensor(col, dtype=torch.long),
        role_ids=torch.tensor(role, dtype=torch.long),
        grid_in_sample=torch.tensor(grid_in_sample, dtype=torch.long),
        cell_mask=torch.tensor(cell_mask, dtype=torch.bool),
        target_mask=target_mask,
        grids=grid_tensor,
        grid_sizes=torch.tensor(grid_sizes, dtype=torch.long),
    )


def collate_batch(packed: list[PackedTask], max_grid: int = 30) -> dict[str, torch.Tensor]:
    """Pad a list of PackedTask to common L and G; produce a key-padding mask."""
    B = len(packed)
    L_max = max(p.token_ids.size(0) for p in packed)
    G_max = max(p.grids.size(0) for p in packed)

    token_ids = torch.full((B, L_max), PAD, dtype=torch.long)
    row_ids = torch.full((B, L_max), max_grid, dtype=torch.long)
    col_ids = torch.full((B, L_max), max_grid, dtype=torch.long)
    role_ids = torch.full((B, L_max), ROLE_SPECIAL, dtype=torch.long)
    grid_in_sample = torch.full((B, L_max), -1, dtype=torch.long)
    cell_mask = torch.zeros((B, L_max), dtype=torch.bool)
    target_mask = torch.zeros((B, L_max), dtype=torch.bool)
    pad_mask = torch.ones((B, L_max), dtype=torch.bool)  # True = real, False = padding
    grids = torch.full((B, G_max, max_grid, max_grid), PAD, dtype=torch.long)
    grid_sizes = torch.zeros((B, G_max, 2), dtype=torch.long)

    for i, p in enumerate(packed):
        L_i = p.token_ids.size(0)
        G_i = p.grids.size(0)
        token_ids[i, :L_i] = p.token_ids
        row_ids[i, :L_i] = p.row_ids
        col_ids[i, :L_i] = p.col_ids
        role_ids[i, :L_i] = p.role_ids
        grid_in_sample[i, :L_i] = p.grid_in_sample
        cell_mask[i, :L_i] = p.cell_mask
        target_mask[i, :L_i] = p.target_mask
        pad_mask[i, L_i:] = False
        grids[i, :G_i] = p.grids
        grid_sizes[i, :G_i] = p.grid_sizes

    return {
        "token_ids": token_ids,
        "row_ids": row_ids,
        "col_ids": col_ids,
        "role_ids": role_ids,
        "grid_in_sample": grid_in_sample,
        "cell_mask": cell_mask,
        "target_mask": target_mask,
        "pad_mask": pad_mask,
        "grids": grids,
        "grid_sizes": grid_sizes,
    }
