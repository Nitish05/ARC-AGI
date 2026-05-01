"""End-to-end shape test of the hybrid model on a synthetic mini-task.

Runs on CPU. If torch isn't installed (e.g. lint pass), the module is skipped.
"""
import pytest

torch = pytest.importorskip("torch")
import numpy as np


def _tiny_task():
    from arc_hybrid.data.arc_loader import Pair, Task

    rng = np.random.default_rng(0)

    def grid(h, w):
        return rng.integers(0, 10, size=(h, w), dtype=np.int8)

    return Task(
        task_id="tiny",
        train=[Pair(grid(3, 3), grid(3, 3)), Pair(grid(2, 4), grid(4, 2))],
        test=[Pair(grid(3, 3), grid(3, 3))],
    )


def test_pack_and_collate_shapes():
    from arc_hybrid.data.tokenize import collate_batch, pack_task

    packed = pack_task(_tiny_task(), test_idx=0, include_test_output=True, max_grid=30)
    assert packed.token_ids.dim() == 1
    L = packed.token_ids.size(0)
    for f in ("row_ids", "col_ids", "role_ids", "grid_in_sample", "cell_mask", "target_mask"):
        assert getattr(packed, f).size(0) == L

    batch = collate_batch([packed, packed], max_grid=30)
    assert batch["token_ids"].shape == (2, L)
    assert batch["grids"].dim() == 4 and batch["grids"].size(0) == 2


def test_hybrid_forward_shape():
    from arc_hybrid.data.tokenize import VOCAB_SIZE, collate_batch, pack_task
    from arc_hybrid.model.hybrid import HybridModel

    packed = pack_task(_tiny_task(), test_idx=0, include_test_output=True, max_grid=30)
    batch = collate_batch([packed], max_grid=30)

    model = HybridModel(
        d_model=64, n_heads=4, n_layers=2, d_ff=128,
        cnn_channels=16, cnn_blocks=2, max_grid=30, vocab_size=VOCAB_SIZE,
    )
    model.eval()
    with torch.no_grad():
        logits = model(
            token_ids=batch["token_ids"],
            row_ids=batch["row_ids"],
            col_ids=batch["col_ids"],
            role_ids=batch["role_ids"],
            grid_in_sample=batch["grid_in_sample"],
            cell_mask=batch["cell_mask"],
            pad_mask=batch["pad_mask"],
            grids=batch["grids"],
            grid_sizes=batch["grid_sizes"],
        )
    B, L = batch["token_ids"].shape
    assert logits.shape == (B, L, VOCAB_SIZE)
    assert torch.isfinite(logits).all()
