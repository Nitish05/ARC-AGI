"""Test-time augmentation voting and two-attempt aggregation.

Applies each D8 transform to a task, predicts on each, inverts the transform on
the prediction, then majority-votes per cell. Two attempts are produced for the
ARC Prize submission format: attempt_1 = TTA-voted, attempt_2 = greedy without
TTA (a different mode of failure to maximise the chance one of the two hits).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from ..data.arc_loader import Pair, Task
from ..data.augment import apply_color_perm, dihedral, invert_dihedral


def _apply_dihedral_to_task(task: Task, k: int) -> Task:
    def t(p: Pair) -> Pair:
        return Pair(
            input=dihedral(p.input, k),
            output=dihedral(p.output, k) if p.output is not None else None,
        )
    return Task(task_id=task.task_id, train=[t(p) for p in task.train], test=[t(p) for p in task.test])


def majority_vote(grids: list[np.ndarray]) -> np.ndarray | None:
    grids = [g for g in grids if g is not None]
    if not grids:
        return None
    shapes = [tuple(g.shape) for g in grids]
    most_common_shape, _ = Counter(shapes).most_common(1)[0]
    grids = [g for g in grids if tuple(g.shape) == most_common_shape]
    H, W = most_common_shape
    stack = np.stack(grids).astype(np.int64)
    out = np.empty((H, W), dtype=np.int8)
    for r in range(H):
        for c in range(W):
            vals, counts = np.unique(stack[:, r, c], return_counts=True)
            out[r, c] = int(vals[int(np.argmax(counts))])
    return out


def tta_predict(generate_fn, task: Task, *, n_aug: int = 8) -> np.ndarray | None:
    """generate_fn(task) must return an np.ndarray (or None) for the task's first test pair."""
    preds: list[np.ndarray] = []
    for k in range(n_aug):
        aug = _apply_dihedral_to_task(task, k)
        out = generate_fn(aug)
        if out is None:
            continue
        preds.append(invert_dihedral(out, k))
    return majority_vote(preds)
