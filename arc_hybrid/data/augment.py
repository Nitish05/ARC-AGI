"""Task-level augmentations: D8 dihedral, color permutation, demo dropout.

All transforms are applied *consistently* across every grid in a task — a
mismatch (e.g. rotating the input but not the output) silently destroys
training signal, so the API forces task-scope application.
"""
from __future__ import annotations

import numpy as np

from .arc_loader import Pair, Task


def dihedral(grid: np.ndarray, k: int) -> np.ndarray:
    """k in 0..7. 0..3 = rotations, 4..7 = horizontal flip then rotation."""
    g = grid
    if k >= 4:
        g = np.fliplr(g)
        k -= 4
    return np.ascontiguousarray(np.rot90(g, k))


def invert_dihedral(grid: np.ndarray, k: int) -> np.ndarray:
    """Inverse of dihedral(_, k). Used for test-time augmentation voting."""
    g = grid
    if k >= 4:
        k -= 4
        g = np.rot90(g, -k)
        g = np.fliplr(g)
    else:
        g = np.rot90(g, -k)
    return np.ascontiguousarray(g)


def random_color_perm(rng: np.random.Generator, fix_zero: bool = True) -> np.ndarray:
    """Random bijection over [0..9]. By convention 0 is most often background;
    fix_zero=True keeps it stable so background-vs-foreground semantics survive."""
    p = np.arange(10, dtype=np.int8)
    if fix_zero:
        p[1:] = rng.permutation(p[1:])
    else:
        p = rng.permutation(p).astype(np.int8)
    return p


def apply_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    return perm[grid].astype(np.int8)


def _augment_pair(p: Pair, k: int, perm: np.ndarray) -> Pair:
    inp = apply_color_perm(dihedral(p.input, k), perm)
    out = apply_color_perm(dihedral(p.output, k), perm) if p.output is not None else None
    return Pair(input=inp, output=out)


def augment_task(
    task: Task,
    rng: np.random.Generator,
    *,
    d8: bool = True,
    color_perm: bool = True,
    demo_dropout: bool = True,
) -> Task:
    k = int(rng.integers(0, 8)) if d8 else 0
    perm = random_color_perm(rng) if color_perm else np.arange(10, dtype=np.int8)
    train = [_augment_pair(p, k, perm) for p in task.train]
    if demo_dropout and len(train) > 1:
        keep = int(rng.integers(1, len(train) + 1))
        idx = rng.choice(len(train), size=keep, replace=False)
        train = [train[i] for i in sorted(idx.tolist())]
    test = [_augment_pair(p, k, perm) for p in task.test]
    return Task(task_id=task.task_id, train=train, test=test)
