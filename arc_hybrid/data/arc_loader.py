"""Loads ARC-AGI / ARC-AGI-2 / RE-ARC tasks from JSON.

Each task JSON follows the standard ARC schema:
    {"train": [{"input": [[...]], "output": [[...]]}, ...],
     "test":  [{"input": [[...]], "output": [[...]]}, ...]}

Test outputs may be missing on hidden Kaggle splits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Pair:
    input: np.ndarray
    output: np.ndarray | None


@dataclass
class Task:
    task_id: str
    train: list[Pair]
    test: list[Pair]


def _to_grid(arr) -> np.ndarray:
    g = np.asarray(arr, dtype=np.int8)
    if g.ndim != 2:
        raise ValueError(f"expected 2D grid, got shape {g.shape}")
    if g.min() < 0 or g.max() > 9:
        raise ValueError(f"grid values out of range [0,9]: min={g.min()} max={g.max()}")
    return g


def _parse_pair(p: dict) -> Pair:
    out = p.get("output")
    return Pair(input=_to_grid(p["input"]), output=_to_grid(out) if out is not None else None)


def load_task(path: Path) -> Task:
    data = json.loads(Path(path).read_text())
    return Task(
        task_id=Path(path).stem,
        train=[_parse_pair(p) for p in data["train"]],
        test=[_parse_pair(p) for p in data["test"]],
    )


def load_split(root: Path, pattern: str = "*.json") -> list[Task]:
    """Load every JSON task under `root` recursively.

    Works for ARC-AGI(-2) directory layouts (one file per task) and for RE-ARC
    (directory of generated tasks in the same per-task schema).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root not found: {root}")
    paths = sorted(root.rglob(pattern))
    return [load_task(p) for p in paths]


def filter_max_grid(tasks: list[Task], max_grid: int) -> list[Task]:
    """Drop tasks whose any grid exceeds max_grid on either side."""
    def fits(p: Pair) -> bool:
        if max(p.input.shape) > max_grid:
            return False
        if p.output is not None and max(p.output.shape) > max_grid:
            return False
        return True

    return [t for t in tasks if all(fits(p) for p in t.train + t.test)]
