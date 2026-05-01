"""Minimal training logger.

Writes a JSONL log per run plus stdout. Avoids dragging in wandb / TensorBoard
for v1 — Colab notebooks render the JSONL inline cheaply.
"""
from __future__ import annotations

import json
import time
from pathlib import Path


class JsonlLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)
        self._t0 = time.time()

    def log(self, **fields) -> None:
        row = {"t": round(time.time() - self._t0, 3), **fields}
        self._fh.write(json.dumps(row) + "\n")
        keys = " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in fields.items())
        print(keys, flush=True)

    def close(self) -> None:
        self._fh.close()
