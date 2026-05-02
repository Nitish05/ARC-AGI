"""Checkpoint save/load — full (with optimizer) and slim (inference only).

The slim variant is what gets shipped to a Kaggle environment: smaller, no
optimizer state, no random-state baggage.

`torch.compile`-wrapped models prefix their state_dict keys with `_orig_mod.`.
We strip that on save so checkpoints are portable, and tolerate it on load so
older checkpoints that still carry the prefix can be loaded cleanly into a
fresh (uncompiled) model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

_COMPILE_PREFIX = "_orig_mod."


def _clean_state_dict(sd: dict) -> dict:
    """Strip a leading `_orig_mod.` from every key, if present."""
    if any(k.startswith(_COMPILE_PREFIX) for k in sd.keys()):
        return {(k[len(_COMPILE_PREFIX):] if k.startswith(_COMPILE_PREFIX) else k): v for k, v in sd.items()}
    return sd


def save_full(path: str | Path, *, model, optimizer, scheduler, step: int, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": _clean_state_dict(model.state_dict()),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "config": config,
        },
        path,
    )


def save_slim(path: str | Path, *, model, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": _clean_state_dict(model.state_dict()), "config": config}, path)


def load_into(path: str | Path, *, model, optimizer=None, scheduler=None, map_location: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(_clean_state_dict(ckpt["model"]))
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
