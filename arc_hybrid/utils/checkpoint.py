"""Checkpoint save/load — full (with optimizer) and slim (inference only).

The slim variant is what gets shipped to a Kaggle environment: smaller, no
optimizer state, no random-state baggage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_full(path: str | Path, *, model, optimizer, scheduler, step: int, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "step": step,
            "config": config,
        },
        path,
    )


def save_slim(path: str | Path, *, model, config: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": config}, path)


def load_into(path: str | Path, *, model, optimizer=None, scheduler=None, map_location: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt
