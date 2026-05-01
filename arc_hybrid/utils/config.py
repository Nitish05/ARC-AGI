"""YAML config loading with attribute access."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


def to_namespace(d: Any) -> Any:
    if isinstance(d, dict):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [to_namespace(v) for v in d]
    return d


def load_config(path: str | Path) -> SimpleNamespace:
    raw = yaml.safe_load(Path(path).read_text())
    return to_namespace(raw)


def asdict(ns: SimpleNamespace) -> dict:
    if isinstance(ns, SimpleNamespace):
        return {k: asdict(v) for k, v in vars(ns).items()}
    if isinstance(ns, list):
        return [asdict(x) for x in ns]
    return ns
