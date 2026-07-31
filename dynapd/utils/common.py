"""Small utilities shared by DynaPD modules."""

from __future__ import annotations

import csv
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def log(message: str, enabled: bool = True) -> None:
    if enabled:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] {message}", flush=True)


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except Exception:
        pass


def resolve_device(value: str):
    import torch

    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(value).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(value)


def write_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def as_serializable_config(obj: Any) -> dict[str, Any]:
    result = {}
    for key, value in vars(obj).items():
        if isinstance(value, Path):
            result[key] = str(value)
        else:
            result[key] = value
    return result

