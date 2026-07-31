"""TAM data loading and resizing utilities for Stage A."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..data import choose_stratified_subset, load_cw_data
from ..evaluation.attack_models import build_rf_tam_input


@dataclass
class StageATamDataset:
    tam: np.ndarray
    labels: np.ndarray
    sample_ids: np.ndarray
    source_indices: np.ndarray
    split: str
    data_source: str


def downsample_tam(tam: np.ndarray, width: int) -> np.ndarray:
    """Resample a TAM array to ``[N, 2, width]`` while preserving rough mass."""
    values = np.asarray(tam, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected TAM [N, 2, W], got {values.shape}")
    target = int(width)
    if values.shape[2] == target:
        return values.astype(np.float32, copy=False)
    if target <= 0:
        raise ValueError(f"width must be positive, got {width}")
    source_x = np.linspace(0.0, 1.0, values.shape[2], dtype=np.float32)
    target_x = np.linspace(0.0, 1.0, target, dtype=np.float32)
    out = np.empty((values.shape[0], 2, target), dtype=np.float32)
    for row in range(values.shape[0]):
        for direction in range(2):
            out[row, direction] = np.interp(target_x, source_x, values[row, direction]).astype(np.float32)
    out *= float(values.shape[2]) / float(target)
    return out


def raw_or_tam_to_tam(
    raw_or_tam: np.ndarray,
    *,
    width: int,
    max_trace_length: int = 5000,
    max_load_time: float = 80.0,
) -> np.ndarray:
    """Convert raw signed timestamps or existing TAM tensors into Stage A TAM."""
    values = np.asarray(raw_or_tam)
    if values.ndim == 3 and values.shape[1] == 2:
        return downsample_tam(values, int(width))
    return build_rf_tam_input(
        values,
        max_len=int(max_trace_length),
        max_load_time=float(max_load_time),
        num_slots=int(width),
    ).astype(np.float32)


def _make_data_cfg(
    data_root: str | Path,
    *,
    seed: int,
    val_ratio: float,
    test_ratio: float,
    max_samples: int,
    max_classes: int,
):
    return SimpleNamespace(
        data_root=str(data_root),
        seed=int(seed),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        max_samples=int(max_samples),
        max_classes=int(max_classes),
    )


def _select_split_indices(splits: dict[str, np.ndarray], split: str, total: int) -> np.ndarray:
    name = str(split).strip().lower()
    if name in {"all", "*"}:
        return np.arange(int(total), dtype=np.int64)
    if name == "valid":
        name = "val"
    if name not in splits:
        raise ValueError(f"Unknown split {split!r}; available={sorted(splits)}")
    return np.asarray(splits[name], dtype=np.int64)


def _choose_per_class_subset(labels: np.ndarray, samples_per_class: int, seed: int) -> np.ndarray:
    local_labels = np.asarray(labels, dtype=np.int64)
    if int(samples_per_class) <= 0:
        return np.arange(local_labels.size, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for label in np.unique(local_labels):
        positions = np.flatnonzero(local_labels == label)
        take = min(int(samples_per_class), int(positions.size))
        if take:
            selected.extend(rng.choice(positions, size=take, replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def load_stage_a_tam_dataset(
    data_root: str | Path,
    *,
    split: str = "test",
    width: int = 200,
    seed: int = 0,
    max_samples: int = 0,
    samples_per_class: int = 0,
    max_classes: int = 0,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    max_trace_length: int = 5000,
    max_load_time: float = 80.0,
) -> StageATamDataset:
    """Load CW-style data and return a Stage A TAM subset."""
    cfg = _make_data_cfg(
        data_root,
        seed=int(seed),
        val_ratio=float(val_ratio),
        test_ratio=float(test_ratio),
        max_samples=0,
        max_classes=int(max_classes),
    )
    raw, labels, trace_ids, splits, data_source = load_cw_data(cfg)
    indices = _select_split_indices(splits, split, len(labels))
    if int(samples_per_class) > 0:
        local = _choose_per_class_subset(labels[indices], int(samples_per_class), int(seed))
        indices = indices[local]
    elif int(max_samples) > 0 and len(indices) > int(max_samples):
        local = choose_stratified_subset(labels[indices], int(max_samples), int(seed))
        indices = indices[local]
    selected = raw[indices]
    tam = raw_or_tam_to_tam(
        selected,
        width=int(width),
        max_trace_length=int(max_trace_length),
        max_load_time=float(max_load_time),
    )
    return StageATamDataset(
        tam=tam.astype(np.float32),
        labels=np.asarray(labels[indices], dtype=np.int64),
        sample_ids=np.asarray(trace_ids[indices]).astype(str),
        source_indices=np.asarray(indices, dtype=np.int64),
        split=str(split),
        data_source=str(data_source),
    )
