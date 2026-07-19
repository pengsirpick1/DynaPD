"""Preference pool and random mixer used by DMMPv3."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PREFERENCE_ALIASES = {
    "interval": "interval",
    "interval_filling": "interval",
    "early": "spread",
    "early_dispersion": "spread",
    "spread": "spread",
    "burst": "boundary",
    "burst_boundary": "boundary",
    "boundary": "boundary",
    "direction": "direction",
    "directional_balance": "direction",
    "shape": "shape",
    "distribution_shaping": "shape",
}

PREFERENCE_NAMES = {
    "interval": "Interval Filling",
    "spread": "Candidate Spread",
    "boundary": "Executable Boundary",
    "direction": "Directional Balance",
    "shape": "Distribution Shaping",
}


def canonical_preference(name: str) -> str:
    key = str(name).strip().lower().replace("-", "_")
    if key not in PREFERENCE_ALIASES:
        raise ValueError(f"Unsupported preference primitive: {name!r}")
    return PREFERENCE_ALIASES[key]


def _as_float_map(values: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape == shape:
        return arr.copy()
    if arr.ndim == 1 and arr.shape[0] == shape[1]:
        return np.repeat(arr.reshape(1, -1), shape[0], axis=0).astype(np.float32)
    return np.resize(arr, shape).astype(np.float32)


def _normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    allowed = np.asarray(mask, dtype=np.float32)
    result = np.maximum(result, 0.0) * allowed
    valid = allowed > 0
    if np.any(valid):
        peak = float(result[valid].max())
        if peak > 1e-8:
            result = result / peak
        else:
            result = allowed.copy()
    return result.astype(np.float32)


def _condition_1d(condition: Any, attr: str, patch_num: int) -> np.ndarray:
    value = getattr(condition, attr, None)
    if value is None:
        return np.zeros(int(patch_num), dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size != int(patch_num):
        arr = np.resize(arr, int(patch_num))
    return arr.astype(np.float32)


def _metadata(condition: Any) -> Mapping[str, Any]:
    return getattr(condition, "metadata", {}) or {}


@dataclass
class PreferencePool:
    patch_num: int = 200

    def compute(
        self,
        name: str,
        condition: Any,
        topk_mask: np.ndarray,
        s_cell: np.ndarray,
        allowed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        key = canonical_preference(name)
        mask = _as_float_map(
            getattr(condition, "allowed_mask", np.ones((2, self.patch_num), dtype=np.float32))
            if allowed_mask is None
            else allowed_mask,
            (2, self.patch_num),
        )
        topk = _as_float_map(topk_mask, mask.shape)
        leakage = _normalize(_as_float_map(s_cell, mask.shape), mask)
        if key == "interval":
            result = self._interval(condition, leakage, topk, mask)
        elif key == "spread":
            result = self._early_dispersion(condition, leakage, topk, mask)
        elif key == "boundary":
            result = self._burst_boundary(condition, leakage, topk, mask)
        elif key == "direction":
            result = self._direction_balance(condition, leakage, topk, mask)
        elif key == "shape":
            result = self._distribution_shaping(condition, leakage, topk, mask)
        else:
            raise ValueError(key)
        return _normalize(result, mask)

    def compute_all(
        self,
        condition: Any,
        topk_mask: np.ndarray,
        s_cell: np.ndarray,
        allowed_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        return {key: self.compute(key, condition, topk_mask, s_cell, allowed_mask) for key in PREFERENCE_NAMES}

    def _interval(self, condition: Any, leakage: np.ndarray, topk: np.ndarray, mask: np.ndarray) -> np.ndarray:
        gap = _condition_1d(condition, "gap_saliency", self.patch_num)
        rate = _condition_1d(condition, "rate_saliency", self.patch_num)
        base = 0.62 * gap.reshape(1, -1) + 0.28 * (1.0 - np.clip(rate, 0.0, 1.0)).reshape(1, -1) + 0.10 * leakage
        return mask * (base + 0.35 * topk)

    def _early_dispersion(self, condition: Any, leakage: np.ndarray, topk: np.ndarray, mask: np.ndarray) -> np.ndarray:
        meta = _metadata(condition)
        observed = int(meta.get("observed_patch", 0))
        early_end = int(meta.get("early_end_patch", max(1, int(0.4 * self.patch_num))))
        grid = np.arange(self.patch_num, dtype=np.float32)
        early = ((grid >= observed) & (grid < max(observed + 1, early_end))).astype(np.float32)
        if float(early.sum()) <= 0.0:
            early[: max(1, int(0.4 * self.patch_num))] = 1.0
        return mask * (0.55 * early.reshape(1, -1) + 0.45 * np.sqrt(np.maximum(leakage + topk, 0.0)))

    def _burst_boundary(self, condition: Any, leakage: np.ndarray, topk: np.ndarray, mask: np.ndarray) -> np.ndarray:
        burst = _condition_1d(condition, "burst_saliency", self.patch_num)
        if float(burst.max()) <= 1e-8:
            burst = _condition_1d(condition, "gap_saliency", self.patch_num)
        return mask * (0.72 * burst.reshape(1, -1) + 0.18 * leakage + 0.10 * topk)

    def _direction_balance(self, condition: Any, leakage: np.ndarray, topk: np.ndarray, mask: np.ndarray) -> np.ndarray:
        meta = _metadata(condition)
        out_ratio = float(meta.get("out_ratio", 0.5))
        in_ratio = float(meta.get("in_ratio", 0.5))
        direction_weight = np.asarray(
            [1.35 if in_ratio > out_ratio else 0.75, 1.35 if out_ratio > in_ratio else 0.75],
            dtype=np.float32,
        ).reshape(2, 1)
        saliency = _as_float_map(getattr(condition, "saliency", leakage), mask.shape)
        return mask * direction_weight * (0.60 * saliency + 0.25 * leakage + 0.15 * topk + 0.05)

    def _distribution_shaping(self, condition: Any, leakage: np.ndarray, topk: np.ndarray, mask: np.ndarray) -> np.ndarray:
        prototype = _condition_1d(condition, "public_prototype", self.patch_num)
        rate = _condition_1d(condition, "rate_saliency", self.patch_num)
        target = 0.65 * prototype + 0.35 * (1.0 - np.clip(rate, 0.0, 1.0))
        if float(target.max()) <= 1e-8:
            target = mask.mean(axis=0)
        return mask * (0.68 * target.reshape(1, -1) + 0.20 * leakage + 0.12 * topk)


@dataclass
class RandomPreferenceMixer:
    combination_sizes: Sequence[int] = (2,)
    seed: int = 0
    record_samples: bool = False
    records: list[dict] = field(default_factory=list)
    _usage: Counter = field(default_factory=Counter, init=False, repr=False)
    _combos: Counter = field(default_factory=Counter, init=False, repr=False)
    _sizes: Counter = field(default_factory=Counter, init=False, repr=False)
    _num_records: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        sizes = [int(item) for item in self.combination_sizes] or [2]
        self.combination_sizes = tuple(max(1, min(5, item)) for item in sizes)
        self.rng = np.random.default_rng(int(self.seed))

    def sample(
        self,
        preference_maps: dict[str, np.ndarray],
        *,
        sample_id: str = "",
        budget: float | None = None,
        combination_size: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        present = {canonical_preference(key) for key in preference_maps}
        keys = [key for key in PREFERENCE_NAMES if key in present]
        if not keys:
            raise ValueError("preference_maps is empty")
        size = int(combination_size) if combination_size is not None else int(self.rng.choice(self.combination_sizes))
        size = max(1, min(size, len(keys)))
        subset = self.rng.choice(keys, size=size, replace=False).tolist()
        weights = self.rng.dirichlet(np.ones(size, dtype=np.float64)).astype(np.float32)
        mixed = np.zeros_like(preference_maps[subset[0]], dtype=np.float32)
        for key, weight in zip(subset, weights):
            mixed += float(weight) * np.asarray(preference_maps[key], dtype=np.float32)
        record = {
            "sample_id": str(sample_id),
            "budget": None if budget is None else float(budget),
            "subset": subset,
            "subset_label": "+".join(subset),
            "weights": [float(item) for item in weights],
            "combination_size": int(size),
        }
        self._num_records += 1
        self._combos[record["subset_label"]] += 1
        self._sizes[str(record["combination_size"])] += 1
        for item in record["subset"]:
            self._usage[item] += 1
        if self.record_samples:
            self.records.append(record)
        return mixed.astype(np.float32), record

    def stats(self) -> dict:
        total = sum(self._combos.values())
        if total and len(self._combos) > 1:
            probs = np.asarray(list(self._combos.values()), dtype=np.float64) / float(total)
            entropy = float(-(probs * np.log(probs + 1e-12)).sum() / math.log(len(self._combos)))
        else:
            entropy = 0.0
        return {
            "num_records": int(self._num_records),
            "sample_records_saved": int(len(self.records)),
            "preference_usage": dict(self._usage),
            "combination_size_distribution": dict(self._sizes),
            "combination_counts": dict(self._combos),
            "combination_entropy": entropy,
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"stats": self.stats()}
        if self.record_samples:
            payload["records"] = self.records
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

