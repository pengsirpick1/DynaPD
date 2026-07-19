"""Self-contained prefix condition extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PrefixCondition:
    vector: np.ndarray
    allowed_mask: np.ndarray
    optimistic_allowed_mask: np.ndarray
    saliency: np.ndarray
    gap_saliency: np.ndarray
    rate_saliency: np.ndarray
    burst_saliency: np.ndarray
    public_prototype: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)


def nonzero_trace(trace: np.ndarray) -> np.ndarray:
    values = np.asarray(trace, dtype=np.float32).reshape(-1)
    return values[values != 0]


def normalize01(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float32)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _resample(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(int(size), dtype=np.float32)
    if values.size == 1:
        return np.full(int(size), float(values[0]), dtype=np.float32)
    source = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    target = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
    return np.interp(target, source, values).astype(np.float32)


def tam_patch_ids(values: np.ndarray, patch_num: int, *, max_load_time: float = 80.0) -> np.ndarray:
    times = np.abs(np.asarray(values, dtype=np.float32).reshape(-1))
    if times.size == 0:
        return np.asarray([], dtype=np.int64)
    scale = float(int(patch_num)) / max(float(max_load_time), 1e-6)
    patch_ids = np.floor(times * scale).astype(np.int64)
    patch_ids[times >= float(max_load_time)] = int(patch_num) - 1
    return np.clip(patch_ids, 0, int(patch_num) - 1)


def tam_patch_center_slots(patch_num: int, num_slots: int, *, max_load_time: float = 80.0) -> np.ndarray:
    centers = (np.arange(int(patch_num), dtype=np.float32) + 0.5) * float(max_load_time) / max(int(patch_num), 1)
    scale = float(int(num_slots) - 1) / max(float(max_load_time), 1e-6)
    slots = np.floor(centers * scale).astype(np.int64)
    return np.clip(slots, 0, int(num_slots) - 1)


def observed_patch_from_prefix(prefix: np.ndarray, patch_num: int, *, max_load_time: float = 80.0) -> int:
    values = nonzero_trace(prefix)
    if values.size == 0:
        return 0
    latest_time = float(np.max(np.abs(values)))
    patch = int(tam_patch_ids(np.asarray([latest_time], dtype=np.float32), int(patch_num), max_load_time=float(max_load_time))[0])
    return int(np.clip(patch + 1, 0, int(patch_num)))


def prefix_patch_counts(
    trace: np.ndarray,
    prefix_n: int,
    patch_num: int,
    *,
    max_trace_length: int = 5000,
    max_load_time: float = 80.0,
) -> np.ndarray:
    del max_trace_length
    prefix = nonzero_trace(trace)[: int(prefix_n)]
    counts = np.zeros((2, int(patch_num)), dtype=np.float32)
    if prefix.size == 0:
        return counts
    directions = np.sign(prefix)
    patch_ids = tam_patch_ids(prefix, int(patch_num), max_load_time=float(max_load_time))
    for direction, patch in zip(directions, patch_ids):
        counts[0 if direction > 0 else 1, int(patch)] += 1.0
    return counts


def _smooth(values: np.ndarray, radius: int = 2) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0 or radius <= 0:
        return arr.astype(np.float32)
    kernel = np.ones(2 * int(radius) + 1, dtype=np.float32)
    kernel /= float(kernel.sum())
    return np.convolve(arr, kernel, mode="same").astype(np.float32)


def extract_prefix_condition(
    trace: np.ndarray,
    prefix_n: int = 500,
    patch_num: int = 200,
    *,
    max_trace_length: int = 5000,
    max_load_time: float = 80.0,
    early_fraction: float = 0.40,
) -> PrefixCondition:
    clean = nonzero_trace(trace)
    prefix = clean[: int(prefix_n)]
    directions = np.sign(prefix).astype(np.float32)
    counts = prefix_patch_counts(
        prefix,
        int(prefix_n),
        int(patch_num),
        max_trace_length=int(max_trace_length),
        max_load_time=float(max_load_time),
    )
    patch_total = counts.sum(axis=0).astype(np.float32)
    prefix_len = int(prefix.size)
    observed_patch = observed_patch_from_prefix(prefix, int(patch_num), max_load_time=float(max_load_time))
    early_end = max(observed_patch + 1, int(math.ceil(float(early_fraction) * int(patch_num))))
    early_end = min(int(patch_num), early_end)

    allowed = np.zeros((2, int(patch_num)), dtype=np.float32)
    optimistic = np.zeros_like(allowed, dtype=np.float32)
    optimistic[:, :early_end] = 1.0
    if observed_patch < early_end:
        allowed[:, observed_patch:early_end] = 1.0
        mask_fallback = False
    else:
        allowed[:, min(observed_patch, int(patch_num) - 1)] = 1.0
        mask_fallback = True

    rate = normalize01(_smooth(patch_total, radius=2))
    gap = 1.0 - rate
    gap[: min(observed_patch, gap.size)] *= 0.25
    if np.any(allowed[0] > 0):
        gap = normalize01(gap * allowed.mean(axis=0))

    burst = np.zeros(int(patch_num), dtype=np.float32)
    if directions.size > 1:
        changes = np.flatnonzero(directions[1:] != directions[:-1]) + 1
        patch_ids = tam_patch_ids(prefix[changes], int(patch_num), max_load_time=float(max_load_time))
        for patch in patch_ids:
            burst[int(patch)] += 1.0
    burst = normalize01(_smooth(burst, radius=3))

    saliency = np.zeros((2, int(patch_num)), dtype=np.float32)
    if float(counts.max()) > 1e-8:
        saliency = counts / float(counts.max())
    saliency = 0.55 * saliency + 0.45 * np.repeat(rate.reshape(1, -1), 2, axis=0)
    saliency = saliency.astype(np.float32)

    prototype = normalize01(_smooth(patch_total + 1.0, radius=4))
    if float(prototype.max()) <= 1e-8:
        prototype = np.ones(int(patch_num), dtype=np.float32)

    out_count = float(np.sum(directions > 0))
    in_count = float(np.sum(directions < 0))
    denom = max(float(prefix_len), 1.0)
    scalars = np.asarray(
        [
            prefix_len / max(float(prefix_n), 1.0),
            prefix_len / max(float(max_trace_length), 1.0),
            out_count / denom,
            in_count / denom,
            (out_count - in_count) / denom,
            float(observed_patch) / max(float(patch_num), 1.0),
            float(early_end) / max(float(patch_num), 1.0),
            float(np.mean(np.abs(prefix))) if prefix_len else 0.0,
        ],
        dtype=np.float32,
    )
    vector = np.concatenate(
        [
            scalars,
            (counts.reshape(-1) / max(float(prefix_n), 1.0)).astype(np.float32),
            gap.astype(np.float32),
            rate.astype(np.float32),
            burst.astype(np.float32),
            prototype.astype(np.float32),
        ],
        axis=0,
    ).astype(np.float32)
    return PrefixCondition(
        vector=vector,
        allowed_mask=allowed,
        optimistic_allowed_mask=optimistic,
        saliency=saliency,
        gap_saliency=gap.astype(np.float32),
        rate_saliency=rate.astype(np.float32),
        burst_saliency=burst.astype(np.float32),
        public_prototype=prototype.astype(np.float32),
        metadata={
            "prefix_len": prefix_len,
            "observed_patch": observed_patch,
            "early_end_patch": early_end,
            "out_ratio": out_count / denom,
            "in_ratio": in_count / denom,
            "mask_fallback": mask_fallback,
            "patch_coordinate": "rf_tam_time",
            "tam_patch_max_load_time": float(max_load_time),
        },
    )

