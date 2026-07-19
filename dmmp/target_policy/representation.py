"""Numerically stable x0* representation helpers.

x0* is stored as a masked centered log-ratio (CLR) transform of a continuous
allocation over the legal [2, patch_num] padding positions.
"""

from __future__ import annotations

import numpy as np


def _as_mask(mask: np.ndarray, shape: tuple[int, ...] | None = None) -> np.ndarray:
    arr = np.asarray(mask, dtype=np.float32)
    if shape is not None and arr.shape != shape:
        arr = np.resize(arr, shape).astype(np.float32)
    return (arr > 0).astype(np.float32)


def normalize_positive(values: np.ndarray, mask: np.ndarray, eps: float = 1.0e-12) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    mask_arr = _as_mask(mask, arr.shape).astype(np.float64)
    arr *= mask_arr
    total = float(arr.sum())
    if total <= float(eps):
        valid = mask_arr > 0
        out = np.zeros_like(arr, dtype=np.float64)
        if np.any(valid):
            out[valid] = 1.0 / float(valid.sum())
        return out.astype(np.float32)
    return (arr / total).astype(np.float32)


def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    mask_arr = _as_mask(mask, values.shape).astype(np.float64)
    valid = mask_arr > 0
    output = np.zeros_like(values, dtype=np.float64)
    if not np.any(valid):
        return output.astype(np.float32)
    temp = max(float(temperature), 1.0e-6)
    selected = values[valid] / temp
    selected = selected - float(np.max(selected))
    exp = np.exp(np.clip(selected, -60.0, 60.0))
    denom = float(exp.sum())
    output[valid] = exp / max(denom, 1.0e-12)
    return output.astype(np.float32)


def largest_remainder_rounding(allocation: np.ndarray, target_count: int, mask: np.ndarray) -> np.ndarray:
    target = max(0, int(target_count))
    mask_arr = _as_mask(mask, np.asarray(allocation).shape)
    valid = mask_arr.reshape(-1) > 0
    counts = np.zeros(mask_arr.size, dtype=np.int32)
    if target == 0 or not np.any(valid):
        return counts.reshape(mask_arr.shape)
    probs = normalize_positive(allocation, mask_arr).reshape(-1)
    raw = probs * float(target)
    base = np.floor(raw).astype(np.int32)
    base[~valid] = 0
    remaining = int(target - int(base.sum()))
    if remaining > 0:
        fractional = raw - base
        fractional[~valid] = -1.0
        order = np.argsort(-fractional, kind="mergesort")
        base[order[:remaining]] += 1
    elif remaining < 0:
        order = np.argsort(raw - base, kind="mergesort")
        for index in order:
            if remaining == 0:
                break
            if valid[index] and base[index] > 0:
                base[index] -= 1
                remaining += 1
    counts[:] = base
    return counts.reshape(mask_arr.shape).astype(np.int32)


def counts_to_allocation(
    counts: np.ndarray,
    mask: np.ndarray,
    clr_epsilon: float = 1.0e-6,
) -> np.ndarray:
    counts_arr = np.maximum(np.asarray(counts, dtype=np.float64), 0.0)
    mask_arr = _as_mask(mask, counts_arr.shape).astype(np.float64)
    allocation = (counts_arr + float(clr_epsilon)) * mask_arr
    return normalize_positive(allocation, mask_arr)


def allocation_to_x0_star(
    allocation: np.ndarray,
    mask: np.ndarray,
    clr_epsilon: float = 1.0e-6,
) -> np.ndarray:
    alloc = normalize_positive(allocation, mask, eps=float(clr_epsilon) * 0.01).astype(np.float64)
    mask_arr = _as_mask(mask, alloc.shape).astype(np.float64)
    valid = mask_arr > 0
    out = np.zeros_like(alloc, dtype=np.float64)
    if not np.any(valid):
        return out.astype(np.float32)
    log_a = np.log(np.maximum(alloc, 0.0) + float(clr_epsilon))
    mean_log = float(log_a[valid].mean())
    out[valid] = log_a[valid] - mean_log
    return out.astype(np.float32)


def counts_to_x0_star(
    counts: np.ndarray,
    mask: np.ndarray,
    clr_epsilon: float = 1.0e-6,
) -> tuple[np.ndarray, np.ndarray]:
    allocation = counts_to_allocation(counts, mask, clr_epsilon=float(clr_epsilon))
    return allocation_to_x0_star(allocation, mask, clr_epsilon=float(clr_epsilon)), allocation


def x0_star_to_allocation(x0_star: np.ndarray, mask: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return masked_softmax(x0_star, mask, temperature=float(temperature))


def x0_star_to_counts(
    x0_star: np.ndarray,
    mask: np.ndarray,
    target_count: int,
    temperature: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    allocation = x0_star_to_allocation(x0_star, mask, temperature=float(temperature))
    counts = largest_remainder_rounding(allocation, int(target_count), mask)
    return counts, allocation


def validate_counts(counts: np.ndarray, mask: np.ndarray, target_count: int) -> dict[str, int | bool]:
    counts_arr = np.asarray(counts, dtype=np.int64)
    mask_arr = _as_mask(mask, counts_arr.shape)
    violation = int(np.maximum(counts_arr, 0)[mask_arr <= 0].sum())
    negative = int(np.abs(np.minimum(counts_arr, 0)).sum())
    total = int(counts_arr.sum())
    return {
        "valid": bool(violation == 0 and negative == 0 and total == int(target_count)),
        "allowed_violation_count": violation,
        "negative_count": negative,
        "actual_count": total,
        "target_count": int(target_count),
        "budget_error": total - int(target_count),
    }
