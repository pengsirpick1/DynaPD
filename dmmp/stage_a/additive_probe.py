"""Additive intervention probing utilities for Stage A masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .faithfulness import hard_top_ratio_masks
from .mask_ops import entropy_np, js_divergence_np


DIRECTION_NAMES = ("out", "in")
DIRECTION_MODES = ("out-only", "in-only", "both-equal", "current-ratio")
BASELINE_METHODS = (
    "random",
    "random_window",
    "early",
    "magnitude",
    "dynamask_same",
    "dynamask_causal",
)


@dataclass(frozen=True)
class CandidateWindow:
    sample_index: int
    window_id: int
    direction: int
    start: int
    end: int
    center: int
    mask_mass: float
    length: int

    @property
    def direction_name(self) -> str:
        return DIRECTION_NAMES[int(self.direction)]


@dataclass(frozen=True)
class ActionSpec:
    sample_index: int
    window_id: int
    affected_direction: int
    affected_start: int
    affected_end: int
    affected_center: int
    insert_start: int
    insert_end: int
    insert_center: int
    offset: int
    dose: int
    direction_mode: str
    counts: np.ndarray
    mask_mass: float
    local_rate_peak: int
    requires_incoming_capability: bool
    causal_violation: bool
    allowed_violation_count: int


def _as_bool_1d(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=bool).reshape(-1)


def close_binary_1d(values: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Morphological 1D closing with a small flat kernel."""
    binary = _as_bool_1d(values)
    width = max(1, int(kernel))
    if width <= 1 or binary.size == 0:
        return binary
    filt = np.ones(width, dtype=np.int32)
    dilated = np.convolve(binary.astype(np.int32), filt, mode="same") > 0
    closed = np.convolve(dilated.astype(np.int32), filt, mode="same") >= width
    return closed.astype(bool)


def intervals_from_binary(values: np.ndarray) -> list[tuple[int, int]]:
    binary = _as_bool_1d(values)
    if binary.size == 0 or not np.any(binary):
        return []
    padded = np.concatenate([[False], binary, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[index]), int(changes[index + 1])) for index in range(0, len(changes), 2)]


def merge_intervals(intervals: Iterable[tuple[int, int]], gap: int = 8) -> list[tuple[int, int]]:
    ordered = sorted((int(start), int(end)) for start, end in intervals if int(end) > int(start))
    if not ordered:
        return []
    merged = [ordered[0]]
    max_gap = int(gap)
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if int(start) - int(prev_end) <= max_gap:
            merged[-1] = (prev_start, max(prev_end, int(end)))
        else:
            merged.append((int(start), int(end)))
    return merged


def weighted_center(scores: np.ndarray, start: int, end: int) -> int:
    lo, hi = int(start), int(end)
    if hi <= lo:
        return lo
    weights = np.asarray(scores[lo:hi], dtype=np.float64)
    xs = np.arange(lo, hi, dtype=np.float64)
    total = float(weights.sum())
    if total <= 1e-12:
        return int(round((lo + hi - 1) / 2.0))
    return int(np.clip(round(float((weights * xs).sum() / total)), lo, hi - 1))


def candidate_windows_for_sample(
    soft_mask: np.ndarray,
    *,
    ratio: float = 0.10,
    closing_kernel: int = 5,
    merge_gap: int = 8,
    min_length: int = 1,
    min_mass: float = 0.0,
    max_windows: int = 6,
    sample_index: int = 0,
) -> list[CandidateWindow]:
    scores = np.asarray(soft_mask, dtype=np.float32)
    if scores.shape[0] != 2:
        raise ValueError(f"Expected soft mask [2, W], got {scores.shape}")
    hard = hard_top_ratio_masks(scores[None, :, :], float(ratio))[0] > 0
    windows: list[CandidateWindow] = []
    for direction in range(2):
        closed = close_binary_1d(hard[direction], int(closing_kernel))
        intervals = merge_intervals(intervals_from_binary(closed), int(merge_gap))
        for start, end in intervals:
            length = int(end) - int(start)
            mass = float(scores[direction, start:end].sum())
            if length < int(min_length) or mass < float(min_mass):
                continue
            windows.append(
                CandidateWindow(
                    sample_index=int(sample_index),
                    window_id=-1,
                    direction=int(direction),
                    start=int(start),
                    end=int(end),
                    center=weighted_center(scores[direction], int(start), int(end)),
                    mask_mass=mass,
                    length=length,
                )
            )
    windows = sorted(windows, key=lambda item: (-float(item.mask_mass), int(item.start), int(item.direction)))
    result = []
    for window_id, window in enumerate(windows[: max(0, int(max_windows))]):
        result.append(
            CandidateWindow(
                sample_index=window.sample_index,
                window_id=int(window_id),
                direction=window.direction,
                start=window.start,
                end=window.end,
                center=window.center,
                mask_mass=window.mask_mass,
                length=window.length,
            )
        )
    return result


def allocate_integer(total: int, weights: np.ndarray) -> np.ndarray:
    count = max(0, int(total))
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 0.0)
    if count <= 0:
        return np.zeros_like(w, dtype=np.int32)
    if float(w.sum()) <= 1e-12:
        w = np.ones_like(w, dtype=np.float64)
    w = w / float(w.sum())
    raw = w * float(count)
    out = np.floor(raw).astype(np.int32)
    remaining = int(count - int(out.sum()))
    if remaining > 0:
        order = np.argsort(-(raw - out), kind="mergesort")
        out[order[:remaining]] += 1
    return out.astype(np.int32)


def direction_counts_for_mode(
    total: int,
    mode: str,
    tam: np.ndarray,
    affected_start: int,
    affected_end: int,
    affected_direction: int,
) -> np.ndarray:
    name = str(mode)
    count = max(0, int(total))
    if name == "out-only":
        return np.asarray([count, 0], dtype=np.int32)
    if name == "in-only":
        return np.asarray([0, count], dtype=np.int32)
    if name == "both-equal":
        return allocate_integer(count, np.asarray([1.0, 1.0], dtype=np.float32))
    if name == "current-ratio":
        lo, hi = int(affected_start), int(affected_end)
        masses = np.asarray(np.asarray(tam, dtype=np.float32)[:, lo:hi].sum(axis=1), dtype=np.float64)
        if float(masses.sum()) <= 1e-12:
            masses = np.asarray([1.0, 1.0], dtype=np.float64)
            masses[int(affected_direction)] += 1.0
        return allocate_integer(count, masses)
    raise ValueError(f"Unknown direction mode: {mode!r}")


def counts_for_action(
    tam: np.ndarray,
    window: CandidateWindow,
    *,
    offset: int,
    dose: int,
    direction_mode: str,
) -> ActionSpec:
    values = np.asarray(tam, dtype=np.float32)
    width = int(values.shape[-1])
    insert_center = int(np.clip(int(window.center) + int(offset), 0, width - 1))
    action_width = max(1, min(width, int(window.length)))
    start = int(np.clip(insert_center - action_width // 2, 0, max(0, width - action_width)))
    end = int(start + action_width)
    per_direction = direction_counts_for_mode(
        int(dose),
        str(direction_mode),
        values,
        int(window.start),
        int(window.end),
        int(window.direction),
    )
    counts = np.zeros((2, width), dtype=np.int32)
    for direction in range(2):
        slot_counts = allocate_integer(int(per_direction[direction]), np.ones(action_width, dtype=np.float32))
        counts[direction, start:end] += slot_counts
    local_rate_peak = int(counts.sum(axis=0).max()) if counts.size else 0
    return ActionSpec(
        sample_index=int(window.sample_index),
        window_id=int(window.window_id),
        affected_direction=int(window.direction),
        affected_start=int(window.start),
        affected_end=int(window.end),
        affected_center=int(window.center),
        insert_start=int(start),
        insert_end=int(end),
        insert_center=int(insert_center),
        offset=int(offset),
        dose=int(dose),
        direction_mode=str(direction_mode),
        counts=counts,
        mask_mass=float(window.mask_mass),
        local_rate_peak=local_rate_peak,
        requires_incoming_capability=bool(counts[1].sum() > 0),
        causal_violation=bool(insert_center > int(window.center)),
        allowed_violation_count=0,
    )


def apply_counts(tam: np.ndarray, counts: np.ndarray) -> np.ndarray:
    return (np.asarray(tam, dtype=np.float32) + np.asarray(counts, dtype=np.float32)).astype(np.float32)


def probability_metrics(
    original_prob: np.ndarray,
    evaluated_prob: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    original = np.asarray(original_prob, dtype=np.float32)
    evaluated = np.asarray(evaluated_prob, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    original_pred = original.argmax(axis=1)
    evaluated_pred = evaluated.argmax(axis=1)
    original_top1 = original[np.arange(len(original)), original_pred]
    evaluated_original_top1 = evaluated[np.arange(len(original)), original_pred]
    original_sorted = np.sort(original, axis=1)
    evaluated_sorted = np.sort(evaluated, axis=1)
    original_margin = original_sorted[:, -1] - original_sorted[:, -2]
    evaluated_margin = evaluated_sorted[:, -1] - evaluated_sorted[:, -2]
    return {
        "accuracy": (evaluated_pred == y).astype(np.float32),
        "flip": (evaluated_pred != original_pred).astype(np.float32),
        "js_div": js_divergence_np(original, evaluated).astype(np.float32),
        "top1_drop": (original_top1 - evaluated_original_top1).astype(np.float32),
        "margin_drop": (original_margin - evaluated_margin).astype(np.float32),
        "entropy_gain": (entropy_np(evaluated) - entropy_np(original)).astype(np.float32),
        "original_pred": original_pred.astype(np.int64),
        "evaluated_pred": evaluated_pred.astype(np.int64),
    }


def channel_cosine_similarity(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    out = values[:, 0, :].reshape(values.shape[0], -1)
    inc = values[:, 1, :].reshape(values.shape[0], -1)
    denom = np.maximum(np.linalg.norm(out, axis=1) * np.linalg.norm(inc, axis=1), 1e-12)
    return ((out * inc).sum(axis=1) / denom).astype(np.float32)


def channel_l1_gap(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    return np.mean(np.abs(values[:, 0, :] - values[:, 1, :]), axis=1).astype(np.float32)


def nested_topr_audit(mask: np.ndarray, ratios: list[float]) -> tuple[list[dict], dict]:
    values = np.asarray(mask, dtype=np.float32)
    hard = [hard_top_ratio_masks(values, float(ratio)) > 0 for ratio in ratios]
    rows = []
    for index in range(len(ratios) - 1):
        prev = hard[index].reshape(values.shape[0], -1)
        curr = hard[index + 1].reshape(values.shape[0], -1)
        subset = np.all(np.logical_or(~prev, curr), axis=1)
        missing = np.logical_and(prev, ~curr).sum(axis=1)
        for sample in range(values.shape[0]):
            rows.append(
                {
                    "sample_index": int(sample),
                    "small_ratio": float(ratios[index]),
                    "large_ratio": float(ratios[index + 1]),
                    "nested": int(bool(subset[sample])),
                    "missing_positions": int(missing[sample]),
                }
            )
    summary = {
        "checked_pairs": int(len(rows)),
        "nested_pair_rate": float(np.mean([row["nested"] for row in rows])) if rows else 0.0,
        "violation_count": int(sum(1 for row in rows if int(row["nested"]) == 0)),
    }
    return rows, summary


def monotonic_violation_rows(sample_metrics: dict[str, np.ndarray], ratios: np.ndarray, sample_ids: np.ndarray) -> tuple[list[dict], dict]:
    ratio_values = np.asarray(ratios, dtype=np.float32)
    ids = np.asarray(sample_ids).astype(str)
    rows = []
    metric_specs = [
        ("necessity_js_div", "nondecreasing"),
        ("necessity_top1_drop", "nondecreasing"),
        ("necessity_correct", "nonincreasing"),
    ]
    for metric, expected in metric_specs:
        if metric not in sample_metrics:
            continue
        values = np.asarray(sample_metrics[metric], dtype=np.float32)
        for sample in range(values.shape[1]):
            series = values[:, sample]
            diffs = series[1:] - series[:-1]
            if expected == "nondecreasing":
                violated = diffs < -1e-6
            else:
                violated = diffs > 1e-6
            rows.append(
                {
                    "sample_index": int(sample),
                    "sample_id": str(ids[sample]) if sample < len(ids) else str(sample),
                    "metric": metric,
                    "expected": expected,
                    "violation_count": int(np.sum(violated)),
                    "has_violation": int(bool(np.any(violated))),
                    "values": ";".join(f"{float(item):.8f}" for item in series),
                    "ratios": ";".join(f"{float(item):.4f}" for item in ratio_values),
                }
            )
    summary = {}
    for metric, _expected in metric_specs:
        metric_rows = [row for row in rows if row["metric"] == metric]
        summary[f"{metric}_sample_violation_rate"] = (
            float(np.mean([row["has_violation"] for row in metric_rows])) if metric_rows else None
        )
        summary[f"{metric}_total_adjacent_violation_count"] = int(sum(int(row["violation_count"]) for row in metric_rows))
    return rows, summary


def empty_heatmaps(n: int, width: int) -> dict[str, np.ndarray]:
    shape = (int(n), 2, int(width))
    return {
        "additive_efficiency": np.full(shape, np.nan, dtype=np.float32),
        "minimum_effective_budget": np.full(shape, np.nan, dtype=np.float32),
        "best_causal_offset": np.full(shape, np.nan, dtype=np.float32),
        "best_insert_position": np.full(shape, np.nan, dtype=np.float32),
    }


def update_sparse_best(
    current: np.ndarray,
    sample: int,
    direction: int,
    position: int,
    value: float,
    *,
    higher_is_better: bool = True,
) -> None:
    old = current[int(sample), int(direction), int(position)]
    if np.isnan(old) or (higher_is_better and float(value) > float(old)) or ((not higher_is_better) and float(value) < float(old)):
        current[int(sample), int(direction), int(position)] = float(value)
