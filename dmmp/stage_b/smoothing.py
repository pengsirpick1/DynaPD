"""Keypoint-guided causal smoothing transforms for Stage B2-S."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from dmmp.encoders.prefix import nonzero_trace
from dmmp.stage_a.additive_probe import CandidateWindow, allocate_integer, candidate_windows_for_sample


@dataclass(frozen=True)
class SmoothingConfig:
    method: str
    window_length: int
    rho: float
    dummy_budget: float = 0.0
    max_delay: int = 0


@dataclass
class SmoothingResult:
    tam: np.ndarray
    trace: np.ndarray | None
    dummy_counts: np.ndarray
    average_delay_bins: float
    maximum_delay_bins: int
    local_variance_reduction: float
    local_gradient_reduction: float
    actual_bandwidth: float
    renderer_mode: str
    renderer_consistency: float


def keypoint_windows(
    soft_mask: np.ndarray,
    *,
    ratio: float = 0.10,
    max_windows: int = 8,
    closing_kernel: int = 5,
    merge_gap: int = 8,
    sample_index: int = 0,
) -> list[CandidateWindow]:
    return candidate_windows_for_sample(
        np.asarray(soft_mask, dtype=np.float32),
        ratio=float(ratio),
        closing_kernel=int(closing_kernel),
        merge_gap=int(merge_gap),
        max_windows=int(max_windows),
        sample_index=int(sample_index),
    )


def trace_to_tam(trace: np.ndarray, *, width: int, max_load_time: float) -> np.ndarray:
    clean = nonzero_trace(trace)
    tam = np.zeros((2, int(width)), dtype=np.float32)
    if clean.size == 0:
        return tam
    scale = float(int(width) - 1) / max(float(max_load_time), 1e-6)
    outgoing = clean[clean > 0]
    incoming = -clean[clean < 0]
    if outgoing.size:
        slots = np.floor(outgoing * scale).astype(np.int64)
        slots[outgoing >= float(max_load_time)] = int(width) - 1
        np.add.at(tam[0], np.clip(slots, 0, int(width) - 1), 1.0)
    if incoming.size:
        slots = np.floor(incoming * scale).astype(np.int64)
        slots[incoming >= float(max_load_time)] = int(width) - 1
        np.add.at(tam[1], np.clip(slots, 0, int(width) - 1), 1.0)
    return tam.astype(np.float32)


def _bin_centers(width: int, max_load_time: float) -> np.ndarray:
    return (np.arange(int(width), dtype=np.float32) + 0.5) * float(max_load_time) / max(int(width), 1)


def _slot_for_time(times: np.ndarray, *, width: int, max_load_time: float) -> np.ndarray:
    scale = float(int(width) - 1) / max(float(max_load_time), 1e-6)
    slots = np.floor(np.asarray(times, dtype=np.float32) * scale).astype(np.int64)
    slots[np.asarray(times) >= float(max_load_time)] = int(width) - 1
    return np.clip(slots, 0, int(width) - 1)


def _future_region(center: int, length: int, width: int, *, include_center: bool = False) -> tuple[int, int]:
    lo = int(center) if include_center else int(center) + 1
    lo = int(np.clip(lo, 0, int(width) - 1))
    hi = int(np.clip(int(center) + int(length) + 1, lo + 1, int(width)))
    return lo, hi


def _neighborhood(center: int, length: int, width: int, *, causal: bool) -> tuple[int, int]:
    if causal:
        return int(np.clip(center, 0, width - 1)), int(np.clip(center + length + 1, 1, width))
    lo = int(np.clip(center - length, 0, width - 1))
    hi = int(np.clip(center + length + 1, lo + 1, width))
    return lo, hi


def _contrast_need(row: np.ndarray, center: int, length: int, rho: float) -> tuple[np.ndarray, int]:
    width = int(row.size)
    center = int(np.clip(center, 0, width - 1))
    lo, hi = _future_region(center, length, width, include_center=False)
    future = np.asarray(row[lo:hi], dtype=np.float32)
    if future.size == 0:
        return np.zeros(0, dtype=np.float64), lo
    peak = float(row[center])
    current_mean = float(future.mean())
    target_mean = float(rho) * peak + (1.0 - float(rho)) * current_mean
    deficits = np.maximum(float(target_mean) - future.astype(np.float64), 0.0)
    return deficits, lo


def add_only_dummy_counts(
    tam: np.ndarray,
    windows: Iterable[CandidateWindow],
    *,
    length: int,
    rho: float,
    budget: float,
    clean_total: float,
    direct_same_position: bool = False,
) -> np.ndarray:
    values = np.asarray(tam, dtype=np.float32)
    width = int(values.shape[1])
    target_dummy = max(0, int(round(float(clean_total) * float(budget))))
    counts = np.zeros_like(values, dtype=np.int32)
    if target_dummy <= 0:
        return counts
    weighted_cells: list[tuple[float, int, int]] = []
    for window in windows:
        direction = int(window.direction)
        if direct_same_position:
            center = int(np.clip(window.center, 0, width - 1))
            weighted_cells.append((float(window.mask_mass) + float(values[direction, center]) + 1.0, direction, center))
            continue
        deficits, offset = _contrast_need(values[direction], int(window.center), int(length), float(rho))
        for index, deficit in enumerate(deficits.tolist()):
            if float(deficit) <= 0.0:
                continue
            weighted_cells.append((float(deficit) * (float(window.mask_mass) + 1.0), direction, int(offset + index)))
    if not weighted_cells:
        return counts
    weights = np.asarray([max(item[0], 1e-6) for item in weighted_cells], dtype=np.float64)
    allocated = allocate_integer(target_dummy, weights)
    for amount, (_weight, direction, slot) in zip(allocated.tolist(), weighted_cells):
        counts[int(direction), int(slot)] += int(amount)
    return counts.astype(np.int32)


def noncausal_symmetric_tam(
    tam: np.ndarray,
    windows: Iterable[CandidateWindow],
    *,
    length: int,
    rho: float,
) -> np.ndarray:
    result = np.asarray(tam, dtype=np.float32).copy()
    width = int(result.shape[1])
    for window in windows:
        direction = int(window.direction)
        lo, hi = _neighborhood(int(window.center), int(length), width, causal=False)
        patch = result[direction, lo:hi].astype(np.float64)
        total = int(round(float(patch.sum())))
        if total <= 0:
            continue
        uniform = np.ones_like(patch, dtype=np.float64) / max(int(patch.size), 1)
        current = patch / max(float(patch.sum()), 1e-8)
        target = (1.0 - float(rho)) * current + float(rho) * uniform
        result[direction, lo:hi] = allocate_integer(total, target).astype(np.float32)
    return result.astype(np.float32)


def _delay_kernel(max_delay: int, rho: float) -> np.ndarray:
    delay = max(0, int(max_delay))
    if delay <= 0 or float(rho) <= 0.0:
        return np.asarray([1.0], dtype=np.float64)
    weights = np.exp(-np.arange(1, delay + 1, dtype=np.float64) / max(delay / 2.0, 1.0))
    weights = weights / max(float(weights.sum()), 1e-12)
    kernel = np.zeros(delay + 1, dtype=np.float64)
    kernel[0] = 1.0 - float(rho)
    kernel[1:] = float(rho) * weights
    return kernel / max(float(kernel.sum()), 1e-12)


def _empty_delay_stats(direction_policy: str) -> dict:
    return {
        "direction_policy": str(direction_policy),
        "delay_values": [],
        "outgoing_delay_values": [],
        "incoming_delay_values": [],
        "delayed_packet_count": 0,
        "delayed_packet_fraction": 0.0,
        "p95_delay_bins": 0.0,
        "outgoing_packet_count": 0,
        "incoming_packet_count": 0,
        "outgoing_delay_packet_count": 0,
        "incoming_delay_packet_count": 0,
        "outgoing_delay_fraction": 0.0,
        "incoming_delay_fraction": 0.0,
        "outgoing_average_delay_bins": 0.0,
        "incoming_average_delay_bins": 0.0,
        "outgoing_p95_delay_bins": 0.0,
        "incoming_p95_delay_bins": 0.0,
        "outgoing_max_delay_bins": 0,
        "incoming_max_delay_bins": 0,
    }


def _delay_stats(delays: np.ndarray, signs: np.ndarray, *, direction_policy: str) -> dict:
    stats = _empty_delay_stats(str(direction_policy))
    moved = np.asarray(delays, dtype=np.int32) > 0
    all_values = np.asarray(delays[moved], dtype=np.int32)
    outgoing = moved & (np.asarray(signs) == 1.0)
    incoming = moved & (np.asarray(signs) == -1.0)
    outgoing_values = np.asarray(delays[outgoing], dtype=np.int32)
    incoming_values = np.asarray(delays[incoming], dtype=np.int32)
    outgoing_total = int(np.sum(np.asarray(signs) == 1.0))
    incoming_total = int(np.sum(np.asarray(signs) == -1.0))
    stats.update(
        {
            "delay_values": [int(item) for item in all_values.tolist()],
            "outgoing_delay_values": [int(item) for item in outgoing_values.tolist()],
            "incoming_delay_values": [int(item) for item in incoming_values.tolist()],
            "delayed_packet_count": int(all_values.size),
            "delayed_packet_fraction": float(all_values.size / max(int(delays.size), 1)),
            "p95_delay_bins": float(np.percentile(all_values, 95)) if all_values.size else 0.0,
            "outgoing_packet_count": outgoing_total,
            "incoming_packet_count": incoming_total,
            "outgoing_delay_packet_count": int(outgoing_values.size),
            "incoming_delay_packet_count": int(incoming_values.size),
            "outgoing_delay_fraction": float(outgoing_values.size / max(outgoing_total, 1)),
            "incoming_delay_fraction": float(incoming_values.size / max(incoming_total, 1)),
            "outgoing_average_delay_bins": float(outgoing_values.mean()) if outgoing_values.size else 0.0,
            "incoming_average_delay_bins": float(incoming_values.mean()) if incoming_values.size else 0.0,
            "outgoing_p95_delay_bins": float(np.percentile(outgoing_values, 95)) if outgoing_values.size else 0.0,
            "incoming_p95_delay_bins": float(np.percentile(incoming_values, 95)) if incoming_values.size else 0.0,
            "outgoing_max_delay_bins": int(outgoing_values.max()) if outgoing_values.size else 0,
            "incoming_max_delay_bins": int(incoming_values.max()) if incoming_values.size else 0,
        }
    )
    return stats


def causal_delay_trace(
    raw_trace: np.ndarray,
    windows: Iterable[CandidateWindow],
    *,
    width: int,
    length: int,
    rho: float,
    max_delay: int,
    max_load_time: float,
    direction_policy: str = "bidirectional",
    return_stats: bool = False,
) -> tuple[np.ndarray, float, int]:
    clean = nonzero_trace(raw_trace)
    policy = str(direction_policy)
    if policy not in {"outgoing_only", "incoming_only", "bidirectional"}:
        raise ValueError(f"Unsupported direction_policy={direction_policy!r}")
    if clean.size == 0 or int(max_delay) <= 0 or float(rho) <= 0.0:
        if bool(return_stats):
            return clean.astype(np.float32), 0.0, 0, _empty_delay_stats(policy)
        return clean.astype(np.float32), 0.0, 0
    centers = _bin_centers(width, max_load_time)
    times = np.abs(clean)
    signs = np.sign(clean)
    slots = _slot_for_time(times, width=width, max_load_time=max_load_time)
    eligible = np.zeros(clean.size, dtype=bool)
    delay_cap = min(int(length), int(max_delay))
    for window in windows:
        if policy == "outgoing_only" and int(window.direction) != 0:
            continue
        if policy == "incoming_only" and int(window.direction) != 1:
            continue
        direction_sign = 1.0 if int(window.direction) == 0 else -1.0
        lo = int(window.start)
        hi = int(window.end)
        eligible |= (signs == direction_sign) & (slots >= lo) & (slots < hi)
    new_slots = slots.copy()
    delays = np.zeros(clean.size, dtype=np.int32)
    kernel = _delay_kernel(delay_cap, float(rho))
    signs_to_delay = (1.0,) if policy == "outgoing_only" else (-1.0,) if policy == "incoming_only" else (1.0, -1.0)
    for direction_sign in signs_to_delay:
        for slot in np.unique(slots[(signs == direction_sign) & eligible]):
            idx = np.flatnonzero((slots == int(slot)) & (signs == direction_sign) & eligible)
            if idx.size == 0:
                continue
            assigned = allocate_integer(int(idx.size), kernel)
            cursor = 0
            for delay, amount in enumerate(assigned.tolist()):
                if int(amount) <= 0:
                    continue
                take = idx[cursor : cursor + int(amount)]
                cursor += int(amount)
                new_slots[take] = np.clip(int(slot) + int(delay), 0, int(width) - 1)
                delays[take] = int(delay)
    new_times = centers[new_slots]
    defended = signs * new_times
    order = np.argsort(np.abs(defended), kind="mergesort")
    moved = delays[delays > 0]
    avg_delay = float(moved.mean()) if moved.size else 0.0
    max_seen = int(moved.max()) if moved.size else 0
    if bool(return_stats):
        return defended[order].astype(np.float32), avg_delay, max_seen, _delay_stats(delays, signs, direction_policy=policy)
    return defended[order].astype(np.float32), avg_delay, max_seen


def local_reductions(
    before: np.ndarray,
    after: np.ndarray,
    windows: Iterable[CandidateWindow],
    *,
    length: int,
    causal: bool = True,
) -> tuple[float, float]:
    original = np.asarray(before, dtype=np.float32)
    defended = np.asarray(after, dtype=np.float32)
    var_before: list[float] = []
    var_after: list[float] = []
    grad_before: list[float] = []
    grad_after: list[float] = []
    width = int(original.shape[1])
    for window in windows:
        direction = int(window.direction)
        lo, hi = _neighborhood(int(window.center), int(length), width, causal=bool(causal))
        a = original[direction, lo:hi]
        b = defended[direction, lo:hi]
        if a.size <= 1:
            continue
        weight = max(float(window.mask_mass), 1e-6)
        var_before.append(weight * float(np.var(a)))
        var_after.append(weight * float(np.var(b)))
        grad_before.append(weight * float(np.mean(np.abs(np.diff(a)))))
        grad_after.append(weight * float(np.mean(np.abs(np.diff(b)))))
    vb = float(np.sum(var_before))
    va = float(np.sum(var_after))
    gb = float(np.sum(grad_before))
    ga = float(np.sum(grad_after))
    var_reduction = float((vb - va) / max(vb, 1.0))
    grad_reduction = float((gb - ga) / max(gb, 1.0))
    return var_reduction, grad_reduction
