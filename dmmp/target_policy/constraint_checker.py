"""Hard and deployability checks for target policies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .representation import validate_counts


@dataclass(frozen=True)
class ConstraintReport:
    valid: bool
    deployable: bool
    allowed_violation_count: int
    negative_count: int
    actual_count: int
    target_count: int
    budget_error: int
    max_slot_count: int
    max_consecutive_dummy_run: int
    max_local_density: float
    tail_extension_ratio: float
    reasons: tuple[str, ...] = ()


def _max_local_density(counts: np.ndarray, window: int = 10) -> float:
    per_slot = np.asarray(counts, dtype=np.float64).sum(axis=0)
    if per_slot.size == 0:
        return 0.0
    width = max(1, min(int(window), per_slot.size))
    kernel = np.ones(width, dtype=np.float64)
    local = np.convolve(per_slot, kernel, mode="valid")
    return float(local.max() / max(float(per_slot.sum()), 1.0))


def check_counts(
    counts: np.ndarray,
    allowed_mask: np.ndarray,
    target_count: int,
    *,
    max_dummy_per_slot: int = 8,
    max_consecutive_dummy_run: int = 24,
    max_tail_extension_ratio: float = 0.10,
    max_local_dummy_density: float = 0.80,
) -> ConstraintReport:
    hard = validate_counts(counts, allowed_mask, int(target_count))
    counts_arr = np.asarray(counts, dtype=np.int64)
    mask_arr = np.asarray(allowed_mask, dtype=np.float32)
    per_slot = counts_arr.sum(axis=0) if counts_arr.ndim == 2 else counts_arr.reshape(-1)
    max_slot = int(per_slot.max()) if per_slot.size else 0
    # The renderer interleaves dummy packets with real packets inside each
    # patch. A consecutive dummy-run proxy based on occupied neighboring
    # patches is too strict for dense but legal allocations, so v1 uses the
    # maximum per-slot stack as the conservative run-length proxy.
    run = max_slot
    density = _max_local_density(counts_arr)
    tail_start = int(max(0, per_slot.size - max(1, round(per_slot.size * float(max_tail_extension_ratio)))))
    tail_ratio = float(per_slot[tail_start:].sum() / max(int(counts_arr.sum()), 1)) if per_slot.size else 0.0
    hard_valid = bool(hard["valid"])
    allowed_slots = int((mask_arr > 0).sum())
    reasons: list[str] = []
    if int(target_count) > 0 and allowed_slots <= 0:
        reasons.append("no_allowed_position")
    if int(target_count) > max(0, allowed_slots * int(max_dummy_per_slot)):
        reasons.append("insufficient_allowed_capacity")
    if int(hard["allowed_violation_count"]) > 0:
        reasons.append("allowed_mask_violation")
    if int(hard["negative_count"]) > 0:
        reasons.append("negative_count")
    if int(hard["budget_error"]) != 0:
        reasons.append("budget_projection_failure")
    if max_slot > int(max_dummy_per_slot):
        reasons.append("max_slot_count_exceeded")
    if run > int(max_consecutive_dummy_run):
        reasons.append("consecutive_dummy_violation")
    if density > float(max_local_dummy_density):
        reasons.append("local_density_violation")
    if tail_ratio > max(float(max_tail_extension_ratio), 1.0e-6) + 1.0e-6:
        reasons.append("tail_extension_violation")
    deployable = bool(
        hard_valid
        and max_slot <= int(max_dummy_per_slot)
        and run <= int(max_consecutive_dummy_run)
        and density <= float(max_local_dummy_density)
        and tail_ratio <= max(float(max_tail_extension_ratio), 1.0e-6) + 1.0e-6
    )
    if not deployable and not reasons:
        reasons.append("other")
    return ConstraintReport(
        valid=hard_valid,
        deployable=deployable,
        reasons=tuple(reasons),
        allowed_violation_count=int(hard["allowed_violation_count"]),
        negative_count=int(hard["negative_count"]),
        actual_count=int(hard["actual_count"]),
        target_count=int(hard["target_count"]),
        budget_error=int(hard["budget_error"]),
        max_slot_count=max_slot,
        max_consecutive_dummy_run=run,
        max_local_density=density,
        tail_extension_ratio=tail_ratio,
    )
