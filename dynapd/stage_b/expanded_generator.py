"""Expanded causal and structural action generation for Stage B1."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import torch

from dynapd.stage_a.additive_probe import CandidateWindow, allocate_integer, candidate_windows_for_sample
from dynapd.stage_b.action_selector import CandidateAction


DIRECTION_MODES = ("out-only", "in-only", "both-equal", "current-ratio", "direction-balance")
TIERS = ("primary", "secondary", "exploration")
ACTION_TYPE_BONUS = {
    "dynamask_causal": 1.00,
    "shared_predecessor": 0.90,
    "two_window_coordinated_insert": 0.85,
    "gap_fill": 0.75,
    "burst_merge": 0.75,
    "rate_peak_predecessor": 0.65,
    "direction_balance": 0.60,
    "local_rate_smoothing": 0.55,
    "burst_extension": 0.50,
    "cumulative_shift": 0.45,
}
TIER_BONUS = {"primary": 1.0, "secondary": 0.65, "exploration": 0.30}
DETAILED_TIMING_KEYS = {
    "anchor_grid_build_time_sec",
    "action_object_build_time_sec",
    "legality_filter_time_sec",
    "pair_action_build_time_sec",
    "score_hint_sort_time_sec",
    "diverse_limit_time_sec",
}


@dataclass(frozen=True)
class SparseCounts:
    width: int
    directions: np.ndarray
    bins: np.ndarray
    values: np.ndarray
    dummy_count: int
    outgoing_dummy_count: int
    incoming_dummy_count: int
    nonzero_bin_count: int
    active_bin_count: int
    local_rate_peak: int
    count_signature: tuple[int, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (2, int(self.width))

    @property
    def size(self) -> int:
        return int(2 * int(self.width))

    def to_dense(self, dtype=np.int32) -> np.ndarray:
        out = np.zeros(self.shape, dtype=dtype)
        if int(self.values.size) > 0:
            np.add.at(out, (self.directions.astype(np.int64), self.bins.astype(np.int64)), self.values.astype(dtype))
        return out

    def __array__(self, dtype=None) -> np.ndarray:
        return self.to_dense(dtype=np.int32 if dtype is None else dtype)

    def astype(self, dtype) -> np.ndarray:
        return self.to_dense(dtype=dtype)

    def reshape(self, *shape) -> np.ndarray:
        return self.to_dense().reshape(*shape)

    def sum(self, axis=None):
        if axis is None:
            return int(self.dummy_count)
        if int(axis) == 0:
            out = np.zeros(int(self.width), dtype=np.int32)
            if int(self.values.size) > 0:
                np.add.at(out, self.bins.astype(np.int64), self.values.astype(np.int32))
            return out
        if int(axis) == 1:
            return np.asarray([int(self.outgoing_dummy_count), int(self.incoming_dummy_count)], dtype=np.int32)
        return self.to_dense().sum(axis=axis)


@dataclass(frozen=True)
class ExpandedAction:
    sample_index: int
    sample_id: str
    true_label: int
    window_id: int
    action_type: str
    tier: str
    source: str
    affected_direction: str
    affected_start: int
    affected_end: int
    affected_center: int
    insert_start: int
    insert_end: int
    insert_center: int
    dose: int
    direction_mode: str
    counts: np.ndarray | SparseCounts
    mask_mass: float
    local_count: float
    local_rate_peak: int
    requires_incoming_capability: int
    allowed_violation_count: int
    score_hint: float
    parent: str = ""
    dummy_count: int = 0
    outgoing_dummy_count: int = 0
    incoming_dummy_count: int = 0
    nonzero_bin_count: int = 0
    active_bin_count: int = 0
    count_signature: tuple[int, ...] = ()

    @property
    def bandwidth_overhead(self) -> float:
        clean_total = max(float(self.local_count_reference), 1.0)
        dummy = int(self.dummy_count) if int(self.dummy_count) > 0 else int(np.asarray(self.counts).sum())
        return float(dummy / clean_total)

    @property
    def local_count_reference(self) -> float:
        return float(self._clean_total) if hasattr(self, "_clean_total") else float(max(self.local_count, 1.0))

    @property
    def group_key(self) -> tuple[int, str, int]:
        return int(self.window_id), str(self.action_type), int(self.insert_center)

    @property
    def is_client_only(self) -> bool:
        return int(self.requires_incoming_capability) == 0


@dataclass(frozen=True, slots=True)
class CandidateDescriptor:
    action_id: int
    sample_index: int
    sample_id: str
    true_label: int
    window_id: int
    action_type: str
    tier: str
    source: str
    affected_direction: str
    affected_start: int
    affected_end: int
    affected_center: int
    insert_start: int
    insert_end: int
    insert_center: int
    action_width: int
    dose: int
    direction_mode: str
    smoothing: str
    mask_mass: float
    local_count: float
    score_hint: float
    dummy_count: int
    outgoing_dummy_count: int
    incoming_dummy_count: int
    nonzero_bin_count: int
    active_bin_count: int
    requires_incoming_capability: int
    allowed_violation_count: int = 0
    local_rate_peak: int = 0
    count_signature: tuple[int, ...] = ()

    @property
    def local_count_reference(self) -> float:
        return float(max(float(self.local_count), 1.0))

    @property
    def is_client_only(self) -> bool:
        return int(self.requires_incoming_capability) == 0


def _with_clean_total(action: ExpandedAction, clean_total: float) -> ExpandedAction:
    object.__setattr__(action, "_clean_total", float(clean_total))
    return action


def action_cost(action: ExpandedAction, clean_total: float | None = None) -> float:
    total = float(clean_total) if clean_total is not None else float(action.local_count_reference)
    dummy = int(action.dummy_count) if int(action.dummy_count) > 0 else int(np.asarray(action.counts).sum())
    return float(dummy / max(total, 1.0))


def action_identity(action: ExpandedAction) -> tuple:
    if isinstance(action, CandidateDescriptor):
        return descriptor_identity(action)
    signature = tuple(int(i) for i in getattr(action, "count_signature", ())[:16])
    if not signature:
        nz = np.flatnonzero(np.asarray(action.counts).reshape(-1) > 0)
        signature = tuple(int(i) for i in nz[:16])
    return (
        str(action.action_type),
        str(action.tier),
        int(action.window_id),
        int(action.insert_start),
        int(action.insert_end),
        int(action.dose),
        str(action.direction_mode),
        signature,
    )


def descriptor_identity(action: CandidateDescriptor) -> tuple:
    return (
        str(action.action_type),
        str(action.tier),
        int(action.window_id),
        int(action.insert_start),
        int(action.insert_end),
        int(action.dose),
        str(action.direction_mode),
    )


def descriptor_cost(action: CandidateDescriptor, clean_total: float) -> float:
    return float(int(action.dummy_count) / max(float(clean_total), 1.0))


def _count_metadata(counts: np.ndarray) -> tuple[int, int, int, int, tuple[int, ...]]:
    if isinstance(counts, SparseCounts):
        return (
            int(counts.dummy_count),
            int(counts.outgoing_dummy_count),
            int(counts.incoming_dummy_count),
            int(counts.nonzero_bin_count),
            tuple(int(i) for i in counts.count_signature[:16]),
        )
    values = np.asarray(counts, dtype=np.int32)
    outgoing = int(values[0].sum())
    incoming = int(values[1].sum())
    dummy = int(outgoing + incoming)
    nz = np.flatnonzero(values.reshape(-1) > 0)
    return dummy, outgoing, incoming, int(nz.size), tuple(int(i) for i in nz[:16])


def _diag_time_add(diagnostics: dict | None, key: str, seconds: float) -> None:
    if diagnostics is None:
        return
    if str(key) in DETAILED_TIMING_KEYS and not bool(diagnostics.get("profile_detail", False)):
        return
    timing = diagnostics.setdefault("timing_sec", {})
    timing[str(key)] = float(timing.get(str(key), 0.0)) + float(seconds)


def _direction_counts(total: int, mode: str, tam: np.ndarray, start: int, end: int, affected_direction: int) -> np.ndarray:
    count = max(0, int(total))
    name = str(mode)
    if name == "out-only":
        return np.asarray([count, 0], dtype=np.int32)
    if name == "in-only":
        return np.asarray([0, count], dtype=np.int32)
    if name == "both-equal":
        return allocate_integer(count, np.asarray([1.0, 1.0], dtype=np.float32))
    lo, hi = int(start), max(int(start) + 1, int(end))
    masses = np.asarray(tam[:, lo:hi].sum(axis=1), dtype=np.float64)
    if name == "direction-balance":
        total_mass = float(masses.sum())
        if total_mass <= 1e-12:
            target = np.asarray([1.0, 1.0], dtype=np.float64)
        else:
            target = np.asarray([total_mass - masses[0], total_mass - masses[1]], dtype=np.float64)
            target[int(affected_direction)] += 0.25 * total_mass
        return allocate_integer(count, target)
    if float(masses.sum()) <= 1e-12:
        masses = np.asarray([1.0, 1.0], dtype=np.float64)
        masses[int(affected_direction)] += 1.0
    return allocate_integer(count, masses)


def _interval_counts(
    *,
    tam: np.ndarray,
    start: int,
    end: int,
    dose: int,
    direction_mode: str,
    affected_start: int,
    affected_end: int,
    affected_direction: int,
    smoothing: str = "uniform",
) -> SparseCounts:
    values = np.asarray(tam, dtype=np.float32)
    width = int(values.shape[1])
    lo = int(np.clip(start, 0, width - 1))
    hi = int(np.clip(max(int(end), lo + 1), lo + 1, width))
    per_direction = _direction_counts(int(dose), str(direction_mode), values, int(affected_start), int(affected_end), int(affected_direction))
    return _interval_counts_from_per_direction(
        values=values,
        lo=lo,
        hi=hi,
        per_direction=per_direction,
        smoothing=str(smoothing),
    )


def _interval_counts_from_per_direction(
    *,
    values: np.ndarray,
    lo: int,
    hi: int,
    per_direction: np.ndarray,
    smoothing: str = "uniform",
) -> SparseCounts:
    width = int(values.shape[1])
    lo = int(np.clip(lo, 0, width - 1))
    hi = int(np.clip(max(int(hi), lo + 1), lo + 1, width))
    per_direction = np.asarray(per_direction, dtype=np.int32)
    direction_parts: list[np.ndarray] = []
    bin_parts: list[np.ndarray] = []
    value_parts: list[np.ndarray] = []
    local_peak = np.zeros(hi - lo, dtype=np.int32)
    outgoing = 0
    incoming = 0
    for direction in range(2):
        n = int(per_direction[direction])
        if n <= 0:
            continue
        if str(smoothing) == "inverse-rate":
            weights = 1.0 / np.sqrt(values[direction, lo:hi].astype(np.float64) + 1.0)
        elif str(smoothing) == "edge":
            xs = np.linspace(-1.0, 1.0, hi - lo)
            weights = np.maximum(np.abs(xs), 0.20)
        else:
            weights = np.ones(hi - lo, dtype=np.float64)
        slot_counts = allocate_integer(n, weights).astype(np.int32)
        nz = np.flatnonzero(slot_counts > 0)
        if nz.size:
            direction_parts.append(np.full(int(nz.size), int(direction), dtype=np.int16))
            bin_parts.append((lo + nz).astype(np.int32))
            value_parts.append(slot_counts[nz].astype(np.int32))
            local_peak[nz] += slot_counts[nz]
        if direction == 0:
            outgoing += int(slot_counts.sum())
        else:
            incoming += int(slot_counts.sum())
    if value_parts:
        directions = np.concatenate(direction_parts).astype(np.int16)
        bins = np.concatenate(bin_parts).astype(np.int32)
        sparse_values = np.concatenate(value_parts).astype(np.int32)
        flat = directions.astype(np.int64) * int(width) + bins.astype(np.int64)
        signature = tuple(int(i) for i in flat[:16].tolist())
        nonzero = int(sparse_values.size)
        active_bins = int(np.unique(bins).size)
    else:
        directions = np.zeros(0, dtype=np.int16)
        bins = np.zeros(0, dtype=np.int32)
        sparse_values = np.zeros(0, dtype=np.int32)
        signature = ()
        nonzero = 0
        active_bins = 0
    dummy = int(outgoing + incoming)
    return SparseCounts(
        width=int(width),
        directions=directions,
        bins=bins,
        values=sparse_values,
        dummy_count=int(dummy),
        outgoing_dummy_count=int(outgoing),
        incoming_dummy_count=int(incoming),
        nonzero_bin_count=int(nonzero),
        active_bin_count=int(active_bins),
        local_rate_peak=int(local_peak.max()) if local_peak.size else 0,
        count_signature=signature,
    )


def _clip_interval(center: int, action_width: int, width: int) -> tuple[int, int]:
    size = max(1, min(int(action_width), int(width)))
    lo = int(np.clip(int(center) - size // 2, 0, max(0, int(width) - size)))
    return lo, int(lo + size)


def _burst_intervals(row: np.ndarray) -> list[tuple[int, int]]:
    active = np.asarray(row, dtype=np.float32) > 0
    if not np.any(active):
        return []
    padded = np.concatenate([[False], active, [False]])
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    return [(int(changes[i]), int(changes[i + 1])) for i in range(0, len(changes), 2)]


def _gap_centers(tam: np.ndarray, *, max_gap: int, min_gap: int = 2) -> list[int]:
    intervals = sorted(_burst_intervals(tam[0]) + _burst_intervals(tam[1]))
    centers: list[int] = []
    for (_, left_end), (right_start, _) in zip(intervals, intervals[1:]):
        gap = int(right_start) - int(left_end)
        if int(min_gap) <= gap <= int(max_gap):
            centers.append(int(round((left_end + right_start) / 2.0)))
    return centers


def _transition_predecessors(tam: np.ndarray, *, radius: int = 4) -> list[int]:
    out = np.asarray(tam[0], dtype=np.float32)
    inc = np.asarray(tam[1], dtype=np.float32)
    active = (out + inc) > 0
    dominant = np.where(out >= inc, 0, 1)
    change = np.flatnonzero((dominant[1:] != dominant[:-1]) & active[1:] & active[:-1]) + 1
    return [max(0, int(pos) - int(radius)) for pos in change.tolist()]


def _rate_peak_predecessors(tam: np.ndarray, *, count: int, radius: int = 8) -> list[int]:
    total = np.asarray(tam, dtype=np.float32).sum(axis=0)
    if not np.any(total > 0):
        return []
    kernel = np.ones(max(1, int(radius)), dtype=np.float32)
    smoothed = np.convolve(total, kernel, mode="same")
    order = np.argsort(-smoothed, kind="mergesort")
    result: list[int] = []
    for pos in order[: max(int(count) * 3, int(count))]:
        if smoothed[int(pos)] <= 0:
            continue
        pred = max(0, int(pos) - int(radius))
        if all(abs(pred - old) > int(radius) for old in result):
            result.append(pred)
        if len(result) >= int(count):
            break
    return result


def _window_local_count(tam: np.ndarray, start: int, end: int) -> float:
    lo, hi = int(start), max(int(start) + 1, int(end))
    return float(np.asarray(tam, dtype=np.float32)[:, lo:hi].sum())


def _dose_values(local_count: float, absolute_doses: Iterable[int], relative_doses: Iterable[float], max_dose: int) -> list[int]:
    values = {int(v) for v in absolute_doses if int(v) > 0}
    base = max(float(local_count), 1.0)
    for rho in relative_doses:
        values.add(max(1, int(round(base * float(rho)))))
    return sorted(v for v in values if 0 < int(v) <= int(max_dose))


def _make_action(
    *,
    sample_index: int,
    sample_id: str,
    true_label: int,
    window: CandidateWindow,
    tam: np.ndarray,
    clean_total: float,
    action_type: str,
    tier: str,
    source: str,
    center: int,
    action_width: int,
    dose: int,
    direction_mode: str,
    smoothing: str = "uniform",
    parent: str = "",
    local_count: float | None = None,
) -> ExpandedAction | None:
    width = int(tam.shape[1])
    insert_start, insert_end = _clip_interval(int(center), int(action_width), width)
    counts = _interval_counts(
        tam=tam,
        start=insert_start,
        end=insert_end,
        dose=int(dose),
        direction_mode=str(direction_mode),
        affected_start=int(window.start),
        affected_end=int(window.end),
        affected_direction=int(window.direction),
        smoothing=str(smoothing),
    )
    dummy_count, outgoing_count, incoming_count, nonzero_count, count_signature = _count_metadata(counts)
    if int(dummy_count) <= 0:
        return None
    local_rate_peak = int(counts.local_rate_peak) if isinstance(counts, SparseCounts) else int(counts.sum(axis=0).max())
    active_bin_count = int(counts.active_bin_count) if isinstance(counts, SparseCounts) else int(np.count_nonzero(np.asarray(counts, dtype=np.int32).sum(axis=0)))
    requires_incoming = int(incoming_count > 0)
    local_count_value = float(local_count) if local_count is not None else _window_local_count(tam, int(window.start), int(window.end))
    cost = float(dummy_count / max(float(clean_total), 1.0))
    if cost <= 0.0:
        return None
    tier_bonus = {"primary": 1.0, "secondary": 0.65, "exploration": 0.30}.get(str(tier), 0.20)
    type_bonus = {
        "dynamask_causal": 1.00,
        "shared_predecessor": 0.90,
        "two_window_coordinated_insert": 0.85,
        "gap_fill": 0.75,
        "burst_merge": 0.75,
        "rate_peak_predecessor": 0.65,
        "direction_balance": 0.60,
        "local_rate_smoothing": 0.55,
        "burst_extension": 0.50,
        "cumulative_shift": 0.45,
    }.get(str(action_type), 0.35)
    dose_bonus = 1.0 + 0.15 * float(np.log1p(max(int(dose), 1)))
    score = float((float(window.mask_mass) + 1.0) * tier_bonus * type_bonus * dose_bonus / max(np.sqrt(cost), 1e-6))
    action = ExpandedAction(
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        true_label=int(true_label),
        window_id=int(window.window_id),
        action_type=str(action_type),
        tier=str(tier),
        source=str(source),
        affected_direction=str(window.direction_name),
        affected_start=int(window.start),
        affected_end=int(window.end),
        affected_center=int(window.center),
        insert_start=int(insert_start),
        insert_end=int(insert_end),
        insert_center=int(np.clip(center, 0, width - 1)),
        dose=int(dose),
        direction_mode=str(direction_mode),
        counts=counts,
        mask_mass=float(window.mask_mass),
        local_count=float(local_count_value),
        local_rate_peak=int(local_rate_peak),
        requires_incoming_capability=int(requires_incoming),
        allowed_violation_count=0,
        score_hint=float(score),
        parent=str(parent),
        dummy_count=int(dummy_count),
        outgoing_dummy_count=int(outgoing_count),
        incoming_dummy_count=int(incoming_count),
        nonzero_bin_count=int(nonzero_count),
        active_bin_count=int(active_bin_count),
        count_signature=count_signature,
    )
    return _with_clean_total(action, float(clean_total))


def _passes_protocol_and_constraints(
    action: ExpandedAction,
    *,
    protocol: str,
    clean_total: float,
    max_action_budget: float,
    max_local_rate_peak: int,
) -> bool:
    if str(protocol) == "client_only" and not action.is_client_only:
        return False
    if str(protocol) not in {"client_only", "bidirectional_cooperative"}:
        raise ValueError(f"Unknown protocol={protocol!r}")
    if action_cost(action, clean_total) > float(max_action_budget) + 1e-12:
        return False
    if int(action.local_rate_peak) > int(max_local_rate_peak):
        return False
    if int(action.allowed_violation_count) > 0:
        return False
    return True


def _dedupe(actions: list[ExpandedAction], clean_total: float) -> list[ExpandedAction]:
    seen = set()
    result = []
    for action in sorted(actions, key=lambda item: (-float(item.score_hint), action_cost(item, clean_total), str(item.action_type))):
        key = action_identity(action)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _diverse_limit(actions: list[ExpandedAction], limit: int, clean_total: float) -> list[ExpandedAction]:
    if int(limit) <= 0 or len(actions) <= int(limit):
        return actions
    ranked = sorted(actions, key=lambda item: (-float(item.score_hint), action_cost(item, clean_total), str(item.action_type)))
    selected: list[ExpandedAction] = []
    seen: set[tuple] = set()
    quotas = {
        "primary": max(1, int(round(int(limit) * 0.45))),
        "secondary": max(1, int(round(int(limit) * 0.45))),
        "exploration": max(1, int(limit) - 2 * max(1, int(round(int(limit) * 0.45)))),
    }
    for tier in TIERS:
        count = 0
        for action in ranked:
            if len(selected) >= int(limit):
                break
            if action.tier != tier:
                continue
            key = action_identity(action)
            if key in seen:
                continue
            selected.append(action)
            seen.add(key)
            count += 1
            if count >= quotas.get(tier, 0) or len(selected) >= int(limit):
                break
    for action in ranked:
        key = action_identity(action)
        if key in seen:
            continue
        selected.append(action)
        seen.add(key)
        if len(selected) >= int(limit):
            break
    return selected


def _diag_add(diagnostics: dict | None, stage: str, action: ExpandedAction) -> None:
    if diagnostics is None:
        return
    action_type = str(action.action_type)
    by_stage = diagnostics.setdefault(str(stage), {})
    by_stage[action_type] = int(by_stage.get(action_type, 0)) + 1
    dose_key = f"dose_{int(action.dose)}"
    dose_stage = diagnostics.setdefault(f"{stage}_dose", {})
    dose_stage[dose_key] = int(dose_stage.get(dose_key, 0)) + 1
    bin_key = "multi_bin" if int(getattr(action, "nonzero_bin_count", 0)) > 1 else "single_bin"
    bin_stage = diagnostics.setdefault(f"{stage}_bin", {})
    bin_stage[bin_key] = int(bin_stage.get(bin_key, 0)) + 1
    direction_key = (
        "bidirectional"
        if int(getattr(action, "outgoing_dummy_count", 0)) > 0 and int(getattr(action, "incoming_dummy_count", 0)) > 0
        else "incoming"
        if int(getattr(action, "incoming_dummy_count", 0)) > 0
        else "outgoing"
    )
    direction_stage = diagnostics.setdefault(f"{stage}_direction", {})
    direction_stage[direction_key] = int(direction_stage.get(direction_key, 0)) + 1


def _diag_add_many(diagnostics: dict | None, stage: str, actions: list[ExpandedAction]) -> None:
    if diagnostics is None:
        return
    for action in actions:
        _diag_add(diagnostics, stage, action)


def _pair_action(
    left: ExpandedAction,
    right: ExpandedAction,
    *,
    clean_total: float,
) -> ExpandedAction | None:
    counts = np.asarray(left.counts, dtype=np.int32) + np.asarray(right.counts, dtype=np.int32)
    dummy_count, outgoing_count, incoming_count, nonzero_count, count_signature = _count_metadata(counts)
    if int(dummy_count) <= 0:
        return None
    active_bin_count = int(np.count_nonzero(counts.sum(axis=0)))
    start = int(min(left.insert_start, right.insert_start))
    end = int(max(left.insert_end, right.insert_end))
    action = replace(
        left,
        action_type="two_window_coordinated_insert",
        tier="secondary",
        source="paired",
        insert_start=start,
        insert_end=end,
        insert_center=int(round((left.insert_center + right.insert_center) / 2.0)),
        dose=int(dummy_count),
        direction_mode="coordinated",
        counts=counts.astype(np.int32),
        local_rate_peak=int(counts.sum(axis=0).max()),
        requires_incoming_capability=int(incoming_count > 0),
        score_hint=float((left.score_hint + right.score_hint) * 0.60),
        parent=f"{left.action_type}:{left.window_id}+{right.action_type}:{right.window_id}",
        dummy_count=int(dummy_count),
        outgoing_dummy_count=int(outgoing_count),
        incoming_dummy_count=int(incoming_count),
        nonzero_bin_count=int(nonzero_count),
        active_bin_count=int(active_bin_count),
        count_signature=count_signature,
    )
    return _with_clean_total(action, float(clean_total))


def convert_stage_b0_action(action: CandidateAction, tam: np.ndarray, clean_total: float) -> ExpandedAction:
    from dynapd.stage_a.additive_probe import counts_for_action

    direction = 0 if str(action.affected_direction) == "out" else 1
    window = CandidateWindow(
        sample_index=int(action.sample_index),
        window_id=int(action.window_id),
        direction=int(direction),
        start=int(action.affected_start),
        end=int(action.affected_end),
        center=int(action.affected_center),
        mask_mass=float(action.mask_mass),
        length=max(1, int(action.affected_end) - int(action.affected_start)),
    )
    spec = counts_for_action(tam, window, offset=int(action.offset), dose=int(action.dose), direction_mode=str(action.direction_mode))
    dummy_count, outgoing_count, incoming_count, nonzero_count, count_signature = _count_metadata(spec.counts)
    active_bin_count = int(np.count_nonzero(np.asarray(spec.counts, dtype=np.int32).sum(axis=0)))
    cost = float(dummy_count / max(float(clean_total), 1.0))
    score = (
        0.30 * max(float(action.top1_drop), 0.0)
        + 0.50 * max(float(action.margin_drop), 0.0)
        + 0.20 * max(float(action.entropy_gain), 0.0)
    ) / max(cost, 1e-6)
    out = ExpandedAction(
        sample_index=int(action.sample_index),
        sample_id=str(action.sample_id),
        true_label=int(action.true_label),
        window_id=int(action.window_id),
        action_type="stage_b0_causal" if int(action.offset) <= 0 else "stage_b0",
        tier="primary",
        source="stage_b0_action_table",
        affected_direction=str(action.affected_direction),
        affected_start=int(action.affected_start),
        affected_end=int(action.affected_end),
        affected_center=int(action.affected_center),
        insert_start=int(spec.insert_start),
        insert_end=int(spec.insert_end),
        insert_center=int(spec.insert_center),
        dose=int(spec.dose),
        direction_mode=str(spec.direction_mode),
        counts=spec.counts.astype(np.int32),
        mask_mass=float(action.mask_mass),
        local_count=float(_window_local_count(tam, int(action.affected_start), int(action.affected_end))),
        local_rate_peak=int(spec.local_rate_peak),
        requires_incoming_capability=int(incoming_count > 0),
        allowed_violation_count=int(spec.allowed_violation_count),
        score_hint=float(score),
        parent="",
        dummy_count=int(dummy_count),
        outgoing_dummy_count=int(outgoing_count),
        incoming_dummy_count=int(incoming_count),
        nonzero_bin_count=int(nonzero_count),
        active_bin_count=int(active_bin_count),
        count_signature=count_signature,
    )
    return _with_clean_total(out, float(clean_total))


def _descriptor_direction_counts(
    *,
    tam: np.ndarray,
    dose: int,
    direction_mode: str,
    affected_start: int,
    affected_end: int,
    affected_direction: int,
) -> tuple[int, int]:
    per_direction = _direction_counts(
        int(dose),
        str(direction_mode),
        np.asarray(tam, dtype=np.float32),
        int(affected_start),
        int(affected_end),
        int(affected_direction),
    )
    return int(per_direction[0]), int(per_direction[1])


def _descriptor_nonzero_estimate(outgoing: int, incoming: int, insert_width: int) -> tuple[int, int]:
    width = max(1, int(insert_width))
    out_bins = min(max(0, int(outgoing)), width)
    in_bins = min(max(0, int(incoming)), width)
    active_bins = min(max(0, int(outgoing) + int(incoming)), width)
    return int(out_bins + in_bins), int(active_bins)


def _descriptor_score(
    *,
    action_type: str,
    tier: str,
    mask_mass: float,
    dose: int,
    cost: float,
) -> float:
    tier_bonus = TIER_BONUS.get(str(tier), 0.20)
    type_bonus = ACTION_TYPE_BONUS.get(str(action_type), 0.35)
    dose_bonus = 1.0 + 0.15 * float(np.log1p(max(int(dose), 1)))
    return float((float(mask_mass) + 1.0) * tier_bonus * type_bonus * dose_bonus / max(np.sqrt(float(cost)), 1e-6))


def _dedupe_descriptors(actions: list[CandidateDescriptor], clean_total: float) -> list[CandidateDescriptor]:
    seen = set()
    result = []
    for action in sorted(
        actions,
        key=lambda item: (-float(item.score_hint), descriptor_cost(item, clean_total), str(item.action_type), int(item.action_id)),
    ):
        key = descriptor_identity(action)
        if key in seen:
            continue
        seen.add(key)
        result.append(action)
    return result


def _unique_descriptor_extend(
    target: list[CandidateDescriptor],
    seen: set[tuple],
    items: list[CandidateDescriptor],
    limit: int,
) -> None:
    for action in items:
        key = descriptor_identity(action)
        if key in seen:
            continue
        target.append(action)
        seen.add(key)
        if len(target) >= int(limit):
            return


def _estimated_descriptor_gain(action: CandidateDescriptor, clean_total: float) -> float:
    return float(action.score_hint) * float(np.sqrt(max(descriptor_cost(action, clean_total), 1e-8)))


def _diverse_limit_descriptors(
    actions: list[CandidateDescriptor],
    limit: int,
    clean_total: float,
) -> list[CandidateDescriptor]:
    if int(limit) <= 0 or len(actions) <= int(limit):
        return actions
    selected: list[CandidateDescriptor] = []
    seen: set[tuple] = set()

    def add_bucket(items: list[CandidateDescriptor], n: int) -> None:
        _unique_descriptor_extend(selected, seen, items, min(int(limit), len(selected) + int(n)))

    for dose in (32, 16, 8, 4, 2, 1):
        bucket = [a for a in actions if int(a.dummy_count) == int(dose)]
        bucket = sorted(bucket, key=lambda a: (-_estimated_descriptor_gain(a, clean_total), int(a.action_id)))
        add_bucket(bucket, 4)
    for action_type in sorted({a.action_type for a in actions}):
        bucket = sorted(
            [a for a in actions if a.action_type == action_type],
            key=lambda a: (-_estimated_descriptor_gain(a, clean_total), int(a.action_id)),
        )
        add_bucket(bucket, 4)
    multi = sorted(
        [a for a in actions if int(a.nonzero_bin_count) > 1],
        key=lambda a: (-_estimated_descriptor_gain(a, clean_total), int(a.action_id)),
    )
    add_bucket(multi, 12)
    two_window = sorted(
        [a for a in actions if a.action_type == "two_window_coordinated_insert"],
        key=lambda a: (-_estimated_descriptor_gain(a, clean_total), int(a.action_id)),
    )
    add_bucket(two_window, 12)

    def composite(action: CandidateDescriptor) -> float:
        cost = descriptor_cost(action, clean_total)
        dose_bonus = 0.05 * np.log1p(max(int(action.dummy_count), 0))
        bin_bonus = 0.06 if int(action.nonzero_bin_count) > 1 else 0.0
        pair_bonus = 0.08 if action.action_type == "two_window_coordinated_insert" else 0.0
        return float(_estimated_descriptor_gain(action, clean_total) + 0.20 * float(action.score_hint) + dose_bonus + bin_bonus + pair_bonus - 0.05 * cost)

    rest = sorted(actions, key=lambda action: (-composite(action), int(action.action_id)))
    _unique_descriptor_extend(selected, seen, rest, int(limit))
    return selected[: int(limit)]


def _candidate_table_device(candidate_device: str) -> torch.device:
    requested = str(candidate_device).lower()
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _allocate_integer_two_many(doses: np.ndarray, weights0: float, weights1: float) -> tuple[np.ndarray, np.ndarray]:
    count = np.maximum(np.asarray(doses, dtype=np.int32), 0)
    w0 = np.full(count.shape, max(float(weights0), 0.0), dtype=np.float64)
    w1 = np.full(count.shape, max(float(weights1), 0.0), dtype=np.float64)
    empty = (w0 + w1) <= 1e-12
    if np.any(empty):
        w0[empty] = 1.0
        w1[empty] = 1.0
    total_w = np.maximum(w0 + w1, 1e-12)
    raw0 = w0 / total_w * count.astype(np.float64)
    raw1 = w1 / total_w * count.astype(np.float64)
    out0 = np.floor(raw0).astype(np.int32)
    out1 = np.floor(raw1).astype(np.int32)
    remaining = np.maximum(count - out0 - out1, 0)
    frac0 = raw0 - out0.astype(np.float64)
    frac1 = raw1 - out1.astype(np.float64)
    prefer0 = frac0 >= frac1
    out0 += remaining * prefer0.astype(np.int32)
    out1 += remaining * (~prefer0).astype(np.int32)
    return out0.astype(np.int32), out1.astype(np.int32)


def _direction_counts_many(
    doses: np.ndarray,
    mode_ids: np.ndarray,
    *,
    masses: np.ndarray,
    affected_direction: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.maximum(np.asarray(doses, dtype=np.int32), 0)
    modes = np.asarray(mode_ids, dtype=np.int16)
    outgoing = np.zeros_like(values, dtype=np.int32)
    incoming = np.zeros_like(values, dtype=np.int32)
    out_mask = modes == 0
    in_mask = modes == 1
    both_mask = modes == 2
    current_mask = modes == 3
    balance_mask = modes == 4
    outgoing[out_mask] = values[out_mask]
    incoming[in_mask] = values[in_mask]
    outgoing[both_mask] = (values[both_mask] + 1) // 2
    incoming[both_mask] = values[both_mask] // 2
    m = np.asarray(masses, dtype=np.float64).reshape(2)
    total_mass = float(m.sum())
    if np.any(current_mask):
        if total_mass <= 1e-12:
            weights = np.asarray([1.0, 1.0], dtype=np.float64)
            weights[int(affected_direction)] += 1.0
        else:
            weights = m
        out, inc = _allocate_integer_two_many(values[current_mask], float(weights[0]), float(weights[1]))
        outgoing[current_mask] = out
        incoming[current_mask] = inc
    if np.any(balance_mask):
        if total_mass <= 1e-12:
            target = np.asarray([1.0, 1.0], dtype=np.float64)
        else:
            target = np.asarray([total_mass - m[0], total_mass - m[1]], dtype=np.float64)
            target[int(affected_direction)] += 0.25 * total_mass
        out, inc = _allocate_integer_two_many(values[balance_mask], float(target[0]), float(target[1]))
        outgoing[balance_mask] = out
        incoming[balance_mask] = inc
    return outgoing.astype(np.int32), incoming.astype(np.int32)


def _score_candidate_table(
    *,
    mask_mass: np.ndarray,
    tier_ids: np.ndarray,
    type_ids: np.ndarray,
    doses: np.ndarray,
    clean_total: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tier_values = np.asarray([TIER_BONUS["primary"], TIER_BONUS["secondary"], TIER_BONUS["exploration"]], dtype=np.float32)
    action_names = sorted(ACTION_TYPE_BONUS)
    type_values = np.asarray([ACTION_TYPE_BONUS[name] for name in action_names], dtype=np.float32)
    if getattr(device, "type", "") == "cpu":
        dose = np.asarray(doses, dtype=np.float32)
        cost = dose / max(float(clean_total), 1.0)
        score = (
            (np.asarray(mask_mass, dtype=np.float32) + np.float32(1.0))
            * tier_values[np.asarray(tier_ids, dtype=np.int64)]
            * type_values[np.asarray(type_ids, dtype=np.int64)]
            * (np.float32(1.0) + np.float32(0.15) * np.log1p(np.maximum(dose, np.float32(1.0))))
            / np.maximum(np.sqrt(np.maximum(cost, np.float32(1e-12))), np.float32(1e-6))
        )
        return score.astype(np.float32, copy=False), cost.astype(np.float32, copy=False)
    dose_t = torch.as_tensor(np.asarray(doses, dtype=np.float32), device=device)
    cost_t = dose_t / max(float(clean_total), 1.0)
    score_t = (
        (torch.as_tensor(np.asarray(mask_mass, dtype=np.float32), device=device) + 1.0)
        * torch.as_tensor(tier_values[np.asarray(tier_ids, dtype=np.int64)], device=device)
        * torch.as_tensor(type_values[np.asarray(type_ids, dtype=np.int64)], device=device)
        * (1.0 + 0.15 * torch.log1p(torch.clamp(dose_t, min=1.0)))
        / torch.clamp(torch.sqrt(torch.clamp(cost_t, min=1e-12)), min=1e-6)
    )
    score = score_t.detach().cpu().numpy().astype(np.float32)
    cost = cost_t.detach().cpu().numpy().astype(np.float32)
    del dose_t, cost_t, score_t
    return score, cost


def _clip_interval_many(centers: np.ndarray, widths: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    sizes = np.maximum(1, np.minimum(np.asarray(widths, dtype=np.int32), int(width)))
    starts = np.clip(np.asarray(centers, dtype=np.int32) - sizes // 2, 0, np.maximum(0, int(width) - sizes)).astype(np.int32)
    return starts, (starts + sizes).astype(np.int32)


def _candidate_table_concat(chunks: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(chunks[0].keys())
    return {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in keys}


def _select_descriptor_indices(table: dict[str, np.ndarray], limit: int, clean_total: float) -> list[int]:
    n = int(len(table["action_id"]))
    if int(limit) <= 0:
        return []
    action_names = sorted(ACTION_TYPE_BONUS)

    action_id = np.asarray(table["action_id"], dtype=np.int64)
    action_type_id = np.asarray(table["action_type_id"], dtype=np.int64)
    tier_id = np.asarray(table["tier_id"], dtype=np.int64)
    window_id = np.asarray(table["window_id"], dtype=np.int64)
    insert_start = np.asarray(table["insert_start"], dtype=np.int64)
    insert_end = np.asarray(table["insert_end"], dtype=np.int64)
    dose = np.asarray(table["dose"], dtype=np.int64)
    mode_id = np.asarray(table["mode_id"], dtype=np.int64)
    dummy_count = np.asarray(table["dummy_count"], dtype=np.int64)
    nonzero_bin_count = np.asarray(table["nonzero_bin_count"], dtype=np.int64)
    score_hint = np.asarray(table["score_hint"], dtype=np.float64)
    cost = dummy_count.astype(np.float64) / max(float(clean_total), 1.0)
    estimated_gain = score_hint * np.sqrt(np.maximum(cost, 1e-8))

    # The identity fields have small bounded domains in the RF-TAM setting
    # (width=1800, doses <= 96). Packing them keeps duplicate removal and
    # selected-membership checks vectorized without building Python tuples.
    pack_base = 8192
    can_pack_identity = (
        int(action_type_id.max(initial=0)) < 16
        and int(tier_id.max(initial=0)) < 4
        and int(window_id.max(initial=0)) < pack_base
        and int(insert_start.max(initial=0)) < pack_base
        and int(insert_end.max(initial=0)) < pack_base
        and int(dose.max(initial=0)) < pack_base
        and int(mode_id.max(initial=0)) < 8
    )
    if can_pack_identity:
        identity_key = action_type_id
        identity_key = identity_key * 4 + tier_id
        identity_key = identity_key * pack_base + window_id
        identity_key = identity_key * pack_base + insert_start
        identity_key = identity_key * pack_base + insert_end
        identity_key = identity_key * pack_base + dose
        identity_key = identity_key * 8 + mode_id
        identity_key = identity_key.astype(np.int64, copy=False)
    else:
        identity_struct = np.empty(
            n,
            dtype=[
                ("action_type_id", np.int64),
                ("tier_id", np.int64),
                ("window_id", np.int64),
                ("insert_start", np.int64),
                ("insert_end", np.int64),
                ("dose", np.int64),
                ("mode_id", np.int64),
            ],
        )
        identity_struct["action_type_id"] = action_type_id
        identity_struct["tier_id"] = tier_id
        identity_struct["window_id"] = window_id
        identity_struct["insert_start"] = insert_start
        identity_struct["insert_end"] = insert_end
        identity_struct["dose"] = dose
        identity_struct["mode_id"] = mode_id
        identity_key = identity_struct

    ordered = np.lexsort((action_id, action_type_id, cost, -score_hint))
    if n == 0:
        return []
    _unique_keys, first_positions = np.unique(identity_key[ordered], return_index=True)
    deduped_arr = ordered[np.sort(first_positions)].astype(np.int64, copy=False)
    deduped = deduped_arr.tolist()
    if len(deduped) <= int(limit):
        return deduped

    selected: list[int] = []
    selected_seen: set = set()

    def add_bucket(indices: np.ndarray, count: int) -> None:
        target = min(int(limit), len(selected) + int(count))
        for value in np.asarray(indices, dtype=np.int64).tolist():
            idx = int(value)
            key = identity_key[idx]
            if isinstance(key, np.void):
                key = bytes(key)
            else:
                key = int(key)
            if key in selected_seen:
                continue
            selected.append(idx)
            selected_seen.add(key)
            if len(selected) >= target:
                return

    def order_by_estimated_gain(indices: np.ndarray) -> np.ndarray:
        values = np.asarray(indices, dtype=np.int64)
        if values.size <= 1:
            return values
        order = np.lexsort((action_id[values], -estimated_gain[values]))
        return values[order]

    for dose in (32, 16, 8, 4, 2, 1):
        bucket = deduped_arr[dummy_count[deduped_arr] == int(dose)]
        add_bucket(order_by_estimated_gain(bucket), 4)
    for current_action_type_id in np.unique(action_type_id[deduped_arr]):
        bucket = deduped_arr[action_type_id[deduped_arr] == int(current_action_type_id)]
        add_bucket(order_by_estimated_gain(bucket), 4)
    multi = order_by_estimated_gain(deduped_arr[nonzero_bin_count[deduped_arr] > 1])
    add_bucket(multi, 12)
    paired_type_id = action_names.index("two_window_coordinated_insert")
    paired = order_by_estimated_gain(deduped_arr[action_type_id[deduped_arr] == int(paired_type_id)])
    add_bucket(paired, 12)

    bin_bonus = np.where(nonzero_bin_count > 1, 0.06, 0.0)
    pair_bonus = np.where(action_type_id == int(paired_type_id), 0.08, 0.0)
    dose_bonus = 0.05 * np.log1p(np.maximum(dummy_count, 0).astype(np.float64))
    composite = estimated_gain + 0.20 * score_hint + dose_bonus + bin_bonus + pair_bonus - 0.05 * cost
    rest_order = np.lexsort((action_id[deduped_arr], -composite[deduped_arr]))
    rest = deduped_arr[rest_order]
    add_bucket(rest, int(limit))
    return selected[: int(limit)]


def _candidate_descriptors_from_table(
    table: dict[str, np.ndarray],
    indices: list[int],
    *,
    sample_index: int,
    sample_id: str,
    true_label: int,
) -> list[CandidateDescriptor]:
    action_names = sorted(ACTION_TYPE_BONUS)
    tier_names = list(TIERS)
    source_names = ("keypoint_start", "keypoint_center", "keypoint_end", "rate_peak", "transition", "gap", "window_start", "window_end", "smooth", "shift")
    smoothing_names = ("uniform", "inverse-rate", "edge")
    result: list[CandidateDescriptor] = []
    for idx in indices:
        result.append(
            CandidateDescriptor(
                action_id=int(table["action_id"][idx]),
                sample_index=int(sample_index),
                sample_id=str(sample_id),
                true_label=int(true_label),
                window_id=int(table["window_id"][idx]),
                action_type=action_names[int(table["action_type_id"][idx])],
                tier=tier_names[int(table["tier_id"][idx])],
                source=source_names[int(table["source_id"][idx])],
                affected_direction="out" if int(table["affected_direction_id"][idx]) == 0 else "in",
                affected_start=int(table["affected_start"][idx]),
                affected_end=int(table["affected_end"][idx]),
                affected_center=int(table["affected_center"][idx]),
                insert_start=int(table["insert_start"][idx]),
                insert_end=int(table["insert_end"][idx]),
                insert_center=int(table["insert_center"][idx]),
                action_width=int(table["action_width"][idx]),
                dose=int(table["dose"][idx]),
                direction_mode=DIRECTION_MODES[int(table["mode_id"][idx])],
                smoothing=smoothing_names[int(table["smoothing_id"][idx])],
                mask_mass=float(table["mask_mass"][idx]),
                local_count=float(table["local_count"][idx]),
                score_hint=float(table["score_hint"][idx]),
                dummy_count=int(table["dummy_count"][idx]),
                outgoing_dummy_count=int(table["outgoing_dummy_count"][idx]),
                incoming_dummy_count=int(table["incoming_dummy_count"][idx]),
                nonzero_bin_count=int(table["nonzero_bin_count"][idx]),
                active_bin_count=int(table["active_bin_count"][idx]),
                requires_incoming_capability=int(table["requires_incoming_capability"][idx]),
            )
        )
    return result


def _generate_compact_action_descriptors_batched(
    *,
    values: np.ndarray,
    windows: list[CandidateWindow],
    gap_centers: list[int],
    transitions: list[int],
    peaks: list[int],
    sample_index: int,
    sample_id: str,
    true_label: int,
    protocol: str,
    total: float,
    offsets: Iterable[int],
    absolute_doses: Iterable[int],
    relative_doses: Iterable[float],
    max_dose: int,
    max_action_budget: float,
    max_actions: int,
    candidate_batch_size: int,
    candidate_device: str,
    diagnostics: dict | None,
) -> list[CandidateDescriptor]:
    start_total = time.perf_counter()
    width = int(values.shape[1])
    modes = list(DIRECTION_MODES)
    if str(protocol) == "client_only":
        modes = ["out-only"]
    mode_ids_all = np.asarray([DIRECTION_MODES.index(mode) for mode in modes], dtype=np.int16)
    action_names = sorted(ACTION_TYPE_BONUS)
    action_type_to_id = {name: idx for idx, name in enumerate(action_names)}
    tier_to_id = {name: idx for idx, name in enumerate(TIERS)}
    source_names = ("keypoint_start", "keypoint_center", "keypoint_end", "rate_peak", "transition", "gap", "window_start", "window_end", "smooth", "shift")
    source_to_id = {name: idx for idx, name in enumerate(source_names)}
    smoothing_to_id = {"uniform": 0, "inverse-rate": 1, "edge": 2}
    device = _candidate_table_device(candidate_device)
    chunks: list[dict[str, np.ndarray]] = []
    kept_rows = 0
    action_id = 0
    score_time = 0.0
    for window in windows:
        local = _window_local_count(values, int(window.start), int(window.end))
        doses = np.asarray(_dose_values(local, absolute_doses, relative_doses, int(max_dose)), dtype=np.int32)
        if int(doses.size) <= 0:
            continue
        action_widths = np.asarray(sorted({max(2, min(width, int(window.length))), max(4, min(width, int(round(window.length * 0.5)))), 8, 16, 32})[:3], dtype=np.int32)
        anchors: list[tuple[str, str, int, str]] = []
        for base_name, anchor in (
            ("keypoint_start", int(window.start)),
            ("keypoint_center", int(window.center)),
            ("keypoint_end", int(window.end) - 1),
        ):
            for offset in offsets:
                center = int(np.clip(int(anchor) + int(offset), 0, width - 1))
                if center <= int(anchor):
                    anchors.append(("dynamask_causal", "primary", center, base_name))
        for center in peaks:
            if center <= int(window.end):
                anchors.append(("rate_peak_predecessor", "secondary", int(center), "rate_peak"))
        for center in transitions:
            if abs(int(center) - int(window.center)) <= max(96, int(window.length) * 2):
                anchors.append(("direction_balance", "secondary", int(center), "transition"))
        for center in gap_centers:
            if abs(int(center) - int(window.center)) <= max(96, int(window.length) * 2):
                anchors.append(("gap_fill", "secondary", int(center), "gap"))
                anchors.append(("burst_merge", "secondary", int(center), "gap"))
        anchors.append(("burst_extension", "secondary", max(0, int(window.start) - 2), "window_start"))
        anchors.append(("burst_extension", "secondary", min(width - 1, int(window.end)), "window_end"))
        anchors.append(("local_rate_smoothing", "secondary", int(window.center), "smooth"))
        anchors.append(("cumulative_shift", "exploration", max(0, int(window.start) - max(8, int(window.length))), "shift"))
        if not anchors:
            continue
        anchor_type = np.asarray([action_type_to_id[item[0]] for item in anchors], dtype=np.int16)
        anchor_tier = np.asarray([tier_to_id[item[1]] for item in anchors], dtype=np.int16)
        anchor_center = np.asarray([item[2] for item in anchors], dtype=np.int32)
        anchor_source = np.asarray([source_to_id[item[3]] for item in anchors], dtype=np.int16)
        n = int(len(anchors) * len(action_widths) * len(doses) * len(mode_ids_all))
        base_action_ids = np.arange(action_id, action_id + n, dtype=np.int32)
        action_id += n
        anchor_idx = np.repeat(np.arange(len(anchors), dtype=np.int32), len(action_widths) * len(doses) * len(mode_ids_all))
        width_values = np.tile(np.repeat(action_widths, len(doses) * len(mode_ids_all)), len(anchors)).astype(np.int32)
        dose_values = np.tile(np.repeat(doses, len(mode_ids_all)), len(anchors) * len(action_widths)).astype(np.int32)
        mode_values = np.tile(mode_ids_all, len(anchors) * len(action_widths) * len(doses)).astype(np.int16)
        cost_filter = (dose_values.astype(np.float64) / max(float(total), 1.0)) <= float(max_action_budget) + 1e-12
        if not np.any(cost_filter):
            continue
        base_action_ids = base_action_ids[cost_filter]
        anchor_idx = anchor_idx[cost_filter]
        width_values = width_values[cost_filter]
        dose_values = dose_values[cost_filter]
        mode_values = mode_values[cost_filter]
        insert_start, insert_end = _clip_interval_many(anchor_center[anchor_idx], width_values, width)
        insert_width = np.maximum(1, insert_end - insert_start).astype(np.int32)
        masses = np.asarray(values[:, int(window.start) : int(window.end)].sum(axis=1), dtype=np.float64)
        outgoing, incoming = _direction_counts_many(dose_values, mode_values, masses=masses, affected_direction=int(window.direction))
        if str(protocol) == "client_only":
            legal = incoming <= 0
            base_action_ids = base_action_ids[legal]
            anchor_idx = anchor_idx[legal]
            width_values = width_values[legal]
            dose_values = dose_values[legal]
            mode_values = mode_values[legal]
            insert_start = insert_start[legal]
            insert_end = insert_end[legal]
            insert_width = insert_width[legal]
            outgoing = outgoing[legal]
            incoming = incoming[legal]
        if int(base_action_ids.size) <= 0:
            continue
        nonzero = (np.minimum(np.maximum(outgoing, 0), insert_width) + np.minimum(np.maximum(incoming, 0), insert_width)).astype(np.int32)
        active = np.minimum(np.maximum(outgoing + incoming, 0), insert_width).astype(np.int32)
        smoothing = np.full(base_action_ids.shape, smoothing_to_id["uniform"], dtype=np.int16)
        action_type_ids = anchor_type[anchor_idx]
        gap_like = np.isin(action_type_ids, [action_type_to_id["gap_fill"], action_type_to_id["local_rate_smoothing"]])
        edge_like = action_type_ids == action_type_to_id["burst_extension"]
        smoothing[gap_like] = smoothing_to_id["inverse-rate"]
        smoothing[edge_like] = smoothing_to_id["edge"]
        batch_size = max(1, int(candidate_batch_size))
        for batch_start in range(0, int(base_action_ids.size), batch_size):
            batch_end = min(int(base_action_ids.size), batch_start + batch_size)
            batch_slice = slice(batch_start, batch_end)
            score_start = time.perf_counter()
            score, cost = _score_candidate_table(
                mask_mass=np.full(batch_end - batch_start, float(window.mask_mass), dtype=np.float32),
                tier_ids=anchor_tier[anchor_idx][batch_slice],
                type_ids=action_type_ids[batch_slice],
                doses=dose_values[batch_slice],
                clean_total=float(total),
                device=device,
            )
            score_time += time.perf_counter() - score_start
            valid = (cost > 0.0) & (cost <= float(max_action_budget) + 1e-12)
            if not np.any(valid):
                continue
            valid_count = int(np.count_nonzero(valid))
            chunk = {
                "action_id": base_action_ids[batch_slice][valid],
                "window_id": np.full(valid_count, int(window.window_id), dtype=np.int32),
                "action_type_id": action_type_ids[batch_slice][valid].astype(np.int16),
                "tier_id": anchor_tier[anchor_idx][batch_slice][valid].astype(np.int16),
                "source_id": anchor_source[anchor_idx][batch_slice][valid].astype(np.int16),
                "affected_direction_id": np.full(valid_count, int(window.direction), dtype=np.int16),
                "affected_start": np.full(valid_count, int(window.start), dtype=np.int32),
                "affected_end": np.full(valid_count, int(window.end), dtype=np.int32),
                "affected_center": np.full(valid_count, int(window.center), dtype=np.int32),
                "insert_start": insert_start[batch_slice][valid].astype(np.int32),
                "insert_end": insert_end[batch_slice][valid].astype(np.int32),
                "insert_center": np.clip(anchor_center[anchor_idx][batch_slice][valid], 0, width - 1).astype(np.int32),
                "action_width": width_values[batch_slice][valid].astype(np.int32),
                "dose": dose_values[batch_slice][valid].astype(np.int32),
                "mode_id": mode_values[batch_slice][valid].astype(np.int16),
                "smoothing_id": smoothing[batch_slice][valid].astype(np.int16),
                "mask_mass": np.full(valid_count, float(window.mask_mass), dtype=np.float32),
                "local_count": np.full(valid_count, float(local), dtype=np.float32),
                "score_hint": score[valid].astype(np.float32),
                "dummy_count": dose_values[batch_slice][valid].astype(np.int32),
                "outgoing_dummy_count": outgoing[batch_slice][valid].astype(np.int32),
                "incoming_dummy_count": incoming[batch_slice][valid].astype(np.int32),
                "nonzero_bin_count": nonzero[batch_slice][valid].astype(np.int32),
                "active_bin_count": active[batch_slice][valid].astype(np.int32),
                "requires_incoming_capability": (incoming[batch_slice][valid] > 0).astype(np.int16),
            }
            chunks.append(chunk)
            kept_rows += int(len(chunk["action_id"]))
            if kept_rows >= int(max_actions) * 4:
                break
        if kept_rows >= int(max_actions) * 4:
            break
    if not chunks:
        _diag_time_add(diagnostics, "compact_descriptor_generation_total_time_sec", time.perf_counter() - start_total)
        return []
    table = _candidate_table_concat(chunks)
    indices = _select_descriptor_indices(table, max(0, int(max_actions)), float(total))
    result = _candidate_descriptors_from_table(table, indices, sample_index=int(sample_index), sample_id=str(sample_id), true_label=int(true_label))
    if diagnostics is not None:
        diagnostics["compact_tensorized"] = True
        diagnostics["compact_tensorized_device"] = str(device)
        diagnostics["compact_tensorized_rows"] = int(len(table["action_id"]))
    _diag_time_add(diagnostics, "compact_tensorized_score_time_sec", score_time)
    _diag_add_many(diagnostics, "compact_after_max_generated_actions", result)  # type: ignore[arg-type]
    _diag_time_add(diagnostics, "compact_descriptor_generation_total_time_sec", time.perf_counter() - start_total)
    return result


def materialize_candidate_descriptor(
    descriptor: CandidateDescriptor,
    *,
    tam: np.ndarray,
    clean_total: float,
    protocol: str,
    max_action_budget: float,
    max_local_rate_peak: int,
) -> ExpandedAction | None:
    values = np.asarray(tam, dtype=np.float32)
    width = int(values.shape[1])
    if str(protocol) == "client_only" and int(descriptor.requires_incoming_capability) != 0:
        return None
    if str(protocol) not in {"client_only", "bidirectional_cooperative"}:
        raise ValueError(f"Unknown protocol={protocol!r}")
    dummy_count = int(descriptor.dummy_count)
    if dummy_count <= 0:
        return None
    if float(dummy_count / max(float(clean_total), 1.0)) > float(max_action_budget) + 1e-12:
        return None

    insert_start = int(np.clip(int(descriptor.insert_start), 0, max(0, width - 1)))
    insert_end = int(np.clip(max(int(descriptor.insert_end), insert_start + 1), insert_start + 1, width))
    per_direction = np.asarray(
        [int(descriptor.outgoing_dummy_count), int(descriptor.incoming_dummy_count)],
        dtype=np.int32,
    )
    counts = _interval_counts_from_per_direction(
        values=values,
        lo=int(insert_start),
        hi=int(insert_end),
        per_direction=per_direction,
        smoothing=str(descriptor.smoothing),
    )
    exact_dummy, outgoing_count, incoming_count, nonzero_count, count_signature = _count_metadata(counts)
    if int(exact_dummy) <= 0:
        return None
    local_rate_peak = int(counts.local_rate_peak) if isinstance(counts, SparseCounts) else int(counts.sum(axis=0).max())
    if int(local_rate_peak) > int(max_local_rate_peak):
        return None
    cost = float(exact_dummy / max(float(clean_total), 1.0))
    tier_bonus = {"primary": 1.0, "secondary": 0.65, "exploration": 0.30}.get(str(descriptor.tier), 0.20)
    type_bonus = {
        "dynamask_causal": 1.00,
        "shared_predecessor": 0.90,
        "two_window_coordinated_insert": 0.85,
        "gap_fill": 0.75,
        "burst_merge": 0.75,
        "rate_peak_predecessor": 0.65,
        "direction_balance": 0.60,
        "local_rate_smoothing": 0.55,
        "burst_extension": 0.50,
        "cumulative_shift": 0.45,
    }.get(str(descriptor.action_type), 0.35)
    dose_bonus = 1.0 + 0.15 * float(np.log1p(max(int(descriptor.dose), 1)))
    score = float((float(descriptor.mask_mass) + 1.0) * tier_bonus * type_bonus * dose_bonus / max(np.sqrt(cost), 1e-6))
    action = ExpandedAction(
        sample_index=int(descriptor.sample_index),
        sample_id=str(descriptor.sample_id),
        true_label=int(descriptor.true_label),
        window_id=int(descriptor.window_id),
        action_type=str(descriptor.action_type),
        tier=str(descriptor.tier),
        source=str(descriptor.source),
        affected_direction=str(descriptor.affected_direction),
        affected_start=int(descriptor.affected_start),
        affected_end=int(descriptor.affected_end),
        affected_center=int(descriptor.affected_center),
        insert_start=int(insert_start),
        insert_end=int(insert_end),
        insert_center=int(np.clip(int(descriptor.insert_center), 0, width - 1)),
        dose=int(descriptor.dose),
        direction_mode=str(descriptor.direction_mode),
        counts=counts,
        mask_mass=float(descriptor.mask_mass),
        local_count=float(descriptor.local_count),
        local_rate_peak=int(local_rate_peak),
        requires_incoming_capability=int(incoming_count > 0),
        allowed_violation_count=int(descriptor.allowed_violation_count),
        score_hint=float(score),
        parent="",
        dummy_count=int(exact_dummy),
        outgoing_dummy_count=int(outgoing_count),
        incoming_dummy_count=int(incoming_count),
        nonzero_bin_count=int(nonzero_count),
        active_bin_count=int(counts.active_bin_count) if isinstance(counts, SparseCounts) else int(np.count_nonzero(np.asarray(counts, dtype=np.int32).sum(axis=0))),
        count_signature=count_signature,
    )
    action = _with_clean_total(action, float(clean_total))
    if int(action.allowed_violation_count) > 0:
        return None
    object.__setattr__(action, "_descriptor_identity", descriptor_identity(descriptor))
    object.__setattr__(action, "_descriptor_action_id", int(descriptor.action_id))
    return action


def materialize_candidate_descriptors(
    descriptors: Iterable[CandidateDescriptor],
    *,
    tam: np.ndarray,
    clean_total: float,
    protocol: str,
    max_action_budget: float,
    max_local_rate_peak: int,
    limit: int | None = None,
) -> list[ExpandedAction]:
    actions: list[ExpandedAction] = []
    for descriptor in descriptors:
        action = materialize_candidate_descriptor(
            descriptor,
            tam=tam,
            clean_total=float(clean_total),
            protocol=str(protocol),
            max_action_budget=float(max_action_budget),
            max_local_rate_peak=int(max_local_rate_peak),
        )
        if action is None:
            continue
        actions.append(action)
        if limit is not None and len(actions) >= int(limit):
            break
    return actions


def generate_compact_action_descriptors(
    *,
    tam: np.ndarray,
    soft_mask: np.ndarray,
    sample_index: int,
    sample_id: str,
    true_label: int,
    protocol: str,
    clean_total: float | None = None,
    ratio: float = 0.10,
    max_windows: int = 8,
    closing_kernel: int = 5,
    merge_gap: int = 8,
    offsets: Iterable[int] = (0, -4, -8, -16, -32, -64, -96),
    absolute_doses: Iterable[int] = (1, 2, 4, 8, 16, 32),
    relative_doses: Iterable[float] = (0.10, 0.25, 0.50, 1.00),
    max_dose: int = 96,
    max_action_budget: float = 0.035,
    max_actions: int = 256,
    candidate_batch_size: int = 0,
    candidate_device: str = "cpu",
    diagnostics: dict | None = None,
) -> list[CandidateDescriptor]:
    total_start = time.perf_counter()
    profile_detail = bool(diagnostics.get("profile_detail", False)) if diagnostics is not None else False
    values = np.asarray(tam, dtype=np.float32)
    mask = np.asarray(soft_mask, dtype=np.float32)
    total = float(values.sum()) if clean_total is None else float(clean_total)
    total = max(total, 1.0)
    if diagnostics is not None:
        diagnostics["candidate_batch_size"] = int(candidate_batch_size)
        diagnostics["candidate_device"] = str(candidate_device)
    start = time.perf_counter()
    windows = candidate_windows_for_sample(
        mask,
        ratio=float(ratio),
        closing_kernel=int(closing_kernel),
        merge_gap=int(merge_gap),
        max_windows=int(max_windows),
        sample_index=int(sample_index),
    )
    _diag_time_add(diagnostics, "candidate_window_extract_time_sec", time.perf_counter() - start)
    width = int(values.shape[1])
    start = time.perf_counter()
    gap_centers = _gap_centers(values, max_gap=max(12, int(merge_gap) * 8))
    transitions = _transition_predecessors(values, radius=6)
    peaks = _rate_peak_predecessors(values, count=max(4, int(max_windows)))
    _diag_time_add(diagnostics, "structural_anchor_extract_time_sec", time.perf_counter() - start)
    if int(candidate_batch_size) > 0 and not profile_detail:
        return _generate_compact_action_descriptors_batched(
            values=values,
            windows=windows,
            gap_centers=gap_centers,
            transitions=transitions,
            peaks=peaks,
            sample_index=int(sample_index),
            sample_id=str(sample_id),
            true_label=int(true_label),
            protocol=str(protocol),
            total=float(total),
            offsets=offsets,
            absolute_doses=absolute_doses,
            relative_doses=relative_doses,
            max_dose=int(max_dose),
            max_action_budget=float(max_action_budget),
            max_actions=int(max_actions),
            candidate_batch_size=int(candidate_batch_size),
            candidate_device=str(candidate_device),
            diagnostics=diagnostics,
        )
    modes = list(DIRECTION_MODES)
    if str(protocol) == "client_only":
        modes = ["out-only"]
    descriptors: list[CandidateDescriptor] = []
    action_id = 0
    for window in windows:
        window_start = time.perf_counter() if profile_detail else 0.0
        local = _window_local_count(values, int(window.start), int(window.end))
        doses = _dose_values(local, absolute_doses, relative_doses, int(max_dose))
        action_widths = sorted({max(2, min(width, int(window.length))), max(4, min(width, int(round(window.length * 0.5)))), 8, 16, 32})
        anchors: list[tuple[str, str, int, str]] = []
        for base_name, anchor in (
            ("keypoint_start", int(window.start)),
            ("keypoint_center", int(window.center)),
            ("keypoint_end", int(window.end) - 1),
        ):
            for offset in offsets:
                center = int(np.clip(int(anchor) + int(offset), 0, width - 1))
                if center <= int(anchor):
                    anchors.append(("dynamask_causal", "primary", center, base_name))
        for center in peaks:
            if center <= int(window.end):
                anchors.append(("rate_peak_predecessor", "secondary", int(center), "rate_peak"))
        for center in transitions:
            if abs(int(center) - int(window.center)) <= max(96, int(window.length) * 2):
                anchors.append(("direction_balance", "secondary", int(center), "transition"))
        for center in gap_centers:
            if abs(int(center) - int(window.center)) <= max(96, int(window.length) * 2):
                anchors.append(("gap_fill", "secondary", int(center), "gap"))
                anchors.append(("burst_merge", "secondary", int(center), "gap"))
        anchors.append(("burst_extension", "secondary", max(0, int(window.start) - 2), "window_start"))
        anchors.append(("burst_extension", "secondary", min(width - 1, int(window.end)), "window_end"))
        anchors.append(("local_rate_smoothing", "secondary", int(window.center), "smooth"))
        anchors.append(("cumulative_shift", "exploration", max(0, int(window.start) - max(8, int(window.length))), "shift"))
        if profile_detail:
            _diag_time_add(diagnostics, "anchor_grid_build_time_sec", time.perf_counter() - window_start)
        for action_type, tier, center, source in anchors:
            for action_width in action_widths[:3]:
                smoothing = "inverse-rate" if action_type in {"gap_fill", "local_rate_smoothing"} else "edge" if action_type == "burst_extension" else "uniform"
                for dose in doses:
                    dummy_count = int(dose)
                    cost = float(dummy_count / max(float(total), 1.0))
                    if cost <= 0.0 or cost > float(max_action_budget) + 1e-12:
                        action_id += 1
                        continue
                    for mode in modes:
                        insert_start, insert_end = _clip_interval(int(center), int(action_width), width)
                        insert_width = max(1, int(insert_end) - int(insert_start))
                        outgoing, incoming = _descriptor_direction_counts(
                            tam=values,
                            dose=int(dose),
                            direction_mode=str(mode),
                            affected_start=int(window.start),
                            affected_end=int(window.end),
                            affected_direction=int(window.direction),
                        )
                        if str(protocol) == "client_only" and int(incoming) > 0:
                            action_id += 1
                            continue
                        nonzero, active = _descriptor_nonzero_estimate(int(outgoing), int(incoming), int(insert_width))
                        score = _descriptor_score(
                            action_type=str(action_type),
                            tier=str(tier),
                            mask_mass=float(window.mask_mass),
                            dose=int(dose),
                            cost=float(cost),
                        )
                        descriptor = CandidateDescriptor(
                            action_id=int(action_id),
                            sample_index=int(sample_index),
                            sample_id=str(sample_id),
                            true_label=int(true_label),
                            window_id=int(window.window_id),
                            action_type=str(action_type),
                            tier=str(tier),
                            source=str(source),
                            affected_direction=str(window.direction_name),
                            affected_start=int(window.start),
                            affected_end=int(window.end),
                            affected_center=int(window.center),
                            insert_start=int(insert_start),
                            insert_end=int(insert_end),
                            insert_center=int(np.clip(center, 0, width - 1)),
                            action_width=int(action_width),
                            dose=int(dose),
                            direction_mode=str(mode),
                            smoothing=str(smoothing),
                            mask_mass=float(window.mask_mass),
                            local_count=float(local),
                            score_hint=float(score),
                            dummy_count=int(dummy_count),
                            outgoing_dummy_count=int(outgoing),
                            incoming_dummy_count=int(incoming),
                            nonzero_bin_count=int(nonzero),
                            active_bin_count=int(active),
                            requires_incoming_capability=int(incoming > 0),
                        )
                        if profile_detail:
                            _diag_add(diagnostics, "compact_generated_before_filter", descriptor)  # type: ignore[arg-type]
                        descriptors.append(descriptor)
                        action_id += 1
        if len(descriptors) >= int(max_actions) * 4:
            break
    start = time.perf_counter()
    descriptors = _dedupe_descriptors(descriptors, total)
    _diag_time_add(diagnostics, "compact_deduplicate_time_sec", time.perf_counter() - start)
    start = time.perf_counter()
    limited = _diverse_limit_descriptors(descriptors, max(0, int(max_actions)), total)
    _diag_time_add(diagnostics, "compact_diverse_limit_time_sec", time.perf_counter() - start)
    _diag_add_many(diagnostics, "compact_after_max_generated_actions", limited)  # type: ignore[arg-type]
    _diag_time_add(diagnostics, "compact_descriptor_generation_total_time_sec", time.perf_counter() - total_start)
    return limited


def generate_expanded_actions(
    *,
    tam: np.ndarray,
    soft_mask: np.ndarray,
    sample_index: int,
    sample_id: str,
    true_label: int,
    protocol: str,
    clean_total: float | None = None,
    ratio: float = 0.10,
    max_windows: int = 8,
    closing_kernel: int = 5,
    merge_gap: int = 8,
    offsets: Iterable[int] = (0, -4, -8, -16, -32, -64, -96),
    absolute_doses: Iterable[int] = (1, 2, 4, 8, 16, 32),
    relative_doses: Iterable[float] = (0.10, 0.25, 0.50, 1.00),
    max_dose: int = 96,
    max_action_budget: float = 0.035,
    max_local_rate_peak: int = 16,
    include_pairs: bool = True,
    max_pair_actions: int = 64,
    max_actions: int = 256,
    diagnostics: dict | None = None,
) -> list[ExpandedAction]:
    total_start = time.perf_counter()
    profile_detail = bool(diagnostics.get("profile_detail", False)) if diagnostics is not None else False
    values = np.asarray(tam, dtype=np.float32)
    mask = np.asarray(soft_mask, dtype=np.float32)
    total = float(values.sum()) if clean_total is None else float(clean_total)
    total = max(total, 1.0)
    start = time.perf_counter()
    windows = candidate_windows_for_sample(
        mask,
        ratio=float(ratio),
        closing_kernel=int(closing_kernel),
        merge_gap=int(merge_gap),
        max_windows=int(max_windows),
        sample_index=int(sample_index),
    )
    _diag_time_add(diagnostics, "candidate_window_extract_time_sec", time.perf_counter() - start)
    actions: list[ExpandedAction] = []
    width = int(values.shape[1])
    start = time.perf_counter()
    gap_centers = _gap_centers(values, max_gap=max(12, int(merge_gap) * 8))
    transitions = _transition_predecessors(values, radius=6)
    peaks = _rate_peak_predecessors(values, count=max(4, int(max_windows)))
    _diag_time_add(diagnostics, "structural_anchor_extract_time_sec", time.perf_counter() - start)
    modes = list(DIRECTION_MODES)
    if str(protocol) == "client_only":
        modes = ["out-only"]
    for window in windows:
        window_start = time.perf_counter() if profile_detail else 0.0
        local = _window_local_count(values, int(window.start), int(window.end))
        doses = _dose_values(local, absolute_doses, relative_doses, int(max_dose))
        action_widths = sorted({max(2, min(width, int(window.length))), max(4, min(width, int(round(window.length * 0.5)))), 8, 16, 32})
        anchors: list[tuple[str, str, int, str]] = []
        for base_name, anchor in (
            ("keypoint_start", int(window.start)),
            ("keypoint_center", int(window.center)),
            ("keypoint_end", int(window.end) - 1),
        ):
            for offset in offsets:
                center = int(np.clip(int(anchor) + int(offset), 0, width - 1))
                if center <= int(anchor):
                    anchors.append(("dynamask_causal", "primary", center, base_name))
        for center in peaks:
            if center <= int(window.end):
                anchors.append(("rate_peak_predecessor", "secondary", int(center), "rate_peak"))
        for center in transitions:
            if abs(int(center) - int(window.center)) <= max(96, int(window.length) * 2):
                anchors.append(("direction_balance", "secondary", int(center), "transition"))
        for center in gap_centers:
            if abs(int(center) - int(window.center)) <= max(96, int(window.length) * 2):
                anchors.append(("gap_fill", "secondary", int(center), "gap"))
                anchors.append(("burst_merge", "secondary", int(center), "gap"))
        anchors.append(("burst_extension", "secondary", max(0, int(window.start) - 2), "window_start"))
        anchors.append(("burst_extension", "secondary", min(width - 1, int(window.end)), "window_end"))
        anchors.append(("local_rate_smoothing", "secondary", int(window.center), "smooth"))
        anchors.append(("cumulative_shift", "exploration", max(0, int(window.start) - max(8, int(window.length))), "shift"))
        if profile_detail:
            _diag_time_add(diagnostics, "anchor_grid_build_time_sec", time.perf_counter() - window_start)
        for action_type, tier, center, source in anchors:
            for action_width in action_widths[:3]:
                smoothing = "inverse-rate" if action_type in {"gap_fill", "local_rate_smoothing"} else "edge" if action_type == "burst_extension" else "uniform"
                for dose in doses:
                    for mode in modes:
                        start = time.perf_counter() if profile_detail else 0.0
                        action = _make_action(
                            sample_index=int(sample_index),
                            sample_id=str(sample_id),
                            true_label=int(true_label),
                            window=window,
                            tam=values,
                            clean_total=total,
                            action_type=str(action_type),
                            tier=str(tier),
                            source=str(source),
                            center=int(center),
                            action_width=int(action_width),
                            dose=int(dose),
                            direction_mode=str(mode),
                            smoothing=smoothing,
                            local_count=float(local),
                        )
                        if profile_detail:
                            _diag_time_add(diagnostics, "action_object_build_time_sec", time.perf_counter() - start)
                        if action is None:
                            continue
                        _diag_add(diagnostics, "generated_before_filter", action)
                        start = time.perf_counter() if profile_detail else 0.0
                        if _passes_protocol_and_constraints(
                            action,
                            protocol=str(protocol),
                            clean_total=total,
                            max_action_budget=float(max_action_budget),
                            max_local_rate_peak=int(max_local_rate_peak),
                        ):
                            _diag_add(diagnostics, "after_legality", action)
                            actions.append(action)
                        if profile_detail:
                            _diag_time_add(diagnostics, "legality_filter_time_sec", time.perf_counter() - start)
        if len(actions) >= int(max_actions) * 4:
            break
    start = time.perf_counter()
    actions = _dedupe(actions, total)
    _diag_time_add(diagnostics, "deduplicate_time_sec", time.perf_counter() - start)
    if include_pairs and len(windows) >= 2:
        start = time.perf_counter()
        primary = [item for item in actions if item.tier in {"primary", "secondary"}]
        primary = sorted(primary, key=lambda item: -float(item.score_hint))[: max(8, min(48, len(primary)))]
        pair_actions: list[ExpandedAction] = []
        for left_index, left in enumerate(primary):
            for right in primary[left_index + 1 :]:
                if int(left.window_id) == int(right.window_id):
                    continue
                pair = _pair_action(left, right, clean_total=total)
                if pair is None:
                    continue
                _diag_add(diagnostics, "generated_before_filter", pair)
                if _passes_protocol_and_constraints(
                    pair,
                    protocol=str(protocol),
                    clean_total=total,
                    max_action_budget=float(max_action_budget) * 1.75,
                    max_local_rate_peak=int(max_local_rate_peak) * 2,
                ):
                    _diag_add(diagnostics, "after_legality", pair)
                    pair_actions.append(pair)
                if len(pair_actions) >= int(max_pair_actions):
                    break
            if len(pair_actions) >= int(max_pair_actions):
                break
        _diag_time_add(diagnostics, "pair_action_build_time_sec", time.perf_counter() - start)
        start = time.perf_counter()
        actions = _dedupe(actions + pair_actions, total)
        _diag_time_add(diagnostics, "pair_deduplicate_time_sec", time.perf_counter() - start)
    start = time.perf_counter()
    score_hint_limited = sorted(actions, key=lambda item: (-float(item.score_hint), action_cost(item, total), str(item.action_type)))[: max(0, int(max_actions))]
    _diag_time_add(diagnostics, "score_hint_sort_time_sec", time.perf_counter() - start)
    _diag_add_many(diagnostics, "after_score_hint", score_hint_limited)
    start = time.perf_counter()
    limited = _diverse_limit(actions, max(0, int(max_actions)), total)
    _diag_time_add(diagnostics, "diverse_limit_time_sec", time.perf_counter() - start)
    _diag_add_many(diagnostics, "after_max_generated_actions", limited)
    _diag_time_add(diagnostics, "generate_expanded_actions_total_time_sec", time.perf_counter() - total_start)
    return limited
