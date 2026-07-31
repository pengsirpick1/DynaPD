"""Serialization helpers for learned candidate-scoring policy data."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from dynapd.stage_b.expanded_generator import ExpandedAction, action_cost


ACTION_TYPES = (
    "dynamask_causal",
    "shared_predecessor",
    "two_window_coordinated_insert",
    "gap_fill",
    "burst_merge",
    "rate_peak_predecessor",
    "direction_balance",
    "local_rate_smoothing",
    "burst_extension",
    "cumulative_shift",
    "stage_b0_causal",
    "stage_b0",
    "unknown",
)
TIERS = ("primary", "secondary", "exploration", "unknown")
DIRECTION_MODES = ("out-only", "in-only", "both-equal", "current-ratio", "direction-balance", "coordinated", "unknown")
DIRECTION_BUCKETS = ("outgoing", "incoming", "bidirectional", "none")
AFFECTED_DIRECTIONS = ("out", "in", "0", "1", "unknown")

ACTION_FEATURE_NAMES = (
    "action_type_id",
    "tier_id",
    "direction_mode_id",
    "direction_bucket_id",
    "affected_direction_id",
    "affected_start_norm",
    "affected_end_norm",
    "affected_center_norm",
    "insert_start_norm",
    "insert_end_norm",
    "insert_center_norm",
    "insert_width_norm",
    "relative_offset_norm",
    "dose_log",
    "dummy_count_log",
    "nonzero_bins_log",
    "bandwidth_cost",
    "mask_mass_log",
    "local_count_log",
    "local_rate_peak_log",
    "requires_incoming",
    "score_hint_log",
)

STATE_FEATURE_NAMES = (
    "remaining_bandwidth",
    "remaining_delay_fraction",
    "round_fraction",
    "dummy_bandwidth_used",
    "avg_delay_fraction",
    "p95_delay_fraction",
    "max_delay_fraction",
    "rf_top1_confidence",
    "rf_margin",
    "rf_entropy_norm",
)


def _index(value: str, vocab: tuple[str, ...]) -> int:
    text = str(value)
    return int(vocab.index(text)) if text in vocab else int(vocab.index("unknown"))


def action_direction_bucket(action: ExpandedAction) -> str:
    out_count = int(getattr(action, "outgoing_dummy_count", 0))
    inc_count = int(getattr(action, "incoming_dummy_count", 0))
    if out_count <= 0 and inc_count <= 0:
        counts = np.asarray(action.counts, dtype=np.int32)
        out_count = int(counts[0].sum())
        inc_count = int(counts[1].sum())
    out = out_count > 0
    inc = inc_count > 0
    if out and inc:
        return "bidirectional"
    if inc:
        return "incoming"
    if out:
        return "outgoing"
    return "none"


def action_nonzero_bins(action: ExpandedAction) -> int:
    active = int(getattr(action, "active_bin_count", 0))
    return active if active > 0 else int(np.count_nonzero(np.asarray(action.counts, dtype=np.int32).sum(axis=0)))


def action_dummy_count(action: ExpandedAction) -> int:
    dummy = int(getattr(action, "dummy_count", 0))
    return dummy if dummy > 0 else int(np.asarray(action.counts, dtype=np.int32).sum())


def encode_action_features(action: ExpandedAction, *, clean_total: float, width: int) -> np.ndarray:
    w = max(float(width), 1.0)
    dummy = max(int(action_dummy_count(action)), 0)
    nonzero = max(int(action_nonzero_bins(action)), 0)
    insert_width = max(0, int(action.insert_end) - int(action.insert_start))
    relative_offset = float(int(action.insert_center) - int(action.affected_center))
    features = [
        _index(action.action_type, ACTION_TYPES),
        _index(action.tier, TIERS),
        _index(action.direction_mode, DIRECTION_MODES),
        _index(action_direction_bucket(action), DIRECTION_BUCKETS),
        _index(action.affected_direction, AFFECTED_DIRECTIONS),
        float(action.affected_start) / w,
        float(action.affected_end) / w,
        float(action.affected_center) / w,
        float(action.insert_start) / w,
        float(action.insert_end) / w,
        float(action.insert_center) / w,
        float(insert_width) / w,
        relative_offset / w,
        float(np.log1p(max(int(action.dose), 0))),
        float(np.log1p(dummy)),
        float(np.log1p(nonzero)),
        float(action_cost(action, clean_total)),
        float(np.log1p(max(float(action.mask_mass), 0.0))),
        float(np.log1p(max(float(action.local_count), 0.0))),
        float(np.log1p(max(float(action.local_rate_peak), 0.0))),
        float(int(action.requires_incoming_capability) > 0),
        float(np.log1p(max(float(action.score_hint), 0.0))),
    ]
    return np.asarray(features, dtype=np.float32)


def encode_actions(actions: Iterable[ExpandedAction], *, clean_total: float, width: int) -> tuple[np.ndarray, np.ndarray]:
    items = list(actions)
    if not items:
        return (
            np.zeros((0, len(ACTION_FEATURE_NAMES)), dtype=np.float32),
            np.zeros((0, 2, int(width)), dtype=np.int16),
        )
    features = np.vstack([encode_action_features(action, clean_total=clean_total, width=width) for action in items]).astype(np.float32)
    counts = np.stack([np.asarray(action.counts, dtype=np.int16) for action in items], axis=0)
    return features, counts


def probability_summary(prob: np.ndarray, *, original_pred: int | None = None) -> tuple[float, float, float]:
    values = np.asarray(prob, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return 0.0, 0.0, 0.0
    top = int(np.argmax(values)) if original_pred is None else int(original_pred)
    top_conf = float(values[top])
    others = np.delete(values, top)
    margin = float(top_conf - float(np.max(others))) if others.size else float(top_conf)
    entropy = -float(np.sum(values * np.log(np.maximum(values, 1e-8))))
    entropy_norm = entropy / float(np.log(max(values.size, 2)))
    return top_conf, margin, entropy_norm


def encode_state_features(
    *,
    current_prob: np.ndarray,
    original_pred: int,
    remaining_bandwidth: float,
    remaining_delay: float,
    round_index: int,
    rounds: int,
    dummy_bandwidth_used: float,
    avg_delay: float,
    p95_delay: float,
    max_delay: float,
    max_delay_budget: float,
) -> np.ndarray:
    top_conf, margin, entropy_norm = probability_summary(current_prob, original_pred=int(original_pred))
    delay_ref = max(float(max_delay_budget), 1.0)
    return np.asarray(
        [
            float(remaining_bandwidth),
            float(remaining_delay) / delay_ref,
            float(round_index) / max(float(rounds), 1.0),
            float(dummy_bandwidth_used),
            float(avg_delay) / delay_ref,
            float(p95_delay) / delay_ref,
            float(max_delay) / delay_ref,
            float(top_conf),
            float(margin),
            float(entropy_norm),
        ],
        dtype=np.float32,
    )


def load_record(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}
