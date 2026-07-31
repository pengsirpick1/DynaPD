"""Action table loading, filtering, and static ranking helpers for Stage B."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .objectives import ObjectiveWeights


@dataclass(frozen=True)
class CandidateAction:
    sample_index: int
    sample_id: str
    true_label: int
    window_id: int
    affected_direction: str
    affected_start: int
    affected_end: int
    affected_center: int
    insert_start: int
    insert_end: int
    insert_center: int
    offset: int
    dose: int
    direction_mode: str
    mask_mass: float
    dummy_packets: int
    bandwidth_overhead: float
    local_rate_peak: int
    causal_violation: int
    allowed_violation_count: int
    requires_incoming_capability: int
    js_div: float
    top1_drop: float
    margin_drop: float
    entropy_gain: float
    efficiency_top1_drop: float

    @property
    def group_key(self) -> tuple[int, str]:
        return int(self.window_id), str(self.direction_mode)

    @property
    def window_key(self) -> int:
        return int(self.window_id)

    @property
    def is_client_only(self) -> bool:
        return int(self.requires_incoming_capability) == 0


def _get(row: dict, key: str, default: str = "0") -> str:
    value = row.get(key, default)
    return default if value is None or value == "" else str(value)


def _parse_action(row: dict) -> CandidateAction:
    return CandidateAction(
        sample_index=int(float(_get(row, "sample_index"))),
        sample_id=_get(row, "sample_id", ""),
        true_label=int(float(_get(row, "true_label"))),
        window_id=int(float(_get(row, "window_id"))),
        affected_direction=_get(row, "affected_direction", "out"),
        affected_start=int(float(_get(row, "affected_start"))),
        affected_end=int(float(_get(row, "affected_end"))),
        affected_center=int(float(_get(row, "affected_center"))),
        insert_start=int(float(_get(row, "insert_start"))),
        insert_end=int(float(_get(row, "insert_end"))),
        insert_center=int(float(_get(row, "insert_center"))),
        offset=int(float(_get(row, "offset"))),
        dose=int(float(_get(row, "dose"))),
        direction_mode=_get(row, "direction_mode", "out-only"),
        mask_mass=float(_get(row, "mask_mass")),
        dummy_packets=int(float(_get(row, "dummy_packets"))),
        bandwidth_overhead=float(_get(row, "bandwidth_overhead")),
        local_rate_peak=int(float(_get(row, "local_rate_peak"))),
        causal_violation=int(float(_get(row, "causal_violation"))),
        allowed_violation_count=int(float(_get(row, "allowed_violation_count"))),
        requires_incoming_capability=int(float(_get(row, "requires_incoming_capability"))),
        js_div=float(_get(row, "js_div")),
        top1_drop=float(_get(row, "top1_drop")),
        margin_drop=float(_get(row, "margin_drop")),
        entropy_gain=float(_get(row, "entropy_gain")),
        efficiency_top1_drop=float(_get(row, "efficiency_top1_drop")),
    )


def load_action_table(path: str | Path) -> list[CandidateAction]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return [_parse_action(row) for row in csv.DictReader(handle)]


def single_action_utility(action: CandidateAction, *, num_classes: int, weights: ObjectiveWeights | None = None) -> float:
    w = weights or ObjectiveWeights()
    normalized_entropy_gain = float(action.entropy_gain) / max(float(np.log(max(int(num_classes), 2))), 1e-8)
    return float(
        float(w.confidence) * float(action.top1_drop)
        + float(w.margin) * float(action.margin_drop)
        + float(w.entropy) * normalized_entropy_gain
    )


def action_utility_components(action: CandidateAction, *, num_classes: int) -> np.ndarray:
    return np.asarray(
        [
            max(float(action.top1_drop), 0.0),
            max(float(action.margin_drop), 0.0),
            max(float(action.entropy_gain) / max(float(np.log(max(int(num_classes), 2))), 1e-8), 0.0),
            max(float(action.js_div), 0.0),
        ],
        dtype=np.float64,
    )


def pareto_filter(actions: list[CandidateAction], *, num_classes: int, eps: float = 1e-12) -> list[CandidateAction]:
    """Drop actions dominated by no-more-costly actions with all utilities >=."""
    if len(actions) <= 1:
        return actions
    costs = np.asarray([float(action.bandwidth_overhead) for action in actions], dtype=np.float64)
    utils = np.stack([action_utility_components(action, num_classes=int(num_classes)) for action in actions], axis=0)
    keep = np.ones(len(actions), dtype=bool)
    order = np.argsort(costs, kind="mergesort")
    for pos, idx in enumerate(order):
        if not keep[idx]:
            continue
        cheaper = order[: pos + 1]
        better = cheaper[
            (costs[cheaper] <= costs[idx] + float(eps))
            & np.all(utils[cheaper] >= utils[idx] - float(eps), axis=1)
            & np.any(utils[cheaper] > utils[idx] + float(eps), axis=1)
        ]
        if better.size:
            keep[idx] = False
    return [action for action, flag in zip(actions, keep) if bool(flag)]


def filter_protocol(actions: list[CandidateAction], protocol: str) -> list[CandidateAction]:
    name = str(protocol)
    if name == "client_only":
        return [action for action in actions if action.is_client_only]
    if name == "bidirectional_cooperative":
        return list(actions)
    raise ValueError(f"Unknown deployment protocol={protocol!r}")


def static_rank(
    actions: list[CandidateAction],
    method: str,
    *,
    tam: np.ndarray,
    rng: np.random.Generator,
    num_classes: int,
    weights: ObjectiveWeights | None = None,
) -> list[CandidateAction]:
    name = str(method)
    if name == "random":
        shuffled = list(actions)
        rng.shuffle(shuffled)
        return shuffled
    if name == "early":
        return sorted(actions, key=lambda action: (int(action.insert_center), int(action.dose), -float(action.mask_mass)))
    if name == "magnitude":
        values = np.asarray(tam, dtype=np.float32)

        def mag(action: CandidateAction) -> float:
            return float(np.abs(values[:, int(action.insert_start) : int(action.insert_end)]).sum())

        return sorted(actions, key=lambda action: (-mag(action), float(action.bandwidth_overhead), int(action.insert_center)))
    if name == "static_single_action_efficiency":
        return sorted(
            actions,
            key=lambda action: (
                -single_action_utility(action, num_classes=int(num_classes), weights=weights) / max(float(action.bandwidth_overhead), 1e-8),
                float(action.bandwidth_overhead),
            ),
        )
    if name == "dynamask_same_sequential":
        return [action for action in actions if int(action.offset) == 0]
    if name == "dynamask_causal_sequential":
        return list(actions)
    raise ValueError(f"Unsupported selection method={method!r}")


def prefilter_actions(
    actions: list[CandidateAction],
    method: str,
    *,
    tam: np.ndarray,
    rng: np.random.Generator,
    num_classes: int,
    max_candidates: int,
    weights: ObjectiveWeights | None = None,
) -> list[CandidateAction]:
    ranked = static_rank(actions, method, tam=tam, rng=rng, num_classes=int(num_classes), weights=weights)
    if int(max_candidates) > 0:
        return ranked[: int(max_candidates)]
    return ranked
