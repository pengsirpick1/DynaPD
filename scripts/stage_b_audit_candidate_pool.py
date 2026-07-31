# -*- coding: utf-8 -*-
"""Audit Stage B2-D dummy candidate action pools and pair-gain opportunities."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import ExpandedAction, action_cost, action_identity, generate_expanded_actions
from dmmp.stage_b.objectives import ObjectiveWeights, original_class_objective_delta, probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT

from scripts.stage_b_run_dual_actuator import (
    _apply_delay,
    _default_checkpoint,
    _fast_refresh_mask,
    _initial_state,
    _load_archive,
    _load_raw_rows,
    _render_dummy_batch,
)


DEFAULT_ARCHIVE = (
    "results/stage_a_rf_native_w1800_n96_s60_seed0/"
    "stage_a_masks_rf/all_masks.npz"
)
DEFAULT_PREVIOUS_CSV = (
    "results/stage_b2d_dual_best_n96_bidir_d32_b10_v1/"
    "dual_sample_results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--previous-result-csv", default=DEFAULT_PREVIOUS_CSV)
    parser.add_argument("--output-dir", default="results/stage_b2d_candidate_pool_audit")
    parser.add_argument("--sample-indices", default="auto", help="Comma-separated indices, or 'auto' for flipped samples from previous CSV.")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--protocols", default="bidirectional_cooperative,client_only")
    parser.add_argument("--audit-states", default="clean_fast,after_first_delay")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trace-length", type=int, default=5000)
    parser.add_argument("--max-load-time", type=float, default=80.0)
    parser.add_argument("--rf-num-slots", type=int, default=1800)
    parser.add_argument("--df-architecture", default="project")
    parser.add_argument("--df-tam-adapter", default="signed_balance")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--delay-budget", type=int, default=32)
    parser.add_argument("--delay-length", type=int, default=32)
    parser.add_argument("--max-candidates", type=int, default=6, help="Current greedy selector only evaluates this many actions.")
    parser.add_argument("--max-generated-actions", type=int, default=32)
    parser.add_argument("--max-pair-actions", type=int, default=0)
    parser.add_argument("--pair-k", type=int, default=16, help="Top K generated actions used for explicit two-action pair audit.")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--max-action-budget", type=float, default=0.035)
    parser.add_argument("--max-local-rate-peak", type=int, default=16)
    parser.add_argument("--confidence-weight", type=float, default=0.40)
    parser.add_argument("--margin-weight", type=float, default=0.40)
    parser.add_argument("--entropy-weight", type=float, default=0.20)
    parser.add_argument("--renderer-batch-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--renderer-coordinate", default="rf_tam")
    parser.add_argument("--renderer-strategy", default="uniform_in_patch")
    parser.add_argument("--epsilons", default="0.005,0.01,0.02")
    parser.add_argument("--taus", default="0,0.005,0.01")
    return parser.parse_args()


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _auto_indices(path: Path, max_samples: int) -> list[int]:
    indices: list[int] = []
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    if row.get("method") == "alternating3_fast_refresh" and float(row.get("flip", 0.0)) > 0.5:
                        value = int(row["sample_index"])
                        if value not in indices:
                            indices.append(value)
                except (KeyError, TypeError, ValueError):
                    continue
                if len(indices) >= int(max_samples):
                    break
    if not indices:
        indices = list(range(max(1, int(max_samples))))
    return indices[: max(1, int(max_samples))]


def _runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(args.data_root),
        attacker=str(args.attacker),
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        seed=int(args.seed),
        max_trace_length=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
        delay_length=int(args.delay_length),
        delay_budget=int(args.delay_budget),
        delay_rho=1.0,
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_candidates=int(args.max_candidates),
        max_generated_actions=int(args.max_generated_actions),
        max_pair_actions=int(args.max_pair_actions),
        max_action_budget=float(args.max_action_budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
        max_dummy_steps=1,
        confidence_weight=float(args.confidence_weight),
        margin_weight=float(args.margin_weight),
        entropy_weight=float(args.entropy_weight),
        refresh_stride=32,
        renderer_batch_size=int(args.renderer_batch_size),
        batch_size=int(args.batch_size),
        renderer_coordinate=str(args.renderer_coordinate),
        renderer_strategy=str(args.renderer_strategy),
        progress=False,
    )


def _action_positions(counts: np.ndarray, limit: int = 16) -> str:
    arr = np.asarray(counts, dtype=np.int32)
    nz = np.argwhere(arr > 0)
    items = []
    for direction, slot in nz[: int(limit)]:
        name = "out" if int(direction) == 0 else "in"
        items.append(f"{name}:{int(slot)}:{int(arr[int(direction), int(slot)])}")
    if len(nz) > int(limit):
        items.append(f"...(+{len(nz) - int(limit)})")
    return ";".join(items)


def _action_row(
    *,
    action: ExpandedAction,
    rank: int,
    sample_index: int,
    sample_id: str,
    protocol: str,
    audit_state: str,
    clean_total: float,
    single_gain: float | None = None,
    single_efficiency: float | None = None,
    final_pred: int | None = None,
    final_margin: float | None = None,
) -> dict[str, Any]:
    counts = np.asarray(action.counts, dtype=np.int32)
    dummy_count = int(counts.sum())
    nonzero_bins = int(np.count_nonzero(counts))
    affected_center = int(action.affected_center)
    insert_center = int(action.insert_center)
    causal_offset = int(insert_center - affected_center)
    same_position = int(int(action.insert_start) <= affected_center < int(action.insert_end))
    structured = int(str(action.action_type) not in {"dynamask_causal", "stage_b0", "stage_b0_causal"})
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "audit_state": str(audit_state),
        "rank_by_score_hint": int(rank),
        "action_id": f"{sample_index}:{protocol}:{audit_state}:{rank}",
        "action_type": str(action.action_type),
        "tier": str(action.tier),
        "source": str(action.source),
        "parent": str(action.parent),
        "affected_direction": str(action.affected_direction),
        "direction_mode": str(action.direction_mode),
        "source_keypoint": f"{action.affected_direction}:{int(action.affected_start)}-{int(action.affected_end)}@{affected_center}",
        "anchor_bin": int(affected_center),
        "window_start": int(action.insert_start),
        "window_end": int(action.insert_end),
        "window_width": int(action.insert_end) - int(action.insert_start),
        "insert_center": int(insert_center),
        "causal_offset": int(causal_offset),
        "dose": int(action.dose),
        "dummy_count": int(dummy_count),
        "outgoing_dummy_count": int(counts[0].sum()),
        "incoming_dummy_count": int(counts[1].sum()),
        "nonzero_insert_bins": int(nonzero_bins),
        "positions_preview": _action_positions(counts),
        "estimated_cost": float(action_cost(action, clean_total)),
        "score_hint": float(action.score_hint),
        "mask_mass": float(action.mask_mass),
        "local_count": float(action.local_count),
        "local_rate_peak": int(action.local_rate_peak),
        "requires_incoming_capability": int(action.requires_incoming_capability),
        "is_same_position": int(same_position),
        "is_causal_predecessor": int(causal_offset < 0),
        "is_structured_action": int(structured),
        "single_gain": "" if single_gain is None else float(single_gain),
        "single_efficiency": "" if single_efficiency is None else float(single_efficiency),
        "single_final_pred": "" if final_pred is None else int(final_pred),
        "single_final_margin": "" if final_margin is None else float(final_margin),
    }


def _summary_distribution(actions: list[ExpandedAction], clean_total: float, key: str) -> str:
    counter = Counter(str(getattr(action, key)) for action in actions)
    return json.dumps(dict(sorted(counter.items())), ensure_ascii=False, sort_keys=True)


def _ratio(values: list[int] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()) if arr.size else 0.0


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _evaluate_actions(
    *,
    state,
    actions: list[ExpandedAction],
    original_prob: np.ndarray,
    label: int,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not actions:
        return np.zeros(0, dtype=np.float32), np.zeros((0, original_prob.size), dtype=np.float32), {}
    trial_counts = [np.asarray(state.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32) for action in actions]
    _traces, tams, _stats = _render_dummy_batch(base_trace=state.trace, counts_list=trial_counts, args=args)
    probs = predict_probabilities(attacker, tams, device=device, batch_size=int(args.batch_size))
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
    original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
    gains = original_class_objective_delta(original, reference, probs, weights)
    metrics = probability_metrics(original, probs, np.repeat(np.asarray([int(label)], dtype=np.int64), len(probs)))
    return gains.astype(np.float32), probs.astype(np.float32), metrics


def _pair_rows(
    *,
    state,
    actions: list[ExpandedAction],
    single_gains: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    budget: float,
    sample_index: int,
    sample_id: str,
    protocol: str,
    audit_state: str,
    attacker,
    device,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    pairs: list[tuple[int, int, np.ndarray, float]] = []
    max_dummy = int(round(float(clean_total) * float(budget)))
    for left_idx, left in enumerate(actions):
        for right_idx in range(left_idx + 1, len(actions)):
            right = actions[right_idx]
            if action_identity(left) == action_identity(right):
                continue
            counts = np.asarray(left.counts, dtype=np.int32) + np.asarray(right.counts, dtype=np.int32)
            dummy = int(counts.sum())
            if dummy <= 0 or dummy > max_dummy:
                continue
            pairs.append((left_idx, right_idx, counts, float(dummy / max(float(clean_total), 1.0))))
    if not pairs:
        return []
    trial_counts = [np.asarray(state.dummy_counts, dtype=np.int32) + counts for _li, _ri, counts, _cost in pairs]
    _traces, tams, _stats = _render_dummy_batch(base_trace=state.trace, counts_list=trial_counts, args=args)
    probs = predict_probabilities(attacker, tams, device=device, batch_size=int(args.batch_size))
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
    original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
    gains = original_class_objective_delta(original, reference, probs, weights)
    metrics = probability_metrics(original, probs, np.repeat(np.asarray([int(label)], dtype=np.int64), len(probs)))
    rows: list[dict[str, Any]] = []
    for pair_rank, ((left_idx, right_idx, counts, cost), gain, pred, margin) in enumerate(
        zip(pairs, gains.tolist(), metrics["evaluated_pred"].tolist(), metrics["original_class_margin"].tolist()),
        start=1,
    ):
        left = actions[left_idx]
        right = actions[right_idx]
        rows.append(
            {
                "sample_index": int(sample_index),
                "sample_id": str(sample_id),
                "protocol": str(protocol),
                "audit_state": str(audit_state),
                "pair_rank_unsorted": int(pair_rank),
                "left_rank": int(left_idx + 1),
                "right_rank": int(right_idx + 1),
                "left_type": str(left.action_type),
                "right_type": str(right.action_type),
                "left_source": str(left.source),
                "right_source": str(right.source),
                "left_gain": float(single_gains[left_idx]) if left_idx < len(single_gains) else "",
                "right_gain": float(single_gains[right_idx]) if right_idx < len(single_gains) else "",
                "pair_gain": float(gain),
                "pair_cost": float(cost),
                "pair_efficiency": float(float(gain) / max(float(cost), 1e-8)),
                "pair_dummy_count": int(np.asarray(counts).sum()),
                "pair_nonzero_bins": int(np.count_nonzero(counts)),
                "pair_outgoing_dummy_count": int(np.asarray(counts)[0].sum()),
                "pair_incoming_dummy_count": int(np.asarray(counts)[1].sum()),
                "has_negative_member": int((left_idx < len(single_gains) and float(single_gains[left_idx]) <= 0.0) or (right_idx < len(single_gains) and float(single_gains[right_idx]) <= 0.0)),
                "both_members_nonpositive": int(left_idx < len(single_gains) and right_idx < len(single_gains) and float(single_gains[left_idx]) <= 0.0 and float(single_gains[right_idx]) <= 0.0),
                "pair_final_pred": int(pred),
                "pair_final_margin": float(margin),
                "left_window": f"{int(left.insert_start)}-{int(left.insert_end)}",
                "right_window": f"{int(right.insert_start)}-{int(right.insert_end)}",
                "left_parent": str(left.parent),
                "right_parent": str(right.parent),
            }
        )
    rows.sort(key=lambda row: (-float(row["pair_efficiency"]), -float(row["pair_gain"]), float(row["pair_cost"])))
    for rank, row in enumerate(rows, start=1):
        row["pair_rank_by_efficiency"] = int(rank)
    return rows


def _summarize_context(
    *,
    actions: list[ExpandedAction],
    selector_actions: list[ExpandedAction],
    action_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
    sample_index: int,
    sample_id: str,
    protocol: str,
    audit_state: str,
    clean_total: float,
    budget: float,
    epsilons: list[float],
    taus: list[float],
) -> dict[str, Any]:
    dummy_counts = np.asarray([int(np.asarray(action.counts).sum()) for action in actions], dtype=np.float32)
    nonzero_bins = np.asarray([int(np.count_nonzero(np.asarray(action.counts))) for action in actions], dtype=np.float32)
    costs = np.asarray([action_cost(action, clean_total) for action in actions], dtype=np.float32)
    top_selector_rows = action_rows[: len(selector_actions)]
    selector_gains = np.asarray([float(row["single_gain"]) for row in top_selector_rows if row["single_gain"] != ""], dtype=np.float32)
    selector_costs = np.asarray([float(row["estimated_cost"]) for row in top_selector_rows if row["estimated_cost"] != ""], dtype=np.float32)
    if selector_gains.size:
        selected_eff_idx = int(np.argmax(selector_gains / np.maximum(selector_costs, 1e-8)))
        selected_gain = float(selector_gains[selected_eff_idx])
        selected_eff = float((selector_gains / np.maximum(selector_costs, 1e-8))[selected_eff_idx])
    else:
        selected_gain = 0.0
        selected_eff = 0.0
    if not actions:
        stop_reason = "candidate_pool_exhausted"
    elif selected_gain <= 0.0:
        stop_reason = "no_positive_single"
    else:
        stop_reason = "positive_single_available"
    positive_pairs = [row for row in pair_rows if float(row["pair_gain"]) > 0.0]
    if stop_reason == "no_positive_single" and not positive_pairs:
        stop_reason_pair = "no_positive_pair"
    elif positive_pairs:
        stop_reason_pair = "positive_pair_available"
    else:
        stop_reason_pair = stop_reason
    summary: dict[str, Any] = {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "audit_state": str(audit_state),
        "clean_packet_count": float(clean_total),
        "budget_bound": float(budget),
        "budget_dummy_limit": int(round(float(clean_total) * float(budget))),
        "generated_action_count": int(len(actions)),
        "selector_evaluated_action_count": int(len(selector_actions)),
        "action_type_distribution": _summary_distribution(actions, clean_total, "action_type"),
        "tier_distribution": _summary_distribution(actions, clean_total, "tier"),
        "direction_mode_distribution": _summary_distribution(actions, clean_total, "direction_mode"),
        "source_distribution": _summary_distribution(actions, clean_total, "source"),
        "mean_dummy_per_action": float(dummy_counts.mean()) if dummy_counts.size else 0.0,
        "median_dummy_per_action": float(np.median(dummy_counts)) if dummy_counts.size else 0.0,
        "max_dummy_per_action": int(dummy_counts.max()) if dummy_counts.size else 0,
        "mean_nonzero_bins_per_action": float(nonzero_bins.mean()) if nonzero_bins.size else 0.0,
        "mean_action_cost": float(costs.mean()) if costs.size else 0.0,
        "max_action_cost": float(costs.max()) if costs.size else 0.0,
        "same_position_ratio": _ratio([int(row["is_same_position"]) for row in action_rows]),
        "causal_predecessor_ratio": _ratio([int(row["is_causal_predecessor"]) for row in action_rows]),
        "structured_action_ratio": _ratio([int(row["is_structured_action"]) for row in action_rows]),
        "incoming_required_ratio": _ratio([int(action.requires_incoming_capability) for action in actions]),
        "selector_positive_single_count": int(np.sum(selector_gains > 0.0)) if selector_gains.size else 0,
        "selector_best_efficiency_gain": float(selected_gain),
        "selector_best_efficiency": float(selected_eff),
        "all_positive_single_count": int(sum(float(row["single_gain"]) > 0.0 for row in action_rows if row["single_gain"] != "")),
        "best_single_gain": max([float(row["single_gain"]) for row in action_rows if row["single_gain"] != ""], default=0.0),
        "strict_single_stop_reason": str(stop_reason),
        "pair_count_evaluated": int(len(pair_rows)),
        "positive_pair_count": int(len(positive_pairs)),
        "positive_pair_with_negative_member_count": int(sum(float(row["pair_gain"]) > 0.0 and int(row["has_negative_member"]) for row in pair_rows)),
        "positive_pair_both_nonpositive_count": int(sum(float(row["pair_gain"]) > 0.0 and int(row["both_members_nonpositive"]) for row in pair_rows)),
        "best_pair_gain": max([float(row["pair_gain"]) for row in pair_rows], default=0.0),
        "best_pair_efficiency": max([float(row["pair_efficiency"]) for row in pair_rows], default=0.0),
        "pair_stop_reason": str(stop_reason_pair),
    }
    for eps in epsilons:
        eligible = [
            row
            for row in action_rows
            if row["single_gain"] != "" and float(row["single_gain"]) >= -float(eps)
        ]
        summary[f"relaxed_first_step_count_eps_{eps:g}"] = int(len(eligible))
    for tau in taus:
        summary[f"positive_pair_count_tau_{tau:g}"] = int(sum(float(row["pair_gain"]) > float(tau) for row in pair_rows))
        summary[f"valley_pair_count_tau_{tau:g}"] = int(
            sum(float(row["pair_gain"]) > float(tau) and int(row["both_members_nonpositive"]) for row in pair_rows)
        )
    return summary


def _build_audit_state(
    *,
    audit_state: str,
    protocol: str,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[Any, np.ndarray]:
    state = _initial_state(raw_trace, original_tam, original_prob)
    if str(audit_state) == "clean_original":
        return state, np.asarray(original_mask, dtype=np.float32)
    if str(audit_state) == "clean_fast":
        return state, _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
    if str(audit_state) == "after_first_delay":
        mask0 = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
        state = _apply_delay(
            state=state,
            mask=mask0,
            protocol=str(protocol),
            delay_budget=max(1, int(round(int(args.delay_budget) / 3))),
            args=args,
        )
        probs = predict_probabilities(attacker, state.tam.reshape(1, *state.tam.shape), device=device, batch_size=int(args.batch_size))
        state.prob = probs[0].astype(np.float32)
        mask1 = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
        return state, mask1
    raise ValueError(f"Unknown audit_state={audit_state!r}")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    run_args = _runtime_args(args)
    device = resolve_device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if str(args.sample_indices).strip().lower() == "auto":
        sample_indices = _auto_indices(Path(args.previous_result_csv).resolve(), int(args.max_samples))
    else:
        sample_indices = _parse_csv_ints(args.sample_indices)[: max(1, int(args.max_samples))]
    protocols = _parse_csv_strings(args.protocols)
    audit_states = _parse_csv_strings(args.audit_states)
    epsilons = _parse_csv_floats(args.epsilons)
    taus = _parse_csv_floats(args.taus)

    archive = _load_archive(str(Path(args.archive).resolve()), 0)
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, run_args)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(prob.shape[1]),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )

    all_action_rows: list[dict[str, Any]] = []
    all_pair_rows: list[dict[str, Any]] = []
    all_summary_rows: list[dict[str, Any]] = []
    top20_rows: list[dict[str, Any]] = []

    for sample_index in sample_indices:
        if not (0 <= int(sample_index) < int(tam.shape[0])):
            raise IndexError(f"sample_index={sample_index} outside archive size {tam.shape[0]}")
        original_tam = np.asarray(tam[int(sample_index)], dtype=np.float32)
        original_mask = np.asarray(mask[int(sample_index)], dtype=np.float32)
        original_prob = np.asarray(prob[int(sample_index)], dtype=np.float32)
        raw_trace = np.asarray(raw_rows[int(sample_index)], dtype=np.float32)
        label = int(labels[int(sample_index)])
        sample_id = str(sample_ids[int(sample_index)])
        clean_total = max(float(original_tam.sum()), 1.0)
        for protocol in protocols:
            for audit_state in audit_states:
                state, soft_mask = _build_audit_state(
                    audit_state=str(audit_state),
                    protocol=str(protocol),
                    raw_trace=raw_trace,
                    original_tam=original_tam,
                    original_mask=original_mask,
                    original_prob=original_prob,
                    attacker=attacker,
                    device=device,
                    args=run_args,
                )
                remaining_dummy = int(round(clean_total * float(args.budget))) - int(np.asarray(state.dummy_counts).sum())
                actions = generate_expanded_actions(
                    tam=state.tam,
                    soft_mask=soft_mask,
                    sample_index=int(sample_index),
                    sample_id=sample_id,
                    true_label=label,
                    protocol=str(protocol),
                    clean_total=clean_total,
                    ratio=float(args.ratio),
                    max_windows=int(args.max_windows),
                    max_action_budget=float(args.max_action_budget),
                    max_local_rate_peak=int(args.max_local_rate_peak),
                    max_actions=int(args.max_generated_actions),
                    max_pair_actions=int(args.max_pair_actions),
                )
                actions = sorted(actions, key=lambda action: (-float(action.score_hint), action_cost(action, clean_total)))
                actions = [action for action in actions if int(action.counts.sum()) <= int(remaining_dummy)]
                gains, probs_eval, metrics = _evaluate_actions(
                    state=state,
                    actions=actions,
                    original_prob=original_prob,
                    label=label,
                    attacker=attacker,
                    device=device,
                    args=run_args,
                )
                rows: list[dict[str, Any]] = []
                for rank, action in enumerate(actions, start=1):
                    cost = action_cost(action, clean_total)
                    gain = float(gains[rank - 1]) if rank - 1 < len(gains) else None
                    eff = None if gain is None else float(gain / max(cost, 1e-8))
                    final_pred = int(metrics["evaluated_pred"][rank - 1]) if metrics else None
                    final_margin = float(metrics["original_class_margin"][rank - 1]) if metrics else None
                    row = _action_row(
                        action=action,
                        rank=rank,
                        sample_index=int(sample_index),
                        sample_id=sample_id,
                        protocol=str(protocol),
                        audit_state=str(audit_state),
                        clean_total=clean_total,
                        single_gain=gain,
                        single_efficiency=eff,
                        final_pred=final_pred,
                        final_margin=final_margin,
                    )
                    rows.append(row)
                all_action_rows.extend(rows)
                top20_rows.extend(rows[: int(args.top_n)])
                pair_actions = actions[: max(0, int(args.pair_k))]
                pair_gains = gains[: len(pair_actions)]
                pair_rows = _pair_rows(
                    state=state,
                    actions=pair_actions,
                    single_gains=pair_gains,
                    original_prob=original_prob,
                    label=label,
                    clean_total=clean_total,
                    budget=float(args.budget),
                    sample_index=int(sample_index),
                    sample_id=sample_id,
                    protocol=str(protocol),
                    audit_state=str(audit_state),
                    attacker=attacker,
                    device=device,
                    args=run_args,
                )
                all_pair_rows.extend(pair_rows)
                all_summary_rows.append(
                    _summarize_context(
                        actions=actions,
                        selector_actions=actions[: int(args.max_candidates)],
                        action_rows=rows,
                        pair_rows=pair_rows,
                        sample_index=int(sample_index),
                        sample_id=sample_id,
                        protocol=str(protocol),
                        audit_state=str(audit_state),
                        clean_total=clean_total,
                        budget=float(args.budget),
                        epsilons=epsilons,
                        taus=taus,
                    )
                )

    _write_csv(output_dir / "candidate_actions_all.csv", all_action_rows)
    _write_csv(output_dir / "candidate_actions_top20.csv", top20_rows)
    _write_csv(output_dir / "candidate_pair_audit.csv", all_pair_rows)
    _write_csv(output_dir / "candidate_pool_summary.csv", all_summary_rows)
    manifest = {
        "archive": str(Path(args.archive).resolve()),
        "checkpoint": str(Path(checkpoint).resolve()),
        "sample_indices": sample_indices,
        "protocols": protocols,
        "audit_states": audit_states,
        "candidate_actions_all": str(output_dir / "candidate_actions_all.csv"),
        "candidate_actions_top20": str(output_dir / "candidate_actions_top20.csv"),
        "candidate_pair_audit": str(output_dir / "candidate_pair_audit.csv"),
        "candidate_pool_summary": str(output_dir / "candidate_pool_summary.csv"),
        "max_candidates": int(args.max_candidates),
        "max_generated_actions": int(args.max_generated_actions),
        "max_pair_actions": int(args.max_pair_actions),
        "pair_k": int(args.pair_k),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
