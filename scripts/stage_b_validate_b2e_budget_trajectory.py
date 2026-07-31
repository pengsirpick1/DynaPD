# -*- coding: utf-8 -*-
"""Validate budget-curve reuse for Stage B2-E.

The experiment compares independent runs at several dummy budgets against one
search trajectory run at Bmax, with intermediate states snapshotted as the best
prefix that stays within each budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.objectives import probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_b2e_diverse_search import (
    DEFAULT_ARCHIVE,
    MethodConfig,
    _default_checkpoint,
    _float,
    _identity,
    _load_archive,
    _load_raw_rows,
    _margin,
    _method_config,
    _parse_csv_floats,
    _resource_fields,
    _run_controller,
    _runtime_args,
    _write_csv,
)
from scripts.stage_b_run_dual_actuator import EvalState, _initial_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--budget_points", default="0.01,0.02,0.05,0.08,0.10")
    parser.add_argument("--bmax", type=float, default=0.10)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=96)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--true_recall_pool_size", type=int, default=0)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b2e_budget_trajectory_{args.attacker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _clone_state(state: EvalState) -> EvalState:
    return EvalState(
        trace=np.asarray(state.trace, dtype=np.float32).copy(),
        tam=np.asarray(state.tam, dtype=np.float32).copy(),
        prob=np.asarray(state.prob, dtype=np.float32).copy(),
        dummy_counts=np.asarray(state.dummy_counts, dtype=np.int32).copy(),
        dummy_bandwidth=float(state.dummy_bandwidth),
        avg_delay=float(state.avg_delay),
        p95_delay=float(state.p95_delay),
        max_delay=int(state.max_delay),
        delay_values=tuple(int(item) for item in state.delay_values),
        outgoing_delay_values=tuple(int(item) for item in state.outgoing_delay_values),
        incoming_delay_values=tuple(int(item) for item in state.incoming_delay_values),
        selected_actions=list(state.selected_actions),
    )


def _action_ids(state: EvalState) -> list[str]:
    return [repr(_identity(action)) for action in state.selected_actions]


def _action_signature(state: EvalState) -> str:
    payload = "\n".join(_action_ids(state)).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _state_row(
    *,
    sample_index: int,
    sample_id: str,
    protocol: str,
    config: MethodConfig,
    run_kind: str,
    budget: float,
    margin_target: float,
    original_prob: np.ndarray,
    state: EvalState,
    label: int,
    clean_total: float,
    runtime: float,
    stop_reason: str,
    rf_eval_count: int,
    candidate_step_count: int,
) -> dict[str, Any]:
    metrics = probability_metrics(
        original_prob.reshape(1, -1),
        state.prob.reshape(1, -1),
        np.asarray([int(label)], dtype=np.int64),
    )
    resource = _resource_fields(state, clean_total)
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(config.name),
        "run_kind": str(run_kind),
        "dummy_budget_bound": float(budget),
        "margin_target": float(margin_target),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "target_margin_success": int(float(metrics["original_class_margin"][0]) <= float(margin_target)),
        "original_pred": int(metrics["original_pred"][0]),
        "final_pred": int(metrics["evaluated_pred"][0]),
        "true_label": int(label),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "stop_reason": str(stop_reason),
        "rf_eval_count": int(rf_eval_count),
        "candidate_step_count": int(candidate_step_count),
        "runtime_sec": float(runtime),
        "action_signature": _action_signature(state),
        "action_id_sequence": json.dumps(_action_ids(state), ensure_ascii=True),
        **resource,
    }


def _event_row(
    *,
    sample_index: int,
    sample_id: str,
    event_index: int,
    event: dict[str, Any],
    state: EvalState,
    clean_total: float,
) -> dict[str, Any]:
    dummy_count = int(np.asarray(state.dummy_counts, dtype=np.int32).sum())
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "event_index": int(event_index),
        "event_type": str(event.get("event_type", "")),
        "round_index": int(event.get("round_index", -1)),
        "step_index": int(event.get("step_index", -1)),
        "selected_kind": str(event.get("selected_kind", "")),
        "selected_gain": float(event.get("selected_gain", 0.0)),
        "added_dummy": int(event.get("added_dummy", 0)),
        "cumulative_dummy": int(dummy_count),
        "cumulative_bandwidth": float(dummy_count / max(float(clean_total), 1.0)),
        "accepted_action_count": int(len(state.selected_actions)),
        "final_pred": int(np.argmax(state.prob)),
        "original_class_margin": float(_margin(state.prob, event["original_prob"])),
        "average_delay_bins": float(state.avg_delay),
        "p95_delay_bins": float(state.p95_delay),
        "maximum_delay_bins": int(state.max_delay),
        "action_signature": _action_signature(state),
    }


def _run_independent(
    *,
    config: MethodConfig,
    protocol: str,
    budget: float,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[EvalState, dict[str, Any], float]:
    start = time.perf_counter()
    state, aggregate, _rows = _run_controller(
        config=config,
        protocol=str(protocol),
        budget=float(budget),
        raw_trace=np.asarray(raw_trace, dtype=np.float32),
        original_tam=np.asarray(original_tam, dtype=np.float32),
        original_mask=np.asarray(original_mask, dtype=np.float32),
        original_prob=np.asarray(original_prob, dtype=np.float32),
        label=int(label),
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        attacker=attacker,
        device=device,
        args=args,
    )
    return state, aggregate, float(time.perf_counter() - start)


def _run_bmax_trajectory(
    *,
    config: MethodConfig,
    protocol: str,
    bmax: float,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[EvalState, dict[str, Any], float, list[dict[str, Any]], list[dict[str, Any]]]:
    clean_total = max(float(np.asarray(original_tam, dtype=np.float32).sum()), 1.0)
    initial = _initial_state(raw_trace, original_tam, original_prob)
    events: list[dict[str, Any]] = [
        {
            "event": {
                "event_type": "initial",
                "round_index": -1,
                "step_index": -1,
                "selected_kind": "",
                "selected_gain": 0.0,
                "added_dummy": 0,
                "original_prob": np.asarray(original_prob, dtype=np.float32),
            },
            "state": _clone_state(initial),
        }
    ]

    def observe(event: dict[str, Any], state: EvalState) -> None:
        payload = dict(event)
        payload["original_prob"] = np.asarray(original_prob, dtype=np.float32)
        events.append({"event": payload, "state": _clone_state(state)})

    start = time.perf_counter()
    state, aggregate, funnel_rows = _run_controller(
        config=config,
        protocol=str(protocol),
        budget=float(bmax),
        raw_trace=np.asarray(raw_trace, dtype=np.float32),
        original_tam=np.asarray(original_tam, dtype=np.float32),
        original_mask=np.asarray(original_mask, dtype=np.float32),
        original_prob=np.asarray(original_prob, dtype=np.float32),
        label=int(label),
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        attacker=attacker,
        device=device,
        args=args,
        state_observer=observe,
    )
    runtime = float(time.perf_counter() - start)
    if not events or _action_signature(events[-1]["state"]) != _action_signature(state):
        events.append(
            {
                "event": {
                    "event_type": "final",
                    "round_index": int(args.rounds),
                    "step_index": -1,
                    "selected_kind": "",
                    "selected_gain": 0.0,
                    "added_dummy": 0,
                    "original_prob": np.asarray(original_prob, dtype=np.float32),
                },
                "state": _clone_state(state),
            }
        )
    event_rows = [
        _event_row(
            sample_index=int(sample_index),
            sample_id=str(sample_id),
            event_index=idx,
            event=item["event"],
            state=item["state"],
            clean_total=clean_total,
        )
        for idx, item in enumerate(events)
    ]
    return state, aggregate, runtime, events, funnel_rows + event_rows


def _snapshots_from_events(events: list[dict[str, Any]], budgets: list[float], clean_total: float) -> dict[float, dict[str, Any]]:
    thresholds = {float(budget): int(np.floor(float(clean_total) * float(budget) + 1e-9)) for budget in budgets}
    snapshots: dict[float, dict[str, Any]] = {}
    for event_index, item in enumerate(events):
        state = item["state"]
        dummy = int(np.asarray(state.dummy_counts, dtype=np.int32).sum())
        for budget, max_dummy in thresholds.items():
            if dummy <= int(max_dummy):
                snapshots[budget] = {"event_index": int(event_index), "event": item["event"], "state": _clone_state(state)}
    return snapshots


def _compare_states(
    *,
    sample_index: int,
    sample_id: str,
    budget: float,
    clean_total: float,
    independent: EvalState,
    trajectory: EvalState,
    trajectory_event_index: int,
    bmax_state: EvalState,
) -> dict[str, Any]:
    independent_ids = _action_ids(independent)
    trajectory_ids = _action_ids(trajectory)
    bmax_ids = _action_ids(bmax_state)
    tam_diff = np.asarray(independent.tam, dtype=np.float32) - np.asarray(trajectory.tam, dtype=np.float32)
    prob_diff = np.asarray(independent.prob, dtype=np.float32) - np.asarray(trajectory.prob, dtype=np.float32)
    ind_dummy = int(np.asarray(independent.dummy_counts, dtype=np.int32).sum())
    traj_dummy = int(np.asarray(trajectory.dummy_counts, dtype=np.int32).sum())
    threshold = int(np.floor(float(clean_total) * float(budget) + 1e-9))
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "dummy_budget_bound": float(budget),
        "budget_dummy_threshold": int(threshold),
        "trajectory_event_index": int(trajectory_event_index),
        "independent_dummy_count": int(ind_dummy),
        "trajectory_dummy_count": int(traj_dummy),
        "independent_bandwidth": float(ind_dummy / max(float(clean_total), 1.0)),
        "trajectory_bandwidth": float(traj_dummy / max(float(clean_total), 1.0)),
        "dummy_count_equal": int(ind_dummy == traj_dummy),
        "both_within_budget": int(ind_dummy <= threshold and traj_dummy <= threshold),
        "independent_action_count": int(len(independent_ids)),
        "trajectory_action_count": int(len(trajectory_ids)),
        "action_sequence_equal": int(independent_ids == trajectory_ids),
        "independent_is_bmax_prefix": int(independent_ids == bmax_ids[: len(independent_ids)]),
        "trajectory_is_bmax_prefix": int(trajectory_ids == bmax_ids[: len(trajectory_ids)]),
        "prediction_equal": int(int(np.argmax(independent.prob)) == int(np.argmax(trajectory.prob))),
        "independent_pred": int(np.argmax(independent.prob)),
        "trajectory_pred": int(np.argmax(trajectory.prob)),
        "tam_exact_equal": int(np.array_equal(np.asarray(independent.tam), np.asarray(trajectory.tam))),
        "tam_close_1e_6": int(float(np.max(np.abs(tam_diff))) <= 1e-6),
        "tam_max_abs_diff": float(np.max(np.abs(tam_diff))) if tam_diff.size else 0.0,
        "tam_mean_abs_diff": float(np.mean(np.abs(tam_diff))) if tam_diff.size else 0.0,
        "prob_max_abs_diff": float(np.max(np.abs(prob_diff))) if prob_diff.size else 0.0,
    }


def _add_margin_diffs(row: dict[str, Any], independent: EvalState, trajectory: EvalState, original_prob: np.ndarray) -> None:
    ind_margin = float(_margin(independent.prob, original_prob))
    traj_margin = float(_margin(trajectory.prob, original_prob))
    row["independent_margin"] = ind_margin
    row["trajectory_margin"] = traj_margin
    row["margin_abs_diff"] = float(abs(ind_margin - traj_margin))


def _summarize(comparison_rows: list[dict[str, Any]], state_rows: list[dict[str, Any]], trajectory_runtime_by_sample: dict[int, float]) -> list[dict[str, Any]]:
    states_by_budget_kind: dict[tuple[float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in state_rows:
        states_by_budget_kind[(float(row["dummy_budget_bound"]), str(row["run_kind"]))].append(row)
    comparisons_by_budget: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for row in comparison_rows:
        comparisons_by_budget[float(row["dummy_budget_bound"])].append(row)

    out: list[dict[str, Any]] = []
    for budget, rows in sorted(comparisons_by_budget.items()):
        independent = states_by_budget_kind.get((budget, "independent"), [])
        trajectory = states_by_budget_kind.get((budget, "trajectory_prefix"), [])
        out.append(
            {
                "dummy_budget_bound": float(budget),
                "samples": int(len(rows)),
                "independent_accuracy": float(np.mean([_float(row, "accuracy") for row in independent])) if independent else 0.0,
                "trajectory_accuracy": float(np.mean([_float(row, "accuracy") for row in trajectory])) if trajectory else 0.0,
                "accuracy_delta_trajectory_minus_independent": float(
                    (np.mean([_float(row, "accuracy") for row in trajectory]) if trajectory else 0.0)
                    - (np.mean([_float(row, "accuracy") for row in independent]) if independent else 0.0)
                ),
                "independent_mean_bandwidth": float(np.mean([_float(row, "actual_dummy_bandwidth") for row in independent])) if independent else 0.0,
                "trajectory_mean_bandwidth": float(np.mean([_float(row, "actual_dummy_bandwidth") for row in trajectory])) if trajectory else 0.0,
                "independent_mean_dummy_count": float(np.mean([_float(row, "dummy_packet_count") for row in independent])) if independent else 0.0,
                "trajectory_mean_dummy_count": float(np.mean([_float(row, "dummy_packet_count") for row in trajectory])) if trajectory else 0.0,
                "prediction_match_rate": float(np.mean([_float(row, "prediction_equal") for row in rows])) if rows else 0.0,
                "action_sequence_match_rate": float(np.mean([_float(row, "action_sequence_equal") for row in rows])) if rows else 0.0,
                "dummy_count_match_rate": float(np.mean([_float(row, "dummy_count_equal") for row in rows])) if rows else 0.0,
                "tam_close_1e_6_rate": float(np.mean([_float(row, "tam_close_1e_6") for row in rows])) if rows else 0.0,
                "both_within_budget_rate": float(np.mean([_float(row, "both_within_budget") for row in rows])) if rows else 0.0,
                "independent_is_bmax_prefix_rate": float(np.mean([_float(row, "independent_is_bmax_prefix") for row in rows])) if rows else 0.0,
                "mean_margin_abs_diff": float(np.mean([_float(row, "margin_abs_diff") for row in rows])) if rows else 0.0,
                "max_margin_abs_diff": float(np.max([_float(row, "margin_abs_diff") for row in rows])) if rows else 0.0,
                "mean_tam_max_abs_diff": float(np.mean([_float(row, "tam_max_abs_diff") for row in rows])) if rows else 0.0,
                "max_tam_max_abs_diff": float(np.max([_float(row, "tam_max_abs_diff") for row in rows])) if rows else 0.0,
                "independent_mean_runtime_sec": float(np.mean([_float(row, "runtime_sec") for row in independent])) if independent else 0.0,
                "trajectory_bmax_mean_runtime_sec": float(np.mean(list(trajectory_runtime_by_sample.values()))) if trajectory_runtime_by_sample else 0.0,
            }
        )
    return out


def _write_event_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    cleaned = []
    for row in rows:
        if "event" in row and "state" in row:
            continue
        cleaned.append(row)
    if cleaned:
        _write_csv(path, cleaned)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    run_args = _runtime_args(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    config = _method_config(args.method)
    budget_points = sorted(set(float(b) for b in _parse_csv_floats(args.budget_points)))
    if not budget_points:
        raise ValueError("No budget points were provided.")
    if max(budget_points) > float(args.bmax) + 1e-12:
        raise ValueError("All budget points must be <= bmax.")

    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(args.data_root, source_indices, run_args)
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

    state_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    trajectory_runtime_by_sample: dict[int, float] = {}
    total_jobs = int(tam.shape[0])
    for sample_index in range(total_jobs):
        sample_id = str(sample_ids[sample_index])
        clean_total = max(float(np.asarray(tam[sample_index], dtype=np.float32).sum()), 1.0)
        if args.progress:
            print(f"[budget-trajectory] sample {sample_index + 1}/{total_jobs} {sample_id}", flush=True)

        bmax_state, bmax_aggregate, bmax_runtime, events, mixed_event_rows = _run_bmax_trajectory(
            config=config,
            protocol=str(args.protocol),
            bmax=float(args.bmax),
            raw_trace=np.asarray(raw_rows[sample_index], dtype=np.float32),
            original_tam=np.asarray(tam[sample_index], dtype=np.float32),
            original_mask=np.asarray(mask[sample_index], dtype=np.float32),
            original_prob=np.asarray(prob[sample_index], dtype=np.float32),
            label=int(labels[sample_index]),
            sample_index=int(sample_index),
            sample_id=sample_id,
            attacker=attacker,
            device=device,
            args=args,
        )
        trajectory_runtime_by_sample[int(sample_index)] = float(bmax_runtime)
        event_rows.extend([row for row in mixed_event_rows if "event_type" in row])
        snapshots = _snapshots_from_events(events, budget_points, clean_total)

        independent_by_budget: dict[float, tuple[EvalState, dict[str, Any], float]] = {}
        for budget in budget_points:
            state, aggregate, runtime = _run_independent(
                config=config,
                protocol=str(args.protocol),
                budget=float(budget),
                raw_trace=np.asarray(raw_rows[sample_index], dtype=np.float32),
                original_tam=np.asarray(tam[sample_index], dtype=np.float32),
                original_mask=np.asarray(mask[sample_index], dtype=np.float32),
                original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                label=int(labels[sample_index]),
                sample_index=int(sample_index),
                sample_id=sample_id,
                attacker=attacker,
                device=device,
                args=args,
            )
            independent_by_budget[float(budget)] = (state, aggregate, runtime)
            state_rows.append(
                _state_row(
                    sample_index=int(sample_index),
                    sample_id=sample_id,
                    protocol=str(args.protocol),
                    config=config,
                    run_kind="independent",
                    budget=float(budget),
                    margin_target=float(args.margin_target),
                    original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                    state=state,
                    label=int(labels[sample_index]),
                    clean_total=clean_total,
                    runtime=runtime,
                    stop_reason=str(aggregate.get("stop_reason", "")),
                    rf_eval_count=int(aggregate.get("rf_eval_count", 0)),
                    candidate_step_count=int(aggregate.get("candidate_step_count", 0)),
                )
            )

        for budget in budget_points:
            snapshot = snapshots.get(float(budget))
            if snapshot is None:
                snapshot = {"event_index": 0, "event": events[0]["event"], "state": _clone_state(events[0]["state"])}
            traj_state = snapshot["state"]
            state_rows.append(
                _state_row(
                    sample_index=int(sample_index),
                    sample_id=sample_id,
                    protocol=str(args.protocol),
                    config=config,
                    run_kind="trajectory_prefix",
                    budget=float(budget),
                    margin_target=float(args.margin_target),
                    original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                    state=traj_state,
                    label=int(labels[sample_index]),
                    clean_total=clean_total,
                    runtime=0.0,
                    stop_reason=str(bmax_aggregate.get("stop_reason", "")),
                    rf_eval_count=int(bmax_aggregate.get("rf_eval_count", 0)),
                    candidate_step_count=int(bmax_aggregate.get("candidate_step_count", 0)),
                )
            )
            independent_state = independent_by_budget[float(budget)][0]
            comparison = _compare_states(
                sample_index=int(sample_index),
                sample_id=sample_id,
                budget=float(budget),
                clean_total=clean_total,
                independent=independent_state,
                trajectory=traj_state,
                trajectory_event_index=int(snapshot["event_index"]),
                bmax_state=bmax_state,
            )
            _add_margin_diffs(comparison, independent_state, traj_state, np.asarray(prob[sample_index], dtype=np.float32))
            comparison_rows.append(comparison)

    summary_rows = _summarize(comparison_rows, state_rows, trajectory_runtime_by_sample)
    _write_csv(output_dir / "b2e_budget_trajectory_state_rows.csv", state_rows)
    _write_csv(output_dir / "b2e_budget_trajectory_comparison.csv", comparison_rows)
    _write_csv(output_dir / "b2e_budget_trajectory_summary.csv", summary_rows)
    _write_event_csv(output_dir / "b2e_budget_trajectory_events.csv", event_rows)
    manifest = {
        "archive": str(args.archive),
        "checkpoint": str(Path(checkpoint).resolve()),
        "samples": int(tam.shape[0]),
        "protocol": str(args.protocol),
        "method": str(config.name),
        "budget_points": budget_points,
        "bmax": float(args.bmax),
        "max_delay": int(args.max_delay),
        "margin_target": float(args.margin_target),
        "state_rows": str(output_dir / "b2e_budget_trajectory_state_rows.csv"),
        "comparison": str(output_dir / "b2e_budget_trajectory_comparison.csv"),
        "summary": str(output_dir / "b2e_budget_trajectory_summary.csv"),
        "events": str(output_dir / "b2e_budget_trajectory_events.csv"),
    }
    (output_dir / "b2e_budget_trajectory_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
