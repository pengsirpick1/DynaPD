# -*- coding: utf-8 -*-
"""Run closed-loop Stage B controllers backed by a learned candidate policy."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import (
    ExpandedAction,
    action_identity,
    generate_compact_action_descriptors,
    generate_expanded_actions,
    materialize_candidate_descriptors,
)
from dmmp.stage_b.objectives import probability_metrics
from dmmp.stage_b.policy_data import encode_actions, encode_state_features
from dmmp.stage_b.policy_model import CandidateScoringPolicy, PolicyModelConfig
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_b2e_diverse_search import (
    DEFAULT_ARCHIVE,
    MethodConfig,
    _action_dummy_count,
    _default_checkpoint,
    _evaluate_actions,
    _load_archive,
    _load_raw_rows,
    _margin,
    _method_config,
    _parse_csv_floats,
    _parse_csv_strings,
    _prefilter_actions,
    _resource_fields,
    _run_controller,
    _runtime_args,
    _selection_scores,
)
from scripts.stage_b_run_dual_actuator import EvalState, _apply_delay, _fast_refresh_mask, _initial_state, _predict_one, _render_dummy


@dataclass
class StudentControllerResult:
    state: EvalState
    stop_reason: str
    scored_candidate_count: int
    candidate_step_count: int
    candidate_rf_eval_count: int
    state_rf_eval_count: int
    accepted_action_count: int
    exact_positive_count: int
    adaptive_expand_count: int
    no_positive_verify_count: int
    predicted_positive_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--policy_checkpoint", required=True)
    parser.add_argument("--split_file", default="")
    parser.add_argument("--split_name", choices=["archive", "train", "val", "test", "all"], default="archive")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocols", default="bidirectional_cooperative")
    parser.add_argument("--methods", default="oracle_top128,student_only,student_top4_verify,student_top4_to8_verify")
    parser.add_argument("--dummy_budgets", default="0.10")
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--sample_offset", type=int, default=0)
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
    parser.add_argument("--student_threshold", type=float, default=0.0)
    parser.add_argument("--verify_topk", type=int, default=4)
    parser.add_argument("--adaptive_topk", type=int, default=8)
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
    parser.add_argument("--compact_candidate_generation", action="store_true")
    parser.add_argument("--candidate_batch_size", type=int, default=0)
    parser.add_argument("--materialization_batch_size", type=int, default=128)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_score_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b_student_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_policy(path: str | Path, device: torch.device) -> CandidateScoringPolicy:
    payload = torch.load(Path(path), map_location=device)
    config = PolicyModelConfig(**payload.get("config", {}))
    model = CandidateScoringPolicy(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model


def _archive_selection_rows(args: argparse.Namespace) -> np.ndarray:
    with np.load(args.archive, allow_pickle=False) as arrays:
        n = int(arrays["tam"].shape[0])
        archive_source = np.asarray(arrays.get("source_indices", np.arange(n)), dtype=np.int64)
    if str(args.split_name) == "archive":
        rows = np.arange(n, dtype=np.int64)
    else:
        if not str(args.split_file):
            raise ValueError("--split_file is required when --split_name is train/val/test/all.")
        with np.load(args.split_file, allow_pickle=False) as splits:
            if str(args.split_name) == "all":
                requested = np.concatenate([np.asarray(splits[f"{name}_indices"], dtype=np.int64) for name in ("train", "val", "test")], axis=0)
            else:
                requested = np.asarray(splits[f"{args.split_name}_indices"], dtype=np.int64)
        position = {int(source): row for row, source in enumerate(archive_source.tolist())}
        missing = [int(source) for source in requested.tolist() if int(source) not in position]
        if missing:
            raise ValueError(f"Archive is missing {len(missing)} requested split source indices; first missing={missing[:5]}")
        rows = np.asarray([position[int(source)] for source in requested.tolist()], dtype=np.int64)
    start = max(0, int(args.sample_offset))
    if start:
        rows = rows[start:]
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]
    if rows.size == 0:
        raise ValueError("Archive row selection is empty.")
    return rows.astype(np.int64)


def _load_archive_rows(path: str | Path, rows: np.ndarray) -> dict[str, np.ndarray]:
    selected = np.asarray(rows, dtype=np.int64)
    with np.load(path, allow_pickle=False) as arrays:
        original_n = int(arrays["tam"].shape[0])
        payload: dict[str, np.ndarray] = {}
        for key in arrays.files:
            arr = arrays[key]
            if arr.shape[:1] == (original_n,):
                payload[key] = np.asarray(arr[selected])
            else:
                payload[key] = np.asarray(arr)
    return payload


def _score_actions(
    *,
    policy: CandidateScoringPolicy,
    state: EvalState,
    mask: np.ndarray,
    actions: list[ExpandedAction],
    original_prob: np.ndarray,
    budget: float,
    clean_total: float,
    round_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> np.ndarray:
    if not actions:
        return np.zeros(0, dtype=np.float32)
    dummy_used = int(np.asarray(state.dummy_counts, dtype=np.int32).sum())
    remaining_bandwidth = max(float(budget) - float(dummy_used / max(clean_total, 1.0)), 0.0)
    state_features = encode_state_features(
        current_prob=np.asarray(state.prob, dtype=np.float32),
        original_pred=int(np.argmax(original_prob)),
        remaining_bandwidth=remaining_bandwidth,
        remaining_delay=max(0.0, float(args.max_delay) - float(state.max_delay)),
        round_index=int(round_index),
        rounds=int(args.rounds),
        dummy_bandwidth_used=float(dummy_used / max(clean_total, 1.0)),
        avg_delay=float(state.avg_delay),
        p95_delay=float(state.p95_delay),
        max_delay=float(state.max_delay),
        max_delay_budget=float(args.max_delay),
    )
    action_features, action_counts = encode_actions(actions, clean_total=clean_total, width=int(args.rf_num_slots))
    state_tensor = np.concatenate([np.asarray(state.tam, dtype=np.float32), np.asarray(mask, dtype=np.float32)], axis=0)
    with torch.no_grad():
        output = policy(
            state_tensor=torch.as_tensor(state_tensor[None], dtype=torch.float32, device=device),
            state_features=torch.as_tensor(state_features[None], dtype=torch.float32, device=device),
            action_counts=torch.as_tensor(action_counts[None], dtype=torch.float32, device=device),
            action_features=torch.as_tensor(action_features[None], dtype=torch.float32, device=device),
            candidate_mask=torch.ones((1, len(actions)), dtype=torch.bool, device=device),
        )
    return output["scores"].detach().cpu().numpy()[0].astype(np.float32)


def _generate_student_actions(
    *,
    state: EvalState,
    mask: np.ndarray,
    protocol: str,
    budget: float,
    clean_total: float,
    label: int,
    sample_index: int,
    sample_id: str,
    config: MethodConfig,
    used: set[tuple],
    rng: random.Random,
    args: argparse.Namespace,
) -> list[ExpandedAction]:
    max_dummy = int(np.floor(float(clean_total) * float(budget) + 1e-9))
    remaining_dummy = int(max_dummy - int(np.asarray(state.dummy_counts, dtype=np.int32).sum()))
    if remaining_dummy <= 0:
        return []
    if bool(getattr(args, "compact_candidate_generation", False)):
        raw_descriptors = generate_compact_action_descriptors(
            tam=state.tam,
            soft_mask=mask,
            sample_index=int(sample_index),
            sample_id=str(sample_id),
            true_label=int(label),
            protocol=str(protocol),
            clean_total=float(clean_total),
            ratio=float(args.ratio),
            max_windows=int(args.max_windows),
            max_action_budget=float(args.max_action_budget),
            max_actions=int(config.raw_pool),
            candidate_batch_size=int(getattr(args, "candidate_batch_size", 0)),
            candidate_device=str(getattr(args, "candidate_score_device", getattr(args, "candidate_device", "auto"))),
            diagnostics={"profile_detail": False},
        )
        raw_descriptors = [
            action
            for action in raw_descriptors
            if tuple(action_identity(action)) not in used and 0 < int(action.dummy_count) <= remaining_dummy
        ]
        descriptor_selected = _prefilter_actions(raw_descriptors, config=config, clean_total=clean_total, args=args, rng=rng)
        actions: list[ExpandedAction] = []
        materialize_batch = max(1, int(getattr(args, "materialization_batch_size", 128)))
        for offset in range(0, len(descriptor_selected), materialize_batch):
            actions.extend(
                materialize_candidate_descriptors(
                    descriptor_selected[offset : offset + materialize_batch],
                    tam=state.tam,
                    clean_total=float(clean_total),
                    protocol=str(protocol),
                    max_action_budget=float(args.max_action_budget),
                    max_local_rate_peak=int(args.max_local_rate_peak),
                )
            )
        return [action for action in actions if action_identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]
    raw_actions = generate_expanded_actions(
        tam=state.tam,
        soft_mask=mask,
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        true_label=int(label),
        protocol=str(protocol),
        clean_total=float(clean_total),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_action_budget=float(args.max_action_budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
        max_actions=int(config.raw_pool),
        max_pair_actions=int(config.generator_pair_actions),
    )
    raw_actions = [action for action in raw_actions if action_identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]
    actions = _prefilter_actions(raw_actions, config=config, clean_total=clean_total, args=args, rng=rng)
    return [action for action in actions if action_identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]


def _render_accept(
    *,
    current: EvalState,
    action: ExpandedAction,
    attacker,
    device,
    args: argparse.Namespace,
) -> EvalState:
    selected_counts = np.asarray(current.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32)
    trace, tam, stats = _render_dummy(base_trace=current.trace, counts=selected_counts, args=args)
    prob = _predict_one(attacker, tam, device=device, args=args)
    return EvalState(
        trace=trace,
        tam=tam,
        prob=prob,
        dummy_counts=np.asarray(selected_counts, dtype=np.int32),
        dummy_bandwidth=float(stats["raw_bandwidth"]),
        avg_delay=float(current.avg_delay),
        p95_delay=float(current.p95_delay),
        max_delay=int(current.max_delay),
        delay_values=tuple(current.delay_values),
        outgoing_delay_values=tuple(current.outgoing_delay_values),
        incoming_delay_values=tuple(current.incoming_delay_values),
        selected_actions=list(current.selected_actions) + [action],
    )


def _best_verified_action(
    *,
    current: EvalState,
    actions: list[ExpandedAction],
    ranked_indices: np.ndarray,
    topk: int,
    already_verified: int,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[ExpandedAction | None, float, int, int]:
    lo = int(already_verified)
    hi = min(int(topk), len(ranked_indices))
    if hi <= lo:
        return None, 0.0, 0, already_verified
    subset_indices = [int(idx) for idx in ranked_indices[lo:hi].tolist()]
    subset = [actions[idx] for idx in subset_indices]
    gains, _probs, _metrics = _evaluate_actions(
        state=current,
        actions=subset,
        original_prob=original_prob,
        label=int(label),
        attacker=attacker,
        device=device,
        args=args,
    )
    if not len(gains):
        return None, 0.0, 0, hi
    scores = _selection_scores(gains, subset, clean_total, "absolute")
    best_local = int(np.argmax(scores))
    best_gain = float(gains[best_local])
    if best_gain > 0.0:
        return subset[best_local], best_gain, int(len(subset)), hi
    return None, best_gain, int(len(subset)), hi


def _run_student_controller(
    *,
    mode: str,
    policy: CandidateScoringPolicy,
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
) -> StudentControllerResult:
    config = _method_config("stratified_top128")
    clean_total = max(float(np.asarray(original_tam, dtype=np.float32).sum()), 1.0)
    current = _initial_state(raw_trace, original_tam, original_prob)
    used: set[tuple] = set()
    stable_name_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(mode)))
    rng = random.Random(int(args.seed) + int(sample_index) * 1009 + stable_name_seed)
    scored_candidates = 0
    candidate_steps = 0
    candidate_rf = 0
    state_rf = 0
    accepted = 0
    exact_positive = 0
    adaptive_expand = 0
    no_positive_verify = 0
    predicted_positive = 0
    stop_reason = "max_rounds_reached"
    max_dummy = int(np.floor(float(clean_total) * float(budget) + 1e-9))

    for round_index in range(max(1, int(args.rounds))):
        if int(args.max_delay) > 0:
            mask0 = _fast_refresh_mask(attacker, current.tam, original_prob, device=device)
            current = _apply_delay(
                state=current,
                mask=mask0,
                protocol=str(protocol),
                delay_budget=max(1, int(round(int(args.max_delay) / max(1, int(args.rounds))))),
                args=args,
            )
            current.prob = _predict_one(attacker, current.tam, device=device, args=args)
            state_rf += 1
        if _margin(current.prob, original_prob) <= float(args.margin_target):
            stop_reason = "target_reached"
            break
        if float(budget) <= 0.0:
            stop_reason = "no_dummy_budget"
            continue
        for _step_index in range(int(args.max_dummy_steps)):
            if _margin(current.prob, original_prob) <= float(args.margin_target):
                stop_reason = "target_reached"
                break
            if int(np.asarray(current.dummy_counts, dtype=np.int32).sum()) >= max_dummy:
                stop_reason = "bandwidth_10pct_reached"
                break
            mask = _fast_refresh_mask(attacker, current.tam, original_prob, device=device)
            actions = _generate_student_actions(
                state=current,
                mask=mask,
                protocol=str(protocol),
                budget=float(budget),
                clean_total=clean_total,
                label=int(label),
                sample_index=int(sample_index),
                sample_id=str(sample_id),
                config=config,
                used=used,
                rng=rng,
                args=args,
            )
            if not actions:
                stop_reason = "candidate_pool_exhausted"
                break
            policy_scores = _score_actions(
                policy=policy,
                state=current,
                mask=mask,
                actions=actions,
                original_prob=original_prob,
                budget=float(budget),
                clean_total=clean_total,
                round_index=int(round_index),
                args=args,
                device=device,
            )
            scored_candidates += int(len(actions))
            candidate_steps += 1
            ranked = np.argsort(-policy_scores, kind="mergesort")
            selected_action: ExpandedAction | None = None
            selected_gain = 0.0
            if str(mode) == "student_only":
                best_idx = int(ranked[0])
                best_score = float(policy_scores[best_idx])
                if best_score <= float(args.student_threshold):
                    stop_reason = "student_threshold_stop"
                    break
                predicted_positive += 1
                selected_action = actions[best_idx]
            elif str(mode) in {"student_top4_verify", "student_top4_to8_verify"}:
                selected_action, selected_gain, used_rf, verified = _best_verified_action(
                    current=current,
                    actions=actions,
                    ranked_indices=ranked,
                    topk=int(args.verify_topk),
                    already_verified=0,
                    original_prob=original_prob,
                    label=int(label),
                    clean_total=clean_total,
                    attacker=attacker,
                    device=device,
                    args=args,
                )
                candidate_rf += int(used_rf)
                if selected_action is None and str(mode) == "student_top4_to8_verify" and int(args.adaptive_topk) > int(args.verify_topk):
                    adaptive_expand += 1
                    selected_action, selected_gain, used_rf, verified = _best_verified_action(
                        current=current,
                        actions=actions,
                        ranked_indices=ranked,
                        topk=int(args.adaptive_topk),
                        already_verified=int(verified),
                        original_prob=original_prob,
                        label=int(label),
                        clean_total=clean_total,
                        attacker=attacker,
                        device=device,
                        args=args,
                    )
                    candidate_rf += int(used_rf)
                if selected_action is None:
                    no_positive_verify += 1
                    stop_reason = "no_positive_verified_action"
                    break
                exact_positive += int(selected_gain > 0.0)
            else:
                raise ValueError(f"Unknown student mode={mode!r}")

            current = _render_accept(current=current, action=selected_action, attacker=attacker, device=device, args=args)
            state_rf += 1
            used.add(action_identity(selected_action))
            accepted += 1
        if stop_reason in {
            "target_reached",
            "bandwidth_10pct_reached",
            "candidate_pool_exhausted",
            "student_threshold_stop",
            "no_positive_verified_action",
        }:
            if _margin(current.prob, original_prob) <= float(args.margin_target):
                stop_reason = "target_reached"
            break
    if _margin(current.prob, original_prob) <= float(args.margin_target):
        stop_reason = "target_reached"
    return StudentControllerResult(
        state=current,
        stop_reason=stop_reason,
        scored_candidate_count=int(scored_candidates),
        candidate_step_count=int(candidate_steps),
        candidate_rf_eval_count=int(candidate_rf),
        state_rf_eval_count=int(state_rf),
        accepted_action_count=int(accepted),
        exact_positive_count=int(exact_positive),
        adaptive_expand_count=int(adaptive_expand),
        no_positive_verify_count=int(no_positive_verify),
        predicted_positive_count=int(predicted_positive),
    )


def _sample_row(
    *,
    sample_index: int,
    sample_id: str,
    protocol: str,
    method: str,
    budget: float,
    original_prob: np.ndarray,
    state: EvalState,
    label: int,
    clean_total: float,
    runtime: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    metrics = probability_metrics(original_prob.reshape(1, -1), state.prob.reshape(1, -1), np.asarray([int(label)], dtype=np.int64))
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(method),
        "dummy_budget_bound": float(budget),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "target_margin_success": int(float(metrics["original_class_margin"][0]) <= 0.0),
        "original_pred": int(metrics["original_pred"][0]),
        "final_pred": int(metrics["evaluated_pred"][0]),
        "true_label": int(label),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "runtime_sec": float(runtime),
        **_resource_fields(state, clean_total),
        **extra,
    }


def _percentile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["protocol"]), str(row["method"]), float(row["dummy_budget_bound"]))].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, method, budget), items in sorted(groups.items()):
        bw = [_float(row, "actual_dummy_bandwidth") for row in items]
        runtimes = [_float(row, "runtime_sec") for row in items]
        out.append(
            {
                "protocol": protocol,
                "method": method,
                "dummy_budget_bound": float(budget),
                "samples": int(len(items)),
                "accuracy": float(np.mean([_float(row, "accuracy") for row in items])),
                "flip": float(np.mean([_float(row, "flip") for row in items])),
                "margin_success_rate": float(np.mean([_float(row, "target_margin_success") for row in items])),
                "mean_actual_bandwidth": float(np.mean(bw)) if bw else 0.0,
                "p95_actual_bandwidth": _percentile(bw, 95),
                "max_actual_bandwidth": max(bw) if bw else 0.0,
                "mean_delay_bins": float(np.mean([_float(row, "average_delay_bins") for row in items])),
                "mean_p95_delay_bins": float(np.mean([_float(row, "p95_delay_bins") for row in items])),
                "mean_candidate_rf_eval_count": float(np.mean([_float(row, "candidate_rf_eval_count") for row in items])),
                "mean_state_rf_eval_count": float(np.mean([_float(row, "state_rf_eval_count") for row in items])),
                "mean_scored_candidate_count": float(np.mean([_float(row, "scored_candidate_count") for row in items])),
                "mean_candidate_step_count": float(np.mean([_float(row, "candidate_step_count") for row in items])),
                "mean_accepted_actions": float(np.mean([_float(row, "accepted_action_count") for row in items])),
                "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else 0.0,
                "p95_runtime_sec": _percentile(runtimes, 95),
                "stop_reasons": json.dumps(dict(Counter(str(row.get("stop_reason", "")) for row in items)), sort_keys=True),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    run_args = _runtime_args(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    protocols = _parse_csv_strings(args.protocols)
    methods = _parse_csv_strings(args.methods)
    budgets = _parse_csv_floats(args.dummy_budgets)
    archive_rows = _archive_selection_rows(args)
    archive = _load_archive_rows(args.archive, archive_rows)
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
    policy = _load_policy(args.policy_checkpoint, device)
    oracle_config = _method_config("stratified_top128")
    rows: list[dict[str, Any]] = []
    total_jobs = len(protocols) * len(methods) * len(budgets) * int(tam.shape[0])
    done = 0
    for protocol in protocols:
        for method in methods:
            for budget in budgets:
                for sample_index in range(int(tam.shape[0])):
                    done += 1
                    if args.progress:
                        print(f"[student-controller] {done}/{total_jobs} method={method} B={budget:g} sample={sample_index}", flush=True)
                    clean_total = max(float(np.asarray(tam[sample_index], dtype=np.float32).sum()), 1.0)
                    start = time.perf_counter()
                    if str(method) == "oracle_top128":
                        state, aggregate, _funnel = _run_controller(
                            config=oracle_config,
                            protocol=str(protocol),
                            budget=float(budget),
                            raw_trace=np.asarray(raw_rows[sample_index], dtype=np.float32),
                            original_tam=np.asarray(tam[sample_index], dtype=np.float32),
                            original_mask=np.asarray(mask[sample_index], dtype=np.float32),
                            original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                            label=int(labels[sample_index]),
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            attacker=attacker,
                            device=device,
                            args=args,
                        )
                        runtime = time.perf_counter() - start
                        extra = {
                            "stop_reason": str(aggregate.get("stop_reason", "")),
                            "candidate_rf_eval_count": int(aggregate.get("rf_eval_count", 0)),
                            "state_rf_eval_count": 0,
                            "scored_candidate_count": int(aggregate.get("rf_eval_count", 0)),
                            "candidate_step_count": int(aggregate.get("candidate_step_count", 0)),
                            "accepted_action_count": int(len(state.selected_actions)),
                            "exact_positive_count": int(aggregate.get("accepted_single_count", 0)),
                            "adaptive_expand_count": 0,
                            "no_positive_verify_count": 0,
                            "predicted_positive_count": 0,
                        }
                    else:
                        result = _run_student_controller(
                            mode=str(method),
                            policy=policy,
                            protocol=str(protocol),
                            budget=float(budget),
                            raw_trace=np.asarray(raw_rows[sample_index], dtype=np.float32),
                            original_tam=np.asarray(tam[sample_index], dtype=np.float32),
                            original_mask=np.asarray(mask[sample_index], dtype=np.float32),
                            original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                            label=int(labels[sample_index]),
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            attacker=attacker,
                            device=device,
                            args=args,
                        )
                        runtime = time.perf_counter() - start
                        state = result.state
                        extra = {
                            "stop_reason": str(result.stop_reason),
                            "candidate_rf_eval_count": int(result.candidate_rf_eval_count),
                            "state_rf_eval_count": int(result.state_rf_eval_count),
                            "scored_candidate_count": int(result.scored_candidate_count),
                            "candidate_step_count": int(result.candidate_step_count),
                            "accepted_action_count": int(result.accepted_action_count),
                            "exact_positive_count": int(result.exact_positive_count),
                            "adaptive_expand_count": int(result.adaptive_expand_count),
                            "no_positive_verify_count": int(result.no_positive_verify_count),
                            "predicted_positive_count": int(result.predicted_positive_count),
                        }
                    rows.append(
                        _sample_row(
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            protocol=str(protocol),
                            method=str(method),
                            budget=float(budget),
                            original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                            state=state,
                            label=int(labels[sample_index]),
                            clean_total=clean_total,
                            runtime=float(runtime),
                            extra=extra,
                        )
                    )
    summary = _summarize(rows)
    _write_csv(output_dir / "student_policy_sample_results.csv", rows)
    _write_csv(output_dir / "student_policy_summary.csv", summary)
    manifest = {
        "archive": str(args.archive),
        "policy_checkpoint": str(Path(args.policy_checkpoint).resolve()),
        "attacker_checkpoint": str(Path(checkpoint).resolve()),
        "samples": int(tam.shape[0]),
        "sample_offset": int(args.sample_offset),
        "split_file": str(args.split_file),
        "split_name": str(args.split_name),
        "archive_row_min": int(np.min(archive_rows)),
        "archive_row_max": int(np.max(archive_rows)),
        "protocols": protocols,
        "methods": methods,
        "dummy_budgets": budgets,
        "summary": str(output_dir / "student_policy_summary.csv"),
        "sample_results": str(output_dir / "student_policy_sample_results.csv"),
    }
    (output_dir / "student_policy_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
