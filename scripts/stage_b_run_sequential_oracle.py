"""Run Stage B0 sequential budgeted action-selection oracle."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.data import load_cw_data
from dmmp.evaluation.attack_models import build_rf_tam_input, crop_or_pad_2d
from dmmp.projection.padding import PaddingTemplate, render_batch_variable
from dmmp.stage_a.additive_probe import CandidateWindow, counts_for_action
from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.action_selector import (
    CandidateAction,
    filter_protocol,
    load_action_table,
    pareto_filter,
    prefilter_actions,
    single_action_utility,
)
from dmmp.stage_b.objectives import ObjectiveWeights, objective_delta, probability_metrics
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, parse_csv_floats, parse_csv_strings


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"


METHODS = (
    "random",
    "early",
    "magnitude",
    "static_single_action_efficiency",
    "dynamask_same_sequential",
    "dynamask_causal_sequential",
)
PROTOCOLS = ("client_only", "bidirectional_cooperative")


def _default_checkpoint(attacker: str) -> str:
    return DEFAULT_FIXED_DF if str(attacker).lower() == "df" else DEFAULT_FIXED_RF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--action_table", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--budgets", default="0.02,0.05,0.10,0.15")
    parser.add_argument("--protocols", default="client_only,bidirectional_cooperative")
    parser.add_argument("--methods", default="random,early,magnitude,static_single_action_efficiency,dynamask_same_sequential,dynamask_causal_sequential")
    parser.add_argument("--max_candidates_per_sample", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--confidence_weight", type=float, default=0.30)
    parser.add_argument("--margin_weight", type=float, default=0.50)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--min_marginal_gain", type=float, default=0.0)
    parser.add_argument("--renderer_batch_size", type=int, default=64)
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
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage_b0_sequential_oracle_{args.attacker}_{stamp}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in parse_csv_strings(value)]
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    return methods


def _parse_protocols(value: str) -> list[str]:
    protocols = [item.strip() for item in parse_csv_strings(value)]
    invalid = sorted(set(protocols) - set(PROTOCOLS))
    if invalid:
        raise ValueError(f"Unknown protocols: {invalid}")
    return protocols


def _load_archive(path: str, max_samples: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    if int(max_samples) > 0:
        original_n = int(payload["tam"].shape[0])
        n = min(int(max_samples), original_n)
        for key, value in list(payload.items()):
            arr = np.asarray(value)
            if arr.shape[:1] == (original_n,):
                payload[key] = arr[:n]
    return payload


def _load_raw_rows(data_root: str, source_indices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    cfg = SimpleNamespace(
        data_root=str(data_root),
        seed=int(args.seed),
        val_ratio=0.10,
        test_ratio=0.10,
        max_samples=0,
        max_classes=0,
    )
    raw, _labels, _trace_ids, _splits, _source = load_cw_data(cfg)
    return np.asarray(raw[np.asarray(source_indices, dtype=np.int64)], dtype=np.float32)


def _candidate_counts(action: CandidateAction, tam_row: np.ndarray) -> np.ndarray:
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
    return counts_for_action(
        tam_row,
        window,
        offset=int(action.offset),
        dose=int(action.dose),
        direction_mode=str(action.direction_mode),
    ).counts.astype(np.int32)


def _render_evaluate_counts(
    *,
    attacker,
    raw_trace: np.ndarray,
    counts_list: list[np.ndarray],
    args: argparse.Namespace,
    device,
) -> tuple[np.ndarray, list[dict]]:
    probs: list[np.ndarray] = []
    stats_rows: list[dict] = []
    if not counts_list:
        return np.zeros((0, 0), dtype=np.float32), []
    clean = np.asarray(raw_trace, dtype=np.float32)
    for start in range(0, len(counts_list), max(1, int(args.renderer_batch_size))):
        batch_counts = counts_list[start : start + max(1, int(args.renderer_batch_size))]
        clean_batch = np.repeat(clean.reshape(1, -1), len(batch_counts), axis=0)
        templates = [
            PaddingTemplate(
                counts=np.asarray(counts, dtype=np.int32),
                target_n_pad=int(np.asarray(counts).sum()),
                actual_n_pad=int(np.asarray(counts).sum()),
                target_bandwidth=0.0,
                metadata={"method": "stage_b0_sequential_oracle"},
            )
            for counts in batch_counts
        ]
        traces, _origins, stats = render_batch_variable(
            clean_batch,
            templates,
            seed=int(args.seed) + int(start),
            strategy=str(args.renderer_strategy),
            coordinate=str(args.renderer_coordinate),
            coordinate_length=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
        )
        padded = np.vstack([crop_or_pad_2d(trace, int(args.max_trace_length))[0] for trace in traces]).astype(np.float32)
        tam = build_rf_tam_input(
            padded,
            max_len=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
            num_slots=int(args.rf_num_slots),
        )
        probs.append(predict_probabilities(attacker, tam, device=device, batch_size=int(args.batch_size)))
        for local in range(len(batch_counts)):
            stats_rows.append(
                {
                    "raw_bandwidth": float(stats["raw_bandwidth"][local]),
                    "raw_real_packet_retention": float(stats["raw_real_packet_retention"][local]),
                    "raw_length": int(stats["raw_lengths"][local]),
                }
            )
    return np.concatenate(probs, axis=0).astype(np.float32), stats_rows


def _metric_row(original_prob: np.ndarray, prob: np.ndarray, label: int) -> dict:
    metrics = probability_metrics(original_prob.reshape(1, -1), prob.reshape(1, -1), np.asarray([int(label)], dtype=np.int64))
    return {key: (float(value[0]) if np.asarray(value).dtype.kind != "i" else int(value[0])) for key, value in metrics.items()}


def _empty_state(original_prob: np.ndarray, label: int, original_utility: float) -> dict:
    row = _metric_row(original_prob, original_prob, int(label))
    row.update(
        {
            "step": 0,
            "action": None,
            "cumulative_cost": 0.0,
            "incremental_cost": 0.0,
            "marginal_gain": 0.0,
            "marginal_efficiency": 0.0,
            "utility_before": float(original_utility),
            "utility_after": float(original_utility),
            "selected_action_count": 0,
            "incoming_action_count": 0,
            "overlap_window_count": 0,
            "raw_bandwidth": 0.0,
            "raw_real_packet_retention": 1.0,
            "raw_length": 0,
        }
    )
    return row


def _action_fields(action: CandidateAction | None) -> dict:
    if action is None:
        return {
            "window_id": "",
            "insert_center": "",
            "affected_center": "",
            "offset": "",
            "dose": "",
            "direction_mode": "",
            "requires_incoming_capability": "",
            "single_action_top1_drop": "",
            "single_action_margin_drop": "",
            "single_action_entropy_gain": "",
            "single_action_js_div": "",
        }
    return {
        "window_id": int(action.window_id),
        "insert_center": int(action.insert_center),
        "affected_center": int(action.affected_center),
        "offset": int(action.offset),
        "dose": int(action.dose),
        "direction_mode": str(action.direction_mode),
        "requires_incoming_capability": int(action.requires_incoming_capability),
        "single_action_top1_drop": float(action.top1_drop),
        "single_action_margin_drop": float(action.margin_drop),
        "single_action_entropy_gain": float(action.entropy_gain),
        "single_action_js_div": float(action.js_div),
    }


def _conflicts(action: CandidateAction, used_groups: set[tuple[int, str]]) -> bool:
    return action.group_key in used_groups


def _history_state(
    *,
    action: CandidateAction,
    step: int,
    cumulative_cost: float,
    incremental_cost: float,
    marginal_gain: float,
    marginal_efficiency: float,
    utility_before: float,
    utility_after: float,
    original_prob: np.ndarray,
    prob: np.ndarray,
    label: int,
    selected_actions: list[CandidateAction],
    stats: dict,
) -> dict:
    row = _metric_row(original_prob, prob, int(label))
    incoming = int(sum(int(item.requires_incoming_capability) > 0 for item in selected_actions))
    windows = [int(item.window_id) for item in selected_actions]
    overlap = int(len(windows) - len(set(windows)))
    row.update(
        {
            "step": int(step),
            "action": action,
            "cumulative_cost": float(cumulative_cost),
            "incremental_cost": float(incremental_cost),
            "marginal_gain": float(marginal_gain),
            "marginal_efficiency": float(marginal_efficiency),
            "utility_before": float(utility_before),
            "utility_after": float(utility_after),
            "selected_action_count": int(len(selected_actions)),
            "incoming_action_count": incoming,
            "overlap_window_count": overlap,
            "raw_bandwidth": float(stats.get("raw_bandwidth", cumulative_cost)),
            "raw_real_packet_retention": float(stats.get("raw_real_packet_retention", 1.0)),
            "raw_length": int(stats.get("raw_length", 0)),
        }
    )
    return row


def _run_static_sequence(
    *,
    actions: list[CandidateAction],
    method: str,
    tam_row: np.ndarray,
    raw_trace: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    max_budget: float,
    attacker,
    args: argparse.Namespace,
    device,
    weights: ObjectiveWeights,
) -> list[dict]:
    original_utility = 0.0
    history = [_empty_state(original_prob, int(label), original_utility)]
    current_counts = np.zeros_like(tam_row, dtype=np.int32)
    current_prob = original_prob.astype(np.float32)
    current_utility = original_utility
    used_groups: set[tuple[int, str]] = set()
    selected: list[CandidateAction] = []
    for action in actions:
        if _conflicts(action, used_groups):
            continue
        counts = _candidate_counts(action, tam_row)
        next_counts = current_counts + counts
        incremental_cost = float(counts.sum() / max(float(tam_row.sum()), 1.0))
        cumulative_cost = float(next_counts.sum() / max(float(tam_row.sum()), 1.0))
        if cumulative_cost > float(max_budget) + 1e-9 or incremental_cost <= 0.0:
            continue
        probs, stats_rows = _render_evaluate_counts(attacker=attacker, raw_trace=raw_trace, counts_list=[next_counts], args=args, device=device)
        next_prob = probs[0]
        gain = float(objective_delta(current_prob.reshape(1, -1), next_prob.reshape(1, -1), weights)[0])
        next_utility = float(current_utility + gain)
        selected.append(action)
        used_groups.add(action.group_key)
        current_counts = next_counts
        current_prob = next_prob
        previous_utility = current_utility
        current_utility = next_utility
        history.append(
            _history_state(
                action=action,
                step=len(history),
                cumulative_cost=cumulative_cost,
                incremental_cost=incremental_cost,
                marginal_gain=gain,
                marginal_efficiency=gain / max(incremental_cost, 1e-8),
                utility_before=previous_utility,
                utility_after=current_utility,
                original_prob=original_prob,
                prob=current_prob,
                label=int(label),
                selected_actions=selected,
                stats=stats_rows[0],
            )
        )
    return history


def _run_dynamic_sequence(
    *,
    actions: list[CandidateAction],
    tam_row: np.ndarray,
    raw_trace: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    max_budget: float,
    attacker,
    args: argparse.Namespace,
    device,
    weights: ObjectiveWeights,
) -> list[dict]:
    original_utility = 0.0
    history = [_empty_state(original_prob, int(label), original_utility)]
    current_counts = np.zeros_like(tam_row, dtype=np.int32)
    current_prob = original_prob.astype(np.float32)
    current_utility = original_utility
    used_groups: set[tuple[int, str]] = set()
    selected: list[CandidateAction] = []
    remaining = list(actions)
    while remaining:
        trial_actions, trial_counts, trial_incremental = [], [], []
        for action in remaining:
            if _conflicts(action, used_groups):
                continue
            counts = _candidate_counts(action, tam_row)
            incremental_cost = float(counts.sum() / max(float(tam_row.sum()), 1.0))
            next_counts = current_counts + counts
            cumulative_cost = float(next_counts.sum() / max(float(tam_row.sum()), 1.0))
            if incremental_cost <= 0.0 or cumulative_cost > float(max_budget) + 1e-9:
                continue
            trial_actions.append(action)
            trial_counts.append(next_counts)
            trial_incremental.append(incremental_cost)
        if not trial_actions:
            break
        probs, stats_rows = _render_evaluate_counts(attacker=attacker, raw_trace=raw_trace, counts_list=trial_counts, args=args, device=device)
        reference = np.repeat(current_prob.reshape(1, -1), len(probs), axis=0)
        gains = objective_delta(reference, probs, weights)
        efficiencies = gains / np.maximum(np.asarray(trial_incremental, dtype=np.float32), 1e-8)
        best_index = int(np.argmax(efficiencies))
        if float(gains[best_index]) <= float(args.min_marginal_gain):
            break
        action = trial_actions[best_index]
        current_counts = trial_counts[best_index]
        previous_utility = current_utility
        current_utility = float(current_utility + gains[best_index])
        current_prob = probs[best_index]
        selected.append(action)
        used_groups.add(action.group_key)
        remaining = [item for item in remaining if item != action]
        history.append(
            _history_state(
                action=action,
                step=len(history),
                cumulative_cost=float(current_counts.sum() / max(float(tam_row.sum()), 1.0)),
                incremental_cost=float(trial_incremental[best_index]),
                marginal_gain=float(gains[best_index]),
                marginal_efficiency=float(efficiencies[best_index]),
                utility_before=previous_utility,
                utility_after=current_utility,
                original_prob=original_prob,
                prob=current_prob,
                label=int(label),
                selected_actions=selected,
                stats=stats_rows[best_index],
            )
        )
    return history


def _state_for_budget(history: list[dict], budget: float) -> dict:
    eligible = [state for state in history if float(state["cumulative_cost"]) <= float(budget) + 1e-9]
    return eligible[-1] if eligible else history[0]


def _step_row(
    *,
    state: dict,
    sample_index: int,
    sample_id: str,
    protocol: str,
    method: str,
    max_budget: float,
) -> dict:
    action = state.get("action")
    row = {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(method),
        "max_budget": float(max_budget),
        "step": int(state["step"]),
        "incremental_cost": float(state["incremental_cost"]),
        "cumulative_cost": float(state["cumulative_cost"]),
        "marginal_gain": float(state["marginal_gain"]),
        "marginal_efficiency": float(state["marginal_efficiency"]),
        "utility_before": float(state["utility_before"]),
        "utility_after": float(state["utility_after"]),
        "accuracy": float(state["accuracy"]),
        "flip": float(state["flip"]),
        "top1_drop": float(state["top1_drop"]),
        "max_confidence_drop": float(state["max_confidence_drop"]),
        "margin_drop": float(state["margin_drop"]),
        "entropy_gain": float(state["entropy_gain"]),
        "normalized_entropy_gain": float(state["normalized_entropy_gain"]),
        "js_div": float(state["js_div"]),
        "selected_action_count": int(state["selected_action_count"]),
        "incoming_action_count": int(state["incoming_action_count"]),
        "incoming_action_ratio": float(state["incoming_action_count"] / max(int(state["selected_action_count"]), 1)),
        "overlap_window_count": int(state["overlap_window_count"]),
        "raw_bandwidth": float(state["raw_bandwidth"]),
        "raw_real_packet_retention": float(state["raw_real_packet_retention"]),
        "raw_length": int(state["raw_length"]),
        **_action_fields(action),
    }
    return row


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    budgets = sorted(parse_csv_floats(args.budgets))
    protocols = _parse_protocols(args.protocols)
    methods = _parse_methods(args.methods)
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    original_prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, args)
    actions = load_action_table(args.action_table)
    actions = [action for action in actions if int(action.sample_index) < int(tam.shape[0])]
    actions_by_sample: dict[int, list[CandidateAction]] = {}
    for action in actions:
        actions_by_sample.setdefault(int(action.sample_index), []).append(action)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=original_prob.shape[1],
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    max_budget = float(max(budgets))
    step_rows: list[dict] = []
    sample_rows: list[dict] = []
    filter_rows: list[dict] = []
    rng_root = np.random.default_rng(int(args.seed))
    for sample_index in range(tam.shape[0]):
        if args.progress and (sample_index == 0 or (sample_index + 1) % 10 == 0 or sample_index + 1 == tam.shape[0]):
            print(f"[stage_b0] sample {sample_index + 1}/{tam.shape[0]}", flush=True)
        sample_actions = actions_by_sample.get(int(sample_index), [])
        for protocol in protocols:
            protocol_actions = filter_protocol(sample_actions, str(protocol))
            pareto_actions = pareto_filter(protocol_actions, num_classes=int(original_prob.shape[1]))
            filter_rows.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": str(sample_ids[sample_index]),
                    "protocol": str(protocol),
                    "raw_actions": int(len(protocol_actions)),
                    "pareto_actions": int(len(pareto_actions)),
                }
            )
            for method in methods:
                method_rng = np.random.default_rng(int(rng_root.integers(0, 2**31 - 1)))
                candidates = prefilter_actions(
                    pareto_actions,
                    str(method),
                    tam=tam[sample_index],
                    rng=method_rng,
                    num_classes=int(original_prob.shape[1]),
                    max_candidates=int(args.max_candidates_per_sample),
                    weights=weights,
                )
                if str(method) in {"dynamask_same_sequential", "dynamask_causal_sequential"}:
                    history = _run_dynamic_sequence(
                        actions=candidates,
                        tam_row=tam[sample_index],
                        raw_trace=raw_rows[sample_index],
                        original_prob=original_prob[sample_index],
                        label=int(labels[sample_index]),
                        max_budget=max_budget,
                        attacker=attacker,
                        args=args,
                        device=device,
                        weights=weights,
                    )
                else:
                    history = _run_static_sequence(
                        actions=candidates,
                        method=str(method),
                        tam_row=tam[sample_index],
                        raw_trace=raw_rows[sample_index],
                        original_prob=original_prob[sample_index],
                        label=int(labels[sample_index]),
                        max_budget=max_budget,
                        attacker=attacker,
                        args=args,
                        device=device,
                        weights=weights,
                    )
                for state in history[1:]:
                    step_rows.append(
                        _step_row(
                            state=state,
                            sample_index=sample_index,
                            sample_id=str(sample_ids[sample_index]),
                            protocol=str(protocol),
                            method=str(method),
                            max_budget=max_budget,
                        )
                    )
                for budget in budgets:
                    state = _state_for_budget(history, float(budget))
                    row = _step_row(
                        state=state,
                        sample_index=sample_index,
                        sample_id=str(sample_ids[sample_index]),
                        protocol=str(protocol),
                        method=str(method),
                        max_budget=max_budget,
                    )
                    row["budget"] = float(budget)
                    row["budget_utilization"] = float(row["cumulative_cost"] / max(float(budget), 1e-8))
                    row["candidate_count"] = int(len(candidates))
                    sample_rows.append(row)
    write_csv(output_dir / "oracle_filter_counts.csv", filter_rows)
    write_csv(output_dir / "oracle_step_results.csv", step_rows)
    write_csv(output_dir / "oracle_sample_results.csv", sample_rows)
    summary_rows = []
    for protocol in protocols:
        for method in methods:
            for budget in budgets:
                matched = [
                    row
                    for row in sample_rows
                    if row["protocol"] == protocol and row["method"] == method and abs(float(row["budget"]) - float(budget)) < 1e-9
                ]
                if not matched:
                    continue
                summary_rows.append(
                    {
                        "protocol": str(protocol),
                        "method": str(method),
                        "budget": float(budget),
                        "samples": int(len(matched)),
                        "accuracy": float(np.mean([row["accuracy"] for row in matched])),
                        "flip": float(np.mean([row["flip"] for row in matched])),
                        "top1_drop": float(np.mean([row["top1_drop"] for row in matched])),
                        "max_confidence_drop": float(np.mean([row["max_confidence_drop"] for row in matched])),
                        "margin_drop": float(np.mean([row["margin_drop"] for row in matched])),
                        "entropy_gain": float(np.mean([row["entropy_gain"] for row in matched])),
                        "normalized_entropy_gain": float(np.mean([row["normalized_entropy_gain"] for row in matched])),
                        "js_div": float(np.mean([row["js_div"] for row in matched])),
                        "actual_bandwidth": float(np.mean([row["cumulative_cost"] for row in matched])),
                        "raw_bandwidth": float(np.mean([row["raw_bandwidth"] for row in matched])),
                        "selected_action_count": float(np.mean([row["selected_action_count"] for row in matched])),
                        "incoming_action_ratio": float(np.mean([row["incoming_action_ratio"] for row in matched])),
                        "overlap_window_count": float(np.mean([row["overlap_window_count"] for row in matched])),
                        "budget_utilization": float(np.mean([row["budget_utilization"] for row in matched])),
                    }
                )
    write_csv(output_dir / "oracle_summary.csv", summary_rows)
    write_json(
        output_dir / "sequential_oracle_summary.json",
        {
            "archive": str(args.archive),
            "action_table": str(args.action_table),
            "checkpoint": str(checkpoint),
            "attacker": str(args.attacker),
            "samples": int(tam.shape[0]),
            "budgets": [float(item) for item in budgets],
            "protocols": protocols,
            "methods": methods,
            "max_candidates_per_sample": int(args.max_candidates_per_sample),
            "objective": {
                "confidence": float(weights.confidence),
                "margin": float(weights.margin),
                "entropy": float(weights.entropy),
                "definition": "0.3*top1_confidence_drop + 0.5*top1_top2_margin_drop + 0.2*normalized_entropy_gain, recomputed as a label-free marginal delta at each sequential step; labels are post-hoc only.",
            },
            "outputs": {
                "filter_counts": str(output_dir / "oracle_filter_counts.csv"),
                "step_results": str(output_dir / "oracle_step_results.csv"),
                "sample_results": str(output_dir / "oracle_sample_results.csv"),
                "summary": str(output_dir / "oracle_summary.csv"),
            },
        },
    )
    print(f"Stage B0 sequential oracle complete: {output_dir}")


if __name__ == "__main__":
    main()
