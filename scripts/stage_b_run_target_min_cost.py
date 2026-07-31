"""Target-accuracy minimum-cost evaluation for Stage B2-D."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import ExpandedAction, action_cost, action_identity, generate_expanded_actions
from dmmp.stage_b.objectives import ObjectiveWeights, original_class_margin, original_class_utility, probability_metrics
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, parse_csv_floats, parse_csv_ints, parse_csv_strings

from scripts.stage_b_run_dual_actuator import (
    EvalState,
    _apply_delay,
    _default_checkpoint,
    _fast_refresh_mask,
    _initial_state,
    _load_archive,
    _load_raw_rows,
    _parse_protocols,
    _p95,
    _predict_one,
    _render_dummy,
    _render_dummy_batch,
)


MARGIN_TARGETS = (0.0, -0.05, -0.10)
ACCURACY_TARGETS = (0.30, 0.20, 0.10)


@dataclass(frozen=True)
class TargetConfig:
    protocol: str
    margin_target: float
    dummy_budget: float
    max_delay: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocols", default="client_only,bidirectional_cooperative")
    parser.add_argument("--dummy_budgets", default="0,0.005,0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.30,0.50")
    parser.add_argument("--max_delays", default="0,4,8,16,32,64")
    parser.add_argument("--margin_targets", default="0,-0.05,-0.10")
    parser.add_argument("--accuracy_targets", default="0.30,0.20,0.10")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=96)
    parser.add_argument("--max_candidates", type=int, default=8)
    parser.add_argument("--max_generated_actions", type=int, default=64)
    parser.add_argument("--max_pair_actions", type=int, default=16)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--max_dummy_steps", type=int, default=12)
    parser.add_argument("--min_margin_gain", type=float, default=1e-5)
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
    name = args.run_name or f"stage_b2d_target_min_cost_{args.attacker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _float(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _margin(prob: np.ndarray, original_prob: np.ndarray) -> float:
    y0 = int(np.argmax(original_prob))
    return float(original_class_margin(np.asarray(prob, dtype=np.float32).reshape(1, -1), np.asarray([y0], dtype=np.int64))[0])


def _action_sequence(actions: list[ExpandedAction]) -> str:
    items = []
    for action in actions:
        items.append(
            f"{action.action_type}@{int(action.insert_center)}:"
            f"dose={int(action.counts.sum())}:mode={action.direction_mode}"
        )
    return ";".join(items)


def _resource_fields(state: EvalState, clean_total: float) -> dict:
    clean = max(float(clean_total), 1.0)
    counts = np.asarray(state.dummy_counts, dtype=np.int32)
    out_dummy = int(counts[0].sum())
    in_dummy = int(counts[1].sum())
    dummy = int(counts.sum())
    defended = float(clean + dummy)
    overhead = float(dummy / clean)
    out_delay = int(len(state.outgoing_delay_values))
    in_delay = int(len(state.incoming_delay_values))
    return {
        "clean_packet_count": float(clean),
        "dummy_packet_count": int(dummy),
        "defended_packet_count": float(defended),
        "actual_dummy_bandwidth": overhead,
        "dummy_overhead": overhead,
        "total_overhead": float((defended - clean) / clean),
        "bandwidth_audit_error": 0.0,
        "outgoing_dummy_packet_count": int(out_dummy),
        "incoming_dummy_packet_count": int(in_dummy),
        "outgoing_dummy_fraction_of_clean": float(out_dummy / clean),
        "incoming_dummy_fraction_of_clean": float(in_dummy / clean),
        "average_delay_bins": float(state.avg_delay),
        "p95_delay_bins": float(state.p95_delay),
        "maximum_delay_bins": int(state.max_delay),
        "delay_packet_count": int(len(state.delay_values)),
        "outgoing_delay_packet_count": int(out_delay),
        "incoming_delay_packet_count": int(in_delay),
        "outgoing_average_delay_bins": float(np.mean(state.outgoing_delay_values)) if out_delay else 0.0,
        "incoming_average_delay_bins": float(np.mean(state.incoming_delay_values)) if in_delay else 0.0,
        "outgoing_p95_delay_bins": _p95(state.outgoing_delay_values),
        "incoming_p95_delay_bins": _p95(state.incoming_delay_values),
        "outgoing_max_delay_bins": int(max(state.outgoing_delay_values)) if out_delay else 0,
        "incoming_max_delay_bins": int(max(state.incoming_delay_values)) if in_delay else 0,
    }


def _select_dummy_target(
    *,
    state: EvalState,
    mask: np.ndarray,
    protocol: str,
    budget: float,
    clean_total: float,
    original_prob: np.ndarray,
    attacker,
    device,
    args: argparse.Namespace,
    margin_target: float,
) -> EvalState:
    current = state
    y0 = int(np.argmax(original_prob))
    used: set[tuple] = {action_identity(action) for action in current.selected_actions}
    max_dummy = int(round(float(clean_total) * float(budget)))
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    for _step in range(int(args.max_dummy_steps)):
        current_margin = _margin(current.prob, original_prob)
        if current_margin <= float(margin_target):
            break
        remaining = max_dummy - int(current.dummy_counts.sum())
        if remaining <= 0:
            break
        actions = generate_expanded_actions(
            tam=current.tam,
            soft_mask=mask,
            sample_index=0,
            sample_id="current",
            true_label=int(y0),
            protocol=str(protocol),
            clean_total=float(clean_total),
            ratio=float(args.ratio),
            max_windows=int(args.max_windows),
            max_action_budget=float(args.max_action_budget),
            max_local_rate_peak=int(args.max_local_rate_peak),
            max_actions=int(args.max_generated_actions),
            max_pair_actions=int(args.max_pair_actions),
        )
        actions = sorted(actions, key=lambda action: (-float(action.score_hint), action_cost(action, clean_total)))[: int(args.max_candidates)]
        trial_actions: list[ExpandedAction] = []
        trial_counts: list[np.ndarray] = []
        for action in actions:
            ident = action_identity(action)
            if ident in used:
                continue
            if int(action.counts.sum()) <= 0 or int(action.counts.sum()) > remaining:
                continue
            trial_actions.append(action)
            trial_counts.append(current.dummy_counts + np.asarray(action.counts, dtype=np.int32))
        if not trial_actions:
            break
        _traces, tams, _stats = _render_dummy_batch(base_trace=current.trace, counts_list=trial_counts, args=args)
        probs = predict_probabilities(attacker, tams, device=device, batch_size=int(args.batch_size))
        margins = original_class_margin(probs, np.asarray([y0], dtype=np.int64))
        utilities = original_class_utility(np.repeat(original_prob.reshape(1, -1), len(probs), axis=0), probs, weights)
        current_utility = float(original_class_utility(original_prob.reshape(1, -1), current.prob.reshape(1, -1), weights)[0])
        costs = np.asarray([max(float(action.counts.sum()) / max(float(clean_total), 1.0), 1e-8) for action in trial_actions], dtype=np.float32)
        margin_gain = float(current_margin) - margins.astype(np.float32)
        utility_gain = utilities.astype(np.float32) - float(current_utility)
        reaches = margins <= float(margin_target)
        score = (margin_gain + 0.25 * utility_gain) / costs
        if np.any(reaches):
            reachable = np.flatnonzero(reaches)
            best_local = reachable[np.lexsort((costs[reachable], margins[reachable]))][0]
            idx = int(best_local)
        else:
            idx = int(np.argmax(score))
            if float(margin_gain[idx]) <= float(args.min_margin_gain):
                break
        action = trial_actions[idx]
        trace, tam, stats = _render_dummy(base_trace=current.trace, counts=trial_counts[idx], args=args)
        prob = _predict_one(attacker, tam, device=device, args=args)
        used.add(action_identity(action))
        current = EvalState(
            trace=trace,
            tam=tam,
            prob=prob,
            dummy_counts=trial_counts[idx].astype(np.int32),
            dummy_bandwidth=float(stats["raw_bandwidth"]),
            avg_delay=current.avg_delay,
            p95_delay=current.p95_delay,
            max_delay=current.max_delay,
            delay_values=tuple(current.delay_values),
            outgoing_delay_values=tuple(current.outgoing_delay_values),
            incoming_delay_values=tuple(current.incoming_delay_values),
            selected_actions=list(current.selected_actions) + [action],
        )
    return current


def _run_target_controller(
    *,
    protocol: str,
    budget: float,
    delay_budget: int,
    margin_target: float,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    attacker,
    device,
    args: argparse.Namespace,
) -> EvalState:
    clean_total = max(float(original_tam.sum()), 1.0)
    state = _initial_state(raw_trace, original_tam, original_prob)
    rounds = max(1, int(args.rounds))
    for round_index in range(rounds):
        if int(delay_budget) > 0:
            mask = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
            state = _apply_delay(
                state=state,
                mask=mask,
                protocol=str(protocol),
                delay_budget=max(1, int(round(int(delay_budget) / rounds))),
                args=args,
            )
            state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        if _margin(state.prob, original_prob) <= float(margin_target):
            break
        mask = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
        round_budget = float(budget) * float(round_index + 1) / float(rounds)
        state = _select_dummy_target(
            state=state,
            mask=mask,
            protocol=str(protocol),
            budget=float(round_budget),
            clean_total=float(clean_total),
            original_prob=original_prob,
            attacker=attacker,
            device=device,
            args=args,
            margin_target=float(margin_target),
        )
        state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        if _margin(state.prob, original_prob) <= float(margin_target):
            break
    return state


def _sample_row(
    *,
    sample_index: int,
    sample_id: str,
    config: TargetConfig,
    original_prob: np.ndarray,
    state: EvalState,
    label: int,
    clean_total: float,
    runtime: float,
) -> dict:
    metrics = probability_metrics(original_prob.reshape(1, -1), state.prob.reshape(1, -1), np.asarray([label], dtype=np.int64))
    y0 = int(np.argmax(original_prob))
    final_pred = int(np.argmax(state.prob))
    final_margin = float(metrics["original_class_margin"][0])
    resource = _resource_fields(state, clean_total)
    legal = int(config.protocol != "client_only" or (resource["incoming_delay_packet_count"] == 0 and resource["incoming_dummy_packet_count"] == 0))
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(config.protocol),
        "margin_target": float(config.margin_target),
        "dummy_budget_bound": float(config.dummy_budget),
        "max_delay_budget": int(config.max_delay),
        "target_margin_success": int(final_margin <= float(config.margin_target)),
        "target_unreachable": int(final_margin > float(config.margin_target)),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "original_pred": int(y0),
        "final_pred": int(final_pred),
        "true_label": int(label),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": final_margin,
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "original_class_utility": float(metrics["original_class_utility"][0]),
        "action_count": int(len(state.selected_actions)),
        "action_sequence": _action_sequence(state.selected_actions),
        "client_only_legal": legal,
        "runtime_sec": float(runtime),
        **resource,
    }


def _percentile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _summarize(sample_rows: list[dict], accuracy_targets: list[float]) -> tuple[list[dict], list[dict], list[dict]]:
    groups: dict[tuple[str, float, int, float], list[dict]] = {}
    for row in sample_rows:
        key = (row["protocol"], float(row["margin_target"]), int(row["max_delay_budget"]), float(row["dummy_budget_bound"]))
        groups.setdefault(key, []).append(row)
    summaries: list[dict] = []
    for (protocol, margin_target, delay, budget), rows in sorted(groups.items()):
        bw = [_float(row, "actual_dummy_bandwidth") for row in rows]
        summaries.append(
            {
                "protocol": protocol,
                "margin_target": margin_target,
                "max_delay_budget": delay,
                "dummy_budget_bound": budget,
                "samples": len(rows),
                "accuracy": float(np.mean([_float(row, "accuracy") for row in rows])),
                "flip": float(np.mean([_float(row, "flip") for row in rows])),
                "margin_success_rate": float(np.mean([_float(row, "target_margin_success") for row in rows])),
                "unreachable_ratio": float(np.mean([_float(row, "target_unreachable") for row in rows])),
                "mean_bandwidth": float(np.mean(bw)) if bw else 0.0,
                "median_bandwidth": _percentile(bw, 50),
                "p90_bandwidth": _percentile(bw, 90),
                "p95_bandwidth": _percentile(bw, 95),
                "max_bandwidth": max(bw) if bw else 0.0,
                "mean_delay": float(np.mean([_float(row, "average_delay_bins") for row in rows])),
                "p95_delay": float(np.mean([_float(row, "p95_delay_bins") for row in rows])),
                "max_delay_seen": float(np.mean([_float(row, "maximum_delay_bins") for row in rows])),
                "mean_action_count": float(np.mean([_float(row, "action_count") for row in rows])),
                "client_only_legal": float(np.mean([_float(row, "client_only_legal") for row in rows])),
            }
        )
    target_rows: list[dict] = []
    for protocol in sorted({row["protocol"] for row in summaries}):
        for margin_target in sorted({float(row["margin_target"]) for row in summaries}):
            for delay in sorted({int(row["max_delay_budget"]) for row in summaries}):
                rows = [row for row in summaries if row["protocol"] == protocol and abs(float(row["margin_target"]) - margin_target) < 1e-9 and int(row["max_delay_budget"]) == delay]
                if not rows:
                    continue
                for acc_target in accuracy_targets:
                    feasible = [row for row in rows if _float(row, "accuracy") <= float(acc_target)]
                    if feasible:
                        best = min(feasible, key=lambda row: (_float(row, "mean_bandwidth"), _float(row, "dummy_budget_bound")))
                        target_rows.append(
                            {
                                "protocol": protocol,
                                "margin_target": margin_target,
                                "max_delay_budget": delay,
                                "accuracy_target": float(acc_target),
                                "target_unreachable": 0,
                                "required_budget_bound": _float(best, "dummy_budget_bound"),
                                "required_mean_bandwidth": _float(best, "mean_bandwidth"),
                                "required_p95_bandwidth": _float(best, "p95_bandwidth"),
                                "accuracy": _float(best, "accuracy"),
                                "flip": _float(best, "flip"),
                                "margin_success_rate": _float(best, "margin_success_rate"),
                            }
                        )
                    else:
                        best = min(rows, key=lambda row: _float(row, "accuracy"))
                        target_rows.append(
                            {
                                "protocol": protocol,
                                "margin_target": margin_target,
                                "max_delay_budget": delay,
                                "accuracy_target": float(acc_target),
                                "target_unreachable": 1,
                                "required_budget_bound": "",
                                "required_mean_bandwidth": "",
                                "required_p95_bandwidth": "",
                                "best_accuracy": _float(best, "accuracy"),
                                "best_budget_bound": _float(best, "dummy_budget_bound"),
                                "best_mean_bandwidth": _float(best, "mean_bandwidth"),
                                "best_p95_bandwidth": _float(best, "p95_bandwidth"),
                                "best_margin_success_rate": _float(best, "margin_success_rate"),
                            }
                        )
    min_rows: list[dict] = []
    sample_keys = sorted({(row["protocol"], float(row["margin_target"]), int(row["max_delay_budget"]), int(row["sample_index"])) for row in sample_rows})
    for protocol, margin_target, delay, sample_index in sample_keys:
        rows = [
            row
            for row in sample_rows
            if row["protocol"] == protocol
            and abs(float(row["margin_target"]) - margin_target) < 1e-9
            and int(row["max_delay_budget"]) == delay
            and int(row["sample_index"]) == sample_index
        ]
        success = [row for row in rows if int(row["target_margin_success"]) == 1]
        best = min(success, key=lambda row: (_float(row, "actual_dummy_bandwidth"), _float(row, "dummy_budget_bound"))) if success else min(rows, key=lambda row: _float(row, "original_class_margin"))
        min_rows.append(
            {
                "protocol": protocol,
                "margin_target": margin_target,
                "max_delay_budget": delay,
                "sample_index": sample_index,
                "minimum_found": int(bool(success)),
                "minimum_dummy_bandwidth": _float(best, "actual_dummy_bandwidth") if success else "",
                "minimum_budget_bound": _float(best, "dummy_budget_bound") if success else "",
                "best_margin": _float(best, "original_class_margin"),
                "best_accuracy": _float(best, "accuracy"),
                "best_flip": _float(best, "flip"),
                "best_action_count": _float(best, "action_count"),
                "best_average_delay_bins": _float(best, "average_delay_bins"),
                "best_p95_delay_bins": _float(best, "p95_delay_bins"),
                "best_max_delay_bins": _float(best, "maximum_delay_bins"),
            }
        )
    return summaries, target_rows, min_rows


def _plot_accuracy(summary_rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for protocol in sorted({row["protocol"] for row in summary_rows}):
        rows_p = [row for row in summary_rows if row["protocol"] == protocol]
        for margin_target in sorted({float(row["margin_target"]) for row in rows_p}):
            rows_m = [row for row in rows_p if abs(float(row["margin_target"]) - margin_target) < 1e-9]
            plt.figure(figsize=(10, 6))
            for delay in sorted({int(row["max_delay_budget"]) for row in rows_m}):
                rows = sorted([row for row in rows_m if int(row["max_delay_budget"]) == delay], key=lambda row: _float(row, "mean_bandwidth"))
                plt.plot([_float(row, "mean_bandwidth") * 100.0 for row in rows], [_float(row, "accuracy") * 100.0 for row in rows], marker="o", label=f"D={delay}")
            for y in (30, 20, 10):
                plt.axhline(float(y), color="gray", linestyle="--", linewidth=0.8)
            plt.xlabel("mean actual dummy bandwidth (%)")
            plt.ylabel("RF accuracy (%)")
            plt.title(f"Accuracy vs Bandwidth | {protocol} | margin <= {margin_target:g}")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(figure_dir / f"accuracy_vs_bandwidth_{protocol}_m{margin_target:g}.png".replace("-", "neg"), dpi=180)
            plt.close()


def _plot_cdf(min_rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for protocol in sorted({row["protocol"] for row in min_rows}):
        for margin_target in sorted({float(row["margin_target"]) for row in min_rows}):
            plt.figure(figsize=(10, 6))
            for delay in sorted({int(row["max_delay_budget"]) for row in min_rows}):
                rows = [row for row in min_rows if row["protocol"] == protocol and abs(float(row["margin_target"]) - margin_target) < 1e-9 and int(row["max_delay_budget"]) == delay and int(row["minimum_found"]) == 1]
                values = sorted(_float(row, "minimum_dummy_bandwidth") * 100.0 for row in rows)
                if not values:
                    continue
                ys = np.arange(1, len(values) + 1, dtype=np.float32) / float(len(values))
                plt.step(values, ys, where="post", label=f"D={delay}")
            plt.xlabel("minimum dummy bandwidth (%)")
            plt.ylabel("coverage among reachable samples")
            plt.title(f"Minimum-Cost CDF | {protocol} | margin <= {margin_target:g}")
            plt.grid(alpha=0.25)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(figure_dir / f"min_cost_cdf_{protocol}_m{margin_target:g}.png".replace("-", "neg"), dpi=180)
            plt.close()


def _plot_unreachable(summary_rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    for protocol in sorted({row["protocol"] for row in summary_rows}):
        rows_p = [row for row in summary_rows if row["protocol"] == protocol]
        plt.figure(figsize=(10, 6))
        for margin_target in sorted({float(row["margin_target"]) for row in rows_p}):
            rows = sorted([row for row in rows_p if abs(float(row["margin_target"]) - margin_target) < 1e-9], key=lambda row: (_float(row, "max_delay_budget"), _float(row, "mean_bandwidth")))
            xs = [_float(row, "mean_bandwidth") * 100.0 for row in rows]
            ys = [_float(row, "unreachable_ratio") * 100.0 for row in rows]
            plt.scatter(xs, ys, s=32, label=f"m<={margin_target:g}")
        plt.xlabel("mean actual dummy bandwidth (%)")
        plt.ylabel("unreachable samples (%)")
        plt.title(f"Unreachable Ratio | {protocol}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(figure_dir / f"unreachable_ratio_{protocol}.png", dpi=180)
        plt.close()


def _write_plots(summary_rows: list[dict], min_rows: list[dict], output_dir: Path) -> None:
    figure_dir = output_dir / "figures"
    _plot_accuracy(summary_rows, figure_dir)
    _plot_cdf(min_rows, figure_dir)
    _plot_unreachable(summary_rows, figure_dir)


def main() -> None:
    import time

    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    protocols = _parse_protocols(args.protocols)
    budgets = parse_csv_floats(args.dummy_budgets)
    delays = parse_csv_ints(args.max_delays)
    margin_targets = parse_csv_floats(args.margin_targets)
    accuracy_targets = parse_csv_floats(args.accuracy_targets)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(args.data_root, source_indices, args)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=prob.shape[1],
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    sample_rows: list[dict] = []
    configs = [TargetConfig(protocol, margin_target, budget, delay) for protocol in protocols for margin_target in margin_targets for delay in delays for budget in budgets]
    for config_index, config in enumerate(configs, start=1):
        if args.progress:
            print(
                f"[target_min_cost] config {config_index}/{len(configs)} "
                f"{config.protocol} margin<={config.margin_target:g} B={config.dummy_budget:g} D={config.max_delay}",
                flush=True,
            )
        for sample_index in range(tam.shape[0]):
            start = time.perf_counter()
            state = _run_target_controller(
                protocol=str(config.protocol),
                budget=float(config.dummy_budget),
                delay_budget=int(config.max_delay),
                margin_target=float(config.margin_target),
                raw_trace=raw_rows[sample_index],
                original_tam=tam[sample_index],
                original_mask=mask[sample_index],
                original_prob=prob[sample_index],
                attacker=attacker,
                device=device,
                args=args,
            )
            sample_rows.append(
                _sample_row(
                    sample_index=sample_index,
                    sample_id=sample_ids[sample_index],
                    config=config,
                    original_prob=prob[sample_index],
                    state=state,
                    label=int(labels[sample_index]),
                    clean_total=float(tam[sample_index].sum()),
                    runtime=float(time.perf_counter() - start),
                )
            )
    summary_rows, target_rows, min_rows = _summarize(sample_rows, accuracy_targets)
    write_csv(output_dir / "target_min_cost_sample_results.csv", sample_rows)
    write_csv(output_dir / "target_min_cost_summary.csv", summary_rows)
    write_csv(output_dir / "target_accuracy_requirements.csv", target_rows)
    write_csv(output_dir / "target_min_cost_samples.csv", min_rows)
    _write_plots(summary_rows, min_rows, output_dir)
    write_json(
        output_dir / "target_min_cost_run.json",
        {
            "archive": str(args.archive),
            "checkpoint": str(checkpoint),
            "samples": int(tam.shape[0]),
            "protocols": protocols,
            "dummy_budgets": budgets,
            "max_delays": delays,
            "margin_targets": margin_targets,
            "accuracy_targets": accuracy_targets,
            "label_free_generation": True,
            "true_labels_used_only_for_final_accuracy": True,
            "controller": "target-aware alternating fast refresh",
            "rounds": int(args.rounds),
        },
    )
    print(f"Target minimum-cost evaluation complete: {output_dir}")


if __name__ == "__main__":
    main()
