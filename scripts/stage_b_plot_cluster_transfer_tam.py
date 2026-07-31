# -*- coding: utf-8 -*-
"""Plot clean and defended TAM figures for one cluster-transfer trace.

The script replays the saved prototype policy on a selected member trace, using
the same transfer loop as stage_b_cluster_prototype_transfer_audit.py, then
saves TAM slot-count plots for the original and defended states.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.objectives import probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT

from scripts.stage_b_cluster_prototype_transfer_audit import (
    DEFAULT_ARCHIVE,
    DEFAULT_FULL_SPLIT,
    SampleInfo,
    _archive_infos,
    _events_by_round,
    _load_archive_keys,
    _load_raw_selected,
    _try_budget_normalized_absolute,
    _try_replay_candidate_queue,
    _try_top4_verify,
)
from scripts.stage_b_run_b2e_diverse_search import _margin, _resource_fields, _runtime_args
from scripts.stage_b_run_dual_actuator import (
    _apply_delay,
    _default_checkpoint,
    _fast_refresh_mask,
    _initial_state,
    _predict_one,
)


DEFAULT_RESULT_DIR = (
    "results/stage_b_cluster_prototype_transfer_audit_20260730/"
    "fullcw_k300_test_absolute_budget_normalized_D64"
)
DEFAULT_POLICY_DIR = (
    "results/stage_b_cluster_prototype_transfer_audit_20260730/"
    "fullcw_k300_prototype_only/prototype_policies"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", default=DEFAULT_RESULT_DIR)
    parser.add_argument("--transfer_results_csv", default="")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default=DEFAULT_FULL_SPLIT)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--prototype_policy_dir", default=DEFAULT_POLICY_DIR)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--sample_index", type=int, default=-1)
    parser.add_argument("--sample_rank", type=int, default=0)
    parser.add_argument("--auto_min_bw", type=float, default=0.01)
    parser.add_argument("--auto_stop_reason", default="target_reached")
    parser.add_argument("--mode", default="absolute_budget_normalized_replay")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--top_relative_keypoints", type=int, default=64)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--compact_candidate_generation", action="store_true", default=True)
    parser.add_argument("--deferred_materialize_oversample", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=8192)
    parser.add_argument("--materialization_batch_size", type=int, default=128)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="gpu_tam")
    parser.add_argument("--profile_candidate_generation", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--zoom_slots", type=int, default=0, help="0 means infer from active TAM range.")
    parser.add_argument("--dpi", type=int, default=190)
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_cell(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _int_cell(row: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _select_result_row(args: argparse.Namespace) -> dict[str, str]:
    result_csv = Path(args.transfer_results_csv).resolve() if args.transfer_results_csv else Path(args.result_dir).resolve() / "transfer_results.csv"
    rows = _read_rows(result_csv)
    if not rows:
        raise ValueError(f"No rows found in {result_csv}")
    if int(args.sample_index) >= 0:
        for row in rows:
            if _int_cell(row, "sample_index", -1) == int(args.sample_index):
                return row
        raise ValueError(f"sample_index={args.sample_index} was not found in {result_csv}")
    hits = [
        row
        for row in rows
        if (not str(args.auto_stop_reason).strip() or str(row.get("stop_reason", "")) == str(args.auto_stop_reason))
        and _float_cell(row, "actual_dummy_bandwidth", 0.0) >= float(args.auto_min_bw)
    ]
    if not hits:
        hits = rows
    rank = min(max(int(args.sample_rank), 0), len(hits) - 1)
    return hits[rank]


def _sample_info_from_archive(args: argparse.Namespace, row: dict[str, str]) -> SampleInfo:
    archive_index = _int_cell(row, "sample_index", -1)
    source_index = _int_cell(row, "source_index", -1)
    if archive_index < 0 or source_index < 0:
        raise ValueError("transfer row must contain sample_index and source_index")
    split = str(row.get("split", "test"))
    label = _int_cell(row, "true_label", -1)
    sample_id = str(row.get("sample_id", archive_index))
    try:
        infos = _archive_infos(args.archive, args.split_file)
        for info in infos:
            if int(info.archive_index) == int(archive_index):
                return info
    except Exception:
        pass
    return SampleInfo(
        local_index=-1,
        archive_index=int(archive_index),
        source_index=int(source_index),
        sample_id=str(sample_id),
        split=str(split),
        label=int(label),
    )


def _policy_path(args: argparse.Namespace, prototype_archive_index: int) -> Path:
    path = Path(args.prototype_policy_dir).resolve() / f"prototype_archive_{int(prototype_archive_index):06d}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prototype policy not found: {path}")
    return path


def _replay_transfer_state(
    *,
    policy_path: Path,
    info: SampleInfo,
    raw_trace: np.ndarray,
    tam: np.ndarray,
    prob: np.ndarray,
    attacker,
    device,
    run_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    by_round = _events_by_round(list(policy.get("events", [])))
    clean_total = max(float(np.asarray(tam, dtype=np.float32).sum()), 1.0)
    current = _initial_state(np.asarray(raw_trace, dtype=np.float32), np.asarray(tam, dtype=np.float32), np.asarray(prob, dtype=np.float32))
    used: set[tuple] = set()
    attempted = 0
    invalid = 0
    executed = 0
    candidate_rf = 0
    state_rf = 0
    no_positive = 0
    zero_dose_skip = 0
    budget_clipped_action = 0
    fully_scaled_action = 0
    prototype_action_dummy_sum = 0
    scaled_action_dummy_sum = 0
    gain_values: list[float] = []
    prototype_gain_values: list[float] = []
    accepted_types: list[str] = []
    invalid_reasons: Counter[str] = Counter()
    stop_reason = "prototype_sequence_exhausted"
    prototype_cumulative_dummy = 0
    member_previous_target_dummy = 0
    replay_log: list[dict[str, Any]] = []

    for round_index in range(max(1, int(args.rounds))):
        if int(args.max_delay) > 0:
            before_margin = _margin(current.prob, prob)
            mask0 = _fast_refresh_mask(attacker, current.tam, np.asarray(prob, dtype=np.float32), device=device)
            current = _apply_delay(
                state=current,
                mask=mask0,
                protocol=str(args.protocol),
                delay_budget=max(1, int(round(int(args.max_delay) / max(1, int(args.rounds))))),
                args=run_args,
            )
            current.prob = _predict_one(attacker, current.tam, device=device, args=run_args)
            state_rf += 1
            replay_log.append(
                {
                    "round_index": int(round_index),
                    "kind": "delay",
                    "margin_before": float(before_margin),
                    "margin_after": float(_margin(current.prob, prob)),
                    "average_delay_bins": float(current.avg_delay),
                    "p95_delay_bins": float(current.p95_delay),
                    "maximum_delay_bins": int(current.max_delay),
                }
            )
        if _margin(current.prob, prob) <= float(args.margin_target):
            stop_reason = "target_reached"
            break

        current_mask = _fast_refresh_mask(attacker, current.tam, np.asarray(prob, dtype=np.float32), device=device)
        round_events = by_round.get(int(round_index), [])
        if not round_events:
            continue
        for event in round_events:
            if int(np.asarray(current.dummy_counts, dtype=np.int32).sum()) >= int(math.floor(clean_total * float(args.budget) + 1e-9)):
                stop_reason = "bandwidth_10pct_reached"
                break
            if str(args.mode) == "absolute_budget_normalized_replay":
                next_state, stats = _try_budget_normalized_absolute(
                    event=event,
                    current=current,
                    original_prob=np.asarray(prob, dtype=np.float32),
                    label=int(info.label),
                    info=info,
                    clean_total=clean_total,
                    used=used,
                    prototype_cumulative_dummy=int(prototype_cumulative_dummy),
                    member_previous_target_dummy=int(member_previous_target_dummy),
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                )
                prototype_cumulative_dummy = int(stats.get("prototype_next_cumulative_dummy", prototype_cumulative_dummy))
                member_previous_target_dummy = int(stats.get("member_next_target_dummy", member_previous_target_dummy))
            elif str(args.mode) == "relative_top4_verify":
                next_state, stats = _try_top4_verify(
                    event=event,
                    current=current,
                    current_mask=current_mask,
                    original_prob=np.asarray(prob, dtype=np.float32),
                    label=int(info.label),
                    info=info,
                    clean_total=clean_total,
                    used=used,
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                )
            else:
                next_state, stats = _try_replay_candidate_queue(
                    mode=str(args.mode),
                    event=event,
                    current=current,
                    current_mask=current_mask,
                    original_prob=np.asarray(prob, dtype=np.float32),
                    label=int(info.label),
                    info=info,
                    clean_total=clean_total,
                    used=used,
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                )

            attempted += int(stats["attempted"])
            invalid += int(stats["invalid"])
            invalid_reasons.update({str(k): int(v) for k, v in dict(stats.get("invalid_reasons", {})).items()})
            candidate_rf += int(stats["candidate_rf_eval_count"])
            state_rf += int(stats["state_rf_eval_count"])
            zero_dose_skip += int(stats.get("zero_dose_skip", 0))
            budget_clipped_action += int(stats.get("budget_clipped_action", 0))
            fully_scaled_action += int(stats.get("fully_scaled_action", 0))
            prototype_action_dummy_sum += int(stats.get("prototype_action_dummy", 0))
            scaled_action_dummy_sum += int(stats.get("scaled_action_dummy", 0))
            replay_log.append(
                {
                    "round_index": int(round_index),
                    "kind": "dummy",
                    "executed": int(stats["executed"]),
                    "attempted": int(stats["attempted"]),
                    "invalid": int(stats["invalid"]),
                    "gain": float(stats.get("gain", 0.0)),
                    "prototype_gain": float(stats.get("prototype_gain", 0.0)),
                    "action_type": str(stats.get("action_type", "")),
                    "scaled_action_dummy": int(stats.get("scaled_action_dummy", 0)),
                }
            )
            if int(stats["executed"]) <= 0:
                no_positive += int(stats.get("no_positive", 0))
                if str(args.mode) == "absolute_budget_normalized_replay" and int(stats.get("zero_dose_skip", 0)) > 0 and int(stats.get("invalid", 0)) <= 0:
                    stop_reason = "prototype_sequence_exhausted"
                    continue
                stop_reason = "no_positive_transfer" if str(args.mode) == "relative_top4_verify" else "invalid_transfer_action"
                break

            current = next_state
            executed += 1
            gain_values.append(float(stats.get("gain", 0.0)))
            prototype_gain_values.append(float(stats.get("prototype_gain", 0.0)))
            if str(stats.get("action_type", "")):
                accepted_types.append(str(stats["action_type"]))
            if _margin(current.prob, prob) <= float(args.margin_target):
                stop_reason = "target_reached"
                break
            current_mask = _fast_refresh_mask(attacker, current.tam, np.asarray(prob, dtype=np.float32), device=device)
        if stop_reason in {"target_reached", "bandwidth_10pct_reached", "invalid_transfer_action", "no_positive_transfer"}:
            break

    summary = {
        "stop_reason": str(stop_reason),
        "attempted_transfer_actions": int(attempted),
        "invalid_transfer_actions": int(invalid),
        "invalid_transfer_reasons": dict(sorted(invalid_reasons.items())),
        "executed_actions": int(executed),
        "no_positive_verify_steps": int(no_positive),
        "candidate_rf_eval_count": int(candidate_rf),
        "state_rf_eval_count": int(state_rf),
        "total_rf_eval_count": int(candidate_rf + state_rf),
        "mean_exact_gain": float(np.mean(gain_values)) if gain_values else 0.0,
        "mean_prototype_gain": float(np.mean(prototype_gain_values)) if prototype_gain_values else 0.0,
        "accepted_action_type_distribution": dict(sorted(Counter(accepted_types).items())),
        "zero_dose_skip": int(zero_dose_skip),
        "budget_clipped_action": int(budget_clipped_action),
        "fully_scaled_action": int(fully_scaled_action),
        "prototype_action_dummy_sum": int(prototype_action_dummy_sum),
        "scaled_action_dummy_sum": int(scaled_action_dummy_sum),
    }
    return current, summary, replay_log


def _metric_summary(original_prob: np.ndarray, defended_prob: np.ndarray, label: int) -> dict[str, Any]:
    metrics = probability_metrics(
        np.asarray(original_prob, dtype=np.float32).reshape(1, -1),
        np.asarray(defended_prob, dtype=np.float32).reshape(1, -1),
        np.asarray([int(label)], dtype=np.int64),
    )
    return {key: (float(value[0]) if np.asarray(value).ndim else float(value)) for key, value in metrics.items()}


def _active_end(*arrays: np.ndarray, default: int = 320) -> int:
    width = int(arrays[0].shape[-1])
    nonzero_bins: list[int] = []
    for arr in arrays:
        nz = np.argwhere(np.asarray(arr, dtype=np.float32) > 0)
        if nz.size:
            nonzero_bins.append(int(nz[:, -1].max()))
    if not nonzero_bins:
        return min(width, int(default))
    return min(width, max(int(default), int(max(nonzero_bins) + 80)))


def _axis_plot(ax: plt.Axes, tam: np.ndarray, *, title: str, ymax: float, x_end: int) -> None:
    width = int(tam.shape[-1])
    x_axis = np.arange(width)
    outgoing = np.asarray(tam[0], dtype=np.float32)
    incoming = -np.asarray(tam[1], dtype=np.float32)
    ax.bar(x_axis, outgoing, width=1.0, color="#1358d8", edgecolor="#1358d8", linewidth=0.0)
    ax.bar(x_axis, incoming, width=1.0, color="#d82424", edgecolor="#d82424", linewidth=0.0)
    ax.axhline(0, color="#c17b99", linewidth=0.9, alpha=0.9)
    ax.grid(True, axis="both", alpha=0.23, linewidth=0.6)
    ax.set_xlim(0, int(x_end))
    ax.set_ylim(-float(ymax), float(ymax))
    ax.set_ylabel("Pkt. Number", fontsize=9)
    ax.set_title(title, fontsize=9.5, loc="left")


def _plot_tam(output_dir: Path, *, clean: np.ndarray, defended: np.ndarray, summary: dict[str, Any], x_end: int, stem: str, dpi: int) -> Path:
    shown_end = min(int(x_end), int(clean.shape[-1]))
    ymax = max(float(np.max(clean[:, :shown_end])), float(np.max(defended[:, :shown_end])), 4.0)
    ymax += max(2.0, 0.08 * ymax)
    fig, axes = plt.subplots(2, 1, figsize=(15.6, 5.5), sharex=True)
    title_common = (
        f"sample {summary['sample_index']} / {summary['sample_id']} | "
        f"y={summary['true_label']} pred {summary['original_pred']}->{summary['final_pred']}"
    )
    _axis_plot(
        axes[0],
        clean,
        title=(
            f"Clean original | {title_common} | "
            f"clean packets={summary['clean_packet_count']:.0f}"
        ),
        ymax=ymax,
        x_end=shown_end,
    )
    _axis_plot(
        axes[1],
        defended,
        title=(
            f"Defended P0+ D64 | dummy={summary['actual_dummy_bandwidth'] * 100:.2f}% "
            f"({summary['dummy_packet_count']:.0f}/{summary['clean_packet_count']:.0f}) | "
            f"delay avg/p95/max={summary['average_delay_bins']:.2f}/"
            f"{summary['p95_delay_bins']:.1f}/{summary['maximum_delay_bins']:.0f} bins | "
            f"actions={summary['executed_actions']}"
        ),
        ymax=ymax,
        x_end=shown_end,
    )
    axes[0].legend(
        handles=[Patch(color="#1358d8", label="Outgoing packets"), Patch(color="#d82424", label="Incoming packets")],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )
    axes[-1].set_xlabel(f"TAM time slots (0-{shown_end - 1})", fontsize=10)
    fig.suptitle("Cluster prototype transfer TAM comparison", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _plot_delta(output_dir: Path, *, clean: np.ndarray, defended: np.ndarray, summary: dict[str, Any], x_end: int, dpi: int) -> Path:
    shown_end = min(int(x_end), int(clean.shape[-1]))
    delta = np.asarray(defended, dtype=np.float32) - np.asarray(clean, dtype=np.float32)
    ymax = max(float(np.max(np.abs(delta[:, :shown_end]))), 2.0)
    ymax += max(1.0, 0.08 * ymax)
    x_axis = np.arange(int(delta.shape[-1]))
    fig, ax = plt.subplots(1, 1, figsize=(15.6, 2.9))
    ax.bar(x_axis, delta[0], width=1.0, color="#17804f", edgecolor="#17804f", linewidth=0.0, label="Outgoing delta")
    ax.bar(x_axis, -delta[1], width=1.0, color="#8d4ac9", edgecolor="#8d4ac9", linewidth=0.0, label="Incoming delta")
    ax.axhline(0, color="#777777", linewidth=0.9, alpha=0.9)
    ax.grid(True, axis="both", alpha=0.23, linewidth=0.6)
    ax.set_xlim(0, shown_end)
    ax.set_ylim(-ymax, ymax)
    ax.set_xlabel(f"TAM time slots (0-{shown_end - 1})", fontsize=10)
    ax.set_ylabel("Defended - Clean", fontsize=9)
    ax.set_title(
        f"Slot-count change | sample {summary['sample_index']} | dummy={summary['actual_dummy_bandwidth'] * 100:.2f}% | "
        f"delay avg={summary['average_delay_bins']:.2f} bins",
        fontsize=10,
        loc="left",
    )
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    path = output_dir / f"sample_{int(summary['sample_index']):06d}_tam_delta_zoom.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    result_dir = Path(args.result_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if str(args.output_dir).strip() else result_dir / "visualizations" / "tam"
    output_dir.mkdir(parents=True, exist_ok=True)
    row = _select_result_row(args)
    info = _sample_info_from_archive(args, row)
    prototype_index = _int_cell(row, "prototype_sample_index", -1)
    policy_path = _policy_path(args, prototype_index)

    arrays = _load_archive_keys(args.archive, [int(info.archive_index)], ["tam", "mask", "pred_prob", "labels"])
    raw = _load_raw_selected(args.data_root, np.asarray([int(info.source_index)], dtype=np.int64), args)
    original_tam = np.asarray(arrays["tam"][0], dtype=np.float32)
    original_prob = np.asarray(arrays["pred_prob"][0], dtype=np.float32)
    label = int(arrays["labels"][0])
    raw_trace = np.asarray(raw[0], dtype=np.float32)

    device = resolve_device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(str(args.attacker))
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(original_prob.shape[0]),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    run_args = _runtime_args(args)
    final_state, replay_summary, replay_log = _replay_transfer_state(
        policy_path=policy_path,
        info=info,
        raw_trace=raw_trace,
        tam=original_tam,
        prob=original_prob,
        attacker=attacker,
        device=device,
        run_args=run_args,
        args=args,
    )
    clean_total = max(float(original_tam.sum()), 1.0)
    metrics = _metric_summary(original_prob, final_state.prob, label)
    resource = _resource_fields(final_state, clean_total)
    summary = {
        "sample_index": int(info.archive_index),
        "source_index": int(info.source_index),
        "sample_id": str(info.sample_id),
        "split": str(info.split),
        "cluster_id": _int_cell(row, "cluster_id", -1),
        "prototype_sample_index": int(prototype_index),
        "prototype_policy_path": str(policy_path),
        "true_label": int(label),
        "original_pred": int(metrics["original_pred"]),
        "final_pred": int(metrics["evaluated_pred"]),
        "accuracy": float(metrics["accuracy"]),
        "flip": float(metrics["flip"]),
        "original_class_probability": float(metrics["original_class_probability"]),
        "original_class_margin": float(metrics["original_class_margin"]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"]),
        **resource,
        **replay_summary,
        "transfer_csv_row": row,
        "replay_log": replay_log,
    }

    inferred_zoom = _active_end(original_tam, final_state.tam, default=320)
    zoom_end = int(args.zoom_slots) if int(args.zoom_slots) > 0 else inferred_zoom
    full_path = _plot_tam(
        output_dir,
        clean=original_tam,
        defended=np.asarray(final_state.tam, dtype=np.float32),
        summary=summary,
        x_end=int(original_tam.shape[-1]),
        stem=f"sample_{int(info.archive_index):06d}_tam_full",
        dpi=int(args.dpi),
    )
    zoom_path = _plot_tam(
        output_dir,
        clean=original_tam,
        defended=np.asarray(final_state.tam, dtype=np.float32),
        summary=summary,
        x_end=zoom_end,
        stem=f"sample_{int(info.archive_index):06d}_tam_zoom",
        dpi=int(args.dpi),
    )
    delta_path = _plot_delta(
        output_dir,
        clean=original_tam,
        defended=np.asarray(final_state.tam, dtype=np.float32),
        summary=summary,
        x_end=zoom_end,
        dpi=int(args.dpi),
    )
    summary_path = output_dir / f"sample_{int(info.archive_index):06d}_tam_summary.json"
    summary.update(
        {
            "full_plot_path": str(full_path),
            "zoom_plot_path": str(zoom_path),
            "delta_plot_path": str(delta_path),
            "zoom_slots": int(zoom_end),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
