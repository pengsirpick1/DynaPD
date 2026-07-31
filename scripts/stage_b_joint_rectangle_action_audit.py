# -*- coding: utf-8 -*-
"""Audit whether current Stage B actions implement joint rectangularization."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.additive_probe import candidate_windows_for_sample
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import (
    CandidateDescriptor,
    ExpandedAction,
    action_cost,
    generate_compact_action_descriptors,
    materialize_candidate_descriptors,
)
from dmmp.stage_b.objectives import probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT

from scripts.stage_b_cluster_prototype_transfer_audit import (
    DEFAULT_ARCHIVE,
    DEFAULT_FULL_SPLIT,
    _archive_infos,
    _load_archive_keys,
    _load_raw_selected,
)
from scripts.stage_b_run_b2e_diverse_search import (
    _evaluate_actions,
    _method_config,
    _prefilter_actions,
    _resource_fields,
    _runtime_args,
)
from scripts.stage_b_run_dual_actuator import (
    _apply_delay,
    _default_checkpoint,
    _fast_refresh_mask,
    _initial_state,
    _predict_one,
)
from scripts.stage_b_run_target_min_cost import _margin


DEFAULT_OUT = "results/stage_b_joint_rectangle_action_audit_20260730"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default=DEFAULT_FULL_SPLIT)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--sample_index", type=int, default=95170)
    parser.add_argument("--split_name", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--method", default="stratified_top128")
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
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--true_recall_pool_size", type=int, default=0)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
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
    parser.add_argument("--run_d64_only", action="store_true")
    parser.add_argument("--d64_max_samples", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _descriptor_row(action: CandidateDescriptor, *, rank: int, clean_total: float, exact_gain: float | str = "") -> dict[str, Any]:
    return {
        "rank": int(rank),
        "action_id": int(action.action_id),
        "window_id": int(action.window_id),
        "action_type": str(action.action_type),
        "tier": str(action.tier),
        "source": str(action.source),
        "affected_direction": str(action.affected_direction),
        "anchor_or_affected_start": int(action.affected_start),
        "anchor_or_affected_end": int(action.affected_end),
        "anchor_or_affected_center": int(action.affected_center),
        "insert_start": int(action.insert_start),
        "insert_end": int(action.insert_end),
        "insert_center": int(action.insert_center),
        "action_width": int(action.action_width),
        "temporal_span_field": "",
        "target_height_field": "",
        "dose": int(action.dose),
        "dummy_count": int(action.dummy_count),
        "outgoing_dummy_count": int(action.outgoing_dummy_count),
        "incoming_dummy_count": int(action.incoming_dummy_count),
        "actual_bandwidth_proxy": float(int(action.dummy_count) / max(float(clean_total), 1.0)),
        "actual_delay_field": "",
        "average_delay_bins_field": "",
        "p95_delay_bins_field": "",
        "maximum_delay_bins_field": "",
        "shape_error_field": "",
        "direction_mode": str(action.direction_mode),
        "smoothing": str(action.smoothing),
        "mask_mass": float(action.mask_mass),
        "local_count": float(action.local_count),
        "score_hint": float(action.score_hint),
        "estimated_gain_proxy": float(float(action.score_hint) * np.sqrt(max(float(int(action.dummy_count) / max(float(clean_total), 1.0)), 1.0e-8))),
        "exact_gain": exact_gain,
        "nonzero_bin_count": int(action.nonzero_bin_count),
        "active_bin_count": int(action.active_bin_count),
        "requires_incoming_capability": int(action.requires_incoming_capability),
        "allowed_violation_count": int(action.allowed_violation_count),
        "local_rate_peak": int(action.local_rate_peak),
        "count_signature": json.dumps(tuple(int(x) for x in action.count_signature), ensure_ascii=False),
    }


def _action_row(action: ExpandedAction, *, rank: int, clean_total: float, exact_gain: float | str = "") -> dict[str, Any]:
    counts = np.asarray(action.counts, dtype=np.int32)
    return {
        "rank": int(rank),
        "window_id": int(action.window_id),
        "action_type": str(action.action_type),
        "tier": str(action.tier),
        "source": str(action.source),
        "affected_direction": str(action.affected_direction),
        "affected_start": int(action.affected_start),
        "affected_end": int(action.affected_end),
        "affected_center": int(action.affected_center),
        "insert_start": int(action.insert_start),
        "insert_end": int(action.insert_end),
        "insert_center": int(action.insert_center),
        "dose": int(action.dose),
        "dummy_count": int(counts.sum()),
        "outgoing_dummy_count": int(counts[0].sum()),
        "incoming_dummy_count": int(counts[1].sum()),
        "actual_bandwidth": float(int(counts.sum()) / max(float(clean_total), 1.0)),
        "actual_delay": "",
        "shape_error": "",
        "direction_mode": str(action.direction_mode),
        "score_hint": float(action.score_hint),
        "estimated_gain_proxy": float(float(action.score_hint) * np.sqrt(max(action_cost(action, clean_total), 1.0e-8))),
        "exact_gain": exact_gain,
        "nonzero_bin_count": int(np.count_nonzero(counts)),
        "local_rate_peak": int(counts.sum(axis=0).max()) if counts.size else 0,
    }


def _sample_payload(args: argparse.Namespace) -> tuple[Any, dict[str, np.ndarray], np.ndarray]:
    infos = _archive_infos(args.archive, args.split_file)
    info_map = {int(info.archive_index): info for info in infos}
    if int(args.sample_index) not in info_map:
        raise ValueError(f"sample_index={args.sample_index} not present in archive/split mapping")
    info = info_map[int(args.sample_index)]
    arrays = _load_archive_keys(args.archive, [int(info.archive_index)], ["tam", "mask", "pred_prob", "labels"])
    raw = _load_raw_selected(args.data_root, np.asarray([int(info.source_index)], dtype=np.int64), args)
    return info, arrays, np.asarray(raw[0], dtype=np.float32)


def audit_candidate_space(args: argparse.Namespace, out_dir: Path, attacker, device) -> dict[str, Any]:
    info, arrays, raw_trace = _sample_payload(args)
    original_tam = np.asarray(arrays["tam"][0], dtype=np.float32)
    original_mask = np.asarray(arrays["mask"][0], dtype=np.float32)
    original_prob = np.asarray(arrays["pred_prob"][0], dtype=np.float32)
    label = int(arrays["labels"][0])
    clean_total = max(float(original_tam.sum()), 1.0)
    run_args = _runtime_args(args)
    state = _initial_state(raw_trace, original_tam, original_prob)
    delay_summary: dict[str, Any] = {"applied_before_candidate_generation": 0}
    if int(args.max_delay) > 0:
        mask0 = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
        state = _apply_delay(
            state=state,
            mask=mask0,
            protocol=str(args.protocol),
            delay_budget=max(1, int(round(int(args.max_delay) / max(1, int(args.rounds))))),
            args=run_args,
        )
        state.prob = _predict_one(attacker, state.tam, device=device, args=run_args)
        delay_summary = {
            "applied_before_candidate_generation": 1,
            "delay_budget_this_round": int(max(1, int(round(int(args.max_delay) / max(1, int(args.rounds)))))),
            "average_delay_bins": float(state.avg_delay),
            "p95_delay_bins": float(state.p95_delay),
            "maximum_delay_bins": int(state.max_delay),
            "delay_packet_count": int(len(state.delay_values)),
            "margin_after_delay": float(_margin(state.prob, original_prob)),
        }
    candidate_mask = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
    windows = candidate_windows_for_sample(
        candidate_mask,
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        sample_index=int(info.archive_index),
    )
    window_rows = [
        {
            "window_id": int(window.window_id),
            "direction": int(window.direction),
            "direction_name": str(window.direction_name),
            "start": int(window.start),
            "end": int(window.end),
            "center": int(window.center),
            "length": int(window.length),
            "mask_mass": float(window.mask_mass),
            "local_tam_count": float(np.asarray(state.tam, dtype=np.float32)[:, int(window.start) : int(window.end)].sum()),
        }
        for window in windows
    ]
    _write_csv(out_dir / "sample_candidate_windows.csv", window_rows)

    config = _method_config(str(args.method))
    full_diag: dict[str, Any] = {"profile_detail": False}
    raw_full = generate_compact_action_descriptors(
        tam=state.tam,
        soft_mask=candidate_mask,
        sample_index=int(info.archive_index),
        sample_id=str(info.sample_id),
        true_label=int(label),
        protocol=str(args.protocol),
        clean_total=float(clean_total),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_action_budget=float(args.max_action_budget),
        max_actions=100000,
        candidate_batch_size=int(args.candidate_batch_size),
        candidate_device=str(args.candidate_device),
        diagnostics=full_diag,
    )
    raw_diag: dict[str, Any] = {"profile_detail": False}
    raw_pool = generate_compact_action_descriptors(
        tam=state.tam,
        soft_mask=candidate_mask,
        sample_index=int(info.archive_index),
        sample_id=str(info.sample_id),
        true_label=int(label),
        protocol=str(args.protocol),
        clean_total=float(clean_total),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_action_budget=float(args.max_action_budget),
        max_actions=int(config.raw_pool),
        candidate_batch_size=int(args.candidate_batch_size),
        candidate_device=str(args.candidate_device),
        diagnostics=raw_diag,
    )
    remaining_dummy = int(np.floor(float(clean_total) * float(args.budget) + 1e-9))
    raw_pool = [item for item in raw_pool if 0 < int(item.dummy_count) <= remaining_dummy]
    descriptor_config = SimpleNamespace(
        name=str(config.name),
        prefilter=str(config.prefilter),
        raw_pool=int(config.raw_pool),
        eval_k=int(config.eval_k),
        objective=str(config.objective),
        pair_enabled=False,
        pair_k=int(config.pair_k),
        epsilon=float(config.epsilon),
        tau=float(config.tau),
        generator_pair_actions=0,
    )
    rng = __import__("random").Random(int(args.seed) + int(info.archive_index) * 1009 + sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(config.name))))
    top_descriptors = _prefilter_actions(raw_pool, config=descriptor_config, clean_total=clean_total, args=args, rng=rng)
    materialized = materialize_candidate_descriptors(
        top_descriptors,
        tam=state.tam,
        clean_total=float(clean_total),
        protocol=str(args.protocol),
        max_action_budget=float(args.max_action_budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
    )
    gains = np.zeros(len(materialized), dtype=np.float32)
    if materialized:
        gains, _probs, _metrics = _evaluate_actions(
            state=state,
            actions=materialized,
            original_prob=original_prob,
            label=int(label),
            attacker=attacker,
            device=device,
            args=run_args,
        )

    field_rows = [
        {"class": "CandidateDescriptor", "ordinal": idx, "field": field.name, "type": str(field.type)}
        for idx, field in enumerate(fields(CandidateDescriptor))
    ]
    field_rows += [
        {"class": "ExpandedAction", "ordinal": idx, "field": field.name, "type": str(field.type)}
        for idx, field in enumerate(fields(ExpandedAction))
    ]
    _write_csv(out_dir / "candidate_descriptor_fields.csv", field_rows)
    raw_type_counts = Counter(str(item.action_type) for item in raw_full)
    pool_type_counts = Counter(str(item.action_type) for item in raw_pool)
    top_type_counts = Counter(str(item.action_type) for item in top_descriptors)
    materialized_type_counts = Counter(str(item.action_type) for item in materialized)
    all_types = sorted(set(raw_type_counts) | set(pool_type_counts) | set(top_type_counts) | set(materialized_type_counts))
    action_type_rows = [
        {
            "action_type": action_type,
            "raw_full_count": int(raw_type_counts.get(action_type, 0)),
            "raw_pool_count": int(pool_type_counts.get(action_type, 0)),
            "top128_descriptor_count": int(top_type_counts.get(action_type, 0)),
            "materialized_top128_count": int(materialized_type_counts.get(action_type, 0)),
        }
        for action_type in all_types
    ]
    _write_csv(out_dir / "sample_action_type_counts.csv", action_type_rows)

    gain_by_identity: dict[tuple, float] = {}
    for action, gain in zip(materialized, gains.tolist()):
        gain_by_identity[
            (
                str(action.action_type),
                str(action.tier),
                int(action.window_id),
                int(action.insert_start),
                int(action.insert_end),
                int(action.dose),
                str(action.direction_mode),
            )
        ] = float(gain)
    top_rows: list[dict[str, Any]] = []
    for rank, desc in enumerate(top_descriptors):
        key = (
            str(desc.action_type),
            str(desc.tier),
            int(desc.window_id),
            int(desc.insert_start),
            int(desc.insert_end),
            int(desc.dose),
            str(desc.direction_mode),
        )
        top_rows.append(_descriptor_row(desc, rank=rank, clean_total=clean_total, exact_gain=gain_by_identity.get(key, "")))
    _write_csv(out_dir / "sample_top128_descriptors.csv", top_rows)

    action_rows = [_action_row(action, rank=rank, clean_total=clean_total, exact_gain=float(gains[rank])) for rank, action in enumerate(materialized)]
    _write_csv(out_dir / "sample_top128_materialized_actions.csv", action_rows)

    primary_window_id = int(windows[0].window_id) if windows else -1
    primary_rows = [
        _descriptor_row(item, rank=rank, clean_total=clean_total)
        for rank, item in enumerate(raw_full)
        if int(item.window_id) == int(primary_window_id)
    ]
    _write_csv(out_dir / "sample_top1_keypoint_raw_descriptors.csv", primary_rows)

    selected_trace: dict[str, Any] = {}
    if materialized and len(gains):
        best_idx = int(np.argmax(gains))
        selected = materialized[best_idx]
        before_margin = float(_margin(state.prob, original_prob))
        selected_counts = np.asarray(state.dummy_counts, dtype=np.int32) + np.asarray(selected.counts, dtype=np.int32)
        from scripts.stage_b_run_dual_actuator import _render_dummy

        rendered_trace, rendered_tam, render_stats = _render_dummy(base_trace=state.trace, counts=selected_counts, args=run_args)
        rendered_prob = _predict_one(attacker, rendered_tam, device=device, args=run_args)
        after_margin = float(_margin(rendered_prob, original_prob))
        metrics = probability_metrics(
            original_prob.reshape(1, -1),
            rendered_prob.reshape(1, -1),
            np.asarray([int(label)], dtype=np.int64),
        )
        selected_trace = {
            "selected_rank": int(best_idx),
            "keypoint_window_id": int(selected.window_id),
            "keypoint_direction": str(selected.affected_direction),
            "keypoint_start": int(selected.affected_start),
            "keypoint_end": int(selected.affected_end),
            "keypoint_center": int(selected.affected_center),
            "descriptor_action_type": str(selected.action_type),
            "descriptor_source": str(selected.source),
            "descriptor_insert_start": int(selected.insert_start),
            "descriptor_insert_end": int(selected.insert_end),
            "descriptor_insert_center": int(selected.insert_center),
            "descriptor_dose": int(selected.dose),
            "renderer_dummy_count": int(np.asarray(selected.counts, dtype=np.int32).sum()),
            "renderer_outgoing_dummy_count": int(np.asarray(selected.counts, dtype=np.int32)[0].sum()),
            "renderer_incoming_dummy_count": int(np.asarray(selected.counts, dtype=np.int32)[1].sum()),
            "renderer_bandwidth": float(render_stats.get("raw_bandwidth", 0.0)),
            "renderer_delay_count": 0,
            "renderer_average_delay_bins": "",
            "renderer_p95_delay_bins": "",
            "renderer_max_delay_bins": "",
            "rf_original_pred": int(np.argmax(original_prob)),
            "rf_after_delay_pred": int(np.argmax(state.prob)),
            "rf_after_action_pred": int(np.argmax(rendered_prob)),
            "margin_before_selected_action": float(before_margin),
            "margin_after_selected_action": float(after_margin),
            "exact_gain": float(before_margin - after_margin),
            "accuracy_after_action": float(metrics["accuracy"][0]),
            "flip_after_action": float(metrics["flip"][0]),
        }
    (out_dir / "selected_action_trace.json").write_text(json.dumps(_jsonable(selected_trace), indent=2, ensure_ascii=False), encoding="utf-8")
    summary = {
        "sample_index": int(info.archive_index),
        "sample_id": str(info.sample_id),
        "source_index": int(info.source_index),
        "split": str(info.split),
        "true_label": int(label),
        "clean_total": float(clean_total),
        "clean_pred_from_archive": int(np.argmax(original_prob)),
        "candidate_generation_state": "after_round0_delay" if int(args.max_delay) > 0 else "clean_state",
        "candidate_descriptor_fields": [field.name for field in fields(CandidateDescriptor)],
        "expanded_action_fields": [field.name for field in fields(ExpandedAction)],
        "missing_formal_rectangle_fields": [
            "keypoint_rank",
            "span_bins",
            "target_height",
            "dummy_count_actual",
            "average_delay_bins",
            "p95_delay_bins",
            "maximum_delay_bins",
            "bandwidth_ratio",
            "shape_error",
            "exact_gain",
        ],
        "delay_summary": delay_summary,
        "window_count": int(len(windows)),
        "primary_window_id": int(primary_window_id),
        "primary_window_raw_descriptor_count": int(len(primary_rows)),
        "raw_full_descriptor_count": int(len(raw_full)),
        "raw_pool_descriptor_count": int(len(raw_pool)),
        "top128_descriptor_count": int(len(top_descriptors)),
        "materialized_top128_action_count": int(len(materialized)),
        "raw_full_action_type_counts": dict(sorted(raw_type_counts.items())),
        "top128_action_type_counts": dict(sorted(top_type_counts.items())),
        "materialized_top128_action_type_counts": dict(sorted(materialized_type_counts.items())),
        "raw_generation_diagnostics": raw_diag,
        "full_generation_diagnostics": full_diag,
        "selected_action_trace": selected_trace,
    }
    (out_dir / "candidate_audit_summary.json").write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _d64_only_row(
    *,
    info,
    state,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    stop_reason: str,
    runtime_sec: float,
    state_rf_eval_count: int,
) -> dict[str, Any]:
    metrics = probability_metrics(
        original_prob.reshape(1, -1),
        np.asarray(state.prob, dtype=np.float32).reshape(1, -1),
        np.asarray([int(label)], dtype=np.int64),
    )
    resource = _resource_fields(state, clean_total)
    return {
        "split": str(info.split),
        "sample_index": int(info.archive_index),
        "sample_id": str(info.sample_id),
        "source_index": int(info.source_index),
        "true_label": int(label),
        "clean_accuracy": float(int(int(np.argmax(original_prob)) == int(label))),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "target_margin_success": int(float(metrics["original_class_margin"][0]) <= 0.0),
        "original_pred": int(metrics["original_pred"][0]),
        "final_pred": int(metrics["evaluated_pred"][0]),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "stop_reason": str(stop_reason),
        "runtime_sec": float(runtime_sec),
        "state_rf_eval_count": int(state_rf_eval_count),
        "candidate_rf_eval_count": 0,
        "total_rf_eval_count": int(state_rf_eval_count),
        **resource,
    }


def run_d64_only(args: argparse.Namespace, out_dir: Path, attacker, device) -> dict[str, Any]:
    rows_path = out_dir / "d64_only_full_test.csv"
    summary_path = out_dir / "d64_only_full_test_summary.json"
    if rows_path.is_file() and summary_path.is_file() and not bool(args.force):
        return json.loads(summary_path.read_text(encoding="utf-8"))
    infos = _archive_infos(args.archive, args.split_file)
    if str(args.split_name) != "all":
        infos = [info for info in infos if str(info.split) == str(args.split_name)]
    if int(args.d64_max_samples) > 0:
        infos = infos[: int(args.d64_max_samples)]
    archive_rows = [int(info.archive_index) for info in infos]
    source_rows = np.asarray([int(info.source_index) for info in infos], dtype=np.int64)
    arrays = _load_archive_keys(args.archive, archive_rows, ["tam", "pred_prob", "labels"])
    raw = _load_raw_selected(args.data_root, source_rows, args)
    run_args = _runtime_args(args)
    rows: list[dict[str, Any]] = []
    total_start = time.perf_counter()
    for idx, info in enumerate(infos):
        sample_start = time.perf_counter()
        tam = np.asarray(arrays["tam"][idx], dtype=np.float32)
        prob = np.asarray(arrays["pred_prob"][idx], dtype=np.float32)
        label = int(arrays["labels"][idx])
        clean_total = max(float(tam.sum()), 1.0)
        state = _initial_state(np.asarray(raw[idx], dtype=np.float32), tam, prob)
        stop_reason = "max_rounds_reached"
        state_rf = 0
        for round_index in range(max(1, int(args.rounds))):
            mask0 = _fast_refresh_mask(attacker, state.tam, prob, device=device)
            state = _apply_delay(
                state=state,
                mask=mask0,
                protocol=str(args.protocol),
                delay_budget=max(1, int(round(int(args.max_delay) / max(1, int(args.rounds))))),
                args=run_args,
            )
            state.prob = _predict_one(attacker, state.tam, device=device, args=run_args)
            state_rf += 1
            if _margin(state.prob, prob) <= float(args.margin_target):
                stop_reason = "target_reached"
                break
        rows.append(
            _d64_only_row(
                info=info,
                state=state,
                original_prob=prob,
                label=label,
                clean_total=clean_total,
                stop_reason=stop_reason,
                runtime_sec=float(time.perf_counter() - sample_start),
                state_rf_eval_count=int(state_rf),
            )
        )
        if bool(args.progress) and (idx + 1) % 250 == 0:
            elapsed = time.perf_counter() - total_start
            print(f"d64-only {idx + 1}/{len(infos)} traces, {elapsed:.1f}s")
    _write_csv(rows_path, rows)
    elapsed = float(time.perf_counter() - total_start)
    summary = {
        "split": str(args.split_name),
        "samples": int(len(rows)),
        "total_wall_time_sec": elapsed,
        "traces_per_hour": float(len(rows) / max(elapsed, 1e-9) * 3600.0),
        "clean_rf_accuracy": float(np.mean([float(row["clean_accuracy"]) for row in rows])) if rows else 0.0,
        "defended_rf_accuracy": float(np.mean([float(row["accuracy"]) for row in rows])) if rows else 0.0,
        "flip_rate": float(np.mean([float(row["flip"]) for row in rows])) if rows else 0.0,
        "target_reached_rate": float(np.mean([float(row["target_margin_success"]) for row in rows])) if rows else 0.0,
        "mean_actual_bw": float(np.mean([float(row["actual_dummy_bandwidth"]) for row in rows])) if rows else 0.0,
        "p95_actual_bw": float(np.percentile([float(row["actual_dummy_bandwidth"]) for row in rows], 95)) if rows else 0.0,
        "mean_average_delay_bins": float(np.mean([float(row["average_delay_bins"]) for row in rows])) if rows else 0.0,
        "p95_delay_bins": float(np.percentile([float(row["p95_delay_bins"]) for row in rows], 95)) if rows else 0.0,
        "mean_maximum_delay_bins": float(np.mean([float(row["maximum_delay_bins"]) for row in rows])) if rows else 0.0,
        "mean_state_rf_eval_count": float(np.mean([float(row["state_rf_eval_count"]) for row in rows])) if rows else 0.0,
        "stop_reason_counts": dict(sorted(Counter(str(row["stop_reason"]) for row in rows).items())),
        "rows_path": str(rows_path),
    }
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(str(args.attacker))
    with np.load(args.archive, allow_pickle=False) as archive:
        num_classes = int(archive["pred_prob"].shape[1])
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=num_classes,
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    candidate_summary = audit_candidate_space(args, out_dir, attacker, device)
    result = {"candidate_audit": candidate_summary}
    if bool(args.run_d64_only):
        result["d64_only"] = run_d64_only(args, out_dir, attacker, device)
    (out_dir / "joint_rectangle_action_audit_result.json").write_text(json.dumps(_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(_jsonable(result), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
