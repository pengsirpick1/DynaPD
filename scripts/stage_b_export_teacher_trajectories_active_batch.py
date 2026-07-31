# -*- coding: utf-8 -*-
"""Export Stage B2-E Teacher trajectories with active-state GPU batching.

This is the batched counterpart of ``stage_b_export_teacher_trajectories.py``
for the current compact single-action B2-E teacher. Several trace states are
kept active at once; each state still generates and prefilters its own
candidate actions, while candidate TAM tensors from all active states are
concatenated into larger RF forward batches.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import (
    CandidateDescriptor,
    ExpandedAction,
    materialize_candidate_descriptors,
    generate_compact_action_descriptors,
)
from dmmp.stage_b.objectives import ObjectiveWeights, original_class_objective_delta
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_export_teacher_trajectories import (
    ManifestWriter,
    _archive_selection_rows,
    _budget_schedule,
    _completed_archive_indices,
    _load_archive_rows,
    _max_existing_record_id,
    _prune_teacher_records_to_completed,
    _save_teacher_record,
    _write_selected_samples,
)
from scripts.stage_b_run_b2e_diverse_search import (
    DEFAULT_ARCHIVE,
    MethodConfig,
    _action_dummy_count,
    _default_checkpoint,
    _estimated_abs_gain,
    _identity,
    _margin,
    _method_config,
    _parse_csv_floats,
    _prefilter_actions,
    _predict_probabilities_tensor,
    _runtime_args,
    _sample_row as _controller_sample_row,
    _selection_scores,
)
from scripts.stage_b_run_dual_actuator import (
    EvalState,
    _apply_delay,
    _fast_refresh_mask,
    _initial_state,
    _load_raw_rows,
    _predict_one,
    _render_dummy,
    _timing_add,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default="")
    parser.add_argument("--split_name", choices=["archive", "train", "val", "test", "all"], default="archive")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--budgets", default="0.01,0.02,0.05,0.08,0.10")
    parser.add_argument("--budget_mode", choices=["balanced", "cycle", "fixed"], default="balanced")
    parser.add_argument("--fixed_budget", type=float, default=0.10)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
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
    parser.add_argument("--active_states", type=int, default=8)
    parser.add_argument("--rf_candidate_batch_size", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--store_next_state", action="store_true")
    parser.add_argument("--storage_mode", choices=["dense", "sparse"], default="sparse")
    parser.add_argument("--profile_candidate_generation", action="store_true")
    parser.add_argument("--compact_candidate_generation", action="store_true", default=True)
    parser.add_argument("--deferred_materialize_oversample", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--materialization_batch_size", type=int, default=128)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_score_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_mode", choices=["gpu_tam"], default="gpu_tam")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b2e_teacher_active_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    (target / "records").mkdir(parents=True, exist_ok=True)
    return target


def _timing_add_to(sink: dict[str, float], key: str, seconds: float) -> None:
    sink[str(key)] = float(sink.get(str(key), 0.0)) + float(seconds)


def _timing_max_to(sink: dict[str, float], key: str, value: float) -> None:
    sink[str(key)] = max(float(sink.get(str(key), 0.0)), float(value))


def _clone_args(base: argparse.Namespace, timing: dict[str, float]) -> SimpleNamespace:
    payload = {key: value for key, value in vars(base).items() if key != "timing_accumulator"}
    payload["timing_accumulator"] = timing
    return SimpleNamespace(**payload)


def _new_aggregate() -> dict[str, Any]:
    return {
        "stop_reason": "max_rounds_reached",
        "accepted_single_count": 0,
        "accepted_pair_count": 0,
        "valley_pair_rescue_count": 0,
        "rf_eval_count": 0,
        "candidate_step_count": 0,
        "best_single_gain_seen": 0.0,
        "best_pair_gain_seen": 0.0,
        "proxy_best_gain_recall_values": [],
        "true_best_gain_recall_values": [],
    }


def _new_teacher_stats() -> dict[str, Any]:
    return {
        "record_rows": 0,
        "action_records": 0,
        "stop_records": 0,
        "candidate_total_count": 0,
        "candidate_positive_count": 0,
        "best_gain_values": [],
        "selected_gain_values": [],
        "record_bytes": 0,
        "serialization_time_sec": 0.0,
    }


@dataclass
class ActiveTeacherState:
    local_index: int
    archive_index: int
    source_index: int
    sample_id: str
    true_label: int
    budget: float
    raw_trace: np.ndarray
    original_tam: np.ndarray
    original_mask: np.ndarray
    original_prob: np.ndarray
    clean_total: float
    state: EvalState
    rng: random.Random
    args: SimpleNamespace
    run_start: float
    used: set[tuple] = field(default_factory=set)
    round_index: int = 0
    dummy_step_index: int = 0
    round_mask: np.ndarray | None = None
    aggregate: dict[str, Any] = field(default_factory=_new_aggregate)
    teacher_stats: dict[str, Any] = field(default_factory=_new_teacher_stats)
    finished: bool = False
    stop_reason: str = ""


@dataclass
class CandidatePack:
    item: ActiveTeacherState
    actions: list[ExpandedAction]
    diagnostics: dict[str, Any]
    remaining_dummy: int
    gains: np.ndarray | None = None
    probs: np.ndarray | None = None
    scores: np.ndarray | None = None


def _candidate_device(device: torch.device, args: argparse.Namespace) -> torch.device:
    requested = str(getattr(args, "candidate_eval_device", "auto")).lower()
    if requested == "auto":
        requested = str(args.candidate_device).lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return device if device.type == "cuda" else torch.device("cpu")


def _candidate_chunk_size(args: argparse.Namespace) -> int:
    explicit = int(getattr(args, "rf_candidate_batch_size", 0))
    if explicit > 0:
        return explicit
    candidate = int(getattr(args, "candidate_batch_size", 0))
    if candidate > 0:
        return candidate
    return max(1, int(getattr(args, "batch_size", 128)))


def _emit_teacher_record(
    *,
    item: ActiveTeacherState,
    payload: dict[str, Any],
    output_dir: Path,
    manifest_writer: ManifestWriter,
    args: argparse.Namespace,
    record_id: int,
) -> int:
    row = _save_teacher_record(
        output_dir=output_dir,
        record_id=int(record_id),
        sample_index=int(item.archive_index),
        sample_id=str(item.sample_id),
        source_index=int(item.source_index),
        budget=float(item.budget),
        payload=payload,
        args=args,
    )
    manifest_writer.write(row)
    stats = item.teacher_stats
    stats["record_rows"] = int(stats["record_rows"]) + 1
    event_type = str(row.get("event_type", ""))
    if event_type == "action":
        stats["action_records"] = int(stats["action_records"]) + 1
    if event_type == "stop":
        stats["stop_records"] = int(stats["stop_records"]) + 1
    candidate_gains = np.asarray(payload.get("candidate_gains", np.zeros(0, dtype=np.float32)), dtype=np.float32)
    stats["candidate_total_count"] = int(stats["candidate_total_count"]) + int(candidate_gains.size)
    stats["candidate_positive_count"] = int(stats["candidate_positive_count"]) + int(np.sum(candidate_gains > 0.0))
    if candidate_gains.size:
        stats["best_gain_values"].append(float(np.max(candidate_gains)))
    stats["selected_gain_values"].append(float(row.get("selected_gain", 0.0)))
    stats["record_bytes"] = int(stats["record_bytes"]) + int(row.get("record_bytes", 0))
    stats["serialization_time_sec"] = float(stats["serialization_time_sec"]) + float(row.get("serialization_sec", 0.0))
    return int(record_id) + 1


def _stop_payload(
    *,
    item: ActiveTeacherState,
    stop_reason: str,
    mask: np.ndarray,
    remaining_dummy: int,
    step_index: int | None = None,
    actions: list[ExpandedAction] | None = None,
    gains: np.ndarray | None = None,
    scores: np.ndarray | None = None,
    selected_index: int = -1,
) -> dict[str, Any]:
    candidate_actions = list(actions or [])
    candidate_gains = np.asarray(gains if gains is not None else np.zeros(len(candidate_actions), dtype=np.float32), dtype=np.float32)
    candidate_scores = np.asarray(scores if scores is not None else np.zeros(len(candidate_actions), dtype=np.float32), dtype=np.float32)
    return {
        "event_type": "stop",
        "stop_reason": str(stop_reason),
        "round_index": int(item.round_index),
        "step_index": int(item.dummy_step_index if step_index is None else step_index),
        "budget": float(item.budget),
        "remaining_dummy": int(remaining_dummy),
        "clean_total": float(item.clean_total),
        "candidate_actions": candidate_actions,
        "candidate_gains": candidate_gains,
        "candidate_scores": candidate_scores,
        "selected_index": int(selected_index),
        "selected_kind": "",
        "selected_gain": 0.0,
        "pre_state": item.state,
        "next_state": item.state,
        "mask": np.asarray(mask, dtype=np.float32),
        "original_prob": np.asarray(item.original_prob, dtype=np.float32),
        "label": int(item.true_label),
    }


def _finish_item(item: ActiveTeacherState, stop_reason: str) -> None:
    item.finished = True
    item.stop_reason = str(stop_reason)
    item.aggregate["stop_reason"] = str(stop_reason)


def _finish_round(item: ActiveTeacherState, stop_reason: str, args: argparse.Namespace) -> None:
    item.aggregate["stop_reason"] = str(stop_reason)
    item.round_index += 1
    item.round_mask = None
    item.dummy_step_index = 0
    if item.round_index >= max(1, int(args.rounds)):
        if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
            _finish_item(item, "target_reached")
        else:
            _finish_item(item, str(stop_reason))


def _enter_round_or_finish(
    item: ActiveTeacherState,
    *,
    attacker,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    manifest_writer: ManifestWriter,
    record_id: int,
) -> int:
    rounds = max(1, int(args.rounds))
    if item.round_index >= rounds:
        if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
            _finish_item(item, "target_reached")
        else:
            _finish_item(item, str(item.aggregate.get("stop_reason", "max_rounds_reached")))
        return int(record_id)

    stop_mask = np.asarray(item.original_mask, dtype=np.float32)
    if int(args.max_delay) > 0:
        start = time.perf_counter()
        mask0 = _fast_refresh_mask(attacker, item.state.tam, item.original_prob, device=device)
        _timing_add(item.args, "keypoint_refresh_time_sec", time.perf_counter() - start)
        stop_mask = np.asarray(mask0, dtype=np.float32)
        item.state = _apply_delay(
            state=item.state,
            mask=mask0,
            protocol=str(args.protocol),
            delay_budget=max(1, int(round(int(args.max_delay) / rounds))),
            args=item.args,
        )
        item.state.prob = _predict_one(attacker, item.state.tam, device=device, args=item.args)

    remaining_dummy = int(np.floor(float(item.clean_total) * float(item.budget) + 1e-9)) - int(np.asarray(item.state.dummy_counts).sum())
    if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
        payload = _stop_payload(
            item=item,
            stop_reason="target_reached",
            mask=stop_mask,
            remaining_dummy=max(0, int(remaining_dummy)),
            step_index=-1,
        )
        record_id = _emit_teacher_record(
            item=item,
            payload=payload,
            output_dir=output_dir,
            manifest_writer=manifest_writer,
            args=args,
            record_id=int(record_id),
        )
        _finish_item(item, "target_reached")
        return int(record_id)

    if float(item.budget) <= 0.0:
        payload = _stop_payload(
            item=item,
            stop_reason="no_dummy_budget",
            mask=stop_mask,
            remaining_dummy=0,
            step_index=-1,
        )
        record_id = _emit_teacher_record(
            item=item,
            payload=payload,
            output_dir=output_dir,
            manifest_writer=manifest_writer,
            args=args,
            record_id=int(record_id),
        )
        _finish_round(item, "no_dummy_budget", args)
        return int(record_id)

    start = time.perf_counter()
    item.round_mask = np.asarray(_fast_refresh_mask(attacker, item.state.tam, item.original_prob, device=device), dtype=np.float32)
    _timing_add(item.args, "keypoint_refresh_time_sec", time.perf_counter() - start)
    item.dummy_step_index = 0
    item.aggregate["stop_reason"] = "max_actions_reached"
    return int(record_id)


def _build_pack(
    item: ActiveTeacherState,
    *,
    config: MethodConfig,
    args: argparse.Namespace,
    output_dir: Path,
    manifest_writer: ManifestWriter,
    record_id: int,
) -> tuple[CandidatePack | None, int]:
    while not item.finished:
        if item.round_mask is None:
            return None, int(record_id)

        max_dummy = int(np.floor(float(item.clean_total) * float(item.budget) + 1e-9))
        if item.dummy_step_index >= int(args.max_dummy_steps):
            if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
                _finish_item(item, "target_reached")
                return None, int(record_id)
            if int(np.asarray(item.state.dummy_counts).sum()) >= max_dummy:
                _finish_round(item, "bandwidth_10pct_reached", args)
                return None, int(record_id)
            _finish_round(item, "max_actions_reached", args)
            return None, int(record_id)

        if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
            _finish_item(item, "target_reached")
            return None, int(record_id)

        remaining_dummy = max_dummy - int(np.asarray(item.state.dummy_counts).sum())
        if remaining_dummy <= 0:
            payload = _stop_payload(
                item=item,
                stop_reason="bandwidth_10pct_reached",
                mask=np.asarray(item.round_mask, dtype=np.float32),
                remaining_dummy=int(remaining_dummy),
            )
            record_id = _emit_teacher_record(
                item=item,
                payload=payload,
                output_dir=output_dir,
                manifest_writer=manifest_writer,
                args=args,
                record_id=int(record_id),
            )
            _finish_round(item, "bandwidth_10pct_reached", args)
            return None, int(record_id)

        diagnostics: dict[str, Any] = {"profile_detail": bool(getattr(args, "profile_candidate_generation", False))}
        start = time.perf_counter()
        raw_descriptors = generate_compact_action_descriptors(
            tam=item.state.tam,
            soft_mask=np.asarray(item.round_mask, dtype=np.float32),
            sample_index=int(item.archive_index),
            sample_id=str(item.sample_id),
            true_label=int(item.true_label),
            protocol=str(args.protocol),
            clean_total=float(item.clean_total),
            ratio=float(args.ratio),
            max_windows=int(args.max_windows),
            max_action_budget=float(args.max_action_budget),
            max_actions=int(config.raw_pool),
            candidate_batch_size=int(getattr(args, "candidate_batch_size", 0)),
            candidate_device=str(getattr(args, "candidate_score_device", getattr(args, "candidate_device", "auto"))),
            diagnostics=diagnostics,
        )
        descriptor_filter_start = time.perf_counter()
        raw_descriptors = [
            action
            for action in raw_descriptors
            if _identity(action) not in item.used and 0 < _action_dummy_count(action) <= remaining_dummy
        ]
        _timing_add(item.args, "compact_descriptor_used_budget_filter_time_sec", time.perf_counter() - descriptor_filter_start)
        materialize_k = max(int(config.eval_k), int(config.eval_k) * max(1, int(getattr(args, "deferred_materialize_oversample", 1))))
        materialize_k = min(max(0, int(materialize_k)), len(raw_descriptors))
        descriptor_config = MethodConfig(
            name=str(config.name),
            prefilter=str(config.prefilter),
            raw_pool=int(config.raw_pool),
            eval_k=int(materialize_k),
            objective=str(config.objective),
            pair_enabled=False,
            pair_k=int(config.pair_k),
            epsilon=float(config.epsilon),
            tau=float(config.tau),
            generator_pair_actions=0,
        )
        descriptor_selected = _prefilter_actions(
            raw_descriptors,
            config=descriptor_config,
            clean_total=float(item.clean_total),
            args=item.args,
            rng=item.rng,
        )
        materialize_start = time.perf_counter()
        raw_actions: list[ExpandedAction] = []
        materialize_batch = max(1, int(getattr(args, "materialization_batch_size", 64)))
        for materialize_start_idx in range(0, len(descriptor_selected), materialize_batch):
            raw_actions.extend(
                materialize_candidate_descriptors(
                    descriptor_selected[materialize_start_idx : materialize_start_idx + materialize_batch],
                    tam=item.state.tam,
                    clean_total=float(item.clean_total),
                    protocol=str(args.protocol),
                    max_action_budget=float(args.max_action_budget),
                    max_local_rate_peak=int(args.max_local_rate_peak),
                )
            )
        _timing_add(item.args, "deferred_materialization_time_sec", time.perf_counter() - materialize_start)
        _timing_add(item.args, "compact_descriptor_count", float(len(raw_descriptors)))
        _timing_add(item.args, "deferred_action_objects_built", float(len(raw_actions)))
        _timing_add(item.args, "deferred_dense_counts_built", float(len(raw_actions)))
        for timing_key, timing_value in dict(diagnostics.get("timing_sec", {})).items():
            _timing_add(item.args, str(timing_key), float(timing_value))
        filter_start = time.perf_counter()
        raw_actions = [action for action in raw_actions if _identity(action) not in item.used and 0 < _action_dummy_count(action) <= remaining_dummy]
        _timing_add(item.args, "candidate_used_budget_filter_time_sec", time.perf_counter() - filter_start)
        _timing_add(item.args, "candidate_generation_time_sec", time.perf_counter() - start)

        if not raw_actions:
            payload = _stop_payload(
                item=item,
                stop_reason="candidate_pool_exhausted",
                mask=np.asarray(item.round_mask, dtype=np.float32),
                remaining_dummy=int(remaining_dummy),
            )
            record_id = _emit_teacher_record(
                item=item,
                payload=payload,
                output_dir=output_dir,
                manifest_writer=manifest_writer,
                args=args,
                record_id=int(record_id),
            )
            _finish_round(item, "candidate_pool_exhausted", args)
            return None, int(record_id)

        start = time.perf_counter()
        actions = _prefilter_actions(raw_actions, config=config, clean_total=float(item.clean_total), args=item.args, rng=item.rng)
        filter_start = time.perf_counter()
        actions = [action for action in actions if _identity(action) not in item.used and 0 < _action_dummy_count(action) <= remaining_dummy]
        _timing_add(item.args, "candidate_selected_used_budget_filter_time_sec", time.perf_counter() - filter_start)
        _timing_add(item.args, "candidate_prefilter_time_sec", time.perf_counter() - start)

        if not actions:
            payload = _stop_payload(
                item=item,
                stop_reason="candidate_pool_exhausted",
                mask=np.asarray(item.round_mask, dtype=np.float32),
                remaining_dummy=int(remaining_dummy),
            )
            record_id = _emit_teacher_record(
                item=item,
                payload=payload,
                output_dir=output_dir,
                manifest_writer=manifest_writer,
                args=args,
                record_id=int(record_id),
            )
            _finish_round(item, "candidate_pool_exhausted", args)
            return None, int(record_id)

        raw_best_proxy = max((_estimated_abs_gain(action, float(item.clean_total)) for action in raw_descriptors), default=0.0)
        selected_best_proxy = max((_estimated_abs_gain(action, float(item.clean_total)) for action in actions), default=0.0)
        item.aggregate["proxy_best_gain_recall_values"].append(float(selected_best_proxy / max(raw_best_proxy, 1e-8)))
        item.aggregate["candidate_step_count"] = int(item.aggregate["candidate_step_count"]) + 1
        return CandidatePack(
            item=item,
            actions=list(actions),
            diagnostics=diagnostics,
            remaining_dummy=int(remaining_dummy),
        ), int(record_id)

    return None, int(record_id)


def _evaluate_packs(
    packs: list[CandidatePack],
    *,
    attacker,
    device: torch.device,
    args: argparse.Namespace,
    global_timing: dict[str, float],
) -> None:
    if not packs:
        return
    rows: list[tuple[int, int]] = []
    for pack_index, pack in enumerate(packs):
        for action_index, _action in enumerate(pack.actions):
            rows.append((pack_index, action_index))
    if not rows:
        return
    candidate_device = _candidate_device(device, args)
    chunk_size = max(1, _candidate_chunk_size(args))
    width = int(packs[0].item.state.tam.shape[-1])
    probs_chunks: list[np.ndarray] = []
    start_total = time.perf_counter()
    max_chunk = 0
    if candidate_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(candidate_device)
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        chunk_rows = rows[start:end]
        max_chunk = max(max_chunk, len(chunk_rows))
        row_counts: dict[int, int] = {}
        for pack_idx, _action_idx in chunk_rows:
            row_counts[pack_idx] = int(row_counts.get(pack_idx, 0)) + 1
        chunk_start = time.perf_counter()
        build_start = time.perf_counter()
        base_np = np.stack([np.asarray(packs[pack_idx].item.state.tam, dtype=np.float32) for pack_idx, _action_idx in chunk_rows], axis=0)
        counts_np = np.stack([np.asarray(packs[pack_idx].actions[action_idx].counts, dtype=np.float32) for pack_idx, action_idx in chunk_rows], axis=0)
        if base_np.shape[-1] != int(width) or counts_np.shape[-1] != int(width):
            raise ValueError(f"Candidate TAM width mismatch: base={base_np.shape}, counts={counts_np.shape}, expected={width}")
        base_t = torch.as_tensor(base_np, dtype=torch.float32, device=candidate_device)
        counts_t = torch.as_tensor(counts_np, dtype=torch.float32, device=candidate_device)
        candidate_tams = base_t + counts_t
        build_sec = time.perf_counter() - build_start
        forward_start = time.perf_counter()
        probs_t = _predict_probabilities_tensor(attacker, candidate_tams, batch_size=chunk_size)
        if candidate_device.type == "cuda":
            torch.cuda.synchronize(candidate_device)
        forward_sec = time.perf_counter() - forward_start
        chunk_sec = time.perf_counter() - chunk_start
        _timing_add_to(global_timing, "candidate_tam_gpu_build_time_sec", build_sec)
        _timing_add_to(global_timing, "rf_forward_time_sec", forward_sec)
        _timing_add_to(global_timing, "candidate_gpu_tam_eval_time_sec", chunk_sec)
        _timing_add_to(global_timing, "candidate_gpu_tam_eval_batches", 1.0)
        for pack_idx, count in row_counts.items():
            share = float(count) / max(float(len(chunk_rows)), 1.0)
            timing = packs[pack_idx].item.args.timing_accumulator
            _timing_add_to(timing, "candidate_tam_gpu_build_time_sec", float(build_sec) * share)
            _timing_add_to(timing, "rf_forward_time_sec", float(forward_sec) * share)
            _timing_add_to(timing, "candidate_gpu_tam_eval_time_sec", float(chunk_sec) * share)
            _timing_add_to(timing, "candidate_gpu_tam_eval_count", float(count))
        probs_chunks.append(probs_t.detach().cpu().numpy().astype(np.float32))
        del base_np, counts_np, base_t, counts_t, candidate_tams, probs_t
    if candidate_device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(candidate_device) / (1024.0 * 1024.0))
        _timing_max_to(global_timing, "candidate_gpu_peak_allocated_mb", peak_mb)
        for pack in packs:
            _timing_max_to(pack.item.args.timing_accumulator, "candidate_gpu_peak_allocated_mb", peak_mb)
    _timing_add_to(global_timing, "candidate_gpu_tam_eval_count", float(len(rows)))
    _timing_max_to(global_timing, "candidate_gpu_tam_max_chunk", float(max_chunk))

    probs = np.concatenate(probs_chunks, axis=0).astype(np.float32)
    cursor = 0
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    for pack in packs:
        n = len(pack.actions)
        pack_probs = probs[cursor : cursor + n]
        cursor += n
        original = np.repeat(pack.item.original_prob.reshape(1, -1), n, axis=0)
        reference = np.repeat(pack.item.state.prob.reshape(1, -1), n, axis=0)
        pack.probs = pack_probs
        pack.gains = original_class_objective_delta(original, reference, pack_probs, weights).astype(np.float32)


def _apply_pack_choice(
    pack: CandidatePack,
    *,
    attacker,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
    manifest_writer: ManifestWriter,
    record_id: int,
) -> int:
    item = pack.item
    gains = np.asarray(pack.gains if pack.gains is not None else np.zeros(0, dtype=np.float32), dtype=np.float32)
    scores = _selection_scores(gains, pack.actions, float(item.clean_total), "absolute")
    pack.scores = np.asarray(scores, dtype=np.float32)
    best_idx = int(np.argmax(scores)) if scores.size else -1
    best_gain = float(gains[best_idx]) if best_idx >= 0 else 0.0
    item.aggregate["rf_eval_count"] = int(item.aggregate["rf_eval_count"]) + int(len(pack.actions))
    if gains.size:
        item.aggregate["best_single_gain_seen"] = max(
            float(item.aggregate["best_single_gain_seen"]),
            max(float(value) for value in gains.tolist()),
        )

    if best_idx < 0 or best_gain <= 0.0:
        payload = _stop_payload(
            item=item,
            stop_reason="no_positive_single",
            mask=np.asarray(item.round_mask, dtype=np.float32),
            remaining_dummy=int(pack.remaining_dummy),
            actions=list(pack.actions),
            gains=gains,
            scores=np.asarray(scores, dtype=np.float32),
        )
        record_id = _emit_teacher_record(
            item=item,
            payload=payload,
            output_dir=output_dir,
            manifest_writer=manifest_writer,
            args=args,
            record_id=int(record_id),
        )
        _finish_round(item, "no_positive_single", args)
        return int(record_id)

    selected_action = pack.actions[best_idx]
    pre_state = item.state
    previous_dummy_count = int(np.asarray(pre_state.dummy_counts, dtype=np.int32).sum())
    selected_counts = np.asarray(pre_state.dummy_counts, dtype=np.int32) + np.asarray(selected_action.counts, dtype=np.int32)
    trace, tam, stats = _render_dummy(base_trace=pre_state.trace, counts=selected_counts, args=item.args)
    prob = _predict_one(attacker, tam, device=device, args=item.args)
    item.used.add(_identity(selected_action))
    item.aggregate["accepted_single_count"] = int(item.aggregate["accepted_single_count"]) + 1
    item.state = EvalState(
        trace=trace,
        tam=tam,
        prob=prob,
        dummy_counts=np.asarray(selected_counts, dtype=np.int32),
        dummy_bandwidth=float(stats["raw_bandwidth"]),
        avg_delay=float(pre_state.avg_delay),
        p95_delay=float(pre_state.p95_delay),
        max_delay=int(pre_state.max_delay),
        delay_values=tuple(pre_state.delay_values),
        outgoing_delay_values=tuple(pre_state.outgoing_delay_values),
        incoming_delay_values=tuple(pre_state.incoming_delay_values),
        selected_actions=list(pre_state.selected_actions) + [selected_action],
    )
    payload = {
        "event_type": "action",
        "stop_reason": "",
        "round_index": int(item.round_index),
        "step_index": int(item.dummy_step_index),
        "budget": float(item.budget),
        "remaining_dummy": int(pack.remaining_dummy),
        "clean_total": float(item.clean_total),
        "candidate_actions": list(pack.actions),
        "candidate_gains": gains,
        "candidate_scores": np.asarray(scores, dtype=np.float32),
        "selected_index": int(best_idx),
        "selected_kind": "single",
        "selected_gain": float(best_gain),
        "pre_state": pre_state,
        "next_state": item.state,
        "mask": np.asarray(item.round_mask, dtype=np.float32),
        "original_prob": np.asarray(item.original_prob, dtype=np.float32),
        "label": int(item.true_label),
        "selected_action_count": 1,
        "added_dummy": int(np.asarray(selected_counts, dtype=np.int32).sum() - previous_dummy_count),
        "cumulative_dummy": int(np.asarray(selected_counts, dtype=np.int32).sum()),
    }
    record_id = _emit_teacher_record(
        item=item,
        payload=payload,
        output_dir=output_dir,
        manifest_writer=manifest_writer,
        args=args,
        record_id=int(record_id),
    )
    item.dummy_step_index += 1
    return int(record_id)


def _sample_summary_row(
    *,
    item: ActiveTeacherState,
    config: MethodConfig,
    args: argparse.Namespace,
    runtime_sec: float,
) -> dict[str, Any]:
    stats = item.teacher_stats
    timing = item.args.timing_accumulator
    sample_row = _controller_sample_row(
        sample_index=int(item.archive_index),
        sample_id=str(item.sample_id),
        protocol=str(args.protocol),
        config=config,
        budget=float(item.budget),
        margin_target=float(args.margin_target),
        original_prob=np.asarray(item.original_prob, dtype=np.float32),
        state=item.state,
        label=int(item.true_label),
        clean_total=float(item.clean_total),
        runtime=float(runtime_sec),
        aggregate=item.aggregate,
    )
    candidate_total = int(stats["candidate_total_count"])
    accounted_timing_keys = [
        "candidate_generation_time_sec",
        "candidate_prefilter_time_sec",
        "candidate_pair_generation_time_sec",
        "renderer_time_sec",
        "tam_rebuild_time_sec",
        "rf_forward_time_sec",
        "candidate_tam_gpu_build_time_sec",
        "keypoint_refresh_time_sec",
        "delay_trace_time_sec",
    ]
    timed_accounted = float(sum(float(timing.get(key, 0.0)) for key in accounted_timing_keys)) + float(stats["serialization_time_sec"])
    sample_row.update(
        {
            "archive_index": int(item.archive_index),
            "source_index": int(item.source_index),
            "budget": float(item.budget),
            "records": int(stats["record_rows"]),
            "action_records": int(stats["action_records"]),
            "stop_records": int(stats["stop_records"]),
            "candidate_total_count": int(candidate_total),
            "candidate_positive_count": int(stats["candidate_positive_count"]),
            "candidate_positive_gain_rate": float(int(stats["candidate_positive_count"]) / max(candidate_total, 1)),
            "mean_best_gain_per_state": float(np.mean(stats["best_gain_values"])) if stats["best_gain_values"] else 0.0,
            "max_best_gain_per_state": float(np.max(stats["best_gain_values"])) if stats["best_gain_values"] else 0.0,
            "mean_selected_gain": float(np.mean(stats["selected_gain_values"])) if stats["selected_gain_values"] else 0.0,
            "record_bytes_per_trace": int(stats["record_bytes"]),
            "serialization_time_sec": float(stats["serialization_time_sec"]),
            "candidate_generation_time_sec": float(timing.get("candidate_generation_time_sec", 0.0)),
            "candidate_prefilter_time_sec": float(timing.get("candidate_prefilter_time_sec", 0.0)),
            "candidate_pair_generation_time_sec": float(timing.get("candidate_pair_generation_time_sec", 0.0)),
            "candidate_used_budget_filter_time_sec": float(timing.get("candidate_used_budget_filter_time_sec", 0.0)),
            "candidate_selected_used_budget_filter_time_sec": float(timing.get("candidate_selected_used_budget_filter_time_sec", 0.0)),
            "compact_descriptor_used_budget_filter_time_sec": float(timing.get("compact_descriptor_used_budget_filter_time_sec", 0.0)),
            "candidate_window_extract_time_sec": float(timing.get("candidate_window_extract_time_sec", 0.0)),
            "structural_anchor_extract_time_sec": float(timing.get("structural_anchor_extract_time_sec", 0.0)),
            "anchor_grid_build_time_sec": float(timing.get("anchor_grid_build_time_sec", 0.0)),
            "action_object_build_time_sec": float(timing.get("action_object_build_time_sec", 0.0)),
            "legality_filter_time_sec": float(timing.get("legality_filter_time_sec", 0.0)),
            "deduplicate_time_sec": float(timing.get("deduplicate_time_sec", 0.0)),
            "compact_deduplicate_time_sec": float(timing.get("compact_deduplicate_time_sec", 0.0)),
            "pair_action_build_time_sec": float(timing.get("pair_action_build_time_sec", 0.0)),
            "pair_deduplicate_time_sec": float(timing.get("pair_deduplicate_time_sec", 0.0)),
            "score_hint_sort_time_sec": float(timing.get("score_hint_sort_time_sec", 0.0)),
            "diverse_limit_time_sec": float(timing.get("diverse_limit_time_sec", 0.0)),
            "generate_expanded_actions_total_time_sec": float(timing.get("generate_expanded_actions_total_time_sec", 0.0)),
            "compact_diverse_limit_time_sec": float(timing.get("compact_diverse_limit_time_sec", 0.0)),
            "compact_descriptor_generation_total_time_sec": float(timing.get("compact_descriptor_generation_total_time_sec", 0.0)),
            "deferred_materialization_time_sec": float(timing.get("deferred_materialization_time_sec", 0.0)),
            "compact_descriptor_count": float(timing.get("compact_descriptor_count", 0.0)),
            "deferred_action_objects_built": float(timing.get("deferred_action_objects_built", 0.0)),
            "deferred_dense_counts_built": float(timing.get("deferred_dense_counts_built", 0.0)),
            "candidate_tam_gpu_build_time_sec": float(timing.get("candidate_tam_gpu_build_time_sec", 0.0)),
            "candidate_gpu_tam_eval_time_sec": float(timing.get("candidate_gpu_tam_eval_time_sec", 0.0)),
            "candidate_gpu_tam_eval_count": float(timing.get("candidate_gpu_tam_eval_count", 0.0)),
            "candidate_gpu_peak_allocated_mb": float(timing.get("candidate_gpu_peak_allocated_mb", 0.0)),
            "renderer_time_sec": float(timing.get("renderer_time_sec", 0.0)),
            "tam_rebuild_time_sec": float(timing.get("tam_rebuild_time_sec", 0.0)),
            "rf_forward_time_sec": float(timing.get("rf_forward_time_sec", 0.0)),
            "keypoint_refresh_time_sec": float(timing.get("keypoint_refresh_time_sec", 0.0)),
            "delay_trace_time_sec": float(timing.get("delay_trace_time_sec", 0.0)),
            "timed_accounted_sec": float(timed_accounted),
            "queue_or_wait_time_sec": float(max(runtime_sec - timed_accounted, 0.0)),
        }
    )
    return sample_row


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    run_args = _runtime_args(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    config = _method_config(args.method)
    if bool(config.pair_enabled):
        raise ValueError("Active-batch teacher exporter currently supports single-action B2-E methods only.")
    if int(args.true_recall_pool_size) > 0:
        raise ValueError("Active-batch teacher exporter does not support --true_recall_pool_size yet.")
    if not bool(getattr(args, "compact_candidate_generation", True)):
        raise ValueError("Active-batch teacher exporter requires compact candidate generation.")
    budgets = _parse_csv_floats(args.budgets)
    archive_rows = _archive_selection_rows(args)
    archive = _load_archive_rows(args.archive, archive_rows)
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(args.data_root, source_indices, run_args)
    schedule = _budget_schedule(int(archive_rows.size), budgets, args)
    selected_samples_path = output_dir / "teacher_selected_samples.csv"
    _write_selected_samples(
        selected_samples_path,
        archive_rows=archive_rows,
        sample_ids=sample_ids,
        source_indices=source_indices,
        schedule=schedule,
    )
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

    teacher_records_path = output_dir / "teacher_records.csv"
    sample_summary_path = output_dir / "teacher_sample_summary.csv"
    completed_before = _completed_archive_indices(sample_summary_path) if bool(args.resume) else set()
    pruned_records = _prune_teacher_records_to_completed(teacher_records_path, completed_before) if bool(args.resume) else 0
    manifest_writer = ManifestWriter(teacher_records_path, append=bool(args.resume))
    sample_writer = ManifestWriter(sample_summary_path, append=bool(args.resume))
    record_count = _max_existing_record_id(teacher_records_path) + 1 if bool(args.resume) else 0
    starting_record_count = int(record_count)
    skipped_samples = 0
    new_samples = 0
    sample_rows: list[dict[str, Any]] = []
    global_timing: dict[str, float] = {}
    active: list[ActiveTeacherState] = []
    pending = list(range(len(tam)))
    stable_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(config.name)))
    run_start = time.perf_counter()

    try:
        while pending or active:
            while pending and len(active) < max(1, int(args.active_states)):
                local_index = pending.pop(0)
                archive_index = int(archive_rows[local_index])
                if archive_index in completed_before:
                    skipped_samples += 1
                    if bool(args.progress):
                        print(
                            f"[teacher-active-export] skip completed {local_index + 1}/{len(archive_rows)} "
                            f"archive_index={archive_index} sample={sample_ids[local_index]}",
                            flush=True,
                        )
                    continue
                timing: dict[str, float] = {}
                item_args = _clone_args(args, timing)
                state = _initial_state(
                    np.asarray(raw_rows[local_index], dtype=np.float32),
                    np.asarray(tam[local_index], dtype=np.float32),
                    np.asarray(prob[local_index], dtype=np.float32),
                )
                item = ActiveTeacherState(
                    local_index=int(local_index),
                    archive_index=int(archive_index),
                    source_index=int(source_indices[local_index]),
                    sample_id=str(sample_ids[local_index]),
                    true_label=int(labels[local_index]),
                    budget=float(schedule[local_index]),
                    raw_trace=np.asarray(raw_rows[local_index], dtype=np.float32),
                    original_tam=np.asarray(tam[local_index], dtype=np.float32),
                    original_mask=np.asarray(mask[local_index], dtype=np.float32),
                    original_prob=np.asarray(prob[local_index], dtype=np.float32),
                    clean_total=max(float(np.asarray(tam[local_index], dtype=np.float32).sum()), 1.0),
                    state=state,
                    rng=random.Random(int(args.seed) + int(archive_index) * 1009 + stable_seed),
                    args=item_args,
                    run_start=time.perf_counter(),
                )
                active.append(item)
                if bool(args.progress):
                    print(
                        f"[teacher-active-export] start {local_index + 1}/{len(archive_rows)} "
                        f"archive_index={archive_index} sample={sample_ids[local_index]} B={float(schedule[local_index]):g}",
                        flush=True,
                    )

            for item in list(active):
                if not item.finished and item.round_mask is None:
                    record_count = _enter_round_or_finish(
                        item,
                        attacker=attacker,
                        device=device,
                        args=args,
                        output_dir=output_dir,
                        manifest_writer=manifest_writer,
                        record_id=int(record_count),
                    )

            packs: list[CandidatePack] = []
            for item in list(active):
                if item.finished:
                    continue
                pack, record_count = _build_pack(
                    item,
                    config=config,
                    args=args,
                    output_dir=output_dir,
                    manifest_writer=manifest_writer,
                    record_id=int(record_count),
                )
                if pack is not None:
                    packs.append(pack)

            _evaluate_packs(packs, attacker=attacker, device=device, args=args, global_timing=global_timing)
            for pack in packs:
                record_count = _apply_pack_choice(
                    pack,
                    attacker=attacker,
                    device=device,
                    args=args,
                    output_dir=output_dir,
                    manifest_writer=manifest_writer,
                    record_id=int(record_count),
                )

            still_active: list[ActiveTeacherState] = []
            for item in active:
                if item.finished:
                    runtime_sec = float(time.perf_counter() - item.run_start)
                    row = _sample_summary_row(item=item, config=config, args=args, runtime_sec=runtime_sec)
                    sample_writer.write(row)
                    sample_rows.append(row)
                    new_samples += 1
                    if bool(args.progress):
                        print(
                            f"[teacher-active-export] done {new_samples + skipped_samples}/{len(archive_rows)} "
                            f"archive_index={item.archive_index} stop={item.stop_reason} records={item.teacher_stats['record_rows']}",
                            flush=True,
                        )
                else:
                    still_active.append(item)
            active = still_active

            if not packs and active and not pending:
                # All active items may have just advanced round boundaries. Loop again
                # so they can enter the next round or finish cleanly.
                continue
    finally:
        manifest_writer.close()
        sample_writer.close()

    manifest = {
        "archive": str(args.archive),
        "checkpoint": str(Path(checkpoint).resolve()),
        "records": int(record_count),
        "new_records": int(record_count - starting_record_count),
        "samples": int(archive_rows.size),
        "new_samples": int(new_samples),
        "skipped_samples": int(skipped_samples),
        "completed_before": int(len(completed_before)),
        "pruned_incomplete_or_orphan_record_rows": int(pruned_records),
        "sample_offset": int(args.sample_offset),
        "max_samples": int(args.max_samples),
        "split_file": str(args.split_file),
        "split_name": str(args.split_name),
        "shard_id": int(args.shard_id),
        "num_shards": int(args.num_shards),
        "archive_row_min": int(np.min(archive_rows)),
        "archive_row_max": int(np.max(archive_rows)),
        "source_index_min": int(np.min(source_indices)),
        "source_index_max": int(np.max(source_indices)),
        "protocol": str(args.protocol),
        "method": str(config.name),
        "budgets": budgets,
        "budget_mode": str(args.budget_mode),
        "fixed_budget": float(args.fixed_budget),
        "max_delay": int(args.max_delay),
        "rounds": int(args.rounds),
        "margin_target": float(args.margin_target),
        "teacher_records": str(teacher_records_path),
        "sample_summary": str(sample_summary_path),
        "selected_samples": str(selected_samples_path),
        "records_dir": str(output_dir / "records"),
        "runtime_sec": float(time.perf_counter() - run_start),
        "traces_per_hour": float(new_samples / max((time.perf_counter() - run_start) / 3600.0, 1e-9)),
        "active_states": int(args.active_states),
        "candidate_batch_size": int(args.candidate_batch_size),
        "rf_candidate_batch_size": int(args.rf_candidate_batch_size),
        "candidate_device": str(args.candidate_device),
        "candidate_score_device": str(args.candidate_score_device),
        "candidate_eval_device": str(args.candidate_eval_device),
        "candidate_eval_mode": str(args.candidate_eval_mode),
        "global_timing": {key: float(value) for key, value in sorted(global_timing.items())},
        "store_next_state": bool(args.store_next_state),
        "storage_mode": str(args.storage_mode),
        "resume": bool(args.resume),
    }
    (output_dir / "teacher_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
