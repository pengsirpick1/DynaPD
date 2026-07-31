# -*- coding: utf-8 -*-
"""Probe active-state GPU batching for Stage B2-E candidate evaluation.

This script is intentionally a throughput probe, not the canonical Teacher
exporter. It keeps several independent trace states active, generates each
state's compact candidate actions, concatenates candidate TAM tensors across
states, performs one shared RF forward batch, then applies the best action per
state with the exact renderer.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import generate_compact_action_descriptors, materialize_candidate_descriptors
from dmmp.stage_b.objectives import ObjectiveWeights, original_class_objective_delta, probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_export_teacher_trajectories import _archive_selection_rows, _load_archive_rows
from scripts.stage_b_run_b2e_diverse_search import (
    DEFAULT_ARCHIVE,
    _action_dummy_count,
    _default_checkpoint,
    _identity,
    _margin,
    _method_config,
    _predict_probabilities_tensor,
    _prefilter_actions,
    _selection_scores,
)
from scripts.stage_b_run_dual_actuator import _initial_state, _load_raw_rows, _predict_one, _render_dummy, _timing_add


def _timing_max(args: argparse.Namespace, key: str, value: float) -> None:
    sink = getattr(args, "timing_accumulator", None)
    if isinstance(sink, dict):
        sink[key] = max(float(sink.get(key, 0.0)), float(value))


@dataclass
class ActiveProbeState:
    local_index: int
    archive_index: int
    sample_id: str
    true_label: int
    raw_trace: np.ndarray
    original_mask: np.ndarray
    original_prob: np.ndarray
    clean_total: float
    state: Any
    rng: random.Random
    used: set[tuple] = field(default_factory=set)
    steps: int = 0
    rf_eval_count: int = 0
    accepted_count: int = 0
    stop_reason: str = ""
    finished: bool = False


@dataclass
class CandidatePack:
    item: ActiveProbeState
    actions: list[Any]
    gains: np.ndarray | None = None
    probs: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default="")
    parser.add_argument("--split_name", choices=["archive", "train", "val", "test", "all"], default="archive")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="stage_b_active_state_gpu_batch_probe")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--fixed_budget", type=float, default=0.10)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--sample_offset", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--active_states", type=int, default=4)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--candidate_batch_size", type=int, default=2048)
    parser.add_argument("--materialization_batch_size", type=int, default=128)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_eval_mode", choices=["gpu_tam"], default="gpu_tam")
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) / str(args.run_name)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _candidate_device(device: torch.device, args: argparse.Namespace) -> torch.device:
    requested = str(args.candidate_device).lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return device if device.type == "cuda" else torch.device("cpu")


def _build_pack(
    item: ActiveProbeState,
    *,
    config,
    args: argparse.Namespace,
    protocol: str,
) -> CandidatePack | None:
    remaining_dummy = int(np.floor(float(item.clean_total) * float(args.fixed_budget) + 1e-9)) - int(np.asarray(item.state.dummy_counts).sum())
    if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
        item.finished = True
        item.stop_reason = "target_reached"
        return None
    if item.steps >= int(args.max_dummy_steps):
        item.finished = True
        item.stop_reason = "max_actions_reached"
        return None
    if remaining_dummy <= 0:
        item.finished = True
        item.stop_reason = "bandwidth_10pct_reached"
        return None
    diagnostics: dict[str, Any] = {"profile_detail": False}
    start = time.perf_counter()
    descriptors = generate_compact_action_descriptors(
        tam=item.state.tam,
        soft_mask=item.original_mask,
        sample_index=int(item.archive_index),
        sample_id=str(item.sample_id),
        true_label=int(item.true_label),
        protocol=str(protocol),
        clean_total=float(item.clean_total),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_action_budget=float(args.max_action_budget),
        max_actions=int(config.raw_pool),
        candidate_batch_size=int(args.candidate_batch_size),
        candidate_device=str(args.candidate_device),
        diagnostics=diagnostics,
    )
    descriptors = [d for d in descriptors if _identity(d) not in item.used and 0 < _action_dummy_count(d) <= remaining_dummy]
    selected = _prefilter_actions(descriptors, config=config, clean_total=item.clean_total, args=args, rng=item.rng)
    selected = [d for d in selected if _identity(d) not in item.used and 0 < _action_dummy_count(d) <= remaining_dummy]
    actions = []
    materialize_batch = max(1, int(args.materialization_batch_size))
    for offset in range(0, len(selected), materialize_batch):
        actions.extend(
            materialize_candidate_descriptors(
                selected[offset : offset + materialize_batch],
                tam=item.state.tam,
                clean_total=float(item.clean_total),
                protocol=str(protocol),
                max_action_budget=float(args.max_action_budget),
                max_local_rate_peak=int(args.max_local_rate_peak),
            )
        )
    _timing_add(args, "candidate_generation_time_sec", time.perf_counter() - start)
    _timing_add(args, "descriptor_count", float(len(descriptors)))
    _timing_add(args, "action_objects_built", float(len(actions)))
    if not actions:
        item.finished = True
        item.stop_reason = "candidate_pool_exhausted"
        return None
    return CandidatePack(item=item, actions=actions)


def _evaluate_packs(
    packs: list[CandidatePack],
    *,
    attacker,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    if not packs:
        return
    rows: list[tuple[int, int]] = []
    for pack_index, pack in enumerate(packs):
        for action_index, action in enumerate(pack.actions):
            rows.append((pack_index, action_index))
    candidate_device = _candidate_device(device, args)
    chunk_size = max(1, int(args.candidate_batch_size))
    probs_chunks: list[np.ndarray] = []
    start_total = time.perf_counter()
    max_chunk = 0
    if candidate_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(candidate_device)
    for start in range(0, len(rows), chunk_size):
        end = min(start + chunk_size, len(rows))
        chunk_rows = rows[start:end]
        max_chunk = max(max_chunk, len(chunk_rows))
        build_start = time.perf_counter()
        base_np = np.stack([np.asarray(packs[pack_idx].item.state.tam, dtype=np.float32) for pack_idx, _action_idx in chunk_rows], axis=0)
        counts_np = np.stack([np.asarray(packs[pack_idx].actions[action_idx].counts, dtype=np.float32) for pack_idx, action_idx in chunk_rows], axis=0)
        base_t = torch.as_tensor(base_np, dtype=torch.float32, device=candidate_device)
        counts_t = torch.as_tensor(counts_np, dtype=torch.float32, device=candidate_device)
        candidate_tams = base_t + counts_t
        _timing_add(args, "candidate_tam_gpu_build_time_sec", time.perf_counter() - build_start)
        forward_start = time.perf_counter()
        probs_t = _predict_probabilities_tensor(attacker, candidate_tams, batch_size=chunk_size)
        if candidate_device.type == "cuda":
            torch.cuda.synchronize(candidate_device)
        _timing_add(args, "rf_forward_time_sec", time.perf_counter() - forward_start)
        probs_chunks.append(probs_t.detach().cpu().numpy().astype(np.float32))
        del base_np, counts_np, base_t, counts_t, candidate_tams, probs_t
    if candidate_device.type == "cuda":
        _timing_max(args, "candidate_gpu_peak_allocated_mb", float(torch.cuda.max_memory_allocated(candidate_device) / (1024.0 * 1024.0)))
    _timing_add(args, "candidate_gpu_tam_eval_time_sec", time.perf_counter() - start_total)
    _timing_add(args, "candidate_gpu_tam_eval_count", float(len(rows)))
    _timing_add(args, "candidate_gpu_tam_eval_batches", float(max(1, int(np.ceil(len(rows) / max(chunk_size, 1))))))
    _timing_max(args, "candidate_gpu_tam_max_chunk", float(max_chunk))
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


def _apply_pack_choice(pack: CandidatePack, *, attacker, device: torch.device, args: argparse.Namespace) -> None:
    item = pack.item
    gains = np.asarray(pack.gains if pack.gains is not None else np.zeros(0, dtype=np.float32), dtype=np.float32)
    scores = _selection_scores(gains, pack.actions, item.clean_total, "absolute")
    best_idx = int(np.argmax(scores)) if scores.size else -1
    best_gain = float(gains[best_idx]) if best_idx >= 0 else 0.0
    item.rf_eval_count += len(pack.actions)
    item.steps += 1
    if best_idx < 0 or best_gain <= 0.0:
        item.finished = True
        item.stop_reason = "no_positive_single"
        return
    action = pack.actions[best_idx]
    selected_counts = np.asarray(item.state.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32)
    trace, tam, stats = _render_dummy(base_trace=item.state.trace, counts=selected_counts, args=args)
    prob = _predict_one(attacker, tam, device=device, args=args)
    for chosen in [action]:
        item.used.add(_identity(chosen))
    item.state = SimpleNamespace(
        trace=trace,
        tam=tam,
        prob=prob,
        dummy_counts=selected_counts.astype(np.int32),
        dummy_bandwidth=float(stats["raw_bandwidth"]),
        avg_delay=float(item.state.avg_delay),
        p95_delay=float(item.state.p95_delay),
        max_delay=int(item.state.max_delay),
        delay_values=tuple(item.state.delay_values),
        outgoing_delay_values=tuple(item.state.outgoing_delay_values),
        incoming_delay_values=tuple(item.state.incoming_delay_values),
        selected_actions=list(item.state.selected_actions) + [action],
    )
    item.accepted_count += 1
    if _margin(item.state.prob, item.original_prob) <= float(args.margin_target):
        item.finished = True
        item.stop_reason = "target_reached"
    elif int(np.asarray(item.state.dummy_counts).sum()) >= int(np.floor(float(item.clean_total) * float(args.fixed_budget) + 1e-9)):
        item.finished = True
        item.stop_reason = "bandwidth_10pct_reached"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = _run_dir(args)
    archive_rows = _archive_selection_rows(args)
    payload = _load_archive_rows(args.archive, archive_rows)
    tam = np.asarray(payload["tam"], dtype=np.float32)
    mask = np.asarray(payload["mask"], dtype=np.float32)
    prob = np.asarray(payload["pred_prob"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    sample_ids = np.asarray(payload.get("sample_ids", [f"sample_{i}" for i in range(len(tam))]))
    source_indices = np.asarray(payload.get("source_indices", archive_rows), dtype=np.int64)
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, args)
    checkpoint = str(args.checkpoint) or _default_checkpoint(str(args.attacker) if hasattr(args, "attacker") else "rf")
    device = resolve_device(str(args.device))
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker="rf",
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture="project",
        df_tam_adapter="signed_balance",
    )
    config = _method_config(str(args.method))
    timing: dict[str, float] = {}
    setattr(args, "timing_accumulator", timing)
    active: list[ActiveProbeState] = []
    pending = list(range(len(tam)))
    completed: list[ActiveProbeState] = []
    stable_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(config.name)))
    run_start = time.perf_counter()
    while pending or active:
        while pending and len(active) < int(args.active_states):
            local_index = pending.pop(0)
            state = _initial_state(np.asarray(raw_rows[local_index], dtype=np.float32), np.asarray(tam[local_index], dtype=np.float32), np.asarray(prob[local_index], dtype=np.float32))
            item = ActiveProbeState(
                local_index=int(local_index),
                archive_index=int(archive_rows[local_index]),
                sample_id=str(sample_ids[local_index]),
                true_label=int(labels[local_index]),
                raw_trace=np.asarray(raw_rows[local_index], dtype=np.float32),
                original_mask=np.asarray(mask[local_index], dtype=np.float32),
                original_prob=np.asarray(prob[local_index], dtype=np.float32),
                clean_total=max(float(np.asarray(tam[local_index], dtype=np.float32).sum()), 1.0),
                state=state,
                rng=random.Random(int(args.seed) + int(archive_rows[local_index]) * 1009 + stable_seed),
            )
            active.append(item)
        packs = [
            pack
            for pack in (
                _build_pack(item, config=config, args=args, protocol=str(args.protocol))
                for item in list(active)
                if not item.finished
            )
            if pack is not None
        ]
        _evaluate_packs(packs, attacker=attacker, device=device, args=args)
        for pack in packs:
            _apply_pack_choice(pack, attacker=attacker, device=device, args=args)
        still_active = []
        for item in active:
            if item.finished:
                completed.append(item)
            else:
                still_active.append(item)
        active = still_active
        if bool(args.progress):
            print(f"[active-probe] completed={len(completed)}/{len(tam)} active={len(active)} pending={len(pending)}", flush=True)
    wall = float(time.perf_counter() - run_start)
    rows = []
    for item in completed:
        clean = max(float(item.clean_total), 1.0)
        dummy = int(np.asarray(item.state.dummy_counts, dtype=np.int32).sum())
        rows.append(
            {
                "archive_index": int(item.archive_index),
                "sample_id": str(item.sample_id),
                "true_label": int(item.true_label),
                "original_pred": int(np.argmax(item.original_prob)),
                "final_pred": int(np.argmax(item.state.prob)),
                "accuracy": int(np.argmax(item.state.prob) == int(item.true_label)),
                "flip": int(np.argmax(item.state.prob) != int(np.argmax(item.original_prob))),
                "stop_reason": str(item.stop_reason),
                "steps": int(item.steps),
                "accepted_action_count": int(item.accepted_count),
                "rf_eval_count": int(item.rf_eval_count),
                "actual_dummy_bandwidth": float(dummy / clean),
                "dummy_packet_count": int(dummy),
            }
        )
    _write_csv(output_dir / "active_state_probe_samples.csv", rows)
    summary = {
        "run_dir": str(output_dir),
        "samples": int(len(rows)),
        "active_states": int(args.active_states),
        "candidate_batch_size": int(args.candidate_batch_size),
        "method": str(args.method),
        "wall_time_sec": float(wall),
        "traces_per_hour": float(len(rows) / max(wall / 3600.0, 1e-9)),
        "defended_rf_accuracy": float(np.mean([row["accuracy"] for row in rows])) if rows else 0.0,
        "flip_rate": float(np.mean([row["flip"] for row in rows])) if rows else 0.0,
        "mean_actual_dummy_bandwidth": float(np.mean([row["actual_dummy_bandwidth"] for row in rows])) if rows else 0.0,
        "p95_actual_dummy_bandwidth": float(np.percentile(np.asarray([row["actual_dummy_bandwidth"] for row in rows], dtype=np.float64), 95)) if rows else 0.0,
        "timing": {key: float(value) for key, value in sorted(timing.items())},
    }
    (output_dir / "active_state_probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
