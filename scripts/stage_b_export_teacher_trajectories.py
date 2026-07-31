# -*- coding: utf-8 -*-
"""Export Stage B2-E Teacher/Oracle trajectories for policy learning."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.policy_data import encode_actions, encode_state_features
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_b2e_diverse_search import (
    DEFAULT_ARCHIVE,
    _default_checkpoint,
    _load_raw_rows,
    _method_config,
    _parse_csv_floats,
    _run_controller,
    _runtime_args,
    _sample_row as _controller_sample_row,
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
    parser.add_argument("--store_next_state", action="store_true")
    parser.add_argument("--storage_mode", choices=["dense", "sparse"], default="dense")
    parser.add_argument("--profile_candidate_generation", action="store_true")
    parser.add_argument("--compact_candidate_generation", action="store_true")
    parser.add_argument("--deferred_materialize_oversample", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=0)
    parser.add_argument("--materialization_batch_size", type=int, default=64)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_score_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="renderer")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b2e_teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()) and not bool(args.resume):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    (target / "records").mkdir(parents=True, exist_ok=True)
    return target


def _budget_schedule(n: int, budgets: list[float], args: argparse.Namespace) -> np.ndarray:
    if str(args.budget_mode) == "fixed":
        return np.full(int(n), float(args.fixed_budget), dtype=np.float32)
    values = np.asarray(budgets, dtype=np.float32)
    if values.size == 0:
        values = np.asarray([float(args.fixed_budget)], dtype=np.float32)
    schedule = np.resize(values, int(n)).astype(np.float32)
    if str(args.budget_mode) == "balanced":
        rng = np.random.default_rng(int(args.seed) + 4099)
        rng.shuffle(schedule)
    return schedule


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
                requested = np.concatenate(
                    [np.asarray(splits[f"{name}_indices"], dtype=np.int64) for name in ("train", "val", "test")],
                    axis=0,
                )
            else:
                requested = np.asarray(splits[f"{args.split_name}_indices"], dtype=np.int64)
        position = {int(source): row for row, source in enumerate(archive_source.tolist())}
        missing = [int(source) for source in requested.tolist() if int(source) not in position]
        if missing:
            raise ValueError(f"Archive is missing {len(missing)} requested split source indices; first missing={missing[:5]}")
        rows = np.asarray([position[int(source)] for source in requested.tolist()], dtype=np.int64)
    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if num_shards > 1:
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError(f"--shard_id must be in [0, {num_shards - 1}], got {shard_id}.")
        rows = np.array_split(rows, num_shards)[shard_id]
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


class ManifestWriter:
    def __init__(self, path: Path, *, append: bool = False) -> None:
        self.path = Path(path)
        self.writer: csv.DictWriter | None = None
        self.fieldnames: list[str] = []
        if append and self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.fieldnames = list(reader.fieldnames or [])
            self.handle = self.path.open("a", newline="", encoding="utf-8")
            if self.fieldnames:
                self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        else:
            self.handle = self.path.open("w", newline="", encoding="utf-8")

    def write(self, row: dict[str, Any]) -> None:
        if self.writer is None:
            self.fieldnames = list(row.keys())
            self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
            self.writer.writeheader()
        self.writer.writerow({key: row.get(key, "") for key in self.fieldnames})
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _completed_archive_indices(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {int(float(row["archive_index"])) for row in reader if str(row.get("archive_index", "")).strip()}


def _max_existing_record_id(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return -1
    max_id = -1
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            value = str(row.get("record_id", "")).strip()
            if value:
                max_id = max(max_id, int(float(value)))
    return max_id


def _prune_teacher_records_to_completed(path: Path, completed: set[int]) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        return 0
    kept = [row for row in rows if int(float(row.get("sample_index", -1))) in completed]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    return len(rows) - len(kept)


def _write_selected_samples(
    path: Path,
    *,
    archive_rows: np.ndarray,
    sample_ids: np.ndarray,
    source_indices: np.ndarray,
    schedule: np.ndarray,
) -> None:
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["local_index", "archive_index", "sample_id", "source_index", "budget"])
        writer.writeheader()
        for local_index, archive_index in enumerate(np.asarray(archive_rows, dtype=np.int64).tolist()):
            writer.writerow(
                {
                    "local_index": int(local_index),
                    "archive_index": int(archive_index),
                    "sample_id": str(sample_ids[local_index]),
                    "source_index": int(source_indices[local_index]),
                    "budget": float(schedule[local_index]),
                }
            )


def _save_teacher_record(
    *,
    output_dir: Path,
    record_id: int,
    sample_index: int,
    sample_id: str,
    source_index: int,
    budget: float,
    payload: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pre_state = payload["pre_state"]
    next_state = payload["next_state"]
    actions = list(payload.get("candidate_actions", []))
    clean_total = max(float(payload.get("clean_total", 1.0)), 1.0)
    width = int(args.rf_num_slots)
    action_features, action_counts = encode_actions(actions, clean_total=clean_total, width=width)
    gains = np.asarray(payload.get("candidate_gains", np.zeros(len(actions), dtype=np.float32)), dtype=np.float32)
    scores = np.asarray(payload.get("candidate_scores", np.zeros(len(actions), dtype=np.float32)), dtype=np.float32)
    selected_index = int(payload.get("selected_index", -1))
    stop_target = int(str(payload.get("event_type", "")) == "stop" or selected_index < 0)
    original_prob = np.asarray(payload["original_prob"], dtype=np.float32)
    original_pred = int(np.argmax(original_prob))
    dummy_used = int(np.asarray(pre_state.dummy_counts, dtype=np.int32).sum())
    remaining_bandwidth = max(float(budget) - float(dummy_used / clean_total), 0.0)
    state_features = encode_state_features(
        current_prob=np.asarray(pre_state.prob, dtype=np.float32),
        original_pred=original_pred,
        remaining_bandwidth=remaining_bandwidth,
        remaining_delay=max(0.0, float(args.max_delay) - float(pre_state.max_delay)),
        round_index=int(payload.get("round_index", 0)),
        rounds=int(args.rounds),
        dummy_bandwidth_used=float(dummy_used / clean_total),
        avg_delay=float(pre_state.avg_delay),
        p95_delay=float(pre_state.p95_delay),
        max_delay=float(pre_state.max_delay),
        max_delay_budget=float(args.max_delay),
    )
    mask = np.asarray(payload.get("mask"), dtype=np.float32)
    current_tam = np.asarray(pre_state.tam, dtype=np.float32)
    state_tensor = np.concatenate([current_tam, mask], axis=0).astype(np.float16)
    record_path = output_dir / "records" / f"record_{record_id:08d}.npz"
    arrays: dict[str, np.ndarray] = {
        "state_tensor": state_tensor,
        "current_tam": current_tam.astype(np.float16),
        "keypoint_map": mask.astype(np.float16),
        "state_features": state_features.astype(np.float32),
        "current_prob": np.asarray(pre_state.prob, dtype=np.float32),
        "original_prob": original_prob.astype(np.float32),
        "action_features": action_features.astype(np.float32),
        "candidate_gains": gains.astype(np.float32),
        "candidate_scores": scores.astype(np.float32),
        "candidate_mask": np.ones(len(actions), dtype=np.bool_),
        "selected_index": np.asarray(selected_index, dtype=np.int64),
        "stop_target": np.asarray(stop_target, dtype=np.float32),
        "budget": np.asarray(float(budget), dtype=np.float32),
        "sample_index": np.asarray(int(sample_index), dtype=np.int64),
        "source_index": np.asarray(int(source_index), dtype=np.int64),
        "true_label_for_eval_only": np.asarray(int(payload.get("label", -1)), dtype=np.int64),
    }
    if str(args.storage_mode) == "dense":
        arrays["action_counts"] = action_counts.astype(np.int16)
    else:
        sparse_action, sparse_direction, sparse_bin, sparse_count = [], [], [], []
        for action_index, counts in enumerate(action_counts):
            nz = np.argwhere(np.asarray(counts, dtype=np.int16) != 0)
            for direction, bin_index in nz.tolist():
                sparse_action.append(int(action_index))
                sparse_direction.append(int(direction))
                sparse_bin.append(int(bin_index))
                sparse_count.append(int(counts[direction, bin_index]))
        arrays["action_count_shape"] = np.asarray([len(actions), 2, width], dtype=np.int64)
        arrays["action_sparse_action"] = np.asarray(sparse_action, dtype=np.int32)
        arrays["action_sparse_direction"] = np.asarray(sparse_direction, dtype=np.int16)
        arrays["action_sparse_bin"] = np.asarray(sparse_bin, dtype=np.int32)
        arrays["action_sparse_count"] = np.asarray(sparse_count, dtype=np.int16)
    if bool(args.store_next_state):
        arrays["next_tam"] = np.asarray(next_state.tam, dtype=np.float16)
        arrays["next_prob"] = np.asarray(next_state.prob, dtype=np.float32)
    start = time.perf_counter()
    np.savez_compressed(record_path, **arrays)
    serialization_sec = float(time.perf_counter() - start)
    best_gain = float(np.max(gains)) if gains.size else 0.0
    return {
        "record_id": int(record_id),
        "record_path": str(record_path),
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "source_index": int(source_index),
        "budget": float(budget),
        "round_index": int(payload.get("round_index", -1)),
        "step_index": int(payload.get("step_index", -1)),
        "event_type": str(payload.get("event_type", "")),
        "stop_reason": str(payload.get("stop_reason", "")),
        "candidate_count": int(len(actions)),
        "storage_mode": str(args.storage_mode),
        "sparse_nonzero_count": int(len(arrays.get("action_sparse_count", []))),
        "record_bytes": int(record_path.stat().st_size),
        "serialization_sec": float(serialization_sec),
        "selected_index": int(selected_index),
        "stop_target": int(stop_target),
        "best_gain": float(best_gain),
        "selected_gain": float(payload.get("selected_gain", 0.0)),
        "dummy_used": int(dummy_used),
        "remaining_bandwidth": float(remaining_bandwidth),
        "final_record_for_sample": 0,
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    run_args = _runtime_args(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    config = _method_config(args.method)
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
    starting_record_count = _max_existing_record_id(teacher_records_path) + 1 if bool(args.resume) else 0
    record_count = int(starting_record_count)
    skipped_samples = 0
    new_samples = 0
    sample_rows: list[dict[str, Any]] = []
    run_start = time.perf_counter()
    try:
        for local_index, archive_index in enumerate(archive_rows.tolist()):
            if int(archive_index) in completed_before:
                skipped_samples += 1
                if args.progress:
                    print(
                        f"[teacher-export] skip completed {local_index + 1}/{len(archive_rows)} "
                        f"archive_index={archive_index} sample={sample_ids[local_index]}",
                        flush=True,
                    )
                continue
            budget = float(schedule[local_index])
            sample_record_start = record_count
            if args.progress:
                print(
                    f"[teacher-export] {local_index + 1}/{len(archive_rows)} archive_index={archive_index} "
                    f"sample={sample_ids[local_index]} B={budget:g}",
                    flush=True,
                )
            timing: dict[str, float] = {}
            setattr(args, "timing_accumulator", timing)
            teacher_stats: dict[str, Any] = {
                "action_records": 0,
                "stop_records": 0,
                "candidate_total_count": 0,
                "candidate_positive_count": 0,
                "best_gain_values": [],
                "selected_gain_values": [],
                "record_bytes": 0,
                "serialization_time_sec": 0.0,
            }

            def observe(payload: dict[str, Any]) -> None:
                nonlocal record_count
                row = _save_teacher_record(
                    output_dir=output_dir,
                    record_id=int(record_count),
                    sample_index=int(archive_index),
                    sample_id=str(sample_ids[local_index]),
                    source_index=int(source_indices[local_index]),
                    budget=float(budget),
                    payload=payload,
                    args=args,
                )
                manifest_writer.write(row)
                event_type = str(row.get("event_type", ""))
                if event_type == "action":
                    teacher_stats["action_records"] = int(teacher_stats["action_records"]) + 1
                if event_type == "stop":
                    teacher_stats["stop_records"] = int(teacher_stats["stop_records"]) + 1
                candidate_gains = np.asarray(payload.get("candidate_gains", np.zeros(0, dtype=np.float32)), dtype=np.float32)
                teacher_stats["candidate_total_count"] = int(teacher_stats["candidate_total_count"]) + int(candidate_gains.size)
                teacher_stats["candidate_positive_count"] = int(teacher_stats["candidate_positive_count"]) + int(np.sum(candidate_gains > 0.0))
                if candidate_gains.size:
                    teacher_stats["best_gain_values"].append(float(np.max(candidate_gains)))
                teacher_stats["selected_gain_values"].append(float(row.get("selected_gain", 0.0)))
                teacher_stats["record_bytes"] = int(teacher_stats["record_bytes"]) + int(row.get("record_bytes", 0))
                teacher_stats["serialization_time_sec"] = float(teacher_stats["serialization_time_sec"]) + float(row.get("serialization_sec", 0.0))
                record_count += 1

            start_time = time.perf_counter()
            state, aggregate, _funnel = _run_controller(
                config=config,
                protocol=str(args.protocol),
                budget=float(budget),
                raw_trace=np.asarray(raw_rows[local_index], dtype=np.float32),
                original_tam=np.asarray(tam[local_index], dtype=np.float32),
                original_mask=np.asarray(mask[local_index], dtype=np.float32),
                original_prob=np.asarray(prob[local_index], dtype=np.float32),
                label=int(labels[local_index]),
                sample_index=int(archive_index),
                sample_id=str(sample_ids[local_index]),
                attacker=attacker,
                device=device,
                args=args,
                teacher_observer=observe,
            )
            runtime_sec = float(time.perf_counter() - start_time)
            clean_total = max(float(np.asarray(tam[local_index], dtype=np.float32).sum()), 1.0)
            sample_row = _controller_sample_row(
                sample_index=int(archive_index),
                sample_id=str(sample_ids[local_index]),
                protocol=str(args.protocol),
                config=config,
                budget=float(budget),
                margin_target=float(args.margin_target),
                original_prob=np.asarray(prob[local_index], dtype=np.float32),
                state=state,
                label=int(labels[local_index]),
                clean_total=float(clean_total),
                runtime=float(runtime_sec),
                aggregate=aggregate,
            )
            candidate_total = int(teacher_stats["candidate_total_count"])
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
            timed_accounted = float(sum(float(timing.get(key, 0.0)) for key in accounted_timing_keys)) + float(teacher_stats["serialization_time_sec"])
            sample_row.update(
                {
                    "archive_index": int(archive_index),
                    "source_index": int(source_indices[local_index]),
                    "budget": float(budget),
                    "records": int(record_count - sample_record_start),
                    "action_records": int(teacher_stats["action_records"]),
                    "stop_records": int(teacher_stats["stop_records"]),
                    "candidate_total_count": int(candidate_total),
                    "candidate_positive_count": int(teacher_stats["candidate_positive_count"]),
                    "candidate_positive_gain_rate": float(int(teacher_stats["candidate_positive_count"]) / max(candidate_total, 1)),
                    "mean_best_gain_per_state": float(np.mean(teacher_stats["best_gain_values"])) if teacher_stats["best_gain_values"] else 0.0,
                    "max_best_gain_per_state": float(np.max(teacher_stats["best_gain_values"])) if teacher_stats["best_gain_values"] else 0.0,
                    "mean_selected_gain": float(np.mean(teacher_stats["selected_gain_values"])) if teacher_stats["selected_gain_values"] else 0.0,
                    "record_bytes_per_trace": int(teacher_stats["record_bytes"]),
                    "serialization_time_sec": float(teacher_stats["serialization_time_sec"]),
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
            sample_writer.write(sample_row)
            sample_rows.append(sample_row)
            new_samples += 1
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
        "max_delay": int(args.max_delay),
        "margin_target": float(args.margin_target),
        "teacher_records": str(teacher_records_path),
        "sample_summary": str(sample_summary_path),
        "selected_samples": str(selected_samples_path),
        "records_dir": str(output_dir / "records"),
        "runtime_sec": float(time.perf_counter() - run_start),
        "store_next_state": bool(args.store_next_state),
        "storage_mode": str(args.storage_mode),
        "resume": bool(args.resume),
    }
    (output_dir / "teacher_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
