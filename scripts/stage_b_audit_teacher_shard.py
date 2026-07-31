# -*- coding: utf-8 -*-
"""Audit one Stage B Teacher shard export."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


TIME_KEYS = [
    "candidate_generation_time_sec",
    "candidate_prefilter_time_sec",
    "candidate_pair_generation_time_sec",
    "candidate_used_budget_filter_time_sec",
    "candidate_selected_used_budget_filter_time_sec",
    "compact_descriptor_used_budget_filter_time_sec",
    "candidate_window_extract_time_sec",
    "structural_anchor_extract_time_sec",
    "anchor_grid_build_time_sec",
    "action_object_build_time_sec",
    "legality_filter_time_sec",
    "deduplicate_time_sec",
    "compact_deduplicate_time_sec",
    "pair_action_build_time_sec",
    "pair_deduplicate_time_sec",
    "score_hint_sort_time_sec",
    "diverse_limit_time_sec",
    "generate_expanded_actions_total_time_sec",
    "compact_diverse_limit_time_sec",
    "compact_descriptor_generation_total_time_sec",
    "deferred_materialization_time_sec",
    "candidate_tam_gpu_build_time_sec",
    "candidate_gpu_tam_eval_time_sec",
    "renderer_time_sec",
    "tam_rebuild_time_sec",
    "rf_forward_time_sec",
    "serialization_time_sec",
    "keypoint_refresh_time_sec",
    "delay_trace_time_sec",
    "queue_or_wait_time_sec",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_shards_for_projection", type=int, default=256)
    parser.add_argument("--budget_bound", type=float, default=0.10)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value in {"", None}:
        return float(default)
    return float(value)


def _int(row: dict[str, Any], key: str, default: int = 0) -> int:
    value = row.get(key, default)
    if value in {"", None}:
        return int(default)
    return int(float(value))


def _stats(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if not arr.size:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def _sum_file_bytes(path: Path) -> int:
    if path.is_file():
        return int(path.stat().st_size)
    if not path.exists():
        return 0
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "teacher_run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    selected_path = Path(manifest.get("selected_samples", run_dir / "teacher_selected_samples.csv"))
    summary_path = Path(manifest.get("sample_summary", run_dir / "teacher_sample_summary.csv"))
    records_path = Path(manifest.get("teacher_records", run_dir / "teacher_records.csv"))
    records_dir = Path(manifest.get("records_dir", run_dir / "records"))
    selected = _read_csv(selected_path)
    samples = _read_csv(summary_path)
    records = _read_csv(records_path)

    expected_ids = [str(row.get("sample_id", "")) for row in selected]
    completed_ids = [str(row.get("sample_id", "")) for row in samples]
    expected_archive = {int(float(row["archive_index"])) for row in selected if str(row.get("archive_index", "")).strip()}
    completed_archive = [int(float(row["archive_index"])) for row in samples if str(row.get("archive_index", "")).strip()]
    completed_archive_set = set(completed_archive)
    duplicate_sample_ids = {key: value for key, value in Counter(completed_ids).items() if key and value > 1}
    duplicate_archive_indices = {str(key): value for key, value in Counter(completed_archive).items() if value > 1}
    missing_archive = sorted(expected_archive - completed_archive_set)
    missing_ids = [
        str(row.get("sample_id", ""))
        for row in selected
        if str(row.get("archive_index", "")).strip() and int(float(row["archive_index"])) in set(missing_archive)
    ]
    failed_rows = [row for row in samples if str(row.get("stop_reason", "")).lower() == "error"]
    runtimes = [_float(row, "runtime_sec") for row in samples]
    bw = [_float(row, "actual_dummy_bandwidth") for row in samples]
    bw_violations = [
        {
            "sample_id": str(row.get("sample_id", "")),
            "archive_index": _int(row, "archive_index", -1),
            "actual_dummy_bandwidth": _float(row, "actual_dummy_bandwidth"),
        }
        for row in samples
        if _float(row, "actual_dummy_bandwidth") > float(args.budget_bound) + 1e-8
    ]
    timing_totals = {key: float(sum(_float(row, key) for row in samples)) for key in TIME_KEYS}
    timing_means = {key.replace("_sec", "_mean_sec"): float(np.mean([_float(row, key) for row in samples])) if samples else 0.0 for key in TIME_KEYS}
    stop_reasons = Counter(str(row.get("stop_reason", "")) or "unknown" for row in samples)
    records_per_trace = [_float(row, "records") for row in samples]
    candidates_per_trace = [_float(row, "candidate_total_count") for row in samples]
    candidates_per_state = [
        _float(row, "candidate_total_count") / max(_float(row, "records"), 1.0)
        for row in samples
    ]
    bytes_per_trace = [_float(row, "record_bytes_per_trace") for row in samples]
    clean_acc_values = [
        1.0 if _int(row, "original_pred", -1) == _int(row, "true_label", -2) else 0.0
        for row in samples
    ]
    shard_size = _sum_file_bytes(records_dir)
    audit = {
        "run_dir": str(run_dir),
        "expected_samples": int(len(selected)),
        "completed_samples": int(len(samples)),
        "skipped_samples": int(manifest.get("skipped_samples", 0)),
        "failed_samples": int(len(failed_rows)),
        "duplicate_sample_ids": int(sum(value - 1 for value in duplicate_sample_ids.values())),
        "duplicate_archive_indices": int(sum(value - 1 for value in duplicate_archive_indices.values())),
        "missing_sample_ids": int(len(missing_ids)),
        "missing_archive_indices": int(len(missing_archive)),
        "missing_sample_id_examples": missing_ids[:20],
        "duplicate_sample_id_examples": dict(list(duplicate_sample_ids.items())[:20]),
        "duplicate_archive_index_examples": dict(list(duplicate_archive_indices.items())[:20]),
        "budget_bound": float(args.budget_bound),
        "bandwidth_violation_count": int(len(bw_violations)),
        "bandwidth_violation_examples": bw_violations[:20],
        "total_wall_time_sec": float(manifest.get("runtime_sec", sum(runtimes))),
        "runtime_stats_sec": _stats(runtimes),
        "timing_totals_sec": timing_totals,
        "timing_means_sec": timing_means,
        "states_per_trace": _stats(records_per_trace),
        "candidates_per_state": _stats(candidates_per_state),
        "candidates_per_trace": _stats(candidates_per_trace),
        "compact_materialization": {
            "descriptor_count_total": float(sum(_float(row, "compact_descriptor_count") for row in samples)),
            "descriptor_count_per_trace": float(np.mean([_float(row, "compact_descriptor_count") for row in samples])) if samples else 0.0,
            "action_objects_built_total": float(sum(_float(row, "deferred_action_objects_built") for row in samples)),
            "action_objects_built_per_trace": float(np.mean([_float(row, "deferred_action_objects_built") for row in samples])) if samples else 0.0,
            "dense_counts_built_total": float(sum(_float(row, "deferred_dense_counts_built") for row in samples)),
            "dense_counts_built_per_trace": float(np.mean([_float(row, "deferred_dense_counts_built") for row in samples])) if samples else 0.0,
        },
        "gpu_candidate_eval": {
            "candidate_gpu_tam_eval_count_total": float(sum(_float(row, "candidate_gpu_tam_eval_count") for row in samples)),
            "candidate_gpu_tam_eval_count_per_trace": float(np.mean([_float(row, "candidate_gpu_tam_eval_count") for row in samples])) if samples else 0.0,
            "candidate_gpu_peak_allocated_mb_max": float(max([_float(row, "candidate_gpu_peak_allocated_mb") for row in samples], default=0.0)),
            "candidate_gpu_peak_allocated_mb_mean": float(np.mean([_float(row, "candidate_gpu_peak_allocated_mb") for row in samples])) if samples else 0.0,
        },
        "action_records": int(sum(_int(row, "action_records") for row in samples)),
        "stop_records": int(sum(_int(row, "stop_records") for row in samples)),
        "bytes_per_trace": _stats(bytes_per_trace),
        "total_shard_size_bytes": int(shard_size),
        "projected_full_records": float(len(records) * int(args.num_shards_for_projection)),
        "projected_full_storage_bytes": float(shard_size * int(args.num_shards_for_projection)),
        "shard_clean_rf_accuracy": float(np.mean(clean_acc_values)) if clean_acc_values else 0.0,
        "defended_rf_accuracy": float(np.mean([_float(row, "accuracy") for row in samples])) if samples else 0.0,
        "flip_rate": float(np.mean([_float(row, "flip") for row in samples])) if samples else 0.0,
        "margin_success_rate": float(np.mean([_float(row, "target_margin_success") for row in samples])) if samples else 0.0,
        "actual_bandwidth": {
            "mean": float(np.mean(bw)) if bw else 0.0,
            "p95": float(np.percentile(np.asarray(bw, dtype=np.float64), 95)) if bw else 0.0,
            "max": float(max(bw)) if bw else 0.0,
        },
        "delay_bins": {
            "mean_average_delay": float(np.mean([_float(row, "average_delay_bins") for row in samples])) if samples else 0.0,
            "mean_p95_delay": float(np.mean([_float(row, "p95_delay_bins") for row in samples])) if samples else 0.0,
            "p95_of_p95_delay": float(np.percentile(np.asarray([_float(row, "p95_delay_bins") for row in samples], dtype=np.float64), 95)) if samples else 0.0,
            "max_delay": float(max([_float(row, "maximum_delay_bins") for row in samples], default=0.0)),
        },
        "mean_action_rounds": float(np.mean([_float(row, "candidate_step_count") for row in samples])) if samples else 0.0,
        "candidate_rf_evaluations_per_sample": float(np.mean([_float(row, "rf_eval_count") for row in samples])) if samples else 0.0,
        "candidate_positive_gain_rate": float(np.mean([_float(row, "candidate_positive_gain_rate") for row in samples])) if samples else 0.0,
        "mean_best_gain_per_state": float(np.mean([_float(row, "mean_best_gain_per_state") for row in samples])) if samples else 0.0,
        "stop_reason_distribution": dict(sorted(stop_reasons.items())),
    }
    (run_dir / "teacher_shard_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    _write_csv(run_dir / "teacher_shard_audit_summary.csv", [audit | {"stop_reason_distribution": json.dumps(audit["stop_reason_distribution"], sort_keys=True)}])
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
