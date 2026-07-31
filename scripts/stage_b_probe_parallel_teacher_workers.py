# -*- coding: utf-8 -*-
"""Probe Teacher-export parallel workers while monitoring GPU memory/utilization."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz")
    parser.add_argument("--split_file", default="results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz")
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--probe_name", default="")
    parser.add_argument("--worker_counts", default="1,2")
    parser.add_argument("--start_shard", type=int, default=20)
    parser.add_argument("--num_shards", type=int, default=256)
    parser.add_argument("--samples_per_worker", type=int, default=4)
    parser.add_argument("--target_gpu_mem_gb", type=float, default=8.0)
    parser.add_argument("--monitor_interval", type=float, default=1.0)
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--compact_candidate_generation", action="store_true")
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_score_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="gpu_tam")
    parser.add_argument("--storage_mode", choices=["dense", "sparse"], default="sparse")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--timeout_sec", type=float, default=1800.0)
    parser.add_argument("--continue_after_target", action="store_true")
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _query_gpu() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {"gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None, "error": str(exc)}
    if not out:
        return {"gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None, "error": "empty nvidia-smi output"}
    parts = [part.strip() for part in out[0].split(",")]
    return {
        "gpu_util_pct": float(parts[0]),
        "gpu_mem_used_mb": float(parts[1]),
        "gpu_mem_total_mb": float(parts[2]),
        "error": "",
    }


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _worker_command(args: argparse.Namespace, *, run_name: str, shard_id: int) -> list[str]:
    cmd = [
        str(args.python_exe),
        "scripts/stage_b_export_teacher_trajectories.py",
        "--archive",
        str(args.archive),
        "--split_file",
        str(args.split_file),
        "--split_name",
        str(args.split_name),
        "--num_shards",
        str(int(args.num_shards)),
        "--shard_id",
        str(int(shard_id)),
        "--max_samples",
        str(int(args.samples_per_worker)),
        "--budget_mode",
        "fixed",
        "--fixed_budget",
        str(float(args.budget)),
        "--method",
        str(args.method),
        "--protocol",
        str(args.protocol),
        "--max_delay",
        str(int(args.max_delay)),
        "--rounds",
        str(int(args.rounds)),
        "--max_dummy_steps",
        str(int(args.max_dummy_steps)),
        "--max_action_budget",
        str(float(args.max_action_budget)),
        "--storage_mode",
        str(args.storage_mode),
        "--device",
        str(args.device),
        "--candidate_batch_size",
        str(int(args.candidate_batch_size)),
        "--candidate_device",
        str(args.candidate_device),
        "--candidate_score_device",
        str(args.candidate_score_device),
        "--candidate_eval_device",
        str(args.candidate_eval_device),
        "--candidate_eval_mode",
        str(args.candidate_eval_mode),
        "--run_name",
        str(run_name),
    ]
    if bool(args.compact_candidate_generation):
        cmd.append("--compact_candidate_generation")
    return cmd


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    probe_name = args.probe_name or f"stage_b_parallel_worker_probe_{stamp}"
    probe_dir = Path(args.output_dir) / probe_name
    probe_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = probe_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    all_monitor_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    worker_counts = _parse_ints(args.worker_counts)
    target_mb = float(args.target_gpu_mem_gb) * 1024.0

    for worker_count in worker_counts:
        workers = []
        group_name = f"p{worker_count}"
        group_start = time.perf_counter()
        for worker_index in range(int(worker_count)):
            shard_id = int(args.start_shard) + int(sum(worker_counts[: worker_counts.index(worker_count)])) + worker_index
            run_name = f"{probe_name}_{group_name}_w{worker_index:02d}_shard{shard_id:03d}"
            stdout_path = logs_dir / f"{run_name}.stdout.log"
            stderr_path = logs_dir / f"{run_name}.stderr.log"
            stdout = stdout_path.open("w", encoding="utf-8")
            stderr = stderr_path.open("w", encoding="utf-8")
            process = subprocess.Popen(
                _worker_command(args, run_name=run_name, shard_id=shard_id),
                cwd=ROOT,
                stdout=stdout,
                stderr=stderr,
                text=True,
            )
            workers.append(
                {
                    "process": process,
                    "stdout": stdout,
                    "stderr": stderr,
                    "stdout_path": stdout_path,
                    "stderr_path": stderr_path,
                    "run_name": run_name,
                    "run_dir": Path(args.output_dir) / run_name,
                    "shard_id": int(shard_id),
                    "worker_index": int(worker_index),
                }
            )

        max_mem = 0.0
        util_values: list[float] = []
        timed_out = False
        while True:
            gpu = _query_gpu()
            now = time.perf_counter()
            mem = float(gpu.get("gpu_mem_used_mb") or 0.0)
            util = gpu.get("gpu_util_pct")
            max_mem = max(max_mem, mem)
            if util is not None:
                util_values.append(float(util))
            alive = [item for item in workers if item["process"].poll() is None]
            row = {
                "group": group_name,
                "worker_count": int(worker_count),
                "elapsed_sec": float(now - group_start),
                "alive_workers": int(len(alive)),
                **gpu,
            }
            all_monitor_rows.append(row)
            print(json.dumps(row), flush=True)
            if not alive:
                break
            if float(now - group_start) > float(args.timeout_sec):
                timed_out = True
                for item in alive:
                    item["process"].terminate()
                break
            time.sleep(max(0.2, float(args.monitor_interval)))

        for item in workers:
            item["process"].wait(timeout=60)
            item["stdout"].close()
            item["stderr"].close()

        group_wall = float(time.perf_counter() - group_start)
        completed_samples = 0
        records = 0
        failed_workers = 0
        run_dirs = []
        for item in workers:
            manifest = _load_json(Path(item["run_dir"]) / "teacher_run.json")
            completed_samples += int(manifest.get("new_samples", manifest.get("samples", 0)) or 0)
            records += int(manifest.get("new_records", manifest.get("records", 0)) or 0)
            failed_workers += int(item["process"].returncode != 0)
            run_dirs.append(str(item["run_dir"]))
        summary = {
            "group": group_name,
            "worker_count": int(worker_count),
            "group_wall_sec": float(group_wall),
            "completed_samples": int(completed_samples),
            "records": int(records),
            "failed_workers": int(failed_workers),
            "timed_out": int(timed_out),
            "traces_per_hour": float(completed_samples / max(group_wall, 1e-9) * 3600.0),
            "max_gpu_mem_mb": float(max_mem),
            "max_gpu_mem_gb": float(max_mem / 1024.0),
            "target_gpu_mem_gb": float(args.target_gpu_mem_gb),
            "mean_gpu_util_pct": float(sum(util_values) / max(len(util_values), 1)),
            "run_dirs": json.dumps(run_dirs),
        }
        summary_rows.append(summary)
        _write_rows(probe_dir / "parallel_probe_monitor.csv", all_monitor_rows)
        _write_rows(probe_dir / "parallel_probe_summary.csv", summary_rows)
        print(json.dumps(summary, indent=2), flush=True)
        if max_mem >= target_mb and not bool(args.continue_after_target):
            break

    manifest = {
        "probe_dir": str(probe_dir),
        "worker_counts": worker_counts,
        "samples_per_worker": int(args.samples_per_worker),
        "target_gpu_mem_gb": float(args.target_gpu_mem_gb),
        "summary_csv": str(probe_dir / "parallel_probe_summary.csv"),
        "monitor_csv": str(probe_dir / "parallel_probe_monitor.csv"),
    }
    (probe_dir / "parallel_probe_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
