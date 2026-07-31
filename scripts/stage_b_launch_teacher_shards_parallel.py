# -*- coding: utf-8 -*-
"""Launch Stage B Teacher export shards with bounded parallel workers."""

from __future__ import annotations

import argparse
import csv
import json
import os
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
    parser.add_argument("--split_name", choices=["train", "val", "test", "all", "archive"], default="train")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--launcher_name", default="")
    parser.add_argument("--run_prefix", default="")
    parser.add_argument("--python_exe", default=sys.executable)
    parser.add_argument("--gpu_ids", default="", help="Comma-separated GPU ids assigned round-robin to workers. Empty means inherit the parent CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--per_worker_threads", type=int, default=0, help="Set OMP/MKL/OpenBLAS/NUMEXPR/TORCH thread counts per worker when > 0.")
    parser.add_argument("--start_shard", type=int, default=0)
    parser.add_argument("--end_shard", type=int, default=256)
    parser.add_argument("--num_shards", type=int, default=256)
    parser.add_argument("--max_parallel_workers", type=int, default=4)
    parser.add_argument("--max_samples_per_shard", type=int, default=-1)
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--candidate_batch_size", type=int, default=4096)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_score_device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--candidate_eval_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="gpu_tam")
    parser.add_argument("--storage_mode", choices=["dense", "sparse"], default="sparse")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--monitor_interval", type=float, default=5.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    return parser.parse_args()


def _parse_gpu_ids(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _query_gpu() -> dict[str, Any]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        lines = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip().splitlines()
    except Exception as exc:  # pragma: no cover - hardware dependent
        return {"gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None, "gpu_details": "[]", "error": str(exc)}
    rows: list[dict[str, float | str]] = []
    for line in lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 4:
            continue
        rows.append(
            {
                "index": parts[0],
                "util_pct": float(parts[1]),
                "mem_used_mb": float(parts[2]),
                "mem_total_mb": float(parts[3]),
            }
        )
    if not rows:
        return {"gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None, "gpu_details": "[]", "error": "unparsed nvidia-smi output"}
    total_mem = sum(float(row["mem_used_mb"]) for row in rows)
    total_cap = sum(float(row["mem_total_mb"]) for row in rows)
    mean_util = sum(float(row["util_pct"]) for row in rows) / max(len(rows), 1)
    return {
        "gpu_util_pct": float(mean_util),
        "gpu_mem_used_mb": float(total_mem),
        "gpu_mem_total_mb": float(total_cap),
        "gpu_details": json.dumps(rows, separators=(",", ":")),
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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        str(int(args.max_samples_per_shard)),
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
        "--compact_candidate_generation",
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
        "--storage_mode",
        str(args.storage_mode),
        "--device",
        str(args.device),
        "--run_name",
        str(run_name),
    ]
    if bool(args.resume):
        cmd.append("--resume")
    if bool(args.progress):
        cmd.append("--progress")
    return cmd


def main() -> None:
    args = parse_args()
    if int(args.start_shard) < 0 or int(args.end_shard) > int(args.num_shards) or int(args.start_shard) >= int(args.end_shard):
        raise ValueError("--start_shard/--end_shard must define a non-empty range inside --num_shards.")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    launcher_name = args.launcher_name or f"stage_b_teacher_parallel_launcher_{stamp}"
    launcher_dir = Path(args.output_dir) / launcher_name
    logs_dir = launcher_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_prefix = args.run_prefix or f"{launcher_name}_{args.split_name}"
    pending = list(range(int(args.start_shard), int(args.end_shard)))
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    active: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    launch_start = time.perf_counter()
    last_monitor = 0.0
    launch_count = 0

    def worker_env(worker_index: int) -> dict[str, str]:
        env = dict(os.environ)
        if gpu_ids:
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[int(worker_index) % len(gpu_ids)])
        if int(args.per_worker_threads) > 0:
            threads = str(int(args.per_worker_threads))
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
                env[key] = threads
        return env

    if bool(args.dry_run):
        rows = []
        for offset, shard_id in enumerate(pending):
            run_name = f"{run_prefix}_shard{int(shard_id):03d}_of{int(args.num_shards):03d}"
            env = worker_env(offset)
            rows.append(
                {
                    "shard_id": int(shard_id),
                    "gpu": env.get("CUDA_VISIBLE_DEVICES", ""),
                    "threads": env.get("OMP_NUM_THREADS", ""),
                    "run_name": run_name,
                    "command": " ".join(_worker_command(args, run_name=run_name, shard_id=int(shard_id))),
                }
            )
        print(json.dumps(rows, indent=2), flush=True)
        return

    def launch_next() -> None:
        nonlocal launch_count
        shard_id = int(pending.pop(0))
        run_name = f"{run_prefix}_shard{shard_id:03d}_of{int(args.num_shards):03d}"
        stdout_path = logs_dir / f"{run_name}.stdout.log"
        stderr_path = logs_dir / f"{run_name}.stderr.log"
        stdout = stdout_path.open("w", encoding="utf-8")
        stderr = stderr_path.open("w", encoding="utf-8")
        env = worker_env(launch_count)
        process = subprocess.Popen(
            _worker_command(args, run_name=run_name, shard_id=shard_id),
            cwd=ROOT,
            env=env,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        assigned_gpu = env.get("CUDA_VISIBLE_DEVICES", "")
        launch_count += 1
        active.append(
            {
                "process": process,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
                "run_name": run_name,
                "run_dir": Path(args.output_dir) / run_name,
                "shard_id": shard_id,
                "gpu": assigned_gpu,
                "start_sec": float(time.perf_counter() - launch_start),
            }
        )

    while pending or active:
        while pending and len(active) < max(1, int(args.max_parallel_workers)):
            launch_next()

        still_active: list[dict[str, Any]] = []
        for item in active:
            process = item["process"]
            if process.poll() is None:
                still_active.append(item)
                continue
            item["stdout"].close()
            item["stderr"].close()
            manifest = _load_json(Path(item["run_dir"]) / "teacher_run.json")
            row = {
                "shard_id": int(item["shard_id"]),
                "run_name": str(item["run_name"]),
                "run_dir": str(item["run_dir"]),
                "returncode": int(process.returncode),
                "gpu": str(item.get("gpu", "")),
                "samples": int(manifest.get("samples", 0) or 0),
                "new_samples": int(manifest.get("new_samples", 0) or 0),
                "records": int(manifest.get("records", 0) or 0),
                "new_records": int(manifest.get("new_records", 0) or 0),
                "runtime_sec": float(manifest.get("runtime_sec", 0.0) or 0.0),
                "start_sec": float(item["start_sec"]),
                "end_sec": float(time.perf_counter() - launch_start),
            }
            completed_rows.append(row)
            if bool(args.progress):
                print(json.dumps(row), flush=True)
            if int(process.returncode) != 0 and bool(args.fail_fast):
                for live in still_active:
                    live["process"].terminate()
                raise SystemExit(f"Shard {item['shard_id']} failed with returncode {process.returncode}")
        active = still_active

        now = time.perf_counter()
        if now - last_monitor >= max(0.5, float(args.monitor_interval)) or not active:
            gpu = _query_gpu()
            monitor_rows.append(
                {
                    "elapsed_sec": float(now - launch_start),
                    "pending_shards": int(len(pending)),
                    "active_workers": int(len(active)),
                    "completed_shards": int(len(completed_rows)),
                    **gpu,
                }
            )
            _write_rows(launcher_dir / "teacher_parallel_monitor.csv", monitor_rows)
            _write_rows(launcher_dir / "teacher_parallel_shards.csv", completed_rows)
            last_monitor = now
        if active:
            time.sleep(0.25)

    wall = float(time.perf_counter() - launch_start)
    failed = [row for row in completed_rows if int(row["returncode"]) != 0]
    completed_samples = sum(int(row["new_samples"] or row["samples"]) for row in completed_rows)
    records = sum(int(row["new_records"] or row["records"]) for row in completed_rows)
    max_mem = max([float(row.get("gpu_mem_used_mb") or 0.0) for row in monitor_rows] or [0.0])
    util_values = [float(row["gpu_util_pct"]) for row in monitor_rows if row.get("gpu_util_pct") is not None]
    summary = {
        "launcher_dir": str(launcher_dir),
        "start_shard": int(args.start_shard),
        "end_shard": int(args.end_shard),
        "num_shards": int(args.num_shards),
        "max_parallel_workers": int(args.max_parallel_workers),
        "gpu_ids": gpu_ids,
        "per_worker_threads": int(args.per_worker_threads),
        "completed_shards": int(len(completed_rows)),
        "failed_shards": int(len(failed)),
        "completed_samples": int(completed_samples),
        "records": int(records),
        "wall_sec": float(wall),
        "traces_per_hour": float(completed_samples / max(wall, 1e-9) * 3600.0),
        "max_gpu_mem_mb": float(max_mem),
        "max_gpu_mem_gb": float(max_mem / 1024.0),
        "mean_gpu_util_pct": float(sum(util_values) / max(len(util_values), 1)),
        "shards_csv": str(launcher_dir / "teacher_parallel_shards.csv"),
        "monitor_csv": str(launcher_dir / "teacher_parallel_monitor.csv"),
    }
    (launcher_dir / "teacher_parallel_run.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
