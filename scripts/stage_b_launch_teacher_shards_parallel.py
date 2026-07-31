# -*- coding: utf-8 -*-
"""Launch Stage B Teacher export shards with bounded parallel workers."""

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
    parser.add_argument("--split_name", choices=["train", "val", "test", "all", "archive"], default="train")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--launcher_name", default="")
    parser.add_argument("--run_prefix", default="")
    parser.add_argument("--python_exe", default=sys.executable)
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
    return parser.parse_args()


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
    active: list[dict[str, Any]] = []
    completed_rows: list[dict[str, Any]] = []
    monitor_rows: list[dict[str, Any]] = []
    launch_start = time.perf_counter()
    last_monitor = 0.0

    def launch_next() -> None:
        shard_id = int(pending.pop(0))
        run_name = f"{run_prefix}_shard{shard_id:03d}_of{int(args.num_shards):03d}"
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
