#!/usr/bin/env python3
"""Generate auditable timeout-driven DynaPD-RT exports from local CW data."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_RHO = 0.35


def init_worker(utility_path: str, rho: float) -> None:
    global _RHO
    _RHO = rho
    import streaming_state_machine as controller

    controller.load_event_utility(utility_path)


def worker(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, dict]:
    source_index, trace = task
    import streaming_state_machine as controller

    defended, debug = controller.defend_stream(trace, seed=source_index, rho=_RHO, debug=True)
    return source_index, defended, {
        "raw_bandwidth": float(debug["raw_bw"]),
        "dummy_packets": int(debug["inj_total"]),
        "delay_rules": int(debug["n_delay"]),
        "event_hits": int(debug["event_keypoint_hits"]),
        "event_fallbacks": int(debug["event_keypoint_fallbacks"]),
        **{key: int(value) for key, value in debug["audit"].items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Local CW .npz containing X and y")
    parser.add_argument("--utility", default=str(ROOT / "configs/dynapd_rt_event_utility_timeout.npy"))
    parser.add_argument("--source-start", type=int, default=96)
    parser.add_argument("--source-end", type=int, default=None)
    parser.add_argument("--rho", type=float, default=0.35)
    parser.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = np.load(args.data)
    traces = np.asarray(payload["X"], dtype=np.float32)
    labels = np.asarray(payload["y"], dtype=np.int64)
    if traces.ndim == 3:
        traces = traces[:, 0, :]
    stop = len(labels) if args.source_end is None else min(args.source_end, len(labels))
    indices = np.arange(args.source_start, stop, dtype=np.int64)
    tasks = [(int(index), traces[int(index), :5000]) for index in indices]
    started = time.monotonic()
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.utility, args.rho)) as pool:
        rows = list(pool.imap_unordered(worker, tasks, chunksize=8))
    elapsed = time.monotonic() - started
    rows.sort(key=lambda row: row[0])
    source = np.asarray([row[0] for row in rows], dtype=np.int64)
    defended = np.stack([row[1] for row in rows]).astype(np.float32)
    metadata = [row[2] for row in rows]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "streaming_timeout_event_keypoint.npz", X=defended[:, None, :], y=labels[source], source_indices=source)
    total = lambda key: int(sum(item[key] for item in metadata))
    summary = {
        "controller": "timeout_event_keypoint",
        "causality_mode": "timer_triggered_forward_only",
        "samples": int(len(rows)),
        "source_start": int(source.min()),
        "source_end_exclusive": int(source.max()) + 1,
        "rho": float(args.rho),
        "mean_actual_bandwidth": float(np.mean([item["raw_bandwidth"] for item in metadata])),
        "audit": {
            "dummy_before_decision": total("dummy_before_decision"),
            "delay_before_activation": total("delay_before_activation"),
            "delay_after_emission": total("delay_after_emission"),
            "future_packet_read": total("future_packet_read"),
        },
        "parallel_generation_ms_per_trace": elapsed * 1000.0 / max(1, len(rows)),
    }
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
