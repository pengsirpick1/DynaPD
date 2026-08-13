#!/usr/bin/env python3
"""Parallel export runner for the strictly causal DynaPD-RT controller."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_UTILITY = ''
_RHO = 0.35


def init_worker(utility: str, rho: float) -> None:
    global _UTILITY, _RHO
    _UTILITY, _RHO = utility, rho
    import streaming_state_machine as controller
    controller.load_utility(utility)


def worker(task: tuple[int, np.ndarray]) -> tuple[int, np.ndarray, dict]:
    index, trace = task
    import streaming_state_machine as controller
    defended, debug = controller.defend_stream(trace, seed=index, rho=_RHO, debug=True)
    return index, np.asarray(defended, dtype=np.float32), debug


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        'mean': float(values.mean()),
        'p50': float(np.quantile(values, 0.50)),
        'p95': float(np.quantile(values, 0.95)),
        'max': float(values.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--utility', default=str(ROOT / 'configs' / 'dynapd_rt_event_utility.npy'))
    parser.add_argument('--data', default=str(ROOT / 'datasets' / 'CW.npz'))
    parser.add_argument('--source-start', type=int, default=1024)
    parser.add_argument('--source-end', type=int, default=1536)
    parser.add_argument('--rho', type=float, default=0.35)
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    payload = np.load(args.data)
    traces = np.asarray(payload['X'], dtype=np.float32)
    labels = np.asarray(payload['y'], dtype=np.int64)
    if traces.ndim == 3:
        traces = traces[:, 0, :]
    source = np.arange(args.source_start, min(args.source_end, len(labels)), dtype=np.int64)
    tasks = [(int(index), traces[int(index), :5000]) for index in source]
    started = time.monotonic()
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.utility, args.rho)) as pool:
        rows = list(pool.imap_unordered(worker, tasks, chunksize=8))
    elapsed = time.monotonic() - started
    rows.sort(key=lambda row: row[0])
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / 'streaming_dynapd_rt.npz',
        X=np.stack([row[1] for row in rows])[:, None, :],
        y=labels[np.asarray([row[0] for row in rows], dtype=np.int64)],
        source_indices=np.asarray([row[0] for row in rows], dtype=np.int64),
    )
    debug_rows = [row[2] for row in rows]
    clean_completion = np.max(np.abs(traces[source, :5000]), axis=1)
    defended_completion = np.asarray([row['defended_completion_time'] for row in debug_rows], dtype=np.float64)
    completion_delta = defended_completion - clean_completion
    real_truncated = np.asarray([row['real_packets_truncated_for_attack_input'] for row in debug_rows], dtype=np.int64)
    dummy_truncated = np.asarray([row['dummy_packets_truncated_for_attack_input'] for row in debug_rows], dtype=np.int64)
    np.savez_compressed(
        out / 'physical_trace_audit.npz',
        source_indices=source,
        clean_completion_time=clean_completion,
        defended_completion_time=defended_completion,
        real_packets_truncated_for_attack_input=real_truncated,
        dummy_packets_truncated_for_attack_input=dummy_truncated,
    )
    audits = {key: int(sum(row['audit'][key] for row in debug_rows)) for key in debug_rows[0]['audit']}
    profiles = {
        name: int(sum(row['profile_counts'].get(name, 0) for row in debug_rows))
        for name in np.load(args.utility, allow_pickle=True).item()['actions']
    }
    summary = {
        'controller': 'timeout_event_keypoint_profiles',
        'causality_mode': 'timer_triggered_forward_only',
        'samples': int(len(rows)),
        'source_start': int(source.min()),
        'source_end_exclusive': int(source.max()) + 1,
        'rho': float(args.rho),
        'utility': str(args.utility),
        'mean_actual_bandwidth': float(np.mean([row['raw_bw'] for row in debug_rows])),
        'mean_dummy_packets': float(np.mean([row['inj_total'] for row in debug_rows])),
        'mean_delay_rules': float(np.mean([row['n_delay'] for row in debug_rows])),
        'mean_delayed_packets': float(np.mean([row['delayed_packets'] for row in debug_rows])),
        'event_keypoint_hits': int(sum(row['event_keypoint_hits'] for row in debug_rows)),
        'event_keypoint_fallbacks': int(sum(row['event_keypoint_fallbacks'] for row in debug_rows)),
        'profile_counts': profiles,
        'physical_trace_audit': {
            'packet_conservation': 'full defended stream retains every input packet; classifier input is its fixed first-5000-packet prefix',
            'completion_delta_seconds': describe(completion_delta),
            'negative_completion_delta_count': int(np.count_nonzero(completion_delta < -1e-5)),
            'attack_input_real_packet_truncation': {
                'affected_traces': int(np.count_nonzero(real_truncated)),
                'mean_packets': float(real_truncated.mean()),
                'p95_packets': float(np.quantile(real_truncated, 0.95)),
                'max_packets': int(real_truncated.max()),
            },
            'attack_input_dummy_packet_truncation': {
                'affected_traces': int(np.count_nonzero(dummy_truncated)),
                'mean_packets': float(dummy_truncated.mean()),
                'p95_packets': float(np.quantile(dummy_truncated, 0.95)),
                'max_packets': int(dummy_truncated.max()),
            },
        },
        'audit': audits,
        'parallel_generation_ms_per_trace': float(elapsed * 1000.0 / max(1, len(rows))),
    }
    (out / 'generation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
