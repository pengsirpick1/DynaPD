#!/usr/bin/env python3
"""Memory-bounded Full-CW exporter for packet-conserving DynaPD-RT."""
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

_RHO = 0.35


def init_worker(utility: str, rho: float) -> None:
    global _RHO
    _RHO = rho
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
    parser.add_argument('--source-start', type=int, default=96)
    parser.add_argument('--source-end', type=int, default=105730)
    parser.add_argument('--rho', type=float, default=0.35)
    parser.add_argument('--workers', type=int, default=18)
    parser.add_argument('--chunk-size', type=int, default=2048)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    payload = np.load(args.data)
    traces = np.asarray(payload['X'], dtype=np.float32)
    labels = np.asarray(payload['y'], dtype=np.int64)
    if traces.ndim == 3:
        traces = traces[:, 0, :]
    source = np.arange(args.source_start, min(args.source_end, len(labels)), dtype=np.int64)
    if not len(source):
        raise ValueError('empty source interval')
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    metadata: list[dict] = []
    chunks_written = 0
    with mp.Pool(args.workers, initializer=init_worker, initargs=(args.utility, args.rho)) as pool:
        for start in range(0, len(source), args.chunk_size):
            chunk_source = source[start : start + args.chunk_size]
            tasks = [(int(index), traces[int(index), :5000]) for index in chunk_source]
            rows = list(pool.imap_unordered(worker, tasks, chunksize=8))
            rows.sort(key=lambda row: row[0])
            indices = np.asarray([row[0] for row in rows], dtype=np.int64)
            np.savez_compressed(
                out / f'streaming_dynapd_rt_shard{chunks_written:03d}.npz',
                X=np.stack([row[1] for row in rows])[:, None, :],
                y=labels[indices],
                source_indices=indices,
            )
            metadata.extend(row[2] for row in rows)
            chunks_written += 1
            completed = start + len(rows)
            elapsed = time.monotonic() - started
            rate = completed / max(elapsed, 1e-9)
            remaining = (len(source) - completed) / max(rate, 1e-9)
            print(
                f'[dynapd-rt] {completed}/{len(source)} traces, {chunks_written} shards, '
                f'{rate:.1f} trace/s, ETA {remaining / 60.0:.1f} min',
                flush=True,
            )

    audits = {key: int(sum(row['audit'][key] for row in metadata)) for key in metadata[0]['audit']}
    profiles = {
        name: int(sum(row['profile_counts'].get(name, 0) for row in metadata))
        for name in np.load(args.utility, allow_pickle=True).item()['actions']
    }
    clean_completion = np.max(np.abs(traces[source, :5000]), axis=1)
    defended_completion = np.asarray([row['defended_completion_time'] for row in metadata], dtype=np.float64)
    completion_delta = defended_completion - clean_completion
    real_truncated = np.asarray([row['real_packets_truncated_for_attack_input'] for row in metadata], dtype=np.int64)
    dummy_truncated = np.asarray([row['dummy_packets_truncated_for_attack_input'] for row in metadata], dtype=np.int64)
    np.savez_compressed(
        out / 'physical_trace_audit.npz',
        source_indices=source,
        clean_completion_time=clean_completion,
        defended_completion_time=defended_completion,
        real_packets_truncated_for_attack_input=real_truncated,
        dummy_packets_truncated_for_attack_input=dummy_truncated,
    )
    elapsed = time.monotonic() - started
    summary = {
        'controller': 'timeout_event_keypoint_profiles_fixed_budget',
        'causality_mode': 'timer_triggered_forward_only',
        'samples': int(len(source)),
        'source_start': int(source.min()),
        'source_end_exclusive': int(source.max()) + 1,
        'rho': float(args.rho),
        'utility': str(args.utility),
        'chunks': int(chunks_written),
        'workers': int(args.workers),
        'mean_actual_bandwidth': float(np.mean([row['raw_bw'] for row in metadata])),
        'mean_dummy_packets': float(np.mean([row['inj_total'] for row in metadata])),
        'mean_delay_rules': float(np.mean([row['n_delay'] for row in metadata])),
        'mean_delayed_packets': float(np.mean([row['delayed_packets'] for row in metadata])),
        'event_keypoint_hits': int(sum(row['event_keypoint_hits'] for row in metadata)),
        'event_keypoint_fallbacks': int(sum(row['event_keypoint_fallbacks'] for row in metadata)),
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
        'parallel_generation_ms_per_trace': float(elapsed * 1000.0 / len(source)),
    }
    (out / 'generation_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
