#!/usr/bin/env python3
"""Build a robust, multi-profile utility table for timeout-driven DynaPD-RT.

Only RF, DF, and AWF are offline surrogates.  TF and VarCNN deliberately do
not appear here: they remain held-out attackers in the final evaluation.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'wflib_copy'))

from event_utility_common import (
    BINS,
    DELAY_WIN,
    GAP_THRESH,
    MAX_DELAY,
    MAX_LOAD_TIME,
    TIMEOUT_BINS,
    TRACE_LENGTH,
    event_contexts,
    extract_outgoing_bursts,
    load_data,
    margin,
    rf_probability,
    wflib_probability,
)
from causal_event_renderer import apply_future_delay, materialize_trace
from dynapd.stage_a.modeling import load_stage_a_attacker
from WFlib import models as wm


ACTIONS = {
    # Dummy timing and bounded delay are both scheduled strictly after timeout.
    'compact': {'dose_scale': 0.80, 'spacing': 1, 'delay_window': 8, 'max_delay': 32},
    'spread': {'dose_scale': 0.80, 'spacing': 3, 'delay_window': 16, 'max_delay': 64},
    'delay_heavy': {'dose_scale': 0.50, 'spacing': 1, 'delay_window': 32, 'max_delay': 64},
    'strong': {'dose_scale': 1.20, 'spacing': 2, 'delay_window': 16, 'max_delay': 64},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=str(ROOT / 'datasets/CW.npz'))
    parser.add_argument('--calibration_start', type=int, default=0)
    parser.add_argument('--calibration_end', type=int, default=96)
    parser.add_argument('--max_bursts_per_trace', type=int, default=48)
    parser.add_argument('--min_support', type=int, default=12)
    parser.add_argument('--rho', type=float, default=0.35)
    parser.add_argument('--dose_multiplier', type=float, default=1.0, help='global multiplier for every causal action dose')
    parser.add_argument(
        '--uniform_dose_scale',
        type=float,
        default=None,
        help='use one absolute allocation scale for every profile, decoupling dummy budget from profile selection',
    )
    parser.add_argument('--delay_multiplier', type=float, default=1.0, help='global multiplier for every forward delay window and cap')
    parser.add_argument('--robust_beta', type=float, default=0.35)
    parser.add_argument('--cost_penalty', type=float, default=0.03)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rf_checkpoint', default=str(ROOT / 'models/attack/fixed_rf_checkpoint.pt'))
    parser.add_argument('--df_checkpoint', default=str(ROOT / 'wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth'))
    parser.add_argument('--awf_checkpoint', default=str(ROOT / 'wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth'))
    return parser.parse_args()


def phase_index(decision_bin: int, phase_edges: np.ndarray) -> int:
    return int(np.digitize(int(decision_bin), phase_edges, right=False))


def event_kind(start: int, end: int, packet_count: int, duration_edges: np.ndarray, volume_edges: np.ndarray) -> str:
    duration = max(1, int(end) - int(start) + 1)
    duration_bin = int(np.digitize(duration, duration_edges, right=False))
    volume_bin = int(np.digitize(max(1, int(packet_count)), volume_edges, right=False))
    return f'd{duration_bin}_v{volume_bin}'


def summarize(values: list[float]) -> dict[str, float | int]:
    vector = np.asarray(values, dtype=np.float64)
    mean = float(vector.mean())
    std = float(vector.std(ddof=1)) if len(vector) > 1 else 0.0
    return {
        'mean': mean,
        'std': std,
        'lcb': float(mean - 1.96 * std / math.sqrt(max(1, len(vector)))),
        'n': int(len(vector)),
    }


def action_trace(trace: np.ndarray, decision_bin: int, dose: int, profile: dict[str, int | float], seed: int) -> np.ndarray:
    delayed, _ = apply_future_delay(
        trace,
        activation_bin=int(decision_bin),
        window_bins=int(profile['delay_window']),
        max_delay_bins=int(profile['max_delay']),
        seed=seed,
    )
    # One-item actions preserve each explicitly requested dummy time.
    injections = [
        (int(decision_bin), int(decision_bin) + 1 + offset * int(profile['spacing']), 1)
        for offset in range(int(dose))
    ]
    defended, _ = materialize_trace(delayed, injections, TRACE_LENGTH)
    return defended


def normalized_gain(clean_margin: float, defended_margin: float) -> float:
    return float((clean_margin - defended_margin) / (abs(clean_margin) + 0.05))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('utility construction requires CUDA')
    if args.calibration_end <= args.calibration_start:
        raise ValueError('calibration interval is empty')
    if args.dose_multiplier <= 0:
        raise ValueError('--dose_multiplier must be positive')
    if args.uniform_dose_scale is not None and args.uniform_dose_scale <= 0:
        raise ValueError('--uniform_dose_scale must be positive')
    if args.delay_multiplier <= 0:
        raise ValueError('--delay_multiplier must be positive')
    # The four action shapes are fixed; this single multiplier calibrates their
    # common bandwidth envelope without changing their relative semantics.
    for profile in ACTIONS.values():
        profile['dose_scale'] = (
            float(args.uniform_dose_scale)
            if args.uniform_dose_scale is not None
            else float(profile['dose_scale']) * float(args.dose_multiplier)
        )
        profile['delay_window'] = max(1, int(round(float(profile['delay_window']) * float(args.delay_multiplier))))
        profile['max_delay'] = max(1, int(round(float(profile['max_delay']) * float(args.delay_multiplier))))

    device = torch.device('cuda')
    traces, labels = load_data(args.data_root)
    source_indices = np.arange(args.calibration_start, min(args.calibration_end, len(labels)), dtype=np.int64)
    raw_events: list[tuple[int, int, int, int, int, int]] = []
    for index in source_indices:
        for start, end, count in extract_outgoing_bursts(traces[index])[:args.max_bursts_per_trace]:
            decision = int(end) + TIMEOUT_BINS
            all_slots = np.sort(np.clip(np.floor(np.abs(traces[index][traces[index] != 0]) * ((BINS - 1) / MAX_LOAD_TIME)).astype(int), 0, BINS - 1))
            observed = int(np.searchsorted(all_slots, decision, side='left'))
            ordinal = sum(1 for item in raw_events if item[0] == int(index)) + 1
            allocation = max(1, int(float(args.rho) * observed / ordinal))
            raw_events.append((int(index), int(start), int(end), int(count), decision, allocation))
    if not raw_events:
        raise RuntimeError('no calibration events')

    duration_edges = np.quantile([end - start + 1 for _, start, end, _, _, _ in raw_events], [1 / 3, 2 / 3]).astype(np.float32)
    volume_edges = np.quantile([count for _, _, _, count, _, _ in raw_events], [1 / 3, 2 / 3]).astype(np.float32)
    phase_edges = np.quantile([decision for _, _, _, _, decision, _ in raw_events], [1 / 3, 2 / 3]).astype(np.float32)

    rf = load_stage_a_attacker(args.rf_checkpoint, attacker='rf', num_classes=95, device=device, max_trace_length=TRACE_LENGTH, rf_num_slots=BINS)
    df = wm.DF(95).to(device)
    df.load_state_dict(torch.load(args.df_checkpoint, map_location=device, weights_only=True))
    df.eval()
    awf = wm.AWF(95).to(device)
    awf.load_state_dict(torch.load(args.awf_checkpoint, map_location=device, weights_only=True))
    awf.eval()

    gains: dict[tuple[int, str, str], dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    fallback: dict[int, dict[str, list[list[float]]]] = defaultdict(lambda: defaultdict(list))
    by_trace: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for index, start, end, count, decision, allocation in raw_events:
        by_trace[index].append((start, end, count, decision, allocation))

    started = time.monotonic()
    for completed, index in enumerate(source_indices, start=1):
        trace, label = traces[index], int(labels[index])
        clean = (
            margin(rf_probability(rf, trace, device), label),
            margin(wflib_probability(df, trace, device, TRACE_LENGTH), label),
            margin(wflib_probability(awf, trace, device, 3000), label),
        )
        for start, end, count, decision, allocation in by_trace[index]:
            phase = phase_index(decision, phase_edges)
            kind = event_kind(start, end, count, duration_edges, volume_edges)
            key = (phase, 'out', kind)
            for action_name, profile in ACTIONS.items():
                dose = max(1, int(round(float(profile['dose_scale']) * allocation)))
                defended = action_trace(trace, decision, dose, profile, args.seed + index * 1009 + decision * 31 + len(action_name))
                defended_margins = (
                    margin(rf_probability(rf, defended, device), label),
                    margin(wflib_probability(df, defended, device, TRACE_LENGTH), label),
                    margin(wflib_probability(awf, defended, device, 3000), label),
                )
                vector = [normalized_gain(before, after) for before, after in zip(clean, defended_margins)]
                gains[key][action_name].append(vector)
                fallback[phase][action_name].append(vector)
        if completed % 16 == 0 or completed == len(source_indices):
            print(f'[dynapd-rt] {completed}/{len(source_indices)} calibration traces, {time.monotonic() - started:.1f}s', flush=True)

    def rows_from(source: dict[str, list[list[float]]]) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        for action_name, vectors in source.items():
            matrix = np.asarray(vectors, dtype=np.float64)
            components = {name: summarize(matrix[:, col].tolist()) for col, name in enumerate(('RF', 'DF', 'AWF'))}
            lcb_values = np.asarray([components[name]['lcb'] for name in ('RF', 'DF', 'AWF')], dtype=np.float64)
            profile = ACTIONS[action_name]
            robust_lcb = float(lcb_values.min() + args.robust_beta * lcb_values.mean())
            rows[action_name] = {
                'components': components,
                'robust_lcb': robust_lcb,
                'cost': float(profile['dose_scale']),
            }
        return rows

    table: dict[tuple[int, str, str], dict[str, dict]] = {}
    support: dict[str, int] = {}
    for key, source in gains.items():
        row = rows_from(source)
        row_support = min(int(value['components']['RF']['n']) for value in row.values())
        support['|'.join(map(str, key))] = row_support
        if row_support >= args.min_support:
            table[key] = row
    fallback_rows = {phase: rows_from(source) for phase, source in fallback.items()}

    artifact = {
        'schema': 'dynapd_event_utility_v4_profiles',
        'calibration_data': str(Path(args.data_root).resolve()),
        'calibration_start': int(args.calibration_start),
        'calibration_end_exclusive': int(source_indices[-1]) + 1,
        'rho': float(args.rho),
        'dose_multiplier': float(args.dose_multiplier),
        'uniform_dose_scale': None if args.uniform_dose_scale is None else float(args.uniform_dose_scale),
        'delay_multiplier': float(args.delay_multiplier),
        'timeout_bins': int(TIMEOUT_BINS),
        'phase_edges': phase_edges,
        'duration_edges': duration_edges,
        'volume_edges': volume_edges,
        'actions': ACTIONS,
        'robust_beta': float(args.robust_beta),
        'cost_penalty': float(args.cost_penalty),
        'table': table,
        'fallback': fallback_rows,
        'support': support,
        'ensemble': {'RF': 'surrogate', 'DF': 'surrogate', 'AWF': 'surrogate', 'TF': 'held_out', 'VarCNN': 'held_out'},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, artifact, allow_pickle=True)
    report = {
        'artifact': str(output),
        'events_total': len(raw_events),
        'retained_event_states': len(table),
        'phase_edges': phase_edges.tolist(),
        'duration_edges': duration_edges.tolist(),
        'volume_edges': volume_edges.tolist(),
        'support': support,
        'elapsed_seconds': time.monotonic() - started,
    }
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
