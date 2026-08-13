#!/usr/bin/env python3
"""Shared helpers for offline DynaPD-RT event-utility calibration.

The table is deliberately label-free at runtime.  Labels and the RF/DF/AWF
surrogates are used only here to estimate which actions reduce the true-class
margin for each observable outgoing-burst shape.
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'wflib_copy'))

from dynapd.evaluation.attack_models import build_rf_tam_input
from dynapd.stage_a.faithfulness import predict_probabilities
from dynapd.stage_a.modeling import load_stage_a_attacker
from causal_event_renderer import apply_future_delay, materialize_trace
from WFlib import models as wm

BINS = 1800
TRACE_LENGTH = 5000
MAX_LOAD_TIME = 80.0
GAP_THRESH = 4
TIMEOUT_BINS = GAP_THRESH + 1
MAX_DELAY = 64
DELAY_WIN = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default=str(ROOT / 'datasets/CW.npz'))
    parser.add_argument('--calibration_start', type=int, default=0)
    parser.add_argument('--calibration_end', type=int, default=96)
    parser.add_argument('--max_bursts_per_trace', type=int, default=16)
    parser.add_argument('--min_support', type=int, default=12)
    parser.add_argument('--rho', type=float, default=0.45, help='token-bucket rate used by the online controller')
    parser.add_argument('--scales', nargs='+', type=float, default=[0.50, 0.75, 1.00, 1.25], help='event allocation scales relative to causal per-burst allocation')
    parser.add_argument(
        '--ensemble_weights',
        nargs=3,
        type=float,
        metavar=('RF', 'DF', 'AWF'),
        default=[0.80, 0.10, 0.10],
        help='offline surrogate weights; normalized before utility aggregation',
    )
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--rf_checkpoint', default=str(ROOT / 'models/attack/fixed_rf_checkpoint.pt'))
    parser.add_argument('--df_checkpoint', default=str(ROOT / 'wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth'))
    parser.add_argument('--awf_checkpoint', default=str(ROOT / 'wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth'))
    return parser.parse_args()


def load_data(path: str) -> tuple[np.ndarray, np.ndarray]:
    payload = np.load(path)
    traces = np.asarray(payload['X'], dtype=np.float32)
    if traces.ndim == 3:
        traces = traces[:, 0, :]
    if traces.ndim != 2:
        raise ValueError(f'expected [N,L] traces, got {traces.shape}')
    return traces[:, :TRACE_LENGTH], np.asarray(payload['y'], dtype=np.int64)


def phase_of_bin(bin_idx: int) -> str:
    if bin_idx < 600:
        return 'early'
    if bin_idx < 1200:
        return 'mid'
    return 'late'


def extract_outgoing_bursts(trace: np.ndarray) -> list[tuple[int, int, int]]:
    indices = np.flatnonzero(trace > 0)
    if not len(indices):
        return []
    slots = np.floor(np.abs(trace[indices]) * ((BINS - 1) / MAX_LOAD_TIME)).astype(int)
    slots = np.clip(np.sort(slots), 0, BINS - 1)
    bursts: list[tuple[int, int, int]] = []
    start = end = int(slots[0])
    count = 1
    for raw_slot in slots[1:]:
        slot = int(raw_slot)
        if slot - end <= GAP_THRESH:
            end = slot
            count += 1
        else:
            bursts.append((start, end, count))
            start = end = slot
            count = 1
    bursts.append((start, end, count))
    return bursts


def event_contexts(trace: np.ndarray, rho: float, max_bursts: int) -> list[tuple[int, int, int, int, int]]:
    """Return timeout-triggered, causally observable outgoing burst contexts.

    Allocation is derived from the same token-bucket state used online: all
    arrived packets contribute to the budget, while only outgoing bursts
    trigger actions.
    """
    all_indices = np.flatnonzero(trace != 0)
    all_slots = np.floor(np.abs(trace[all_indices]) * ((BINS - 1) / MAX_LOAD_TIME)).astype(int)
    all_slots = np.sort(np.clip(all_slots, 0, BINS - 1))
    contexts: list[tuple[int, int, int, int]] = []
    for ordinal, (start, end, count) in enumerate(extract_outgoing_bursts(trace)[:max_bursts], start=1):
        decision_bin = int(end) + TIMEOUT_BINS
        # The timer fires before a packet at its deadline is processed.
        observed_packets = int(np.searchsorted(all_slots, decision_bin, side='left'))
        allocation = max(1, int(float(rho) * observed_packets / ordinal))
        contexts.append((start, end, count, decision_bin, allocation))
    return contexts


def margin(probabilities: np.ndarray, label: int) -> float:
    masked = np.asarray(probabilities, dtype=np.float32).copy()
    true_score = float(masked[label])
    masked[label] = -np.inf
    return true_score - float(masked.max())


def wflib_probability(model: torch.nn.Module, trace: np.ndarray, device: torch.device, length: int) -> np.ndarray:
    x = np.sign(trace[:length])[None, None, :]
    with torch.no_grad():
        logits, _ = model(torch.as_tensor(x, dtype=torch.float32, device=device))
    return torch.softmax(logits, dim=1).cpu().numpy()[0]


def rf_probability(model: torch.nn.Module, trace: np.ndarray, device: torch.device) -> np.ndarray:
    tam = build_rf_tam_input(trace[None], max_len=TRACE_LENGTH, max_load_time=MAX_LOAD_TIME, num_slots=BINS)
    return predict_probabilities(model, tam, device=device, batch_size=1)[0]


def bucket(value: int, edges: np.ndarray) -> int:
    return int(np.digitize(value, edges, right=False))


def event_type(start: int, end: int, packet_count: int, duration_edges: np.ndarray, volume_edges: np.ndarray) -> str:
    duration = max(1, int(end) - int(start) + 1)
    return f'd{bucket(duration, duration_edges)}_v{bucket(max(1, packet_count), volume_edges)}'


def action_trace(trace: np.ndarray, decision_bin: int, dose: int, seed: int) -> np.ndarray:
    """Simulate a timeout action without rewriting already emitted packets."""
    delayed, _audit = apply_future_delay(
        trace,
        activation_bin=int(decision_bin),
        window_bins=DELAY_WIN,
        max_delay_bins=MAX_DELAY,
        seed=seed,
    )
    dummy_start = int(decision_bin) + 1
    defended, _stats = materialize_trace(
        delayed,
        [(int(decision_bin), dummy_start, int(dose))],
        TRACE_LENGTH,
    )
    return defended


def summary_entry(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    lcb = mean - 1.96 * std / math.sqrt(max(1, len(array)))
    return {'mean_gain': mean, 'std_gain': std, 'lcb_gain': float(lcb), 'n': int(len(array))}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('event utility construction requires CUDA for the surrogate ensemble')
    if args.calibration_end <= args.calibration_start:
        raise ValueError('calibration_end must be greater than calibration_start')
    weights = np.asarray(args.ensemble_weights, dtype=np.float64)
    if np.any(weights < 0) or float(weights.sum()) <= 0:
        raise ValueError('--ensemble_weights must be non-negative with a positive sum')
    weights /= weights.sum()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda')
    traces, labels = load_data(args.data_root)
    end = min(args.calibration_end, len(labels))
    source_indices = np.arange(args.calibration_start, end, dtype=np.int64)
    if not len(source_indices):
        raise ValueError('empty calibration interval')

    events: list[tuple[int, int, int, int, int, int]] = []
    for source_index in source_indices:
        for start, stop, count, decision_bin, allocation in event_contexts(traces[source_index], args.rho, args.max_bursts_per_trace):
            events.append((int(source_index), start, stop, count, decision_bin, allocation))
    if not events:
        raise RuntimeError('no outgoing bursts in calibration interval')
    duration_edges = np.quantile([end - start + 1 for _, start, end, _, _, _ in events], [1 / 3, 2 / 3]).astype(np.float32)
    volume_edges = np.quantile([count for _, _, _, count, _, _ in events], [1 / 3, 2 / 3]).astype(np.float32)

    rf = load_stage_a_attacker(args.rf_checkpoint, attacker='rf', num_classes=95, device=device, max_trace_length=TRACE_LENGTH, rf_num_slots=BINS)
    df = wm.DF(95).to(device)
    df.load_state_dict(torch.load(args.df_checkpoint, map_location=device, weights_only=True))
    df.eval()
    awf = wm.AWF(95).to(device)
    awf.load_state_dict(torch.load(args.awf_checkpoint, map_location=device, weights_only=True))
    awf.eval()

    gains: dict[tuple[str, str, str, float], list[float]] = defaultdict(list)
    fallback_gains: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    started = time.monotonic()
    events_by_trace: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for source_index, start, end_bin, count, decision_bin, allocation in events:
        events_by_trace[source_index].append((start, end_bin, count, decision_bin, allocation))

    for completed, source_index in enumerate(source_indices, start=1):
        trace = traces[source_index]
        label = int(labels[source_index])
        clean_margins = (
            margin(rf_probability(rf, trace, device), label),
            margin(wflib_probability(df, trace, device, TRACE_LENGTH), label),
            margin(wflib_probability(awf, trace, device, 3000), label),
        )
        for start, end_bin, count, decision_bin, allocation in events_by_trace[source_index]:
            phase = phase_of_bin(decision_bin)
            kind = event_type(start, end_bin, count, duration_edges, volume_edges)
            for scale in args.scales:
                dose = max(1, int(round(float(scale) * allocation)))
                defended = action_trace(trace, decision_bin, dose, seed=args.seed + int(source_index) * 1009 + int(decision_bin) * 17 + int(round(scale * 1000)))
                defended_margins = (
                    margin(rf_probability(rf, defended, device), label),
                    margin(wflib_probability(df, defended, device, TRACE_LENGTH), label),
                    margin(wflib_probability(awf, defended, device, 3000), label),
                )
                component_gains = np.asarray(clean_margins) - np.asarray(defended_margins)
                gain = float(np.dot(weights, component_gains))
                gains[(phase, 'out', kind, float(scale))].append(float(gain))
                fallback_gains[(phase, 'out', float(scale))].append(float(gain))
        if completed % 16 == 0 or completed == len(source_indices):
            print(f'[event-utility] {completed}/{len(source_indices)} traces, {time.monotonic() - started:.1f}s', flush=True)

    table: dict[tuple[str, str, str], dict[float, dict[str, float | int]]] = {}
    support: dict[str, int] = {}
    keys = sorted({key[:3] for key in gains})
    for key in keys:
        row = {float(scale): summary_entry(gains[(key[0], key[1], key[2], float(scale))]) for scale in args.scales}
        row_support = min(int(value['n']) for value in row.values())
        support['|'.join(key)] = row_support
        if row_support >= args.min_support:
            table[key] = row
    fallback: dict[tuple[str, str], dict[float, dict[str, float | int]]] = {}
    for phase in ('early', 'mid', 'late'):
        key = (phase, 'out')
        fallback[key] = {
            float(scale): summary_entry(fallback_gains[(phase, 'out', float(scale))])
            for scale in args.scales
            if fallback_gains[(phase, 'out', float(scale))]
        }

    artifact = {
        'schema': 'dynapd_event_utility_v3_timeout',
        'calibration_data': str(Path(args.data_root).resolve()),
        'calibration_start': int(args.calibration_start),
        'calibration_end_exclusive': int(end),
        'action_kind': 'allocation_scale',
        'rho': float(args.rho),
        'timeout_bins': int(TIMEOUT_BINS),
        'delay_window_bins': int(DELAY_WIN),
        'max_delay_bins': int(MAX_DELAY),
        'actions': [float(value) for value in args.scales],
        'duration_edges': duration_edges,
        'volume_edges': volume_edges,
        'min_support': int(args.min_support),
        'table': table,
        'fallback': fallback,
        'support': support,
        'ensemble': {'RF': float(weights[0]), 'DF': float(weights[1]), 'AWF': float(weights[2])},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, artifact, allow_pickle=True)
    report = {
        'artifact': str(output),
        'events_total': int(len(events)),
        'event_keys_total': int(len(keys)),
        'event_keys_retained': int(len(table)),
        'duration_edges': duration_edges.tolist(),
        'volume_edges': volume_edges.tolist(),
        'support': support,
        'elapsed_seconds': time.monotonic() - started,
    }
    output.with_suffix('.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
