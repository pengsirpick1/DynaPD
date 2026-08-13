"""Strictly causal, timeout-driven DynaPD-RT state machine.

The controller is a deployment-time state machine: it consumes packet arrivals
in timestamp order, schedules a timer when an outgoing burst becomes idle, and
only acts after that timer has expired.  It never requires a website label, a
complete trace, or an online attacker-model query.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from causal_event_renderer import materialize_trace

BINS = 1800
SL = 5000
MAX_LOAD_TIME = 80.0
GAP_THRESH = 4
TIMEOUT_BINS = GAP_THRESH + 1
RHO = 0.35
UTILITY: dict | None = None

DEFAULT_UTILITY_PATH = ROOT / 'configs' / 'dynapd_rt_event_utility.npy'


def load_utility(path: str | Path) -> None:
    global UTILITY
    utility = np.load(path, allow_pickle=True).item()
    if utility.get('schema') != 'dynapd_event_utility_v4_profiles':
        raise ValueError(f"expected v4 profile utility, got {utility.get('schema')!r}")
    UTILITY = utility


def _extract_packets(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(trace != 0)
    if not len(indices):
        return np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=int)
    slots = np.clip(np.floor(np.abs(trace[indices]) * ((BINS - 1) / MAX_LOAD_TIME)).astype(int), 0, BINS - 1)
    directions = np.sign(trace[indices])
    order = np.argsort(slots, kind='stable')
    return slots[order], directions[order], indices[order]


def _event_kind(start: int, end: int, count: int) -> str:
    if UTILITY is None:
        raise RuntimeError('utility is not loaded')
    duration = max(1, int(end) - int(start) + 1)
    duration_bin = int(np.digitize(duration, UTILITY['duration_edges'], right=False))
    volume_bin = int(np.digitize(max(1, count), UTILITY['volume_edges'], right=False))
    return f'd{duration_bin}_v{volume_bin}'


def _phase(decision_bin: int) -> int:
    if UTILITY is None:
        raise RuntimeError('utility is not loaded')
    return int(np.digitize(int(decision_bin), UTILITY['phase_edges'], right=False))


def _action_row(phase: int, kind: str) -> tuple[dict, bool]:
    if UTILITY is None:
        raise RuntimeError('utility is not loaded')
    row = UTILITY['table'].get((phase, 'out', kind))
    if row is not None:
        return row, True
    return UTILITY['fallback'].get(phase, {}), False


def select_profile(
    phase: int,
    kind: str,
    budget_left: int,
    per_burst: int,
    require_positive_utility: bool = False,
) -> tuple[str | None, int, bool]:
    if UTILITY is None:
        raise RuntimeError('utility is not loaded')
    row, hit = _action_row(phase, kind)
    best_name: str | None = None
    best_dose = 0
    best_score = -np.inf
    for name, entry in row.items():
        profile = UTILITY['actions'][name]
        dose = max(1, int(round(float(profile['dose_scale']) * per_burst)))
        if dose > budget_left:
            continue
        # The conservative minimum surrogate benefit prevents RF-only actions
        # from winning when DF or AWF has negative evidence.
        score = float(entry['robust_lcb']) - float(UTILITY['cost_penalty']) * float(profile['dose_scale'])
        # The standard RT policy preserves coverage and selects the best
        # profile even in weak states. A diagnostic can opt into no-op gating.
        if (not require_positive_utility or score > 0.0) and score > best_score:
            best_name, best_dose, best_score = name, dose, score
    return best_name, best_dose, hit


def defend_stream(
    trace: np.ndarray,
    seed: int = 0,
    rho: float = RHO,
    debug: bool = False,
    require_positive_utility: bool = False,
):
    """Defend one trace using only arrived packets and a wall-clock timeout."""
    if UTILITY is None:
        raise RuntimeError('call load_utility() before defending traces')
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)[:SL]
    trace = np.pad(trace, (0, max(0, SL - len(trace))))
    slots, directions, indices = _extract_packets(trace)
    rng = np.random.default_rng(seed)

    observed = used_dummy = closed_bursts = 0
    burst_start = burst_end = None
    burst_packets = 0
    delayed_trace = trace.copy()
    emitted: set[int] = set()
    # (activation, start, end, max_delay)
    delay_rules: list[tuple[int, int, int, int]] = []
    injections: list[tuple[int, int, int]] = []
    profiles: list[str] = []
    event_hits = event_fallbacks = delayed_packets = 0
    audit = {'dummy_before_decision': 0, 'delay_before_activation': 0, 'delay_after_emission': 0, 'future_packet_read': 0}

    def fire_timeout() -> None:
        nonlocal used_dummy, closed_bursts, event_hits, event_fallbacks
        if burst_end is None:
            return
        decision = int(burst_end) + TIMEOUT_BINS
        phase = _phase(decision)
        kind = _event_kind(int(burst_start), int(burst_end), burst_packets)
        token = int(float(rho) * observed - used_dummy)
        if token > 0:
            per_burst = max(1, int(float(rho) * observed / max(1, closed_bursts + 1)))
            name, dose, hit = select_profile(phase, kind, token, per_burst, require_positive_utility)
            if hit:
                event_hits += 1
            else:
                event_fallbacks += 1
            if name is not None and dose > 0:
                profile = UTILITY['actions'][name]
                spacing = int(profile['spacing'])
                for offset in range(dose):
                    dummy_start = decision + 1 + offset * spacing
                    if dummy_start <= decision:
                        audit['dummy_before_decision'] += 1
                    injections.append((decision, dummy_start, 1))
                used_dummy += dose
                delay_rules.append((decision, decision + 1, decision + 1 + int(profile['delay_window']), int(profile['max_delay'])))
                profiles.append(name)
        closed_bursts += 1

    for slot, direction, index in zip(slots, directions, indices):
        slot, index = int(slot), int(index)
        if burst_end is not None and slot >= int(burst_end) + TIMEOUT_BINS:
            fire_timeout()
            burst_start = burst_end = None
            burst_packets = 0

        for activation, start, end, max_delay in delay_rules:
            if start <= slot < end:
                if slot <= activation:
                    audit['delay_before_activation'] += 1
                elif index in emitted:
                    audit['delay_after_emission'] += 1
                else:
                    delayed_trace[index] = np.sign(delayed_trace[index]) * (
                        np.abs(delayed_trace[index]) + float(rng.integers(1, max_delay + 1)) * (MAX_LOAD_TIME / BINS)
                    )
                    delayed_packets += 1

        observed += 1
        if direction > 0:
            if burst_start is None:
                burst_start = burst_end = slot
                burst_packets = 1
            elif slot - int(burst_end) <= GAP_THRESH:
                burst_end = slot
                burst_packets += 1
            else:
                audit['future_packet_read'] += 1
                burst_start = burst_end = slot
                burst_packets = 1
        emitted.add(index)

    # A real wall-clock timer also fires after the final observed outgoing
    # burst; this action is scheduled from burst_end alone and does not require
    # a trace-end marker or any later packet.
    if burst_end is not None:
        fire_timeout()
    defended, stats = materialize_trace(delayed_trace, injections, SL)
    if not debug:
        return defended
    return defended, {
        'raw_bw': float(stats['raw_bandwidth']),
        'inj_total': int(used_dummy),
        'n_bursts': int(closed_bursts),
        'n_delay': len(delay_rules),
        'delayed_packets': int(delayed_packets),
        'defended_packets_total': int(stats['defended_packets_total']),
        'defended_completion_time': float(stats['defended_completion_time']),
        'attack_input_real_packets': int(stats['attack_input_real_packets']),
        'real_packets_truncated_for_attack_input': int(stats['real_packets_truncated_for_attack_input']),
        'dummy_packets_truncated_for_attack_input': int(stats['dummy_packets_truncated_for_attack_input']),
        'event_keypoint_hits': int(event_hits),
        'event_keypoint_fallbacks': int(event_fallbacks),
        'profile_counts': {name: profiles.count(name) for name in UTILITY['actions']},
        'actions': [{'decision_bin': int(decision), 'dummy_start_bin': int(start), 'dose': int(dose)} for decision, start, dose in injections],
        'audit': audit,
    }
