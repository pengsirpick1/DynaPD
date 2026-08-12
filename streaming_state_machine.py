"""Timeout-driven, forward-only event-keypoint DynaPD-RT controller.

The controller has two causality layers:
1. A positive-direction burst closes only when a real timer deadline has
   elapsed while processing a subsequent observed packet.
2. A perturbation starts after that deadline.  Dummy packets are placed in a
   later bin, and a delay rule can affect only real packets that arrive after
   the rule has been activated.

The final burst is also closed by its wall-clock timeout; no end-of-trace
sentinel or future packet is required.
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
MAX_DELAY = 64
DELAY_WIN = 16
RHO = 0.213
USE_DELAY = True
TAIL_ACTION = True  # Compatibility flag; every action is timeout-triggered.

DEFAULT_UTILITY_PATH = ROOT / "configs" / "dynapd_rt_event_utility_timeout.npy"
EVENT_UTILITY = None
EVENT_UTILITY_PATH = None


def load_event_utility(path: str | Path) -> None:
    """Load only offline aggregate utility; no label or trace state is loaded."""
    global EVENT_UTILITY, EVENT_UTILITY_PATH
    payload = np.load(path, allow_pickle=True).item()
    if payload.get("schema") != "dynapd_event_utility_v3_timeout":
        raise ValueError(
            "timeout controller requires dynapd_event_utility_v3_timeout; "
            f"got {payload.get('schema')!r}"
        )
    EVENT_UTILITY = payload
    EVENT_UTILITY_PATH = str(path)


load_event_utility(DEFAULT_UTILITY_PATH)


def phase_of_bin(bin_idx: int) -> str:
    if bin_idx < 600:
        return "early"
    if bin_idx < 1200:
        return "mid"
    return "late"


def _event_kind(start_bin: int, end_bin: int, packet_count: int) -> str:
    if EVENT_UTILITY is None:
        raise RuntimeError("event utility is not loaded")
    duration = max(1, int(end_bin) - int(start_bin) + 1)
    duration_bin = int(np.digitize(duration, EVENT_UTILITY["duration_edges"], right=False))
    volume_bin = int(np.digitize(max(1, int(packet_count)), EVENT_UTILITY["volume_edges"], right=False))
    return f"d{duration_bin}_v{volume_bin}"


def _event_row(phase: str, event_kind: str) -> tuple[dict, bool]:
    if EVENT_UTILITY is None:
        raise RuntimeError("event utility is not loaded")
    row = EVENT_UTILITY["table"].get((phase, "out", event_kind))
    if row is not None:
        return row, True
    return EVENT_UTILITY["fallback"].get((phase, "out"), {}), False


def best_dose(phase: str, budget_left: int, event_kind: str, per_burst: int) -> tuple[int, bool]:
    """Choose a supported allocation scale without exceeding live tokens."""
    row, event_hit = _event_row(phase, event_kind)
    selected, selected_ratio = 0, -np.inf
    for scale in EVENT_UTILITY["actions"]:
        entry = row.get(float(scale))
        if entry is None:
            continue
        dose = max(1, int(round(float(scale) * float(per_burst))))
        if dose > budget_left:
            continue
        gain = max(float(entry.get("lcb_gain", entry.get("mean_gain", 0.0))), 0.0)
        ratio = gain / max(1, dose)
        if ratio > selected_ratio:
            selected, selected_ratio = dose, ratio
    return selected, event_hit


def _extract_packets(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.flatnonzero(trace != 0)
    if not len(indices):
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0), np.zeros(0, dtype=int)
    times = np.abs(trace[indices])
    slots = np.clip(np.floor(times * ((BINS - 1) / MAX_LOAD_TIME)).astype(int), 0, BINS - 1)
    directions = np.sign(trace[indices])
    order = np.argsort(slots, kind="stable")
    return times[order], slots[order], directions[order], indices[order]


def defend_stream(trace: np.ndarray, seed: int = 0, rho: float = RHO, debug: bool = False):
    """Run a timeout-driven DynaPD-RT policy over one signed-timestamp trace."""
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)[:SL]
    trace = np.pad(trace, (0, max(0, SL - len(trace))))
    _times, slots, directions, indices = _extract_packets(trace)
    rng = np.random.default_rng(seed)

    observed_packets = 0
    used_dummy = 0
    closed_bursts = 0
    burst_start = burst_end = None
    burst_packets = 0
    injections: list[tuple[int, int, int]] = []  # (decision_bin, dummy_start, dose)
    delay_rules: list[tuple[int, int, int]] = []  # (activation, start, end)
    delayed_trace = trace.copy()
    emitted_indices: set[int] = set()
    event_hits = event_fallbacks = 0
    delayed_packets = 0
    audit = {
        "dummy_before_decision": 0,
        "delay_before_activation": 0,
        "delay_after_emission": 0,
        "future_packet_read": 0,
    }

    def fire_timeout() -> None:
        nonlocal used_dummy, closed_bursts, event_hits, event_fallbacks
        if burst_end is None:
            return
        decision_bin = int(burst_end) + TIMEOUT_BINS
        phase = phase_of_bin(decision_bin)
        kind = _event_kind(int(burst_start), int(burst_end), burst_packets)
        token = int(float(rho) * observed_packets - used_dummy)
        if token <= 0:
            closed_bursts += 1
            return
        per_burst = max(1, int(float(rho) * observed_packets / max(1, closed_bursts + 1)))
        dose, event_hit = best_dose(phase, token, kind, per_burst)
        if event_hit:
            event_hits += 1
        else:
            event_fallbacks += 1
        if dose > 0:
            # +1 ensures the rendered dummy bin is strictly after the timeout.
            dummy_start = decision_bin + 1
            if dummy_start <= decision_bin:
                audit["dummy_before_decision"] += 1
            injections.append((decision_bin, dummy_start, dose))
            used_dummy += dose
            if USE_DELAY:
                # This rule applies only to future arrivals, never to the burst
                # that caused its creation or to already emitted packets.
                delay_rules.append((decision_bin, decision_bin + 1, decision_bin + 1 + DELAY_WIN))
        closed_bursts += 1

    for slot, direction, original_index in zip(slots, directions, indices):
        slot = int(slot)
        # A timeout is processed before the current packet. Therefore this
        # packet is not visible when the timer action is selected.
        if burst_end is not None and slot >= int(burst_end) + TIMEOUT_BINS:
            fire_timeout()
            burst_start = burst_end = None
            burst_packets = 0

        # Forward-only delay: the current packet has just arrived and has not
        # yet been emitted. Rules created at earlier timeouts may delay it.
        for activation, start, end in delay_rules:
            if start <= slot < end:
                if slot <= activation:
                    audit["delay_before_activation"] += 1
                    continue
                # `original_index` is the packet that has just arrived. It is
                # delayed before emission; earlier trace entries are never
                # revisited or rewritten.
                if int(original_index) in emitted_indices:
                    audit["delay_after_emission"] += 1
                    continue
                delayed_trace[int(original_index)] = np.sign(delayed_trace[int(original_index)]) * (
                    np.abs(delayed_trace[int(original_index)])
                    + float(rng.integers(1, MAX_DELAY + 1)) * (MAX_LOAD_TIME / BINS)
                )
                delayed_packets += 1

        observed_packets += 1
        if direction <= 0:
            emitted_indices.add(int(original_index))
            continue
        if burst_start is None:
            burst_start = burst_end = slot
            burst_packets = 1
        elif slot - int(burst_end) <= GAP_THRESH:
            burst_end = slot
            burst_packets += 1
        else:
            # This branch should be unreachable because the timeout check above
            # fires before any next outgoing packet separated by a long gap.
            audit["future_packet_read"] += 1
            burst_start = burst_end = slot
            burst_packets = 1
        emitted_indices.add(int(original_index))

    # A real wall-clock timer also fires after the final observed outgoing
    # burst. This action is scheduled from burst_end alone and therefore does
    # not depend on a trace-end marker or future packet.
    tail_timeout_actions = 0
    if burst_end is not None:
        fire_timeout()
        tail_timeout_actions = 1
    counts = np.zeros((2, BINS), dtype=np.int32)
    for decision_bin, dummy_start, dose in injections:
        if dummy_start <= decision_bin:
            audit["dummy_before_decision"] += 1
        counts[0, dummy_start:min(BINS, dummy_start + dose)] += 1
    defended, stats = materialize_trace(delayed_trace, injections, SL)

    if not debug:
        return defended
    return defended, {
        "raw_bw": float(stats.get("raw_bandwidth", 0.0)),
        "inj_total": int(used_dummy),
        "n_bursts": int(closed_bursts),
        "n_tail": int(tail_timeout_actions),
        "n_delay": len(delay_rules),
        "event_keypoint_hits": int(event_hits),
        "event_keypoint_fallbacks": int(event_fallbacks),
        "delayed_packets": int(delayed_packets),
        "actions": [
            {"decision_bin": int(decision), "dummy_start_bin": int(start), "dose": int(dose)}
            for decision, start, dose in injections
        ],
        "audit": audit,
    }
