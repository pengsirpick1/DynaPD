"""DynaPD-RT event-keypoint streaming controller.

This deployment controller is strictly causal.  It uses packets observed so
far to identify an ended outgoing burst, derives its local shape
``(duration-bin, packet-volume-bin)``, and looks up an offline-calibrated
action utility.  It never receives a website label, a complete trace, or an
online attacker-model query.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.stage_b_run_dual_actuator import _render_dummy

ROOT = Path(__file__).resolve().parent
BINS = 1800
TRACE_LENGTH = 5000
SL = TRACE_LENGTH  # Compatibility alias used by the public evaluation runners.
MAX_LOAD_TIME = 80.0
GAP_THRESH = 4
MAX_DELAY = 64
DELAY_WIN = 16
DEFAULT_RHO = 0.213
DEFAULT_UTILITY_PATH = ROOT / "configs" / "dynapd_rt_event_utility.npy"
USE_DELAY = True
TAIL_ACTION = False


def _make_renderer_args(seed: int) -> SimpleNamespace:
    return SimpleNamespace(
        rf_num_slots=BINS,
        max_trace_length=TRACE_LENGTH,
        max_delay=0,
        rounds=8,
        delay_length=MAX_DELAY,
        delay_rho=1.0,
        max_load_time=MAX_LOAD_TIME,
        algorithm="priority",
        seed=seed,
        renderer_strategy="priority",
        renderer_coordinate="absolute",
        ratio=0.10,
        max_windows=8,
    )


def load_event_utility(path: str | Path = DEFAULT_UTILITY_PATH) -> dict:
    """Load an offline aggregate table without any trace-specific state."""
    payload = np.load(Path(path), allow_pickle=True).item()
    if payload.get("schema") != "dynapd_event_utility_v2":
        raise ValueError(f"expected dynapd_event_utility_v2, got {payload.get('schema')}")
    return payload


EVENT_UTILITY = load_event_utility()


def phase_of_bin(bin_idx: int) -> str:
    if bin_idx < 600:
        return "early"
    if bin_idx < 1200:
        return "mid"
    return "late"


def _event_kind(start_bin: int, end_bin: int, packet_count: int) -> str:
    duration = max(1, end_bin - start_bin + 1)
    duration_bin = int(np.digitize(duration, EVENT_UTILITY["duration_edges"], right=False))
    volume_bin = int(np.digitize(max(1, packet_count), EVENT_UTILITY["volume_edges"], right=False))
    return f"d{duration_bin}_v{volume_bin}"


def _row_for(phase: str, event_kind: str) -> tuple[dict, bool]:
    event_row = EVENT_UTILITY["table"].get((phase, "out", event_kind))
    if event_row is not None:
        return event_row, True
    return EVENT_UTILITY["fallback"].get((phase, "out"), {}), False


def _choose_dose(phase: str, event_kind: str, token: int, per_burst: int) -> tuple[int, bool]:
    """Select the best supported allocation scale under the live budget."""
    row, event_hit = _row_for(phase, event_kind)
    best_dose, best_ratio = 0, -np.inf
    for scale in EVENT_UTILITY["actions"]:
        entry = row.get(scale)
        if entry is None:
            continue
        dose = max(1, int(round(float(scale) * per_burst)))
        if dose > token:
            continue
        gain = max(float(entry.get("lcb_gain", entry["mean_gain"])), 0.0)
        ratio = gain / dose
        if ratio > best_ratio:
            best_dose, best_ratio = dose, ratio
    return best_dose, event_hit


def _extract_all_packets(trace: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nonzero = np.flatnonzero(trace != 0)
    if not len(nonzero):
        return np.zeros(0), np.zeros(0, dtype=int), np.zeros(0)
    times = np.abs(trace[nonzero])
    slots = np.clip(np.floor(times * ((BINS - 1) / MAX_LOAD_TIME)).astype(int), 0, BINS - 1)
    directions = np.sign(trace[nonzero])
    order = np.argsort(slots, kind="stable")
    return times[order], slots[order], directions[order]


def defend_stream(trace: np.ndarray, seed: int = 0, rho: float = DEFAULT_RHO, debug: bool = False):
    """Defend one trace using only packet history and timeout-safe tail0 logic.

    A burst action is emitted immediately when the following outgoing packet
    proves that the preceding burst has ended.  The unresolved final burst is
    intentionally left untouched: a deployment can invoke that action only
    after a genuine network timeout.
    """
    trace = np.asarray(trace, dtype=np.float32).reshape(-1)[:TRACE_LENGTH]
    trace = np.pad(trace, (0, max(0, TRACE_LENGTH - len(trace))))
    _, slots, directions = _extract_all_packets(trace)
    rng = np.random.default_rng(seed)

    observed_packets = 0
    used_dummy = 0
    burst_id = 0
    burst_start = burst_end = None
    burst_packets = 0
    injections: list[tuple[int, int]] = []
    delay_windows: list[tuple[int, int, int]] = []
    event_hits = event_fallbacks = 0
    bursts_detected = tail_actions = 0
    audit_past = audit_future = 0

    def close_burst() -> None:
        nonlocal used_dummy, event_hits, event_fallbacks, bursts_detected
        if burst_end is None:
            return
        bursts_detected += 1
        decision_bin = int(burst_end)
        phase = phase_of_bin(decision_bin)
        event_kind = _event_kind(int(burst_start), decision_bin, burst_packets)
        token = int(rho * observed_packets - used_dummy)
        if token <= 0:
            return
        per_burst = max(1, int(rho * observed_packets / (burst_id + 1)))
        dose, event_hit = _choose_dose(phase, event_kind, token, per_burst)
        if event_hit:
            event_hits += 1
        else:
            event_fallbacks += 1
        if dose <= 0:
            return
        injections.append((min(BINS - 1, decision_bin + 1), dose))
        used_dummy += dose
        delay_windows.append((decision_bin, max(0, decision_bin - DELAY_WIN), decision_bin + 1))

    for slot, direction in zip(slots, directions):
        observed_packets += 1
        if direction <= 0:
            continue
        slot = int(slot)
        if burst_start is None:
            burst_start = burst_end = slot
            burst_packets = 1
        elif slot - int(burst_end) <= GAP_THRESH:
            burst_end = slot
            burst_packets += 1
        else:
            close_burst()
            burst_id += 1
            burst_start = burst_end = slot
            burst_packets = 1

    # The deployment default is tail0. A caller with an actual timeout event
    # may opt into this ablation by setting TAIL_ACTION before invocation.
    if TAIL_ACTION and burst_end is not None:
        close_burst()
        tail_actions = 1

    counts = np.zeros((2, BINS), dtype=np.int32)
    for injection_bin, dose in injections:
        counts[0, injection_bin:min(BINS, injection_bin + dose)] += 1
    defended, _, renderer_stats = _render_dummy(
        base_trace=trace,
        counts=counts,
        args=_make_renderer_args(seed),
    )

    if delay_windows and USE_DELAY:
        nonzero = np.flatnonzero(defended != 0)
        times = np.abs(defended[nonzero])
        signs = np.sign(defended[nonzero])
        packet_slots = np.clip(np.floor(times * ((BINS - 1) / MAX_LOAD_TIME)).astype(int), 0, BINS - 1)
        for decision_bin, window_start, window_end in delay_windows:
            in_window = (packet_slots >= window_start) & (packet_slots < window_end)
            if not in_window.any():
                continue
            arrived = packet_slots[in_window] <= decision_bin
            audit_past += int(arrived.sum())
            audit_future += int((~arrived).sum())
            times[in_window] += rng.integers(1, MAX_DELAY + 1, size=int(in_window.sum())) * (MAX_LOAD_TIME / BINS)
        defended[nonzero] = signs * times
        defended = defended[np.argsort(np.abs(defended), kind="stable")]
        defended = np.pad(defended, (0, max(0, TRACE_LENGTH - len(defended))))[:TRACE_LENGTH]

    if not debug:
        return defended.astype(np.float32)
    return defended.astype(np.float32), {
        "raw_bw": float(renderer_stats.get("raw_bandwidth", 0.0)),
        "inj_total": int(used_dummy),
        "n_bursts": bursts_detected,
        "n_tail": tail_actions,
        "n_delay": len(delay_windows),
        "event_keypoint_hits": event_hits,
        "event_keypoint_fallbacks": event_fallbacks,
        "audit": {"delay_past_packet": audit_past, "delay_future_window": audit_future},
    }
