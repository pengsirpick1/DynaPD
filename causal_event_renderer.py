"""Timestamp-faithful materialization for timeout-driven RT actions.

The generic padding renderer maps a two-dimensional count map to trace-index
locations.  That is useful for offline attacks, but cannot certify a real-time
action timestamp.  This module instead creates every dummy directly at its
requested time bin and keeps all original signed timestamps explicit.
"""
from __future__ import annotations

import numpy as np

BINS = 1800
MAX_LOAD_TIME = 80.0


def slot_to_time(slot: int) -> float:
    """Map a non-negative time bin to the CW signed-timestamp time axis.

    Values after the 80-second calibration horizon remain valid timestamps.
    They must not be clipped back into the past merely because the classifier
    feature map has a finite number of bins.
    """
    return float(max(0, int(slot)) * (MAX_LOAD_TIME / (BINS - 1)))


def slot_of(values: np.ndarray) -> np.ndarray:
    slots = np.floor(np.abs(values) * ((BINS - 1) / MAX_LOAD_TIME)).astype(int)
    return np.clip(slots, 0, BINS - 1)


def apply_future_delay(
    trace: np.ndarray,
    activation_bin: int,
    window_bins: int,
    max_delay_bins: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Delay only packets that arrive strictly after a known activation time."""
    output = np.asarray(trace, dtype=np.float32).copy()
    nonzero = np.flatnonzero(output != 0)
    audit = {"delay_before_activation": 0, "delay_after_emission": 0}
    if not len(nonzero):
        return output, audit
    slots = slot_of(output[nonzero])
    allowed = (slots >= int(activation_bin) + 1) & (slots < int(activation_bin) + 1 + int(window_bins))
    # This helper receives a static trace only for offline utility estimation.
    # Every selected packet is after activation, hence it has not been emitted
    # by the timer action at activation_bin.
    selected = nonzero[allowed]
    if len(selected):
        rng = np.random.default_rng(seed)
        output[selected] = np.sign(output[selected]) * (
            np.abs(output[selected])
            + rng.integers(1, int(max_delay_bins) + 1, size=len(selected)) * (MAX_LOAD_TIME / BINS)
        )
    return output, audit


def materialize_trace(
    delayed_real_trace: np.ndarray,
    dummy_actions: list[tuple[int, int, int]],
    trace_length: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Merge delayed real packets with explicitly timestamped positive dummies.

    Each action is ``(decision_bin, dummy_start_bin, dose)``.  The caller must
    audit that `dummy_start_bin > decision_bin`; this function preserves these
    exact time semantics rather than translating bins to packet-index patches.
    """
    real = np.asarray(delayed_real_trace, dtype=np.float32).reshape(-1)
    real_values = real[real != 0]
    values: list[tuple[float, int, int, float]] = []
    for serial, value in enumerate(real_values.tolist()):
        values.append((abs(float(value)), 1, serial, float(value)))
    serial = len(values)
    dummy_count = 0
    for decision_bin, start_bin, dose in dummy_actions:
        if int(start_bin) <= int(decision_bin):
            raise ValueError("dummy start must be after decision time")
        for offset in range(max(0, int(dose))):
            dummy_time = slot_to_time(int(start_bin) + offset)
            values.append((dummy_time, 0, serial, float(dummy_time)))
            serial += 1
            dummy_count += 1
    values.sort(key=lambda item: (item[0], item[1], item[2]))
    merged = np.asarray([item[3] for item in values], dtype=np.float32)
    merged = np.pad(merged, (0, max(0, int(trace_length) - len(merged))))[: int(trace_length)]
    return merged, {
        "raw_bandwidth": float(dummy_count / max(1, len(real_values))),
        "dummy_packets": int(dummy_count),
        "original_packets": int(len(real_values)),
    }
