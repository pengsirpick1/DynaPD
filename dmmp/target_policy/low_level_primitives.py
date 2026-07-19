"""Label-free low-level utility maps for target-policy construction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encoders.prefix import PrefixCondition

PRIMITIVES = ("burst", "gap", "rate", "direction", "shape")


def _normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float32), 0.0) * np.asarray(mask, dtype=np.float32)
    valid = np.asarray(mask) > 0
    if np.any(valid):
        peak = float(arr[valid].max())
        if peak > 1.0e-8:
            arr = arr / peak
    return arr.astype(np.float32)


def _repeat_channels(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    return np.repeat(arr.reshape(1, -1), np.asarray(mask).shape[0], axis=0).astype(np.float32)


@dataclass(frozen=True)
class PrimitiveUtilityBundle:
    burst: np.ndarray
    gap: np.ndarray
    rate: np.ndarray
    direction: np.ndarray
    shape: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in PRIMITIVES}


def burst_utility(condition: PrefixCondition, allowed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=np.float32)
    saliency = np.asarray(condition.burst_saliency, dtype=np.float32)
    if saliency.size != mask.shape[1]:
        saliency = np.resize(saliency, mask.shape[1])
    early = np.linspace(1.0, 0.35, mask.shape[1], dtype=np.float32)
    utility = _repeat_channels(0.75 * saliency + 0.25 * early, mask)
    utility += 0.15 * np.asarray(condition.saliency, dtype=np.float32)
    return _normalize(utility, mask)


def gap_utility(condition: PrefixCondition, allowed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=np.float32)
    gap = np.asarray(condition.gap_saliency, dtype=np.float32)
    if gap.size != mask.shape[1]:
        gap = np.resize(gap, mask.shape[1])
    utility = _repeat_channels(gap, mask)
    return _normalize(utility, mask)


def rate_utility(condition: PrefixCondition, allowed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=np.float32)
    rate = np.asarray(condition.rate_saliency, dtype=np.float32)
    if rate.size != mask.shape[1]:
        rate = np.resize(rate, mask.shape[1])
    low_valley = 1.0 - rate
    smooth_target = 0.65 * low_valley + 0.35 * np.convolve(rate, np.ones(5, dtype=np.float32) / 5.0, mode="same")
    return _normalize(_repeat_channels(smooth_target, mask), mask)


def direction_utility(condition: PrefixCondition, allowed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=np.float32)
    out_ratio = float(condition.metadata.get("out_ratio", 0.5))
    in_ratio = float(condition.metadata.get("in_ratio", 0.5))
    target_in = float(np.clip(0.35 + 0.30 * ((out_ratio + in_ratio) % 1.0), 0.25, 0.75))
    current_in = in_ratio / max(out_ratio + in_ratio, 1.0e-6)
    incoming_need = max(target_in - current_in, 0.0)
    outgoing_need = max((1.0 - target_in) - (1.0 - current_in), 0.0)
    channel_weight = np.asarray([outgoing_need + 0.10, incoming_need + 0.10], dtype=np.float32).reshape(2, 1)
    base = _repeat_channels(0.5 + 0.5 * np.asarray(condition.gap_saliency, dtype=np.float32), mask)
    return _normalize(base * channel_weight, mask)


def shape_utility(condition: PrefixCondition, allowed_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=np.float32)
    prototype = np.asarray(condition.public_prototype, dtype=np.float32)
    rate = np.asarray(condition.rate_saliency, dtype=np.float32)
    if prototype.size != mask.shape[1]:
        prototype = np.resize(prototype, mask.shape[1])
    if rate.size != mask.shape[1]:
        rate = np.resize(rate, mask.shape[1])
    residual = np.maximum(prototype - rate, 0.0)
    return _normalize(_repeat_channels(residual, mask), mask)


def compute_primitive_utilities(condition: PrefixCondition, allowed_mask: np.ndarray) -> PrimitiveUtilityBundle:
    return PrimitiveUtilityBundle(
        burst=burst_utility(condition, allowed_mask),
        gap=gap_utility(condition, allowed_mask),
        rate=rate_utility(condition, allowed_mask),
        direction=direction_utility(condition, allowed_mask),
        shape=shape_utility(condition, allowed_mask),
    )
