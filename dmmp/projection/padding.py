"""Padding policy projection, refinement, and rendering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from ..encoders.prefix import nonzero_trace


@dataclass
class PaddingTemplate:
    counts: np.ndarray
    target_n_pad: int
    actual_n_pad: int
    target_bandwidth: float
    metadata: dict[str, Any] = field(default_factory=dict)


def target_padding_count(clean_trace: np.ndarray, target_bandwidth: float) -> int:
    clean_count = int(nonzero_trace(clean_trace).size)
    if clean_count <= 0 or float(target_bandwidth) <= 0:
        return 0
    return max(1, int(round(clean_count * float(target_bandwidth))))


def exact_round_probabilities(probabilities: np.ndarray, target: int) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = max(0, int(target))
    if target == 0:
        return np.zeros_like(probs, dtype=np.int32)
    probs = np.maximum(probs, 0.0)
    total = float(probs.sum())
    if total <= 1e-12:
        probs = np.ones_like(probs, dtype=np.float64) / max(probs.size, 1)
    else:
        probs = probs / total
    raw = probs * float(target)
    counts = np.floor(raw).astype(np.int32)
    remaining = int(target - counts.sum())
    if remaining > 0:
        order = np.argsort(-(raw - counts), kind="mergesort")
        counts[order[:remaining]] += 1
    return counts.astype(np.int32)


def normalized_template_entropy(counts: np.ndarray) -> float:
    flat = np.asarray(counts, dtype=np.float64).reshape(-1)
    total = float(flat.sum())
    if total <= 0.0:
        return 0.0
    probs = flat[flat > 0] / total
    if probs.size <= 1:
        return 0.0
    return float(-(probs * np.log(probs + 1e-12)).sum() / np.log(flat.size))


def _normalize(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.shape != mask.shape:
        arr = np.resize(arr, mask.shape)
    arr = np.maximum(arr, 0.0) * np.asarray(mask, dtype=np.float32)
    valid = mask > 0
    if np.any(valid):
        peak = float(arr[valid].max())
        if peak > 1e-8:
            arr = arr / peak
    return arr.astype(np.float32)


def _clean_tam_patch_counts(clean_trace: np.ndarray, patch_num: int, max_load_time: float) -> np.ndarray:
    counts = np.zeros((2, int(patch_num)), dtype=np.float64)
    clean = nonzero_trace(clean_trace).astype(np.float32)
    if clean.size == 0:
        return counts
    scale = float(int(patch_num) - 1) / max(float(max_load_time), 1e-6)
    outgoing = clean[clean > 0]
    incoming = -clean[clean < 0]
    if outgoing.size:
        bins = np.floor(outgoing * scale).astype(np.int64)
        bins[outgoing >= float(max_load_time)] = int(patch_num) - 1
        np.add.at(counts[0], np.clip(bins, 0, int(patch_num) - 1), 1.0)
    if incoming.size:
        bins = np.floor(incoming * scale).astype(np.int64)
        bins[incoming >= float(max_load_time)] = int(patch_num) - 1
        np.add.at(counts[1], np.clip(bins, 0, int(patch_num) - 1), 1.0)
    return counts


def _apply_tam_flatten_prior(
    probabilities_2d: np.ndarray,
    mask: np.ndarray,
    clean_trace: np.ndarray,
    *,
    strength: float,
    floor: float,
    max_load_time: float,
) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))
    if strength <= 0.0 or probabilities_2d.ndim != 2 or probabilities_2d.shape[0] < 2:
        return probabilities_2d
    result = np.asarray(probabilities_2d, dtype=np.float64).copy()
    clean_counts = _clean_tam_patch_counts(clean_trace, int(result.shape[1]), float(max_load_time))
    valid_mask = np.asarray(mask, dtype=np.float64)
    density_floor = max(float(floor), 1e-6)
    for row in range(min(result.shape[0], clean_counts.shape[0])):
        valid = valid_mask[row] > 0
        row_sum = float(result[row].sum())
        if row_sum <= 1e-12 or not np.any(valid):
            continue
        current = result[row] / row_sum
        flatten = (1.0 / np.sqrt(clean_counts[row] + density_floor)) * valid.astype(np.float64)
        flatten_sum = float(flatten.sum())
        if flatten_sum <= 1e-12:
            continue
        flatten = flatten / flatten_sum
        result[row] = ((1.0 - strength) * current + strength * flatten) * row_sum
    return result


def project_policy_to_template(
    x0_logits: np.ndarray,
    condition: Any,
    clean_trace: np.ndarray,
    target_bandwidth: float,
    *,
    c_leakage: np.ndarray | None = None,
    preference: np.ndarray | None = None,
    method: str = "topk_random_preference",
    metadata: dict[str, Any] | None = None,
    use_causal_mask: bool = True,
    leakage_weight: float = 1.50,
    preference_weight: float = 0.15,
    logit_temperature: float = 1.0,
    logit_noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    direction_target_incoming_share: float | None = None,
    direction_correction_strength: float = 0.0,
    tam_flatten_strength: float = 0.0,
    tam_flatten_floor: float = 1.0,
    max_load_time: float = 80.0,
) -> PaddingTemplate:
    mask_name = "allowed_mask" if use_causal_mask else "optimistic_allowed_mask"
    mask = np.asarray(getattr(condition, mask_name), dtype=np.float32)
    logits = np.asarray(x0_logits, dtype=np.float32)
    if logits.shape != mask.shape:
        logits = np.resize(logits, mask.shape).astype(np.float32)

    weighted = logits.copy()
    if c_leakage is not None:
        weighted += float(leakage_weight) * _normalize(c_leakage, mask)
    if preference is not None:
        weighted += float(preference_weight) * _normalize(preference, mask)
    if float(logit_noise_std) > 0.0:
        generator = rng if rng is not None else np.random.default_rng(0)
        weighted += generator.normal(0.0, float(logit_noise_std), size=weighted.shape).astype(np.float32) * mask
    weighted = np.where(mask > 0, weighted, -1e9)
    flat = weighted.reshape(-1)
    if not np.any(mask > 0):
        probabilities = np.ones_like(flat, dtype=np.float64)
    else:
        flat = flat - float(np.max(flat))
        temperature = max(float(logit_temperature), 1e-3)
        probabilities = np.exp(np.clip(flat / temperature, -60.0, 0.0)).astype(np.float64)
        probabilities *= mask.reshape(-1).astype(np.float64)
        if float(probabilities.sum()) <= 1e-12:
            probabilities = mask.reshape(-1).astype(np.float64)

    target = target_padding_count(np.asarray(clean_trace), float(target_bandwidth))
    probabilities_2d = probabilities.reshape(mask.shape)
    probabilities_2d = _apply_tam_flatten_prior(
        probabilities_2d,
        mask,
        np.asarray(clean_trace),
        strength=float(tam_flatten_strength),
        floor=float(tam_flatten_floor),
        max_load_time=float(max_load_time),
    )
    probabilities = probabilities_2d.reshape(-1)
    if (
        direction_target_incoming_share is not None
        and float(direction_correction_strength) > 0.0
        and target > 0
        and mask.shape[0] >= 2
    ):
        incoming_target = float(np.clip(direction_target_incoming_share, 0.0, 1.0))
        current_incoming = float(probabilities_2d[1].sum() / max(float(probabilities_2d.sum()), 1e-12))
        desired_incoming = current_incoming + float(direction_correction_strength) * (incoming_target - current_incoming)
        desired_incoming = float(np.clip(desired_incoming, 0.0, 1.0))
        if float(probabilities_2d[1].sum()) <= 1e-12:
            incoming_count = 0
        elif float(probabilities_2d[0].sum()) <= 1e-12:
            incoming_count = target
        else:
            incoming_count = int(round(target * desired_incoming))
        incoming_count = max(0, min(target, incoming_count))
        outgoing_count = target - incoming_count
        counts = np.zeros_like(mask, dtype=np.int32)
        counts[0] = exact_round_probabilities(probabilities_2d[0], outgoing_count).reshape(mask.shape[1])
        counts[1] = exact_round_probabilities(probabilities_2d[1], incoming_count).reshape(mask.shape[1])
    else:
        counts = exact_round_probabilities(probabilities, target).reshape(mask.shape)
    violation = int(counts[mask <= 0].sum())
    total_counts = max(int(counts.sum()), 1)
    incoming_share = float(counts[1].sum() / total_counts) if counts.shape[0] >= 2 else 0.0
    outgoing_share = float(counts[0].sum() / total_counts) if counts.shape[0] >= 1 else 0.0
    info = {
        "method": method,
        "target_bandwidth": float(target_bandwidth),
        "target_n_pad": int(target),
        "actual_n_pad": int(counts.sum()),
        "budget_error": int(counts.sum()) - int(target),
        "allowed_violation_count": violation,
        "allowed_violation_rate": float(violation / max(int(target), 1)),
        "template_entropy": normalized_template_entropy(counts),
        "mask_kind": "causal" if use_causal_mask else "optimistic",
        "dummy_outgoing_share": outgoing_share,
        "dummy_incoming_share": incoming_share,
        "direction_target_incoming_share": None if direction_target_incoming_share is None else float(direction_target_incoming_share),
        "direction_correction_strength": float(direction_correction_strength),
        "tam_flatten_strength": float(tam_flatten_strength),
        "tam_flatten_floor": float(tam_flatten_floor),
        "logit_temperature": float(logit_temperature),
        "logit_noise_std": float(logit_noise_std),
        "invalid": False,
    }
    if metadata:
        info.update(metadata)
    return PaddingTemplate(
        counts=counts.astype(np.int32),
        target_n_pad=int(target),
        actual_n_pad=int(counts.sum()),
        target_bandwidth=float(target_bandwidth),
        metadata=info,
    )


def _keep_top_counts(counts: np.ndarray, score: np.ndarray, keep_total: int) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.int32)
    keep_total = max(0, min(int(keep_total), int(counts.sum())))
    if keep_total == int(counts.sum()):
        return counts.copy()
    if keep_total <= 0:
        return np.zeros_like(counts, dtype=np.int32)
    units: list[tuple[float, int]] = []
    flat_counts = counts.reshape(-1)
    flat_score = score.reshape(-1)
    for index, count in enumerate(flat_counts.tolist()):
        for offset in range(max(0, int(count))):
            units.append((float(flat_score[index]) - offset * 1e-6, index))
    units.sort(key=lambda item: item[0], reverse=True)
    kept = np.zeros_like(flat_counts, dtype=np.int32)
    for _, index in units[:keep_total]:
        kept[int(index)] += 1
    return kept.reshape(counts.shape).astype(np.int32)


def refine_counts(
    counts: np.ndarray,
    c_leakage: np.ndarray,
    preference: np.ndarray,
    *,
    method: Literal["weighted", "greedy"] = "weighted",
    keep_ratio: float = 0.75,
) -> tuple[np.ndarray, dict]:
    counts = np.asarray(counts, dtype=np.int32)
    shape = counts.shape
    total_before = int(counts.sum())
    keep_total = int(round(total_before * float(keep_ratio)))
    leakage = _normalize(c_leakage, np.ones(shape, dtype=np.float32))
    pref = _normalize(preference, np.ones(shape, dtype=np.float32))
    allocation = _normalize(counts, np.ones(shape, dtype=np.float32))
    if method == "weighted":
        score = (counts.astype(np.float32) + 0.10) * (0.55 * leakage + 0.45 * pref + 0.05)
    elif method == "greedy":
        score = (counts.astype(np.float32) + 0.10) * (0.50 * leakage + 0.35 * pref + 0.15 * allocation)
    else:
        raise ValueError(f"Unsupported shrink method: {method!r}")
    refined = _keep_top_counts(counts, score, keep_total)
    total_after = int(refined.sum())
    report = {
        "method": method,
        "packets_before": total_before,
        "packets_after": total_after,
        "shrink_ratio": float(1.0 - total_after / max(total_before, 1)),
        "keep_ratio": float(total_after / max(total_before, 1)),
        "score_mean": float(score[counts > 0].mean()) if np.any(counts > 0) else 0.0,
    }
    return refined.astype(np.int32), report


def _patch_ids(length: int, patch_num: int) -> np.ndarray:
    if int(length) <= 0:
        return np.asarray([], dtype=np.int64)
    ids = np.floor(np.arange(int(length), dtype=np.float32) * int(patch_num) / max(int(length), 1))
    return np.clip(ids.astype(np.int64), 0, int(patch_num) - 1)


def _tam_patch_times(
    patch: int,
    count: int,
    patch_num: int,
    *,
    max_load_time: float,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    if int(count) <= 0:
        return np.asarray([], dtype=np.float32)
    edges = np.linspace(0.0, float(max_load_time), int(patch_num) + 1, dtype=np.float64)
    left = float(edges[int(patch)])
    right = float(edges[int(patch) + 1])
    low = max(left, 1e-6)
    high = max(right, low + 1e-6)
    if strategy == "random_jitter_in_patch":
        times = rng.uniform(low, high, size=int(count))
    elif strategy == "burst_aware":
        left_count = int(count) // 2
        right_count = int(count) - left_count
        eps = min((high - low) * 0.15, 1e-3)
        times = np.concatenate(
            [
                np.full(left_count, low + eps, dtype=np.float64),
                np.full(right_count, max(low + eps, high - eps), dtype=np.float64),
            ]
        )
    else:
        times = np.linspace(low, high, int(count) + 2, dtype=np.float64)[1:-1]
    return np.clip(times, 1e-6, max(float(max_load_time), 1e-6)).astype(np.float32)


def _tam_patch_bounds(patch: int, patch_num: int, *, max_load_time: float) -> tuple[float, float]:
    edges = np.linspace(0.0, float(max_load_time), int(patch_num) + 1, dtype=np.float64)
    left = float(edges[int(patch)])
    right = float(edges[int(patch) + 1])
    low = max(left, 1e-6)
    high = max(right, low + 1e-6)
    return low, high


def _bounded_jitter(width: float, jitter: float) -> float:
    return max(min(float(jitter), max(float(width) / 3.0, 1e-6)), 1e-6)


def _tam_run_cluster_times(
    count: int,
    low: float,
    high: float,
    *,
    rng: np.random.Generator,
    jitter: float,
    local_run_max: int,
    channel_phase: int = 0,
) -> np.ndarray:
    count = max(0, int(count))
    if count <= 0:
        return np.asarray([], dtype=np.float32)
    width = max(float(high) - float(low), 1e-6)
    run_max = max(1, int(local_run_max))
    groups = int(np.ceil(count / float(run_max)))
    edge_span = max(width * 0.18, 1e-6)
    noise = min(_bounded_jitter(width, float(jitter)), width * 0.015)
    phase = int(channel_phase) * min(width * 0.02, 1e-4)
    values: list[np.ndarray] = []
    remaining = count
    left_groups = max(1, int(np.ceil(groups / 2.0)))
    right_groups = max(1, groups // 2)
    for group in range(groups):
        group_size = min(run_max, remaining)
        remaining -= group_size
        if group % 2 == 0:
            step = ((group // 2) + 0.5) / float(left_groups)
            anchor = low + min(edge_span * step, width * 0.45)
        else:
            step = ((group // 2) + 0.5) / float(right_groups)
            anchor = high - min(edge_span * step, width * 0.45)
        offsets = rng.normal(0.0, noise, size=group_size) if noise > 1e-9 else np.zeros(group_size, dtype=np.float64)
        values.append(np.asarray(anchor + phase + offsets, dtype=np.float64))
    return np.clip(np.concatenate(values), low, high - 1e-7).astype(np.float32)


def _tam_obfuscation_times(
    patch: int,
    count: int,
    patch_num: int,
    *,
    max_load_time: float,
    strategy: str,
    slot_jitter: float,
    cluster_ratio: float,
    local_run_max: int,
    rng: np.random.Generator,
    channel_phase: int = 0,
) -> np.ndarray:
    count = max(0, int(count))
    if count <= 0:
        return np.asarray([], dtype=np.float32)
    low, high = _tam_patch_bounds(int(patch), int(patch_num), max_load_time=float(max_load_time))
    width = max(high - low, 1e-6)
    jitter = _bounded_jitter(width, float(slot_jitter))
    mode = str(strategy)
    if mode == "rayleigh_in_slot":
        times = low + rng.rayleigh(jitter, size=count)
    elif mode == "edge_clustered":
        times = _tam_run_cluster_times(
            count,
            low,
            high,
            rng=rng,
            jitter=jitter,
            local_run_max=int(local_run_max),
            channel_phase=int(channel_phase),
        )
    elif mode == "hybrid_clustered":
        clustered = int(round(count * float(np.clip(cluster_ratio, 0.0, 1.0))))
        clustered = max(0, min(count, clustered))
        parts: list[np.ndarray] = []
        if clustered > 0:
            parts.append(
                _tam_run_cluster_times(
                    clustered,
                    low,
                    high,
                    rng=rng,
                    jitter=jitter,
                    local_run_max=int(local_run_max),
                    channel_phase=int(channel_phase),
                )
            )
        remaining = count - clustered
        if remaining > 0:
            left = min(high - 1e-7, low + width * 0.10)
            right = max(left + 1e-7, high - width * 0.05)
            parts.append(rng.uniform(left, right, size=remaining).astype(np.float32))
        times = np.concatenate(parts) if parts else np.asarray([], dtype=np.float32)
    else:
        raise ValueError(
            f"Unsupported tam_obfuscation_strategy={strategy!r}; "
            "expected 'rayleigh_in_slot', 'edge_clustered', or 'hybrid_clustered'."
        )
    return np.clip(np.asarray(times, dtype=np.float32), low, high - 1e-7).astype(np.float32)


def _tam_hybrid_interleaved_events(
    out_count: int,
    in_count: int,
    low: float,
    high: float,
    *,
    rng: np.random.Generator,
    slot_jitter: float,
    cluster_ratio: float,
    local_run_max: int,
) -> tuple[list[float], list[int]]:
    total = max(0, int(out_count)) + max(0, int(in_count))
    if total <= 0:
        return [], [0, 0]
    cluster_total = int(round(total * float(np.clip(cluster_ratio, 0.0, 1.0))))
    cluster_total = max(0, min(total, cluster_total))
    if cluster_total <= 0:
        return [], [0, 0]
    out_cluster = min(int(out_count), int(round(cluster_total * max(int(out_count), 0) / float(total))))
    in_cluster = min(int(in_count), cluster_total - out_cluster)
    while out_cluster + in_cluster < cluster_total:
        if out_cluster < int(out_count):
            out_cluster += 1
        elif in_cluster < int(in_count):
            in_cluster += 1
        else:
            break
    remaining = [out_cluster, in_cluster]
    groups: list[tuple[int, int]] = []
    channel = 0
    run_max = max(1, int(local_run_max))
    while remaining[0] > 0 or remaining[1] > 0:
        if remaining[channel] <= 0:
            channel = 1 - channel
        group_size = min(run_max, remaining[channel])
        groups.append((channel, group_size))
        remaining[channel] -= group_size
        channel = 1 - channel
    if not groups:
        return [], [0, 0]
    width = max(float(high) - float(low), 1e-6)
    span = max(width * 0.35, 1e-6)
    noise = min(_bounded_jitter(width, float(slot_jitter)), width * 0.012)
    events: list[float] = []
    for group_index, (group_channel, group_size) in enumerate(groups):
        anchor = low + ((group_index + 0.5) / float(len(groups))) * span
        anchor += int(group_channel) * min(width * 0.02, 1e-4)
        offsets = rng.normal(0.0, noise, size=int(group_size)) if noise > 1e-9 else np.zeros(int(group_size))
        sign = 1.0 if int(group_channel) == 0 else -1.0
        for timestamp in np.clip(anchor + offsets, low, high - 1e-7).tolist():
            events.append(float(sign * max(float(timestamp), 1e-6)))
    return events, [int(out_cluster), int(in_cluster)]


def _append_tam_obfuscation_insertions(
    insertions: list[list[float]],
    orig_times: np.ndarray,
    counts: np.ndarray,
    *,
    max_load_time: float,
    strategy: str,
    slot_jitter: float,
    cluster_ratio: float,
    local_run_max: int,
    rng: np.random.Generator,
) -> None:
    patch_num = int(counts.shape[1])
    for patch in range(patch_num):
        out_count = int(max(counts[0, patch], 0)) if counts.shape[0] >= 1 else 0
        in_count = int(max(counts[1, patch], 0)) if counts.shape[0] >= 2 else 0
        events: list[float] = []
        used = [0, 0]
        if str(strategy) == "hybrid_clustered" and out_count > 0 and in_count > 0:
            low, high = _tam_patch_bounds(patch, patch_num, max_load_time=float(max_load_time))
            events, used = _tam_hybrid_interleaved_events(
                out_count,
                in_count,
                low,
                high,
                rng=rng,
                slot_jitter=float(slot_jitter),
                cluster_ratio=float(cluster_ratio),
                local_run_max=int(local_run_max),
            )
        for channel, sign, count, already_used in (
            (0, 1.0, out_count, used[0]),
            (1, -1.0, in_count, used[1]),
        ):
            remaining = max(0, int(count) - int(already_used))
            if remaining <= 0:
                continue
            residual_strategy = "rayleigh_in_slot" if str(strategy) == "hybrid_clustered" else str(strategy)
            times = _tam_obfuscation_times(
                patch,
                remaining,
                patch_num,
                max_load_time=float(max_load_time),
                strategy=residual_strategy,
                slot_jitter=float(slot_jitter),
                cluster_ratio=float(cluster_ratio),
                local_run_max=int(local_run_max),
                rng=rng,
                channel_phase=int(channel),
            )
            events.extend(float(sign * max(float(timestamp), 1e-6)) for timestamp in times.tolist())
        if not events:
            continue
        event_times = np.abs(np.asarray(events, dtype=np.float32))
        slots = np.searchsorted(orig_times, event_times, side="right") if orig_times.size else np.zeros(len(events), dtype=np.int64)
        for slot, value in zip(np.clip(slots.astype(np.int64), 0, len(insertions) - 1), events):
            insertions[int(slot)].append(float(value))


def _split_counts_by_shares(
    counts: np.ndarray,
    shares: Sequence[float],
    rng: np.random.Generator,
) -> list[np.ndarray]:
    counts = np.maximum(np.asarray(counts, dtype=np.int32), 0)
    total = int(counts.sum())
    parts = [np.zeros_like(counts, dtype=np.int32) for _ in shares]
    if total <= 0 or not shares:
        return parts
    raw_shares = np.maximum(np.asarray(list(shares), dtype=np.float64), 0.0)
    if float(raw_shares.sum()) <= 1e-12:
        raw_shares = np.ones(len(parts), dtype=np.float64)
    raw_shares = raw_shares / float(raw_shares.sum())
    raw_targets = raw_shares * float(total)
    targets = np.floor(raw_targets).astype(np.int64)
    remaining = int(total - int(targets.sum()))
    if remaining > 0:
        order = np.argsort(-(raw_targets - targets), kind="mergesort")
        targets[order[:remaining]] += 1
    flat_counts = counts.reshape(-1)
    units = np.repeat(np.arange(flat_counts.size, dtype=np.int64), flat_counts)
    if units.size:
        rng.shuffle(units)
    cursor = 0
    for part, target in zip(parts, targets.tolist()):
        take = max(0, int(target))
        chosen = units[cursor : cursor + take]
        cursor += take
        if chosen.size:
            bincount = np.bincount(chosen, minlength=flat_counts.size).astype(np.int32)
            part += bincount.reshape(counts.shape)
    return parts


def _append_trace_index_insertions(
    insertions: list[list[float]],
    clean: np.ndarray,
    counts: np.ndarray,
    *,
    coordinate_length: int,
    strategy: str,
    rng: np.random.Generator,
) -> None:
    counts = np.asarray(counts, dtype=np.int32)
    patch_num = int(counts.shape[1])
    orig_times = np.abs(clean).astype(np.float32)
    orig_dirs = np.sign(clean).astype(np.float32)
    index_edges = np.rint(np.linspace(0, int(coordinate_length), patch_num + 1)).astype(np.int64)
    index_edges = np.clip(index_edges, 0, clean.size)

    def dummy_time(slot: int) -> float:
        if clean.size == 0:
            return 1e-6
        if slot <= 0:
            return max(float(orig_times[0]) * 0.5, 1e-6)
        if slot >= clean.size:
            return max(float(orig_times[-1]) + 1e-6, 1e-6)
        left, right = float(orig_times[slot - 1]), float(orig_times[slot])
        return max((left + right) * 0.5 if right >= left else right, 1e-6)

    for patch in range(patch_num):
        start_idx = int(index_edges[patch])
        end_idx = int(index_edges[patch + 1])
        slot_hi = max(end_idx, start_idx + 1)
        for channel, sign in ((0, 1.0), (1, -1.0)):
            count = int(max(counts[channel, patch], 0))
            if count <= 0:
                continue
            if strategy == "random_jitter_in_patch":
                slots = rng.integers(start_idx, slot_hi + 1, size=count)
            elif strategy == "burst_aware":
                local_dirs = orig_dirs[start_idx:end_idx]
                anchors = np.flatnonzero(local_dirs == sign) + start_idx
                if anchors.size:
                    slots = rng.choice(anchors, size=count, replace=True)
                    slots = np.clip(slots + rng.integers(-1, 2, size=count), start_idx, slot_hi)
                else:
                    slots = rng.integers(start_idx, slot_hi + 1, size=count)
            else:
                slots = np.linspace(start_idx, slot_hi, count + 2, dtype=np.float64)[1:-1]
                slots = np.rint(slots).astype(np.int64)
            for slot in np.clip(slots.astype(np.int64), 0, clean.size):
                insertions[int(slot)].append(float(sign * dummy_time(int(slot))))


def _append_burst_obfuscation_insertions(
    insertions: list[list[float]],
    clean: np.ndarray,
    counts: np.ndarray,
    *,
    coordinate_length: int,
    local_run_max: int,
    rng: np.random.Generator,
) -> None:
    counts = np.asarray(counts, dtype=np.int32)
    patch_num = int(counts.shape[1])
    orig_times = np.abs(clean).astype(np.float32)
    orig_dirs = np.sign(clean).astype(np.float32)
    index_edges = np.rint(np.linspace(0, int(coordinate_length), patch_num + 1)).astype(np.int64)
    index_edges = np.clip(index_edges, 0, clean.size)
    change_points = np.flatnonzero(orig_dirs[1:] != orig_dirs[:-1]) + 1 if orig_dirs.size > 1 else np.asarray([], dtype=np.int64)
    run_max = max(1, int(local_run_max))

    def dummy_time(slot: int, offset: int = 0) -> float:
        if clean.size == 0:
            return 1e-6
        if slot <= 0:
            base = max(float(orig_times[0]) * 0.5, 1e-6)
        elif slot >= clean.size:
            base = max(float(orig_times[-1]) + 1e-6, 1e-6)
        else:
            left, right = float(orig_times[slot - 1]), float(orig_times[slot])
            base = max((left + right) * 0.5 if right >= left else right, 1e-6)
        return max(base + float(offset) * 1e-7, 1e-6)

    for patch in range(patch_num):
        start_idx = int(index_edges[patch])
        end_idx = int(index_edges[patch + 1])
        slot_hi = max(end_idx, start_idx + 1)
        local_changes = change_points[(change_points >= start_idx) & (change_points <= slot_hi)]
        for channel, sign in ((0, 1.0), (1, -1.0)):
            count = int(max(counts[channel, patch], 0))
            if count <= 0:
                continue
            local_dirs = orig_dirs[start_idx:end_idx]
            same_slots = np.flatnonzero(local_dirs == sign) + start_idx
            anchors: list[np.ndarray] = []
            if local_changes.size:
                boundary = np.concatenate([local_changes - 1, local_changes, local_changes + 1])
                anchors.append(np.clip(boundary, start_idx, slot_hi))
            if same_slots.size:
                anchors.append(same_slots)
            if anchors:
                candidates = np.unique(np.concatenate(anchors).astype(np.int64))
            else:
                candidates = np.arange(start_idx, slot_hi + 1, dtype=np.int64)
            candidates = candidates[(candidates >= start_idx) & (candidates <= slot_hi)]
            if candidates.size == 0:
                candidates = np.asarray([start_idx], dtype=np.int64)
            remaining = count
            group_offset = 0
            while remaining > 0:
                group_size = min(run_max, remaining)
                anchor = int(rng.choice(candidates))
                jitter = rng.integers(-1, 2, size=group_size)
                slots = np.clip(anchor + jitter, start_idx, slot_hi)
                for offset, slot in enumerate(slots.astype(np.int64).tolist()):
                    insertions[int(np.clip(slot, 0, clean.size))].append(float(sign * dummy_time(int(slot), group_offset + offset)))
                group_offset += group_size
                remaining -= group_size


def _append_multi_view_insertions(
    insertions: list[list[float]],
    clean: np.ndarray,
    counts: np.ndarray,
    *,
    coordinate_length: int,
    max_load_time: float,
    trace_index_share: float,
    burst_share: float,
    tam_share: float,
    trace_strategy: str,
    tam_strategy: str,
    tam_slot_jitter: float,
    tam_cluster_ratio: float,
    local_run_max: int,
    rng: np.random.Generator,
) -> dict[str, int]:
    df_counts, awf_counts, rf_counts = _split_counts_by_shares(
        counts,
        [float(trace_index_share), float(burst_share), float(tam_share)],
        rng,
    )
    _append_trace_index_insertions(
        insertions,
        clean,
        df_counts,
        coordinate_length=int(coordinate_length),
        strategy=str(trace_strategy),
        rng=rng,
    )
    _append_burst_obfuscation_insertions(
        insertions,
        clean,
        awf_counts,
        coordinate_length=int(coordinate_length),
        local_run_max=int(local_run_max),
        rng=rng,
    )
    _append_tam_obfuscation_insertions(
        insertions,
        np.abs(clean).astype(np.float32),
        rf_counts,
        max_load_time=float(max_load_time),
        strategy=str(tam_strategy),
        slot_jitter=float(tam_slot_jitter),
        cluster_ratio=float(tam_cluster_ratio),
        local_run_max=int(local_run_max),
        rng=rng,
    )
    return {
        "multi_view_df_dummy_count": int(df_counts.sum()),
        "multi_view_awf_dummy_count": int(awf_counts.sum()),
        "multi_view_rf_dummy_count": int(rf_counts.sum()),
    }


def _slot_time(clean: np.ndarray, orig_times: np.ndarray, slot: int) -> float:
    if clean.size == 0:
        return 1e-6
    if slot <= 0:
        return max(float(orig_times[0]) * 0.5, 1e-6)
    if slot >= clean.size:
        return max(float(orig_times[-1]) + 1e-6, 1e-6)
    left, right = float(orig_times[slot - 1]), float(orig_times[slot])
    return max((left + right) * 0.5 if right >= left else right, 1e-6)


def _slot_time_for_local_run(
    clean: np.ndarray,
    orig_times: np.ndarray,
    slot: int,
    local_index: int,
    group_size: int,
) -> float:
    """Return a dummy timestamp that stays inside the insertion gap for slot."""

    if clean.size == 0:
        return 1e-6
    slot = int(slot)
    local_index = int(local_index)
    group_size = max(1, int(group_size))
    frac = float(local_index + 1) / float(group_size + 1)
    if slot <= 0:
        right = float(orig_times[0])
        low = 1e-6
        high = max(right - 1e-8, low)
        return max(low + (high - low) * frac, 1e-6)
    if slot >= clean.size:
        return max(float(orig_times[-1]) + (local_index + 1) * 1e-6, 1e-6)
    left, right = float(orig_times[slot - 1]), float(orig_times[slot])
    low = max(left + 1e-8, 1e-6)
    high = right - 1e-8
    if high <= low:
        return _slot_time(clean, orig_times, slot)
    return max(low + (high - low) * frac, 1e-6)


def _candidate_slots_for_views(
    clean: np.ndarray,
    orig_times: np.ndarray,
    change_points: np.ndarray,
    patch: int,
    patch_num: int,
    *,
    coordinate_length: int,
    max_load_time: float,
) -> np.ndarray:
    index_edges = np.rint(np.linspace(0, int(coordinate_length), patch_num + 1)).astype(np.int64)
    index_edges = np.clip(index_edges, 0, clean.size)
    index_start = int(index_edges[int(patch)])
    index_end = int(index_edges[int(patch) + 1])
    index_hi = max(index_end, index_start + 1)
    index_candidates = np.linspace(index_start, index_hi, min(max(index_hi - index_start + 1, 1), 48), dtype=np.float64)
    index_candidates = np.rint(index_candidates).astype(np.int64)

    tam_low, tam_high = _tam_patch_bounds(int(patch), int(patch_num), max_load_time=float(max_load_time))
    if orig_times.size:
        time_start = int(np.searchsorted(orig_times, tam_low, side="left"))
        time_end = int(np.searchsorted(orig_times, tam_high, side="right"))
    else:
        time_start = 0
        time_end = 0
    time_hi = max(time_end, time_start + 1)
    time_candidates = np.linspace(time_start, time_hi, min(max(time_hi - time_start + 1, 1), 48), dtype=np.float64)
    time_candidates = np.rint(time_candidates).astype(np.int64)

    windows = []
    if change_points.size:
        for start, stop in ((index_start, index_hi), (time_start, time_hi)):
            local = change_points[(change_points >= max(start - 2, 0)) & (change_points <= min(stop + 2, clean.size))]
            if local.size:
                windows.append(np.concatenate([local - 1, local, local + 1]))
    fallback_time_slot = int(np.searchsorted(orig_times, (tam_low + tam_high) * 0.5, side="right")) if orig_times.size else 0
    pieces = [index_candidates, time_candidates, np.asarray([fallback_time_slot], dtype=np.int64), np.asarray([index_start, index_hi], dtype=np.int64)]
    pieces.extend(windows)
    candidates = np.unique(np.concatenate([np.asarray(piece, dtype=np.int64).reshape(-1) for piece in pieces]))
    return np.clip(candidates, 0, clean.size).astype(np.int64)


def _multi_view_slot_scores(
    clean: np.ndarray,
    orig_times: np.ndarray,
    change_points: np.ndarray,
    slots: np.ndarray,
    sign: float,
    patch: int,
    patch_num: int,
    *,
    max_load_time: float,
    df_share: float,
    awf_share: float,
    rf_share: float,
) -> np.ndarray:
    if slots.size == 0:
        return np.asarray([], dtype=np.float64)
    dirs = np.sign(clean).astype(np.float32)
    tam_low, tam_high = _tam_patch_bounds(int(patch), int(patch_num), max_load_time=float(max_load_time))
    tam_center = (tam_low + tam_high) * 0.5
    tam_width = max(tam_high - tam_low, 1e-6)
    raw_weights = np.maximum(np.asarray([df_share, awf_share, rf_share], dtype=np.float64), 0.0)
    if float(raw_weights.sum()) <= 1e-12:
        raw_weights = np.asarray([0.40, 0.30, 0.30], dtype=np.float64)
    weights = raw_weights / float(raw_weights.sum())
    scores = []
    for slot in np.asarray(slots, dtype=np.int64).tolist():
        left = dirs[slot - 1] if 0 < int(slot) <= dirs.size else 0.0
        right = dirs[slot] if 0 <= int(slot) < dirs.size else 0.0
        same_near = float(left == float(sign) or right == float(sign))
        opposite_near = float((left != 0.0 and left != float(sign)) or (right != 0.0 and right != float(sign)))
        boundary_here = float(left != 0.0 and right != 0.0 and left != right)
        df_score = 0.55 * same_near + 0.25 * opposite_near + 0.20 * boundary_here

        if change_points.size:
            distance = float(np.min(np.abs(change_points.astype(np.float64) - float(slot))))
            boundary_score = 1.0 / (1.0 + distance / 2.0)
        else:
            boundary_score = 0.0
        awf_score = 0.70 * boundary_score + 0.30 * same_near

        timestamp = _slot_time(clean, orig_times, int(slot))
        inside = float(tam_low <= timestamp <= tam_high)
        rf_score = inside + (1.0 - inside) * np.exp(-abs(timestamp - tam_center) / tam_width)
        scores.append(weights[0] * df_score + weights[1] * awf_score + weights[2] * rf_score)
    return np.asarray(scores, dtype=np.float64)


def _append_multi_view_fused_insertions(
    insertions: list[list[float]],
    clean: np.ndarray,
    counts: np.ndarray,
    *,
    coordinate_length: int,
    max_load_time: float,
    df_share: float,
    awf_share: float,
    rf_share: float,
    local_run_max: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    counts = np.asarray(counts, dtype=np.int32)
    patch_num = int(counts.shape[1])
    orig_times = np.abs(clean).astype(np.float32)
    orig_dirs = np.sign(clean).astype(np.float32)
    change_points = np.flatnonzero(orig_dirs[1:] != orig_dirs[:-1]) + 1 if orig_dirs.size > 1 else np.asarray([], dtype=np.int64)
    run_max = max(1, int(local_run_max))
    placed = 0
    score_sum = 0.0
    for patch in range(patch_num):
        candidates = _candidate_slots_for_views(
            clean,
            orig_times,
            change_points,
            patch,
            patch_num,
            coordinate_length=int(coordinate_length),
            max_load_time=float(max_load_time),
        )
        for channel, sign in ((0, 1.0), (1, -1.0)):
            count = int(max(counts[channel, patch], 0))
            if count <= 0:
                continue
            scores = _multi_view_slot_scores(
                clean,
                orig_times,
                change_points,
                candidates,
                sign,
                patch,
                patch_num,
                max_load_time=float(max_load_time),
                df_share=float(df_share),
                awf_share=float(awf_share),
                rf_share=float(rf_share),
            )
            if candidates.size == 0 or scores.size == 0:
                candidates = np.asarray([0], dtype=np.int64)
                scores = np.asarray([1.0], dtype=np.float64)
            order = np.argsort(-(scores + rng.normal(0.0, 1e-6, size=scores.shape)), kind="mergesort")
            top = candidates[order[: max(1, min(order.size, int(np.ceil(count / float(run_max))) + 2))]]
            remaining = count
            while remaining > 0:
                group_size = min(run_max, remaining)
                anchor = int(rng.choice(top))
                slots = np.clip(anchor + rng.integers(-1, 2, size=group_size), 0, clean.size)
                for local_index, slot in enumerate(slots.astype(np.int64).tolist()):
                    timestamp = _slot_time_for_local_run(clean, orig_times, int(slot), local_index, group_size)
                    insertions[int(slot)].append(float(sign * max(timestamp, 1e-6)))
                    placed += 1
                    score_sum += float(scores[np.argmin(np.abs(candidates - int(slot)))]) if candidates.size else 0.0
                remaining -= group_size
    return {
        "multi_view_mode": "fused",
        "multi_view_shared_dummy_count": int(placed),
        "multi_view_mean_slot_score": float(score_sum / max(int(placed), 1)),
    }


def _append_tam_insertions(
    insertions: list[list[float]],
    orig_times: np.ndarray,
    counts: np.ndarray,
    *,
    max_load_time: float,
    strategy: str,
    rng: np.random.Generator,
) -> None:
    patch_num = int(counts.shape[1])
    for patch in range(patch_num):
        for channel, sign in ((0, 1.0), (1, -1.0)):
            count = int(max(counts[channel, patch], 0))
            if count <= 0:
                continue
            times = _tam_patch_times(
                patch,
                count,
                patch_num,
                max_load_time=float(max_load_time),
                strategy=strategy,
                rng=rng,
            )
            slots = np.searchsorted(orig_times, times, side="right") if orig_times.size else np.zeros(len(times), dtype=np.int64)
            for slot, timestamp in zip(np.clip(slots.astype(np.int64), 0, len(insertions) - 1), times):
                insertions[int(slot)].append(float(sign * max(float(timestamp), 1e-6)))


def renderer_options_from_config(cfg: Any) -> dict[str, Any]:
    return {
        "strategy": str(getattr(cfg, "insertion_strategy", "uniform_in_patch")),
        "coordinate": str(getattr(cfg, "render_coordinate", "rf_tam")),
        "max_load_time": float(getattr(cfg, "surrogate_rf_max_load_time", 80.0)),
        "tam_obfuscation_strategy": str(getattr(cfg, "tam_obfuscation_strategy", "hybrid_clustered")),
        "tam_slot_jitter": float(getattr(cfg, "tam_slot_jitter", 0.03)),
        "tam_cluster_ratio": float(getattr(cfg, "tam_cluster_ratio", 0.70)),
        "tam_local_run_max": int(getattr(cfg, "tam_local_run_max", 8)),
        "tam_preserve_real_timestamps": bool(getattr(cfg, "tam_preserve_real_timestamps", True)),
        "multi_view_mode": str(getattr(cfg, "multi_view_mode", "fused")),
        "multi_view_df_share": float(getattr(cfg, "multi_view_df_share", 0.40)),
        "multi_view_awf_share": float(getattr(cfg, "multi_view_awf_share", 0.30)),
        "multi_view_rf_share": float(getattr(cfg, "multi_view_rf_share", 0.30)),
    }


def render_trace(
    clean_trace: np.ndarray,
    template: PaddingTemplate,
    *,
    seed: int = 0,
    max_trace_length: int = 5000,
    strategy: str = "uniform_in_patch",
    coordinate: str = "rf_tam",
    max_load_time: float = 80.0,
    tam_obfuscation_strategy: str = "hybrid_clustered",
    tam_slot_jitter: float = 0.03,
    tam_cluster_ratio: float = 0.70,
    tam_local_run_max: int = 8,
    tam_preserve_real_timestamps: bool = True,
    multi_view_mode: str = "fused",
    multi_view_df_share: float = 0.40,
    multi_view_awf_share: float = 0.30,
    multi_view_rf_share: float = 0.30,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    if str(coordinate) in {"tam_obfuscation", "multi_view"} and not bool(tam_preserve_real_timestamps):
        raise ValueError(f"{coordinate} renderer requires tam_preserve_real_timestamps=True.")
    rng = np.random.default_rng(int(seed))
    clean = nonzero_trace(clean_trace)
    counts = np.asarray(template.counts, dtype=np.int32)
    patch_num = int(counts.shape[1])
    orig_times = np.abs(clean).astype(np.float32)
    index_edges = np.rint(np.linspace(0, clean.size, patch_num + 1)).astype(np.int64)
    insertions: list[list[float]] = [[] for _ in range(clean.size + 1)]
    renderer_stats: dict[str, Any] = {}

    def dummy_time(slot: int) -> float:
        if clean.size == 0:
            return 1e-6
        if slot <= 0:
            return max(float(orig_times[0]) * 0.5, 1e-6)
        if slot >= clean.size:
            return max(float(orig_times[-1]) + 1e-6, 1e-6)
        left = float(orig_times[slot - 1])
        right = float(orig_times[slot])
        if right >= left:
            return max((left + right) * 0.5, 1e-6)
        return max(right, 1e-6)

    if str(coordinate) == "rf_tam":
        _append_tam_insertions(
            insertions,
            orig_times,
            counts,
            max_load_time=float(max_load_time),
            strategy=strategy,
            rng=rng,
        )
    elif str(coordinate) == "tam_obfuscation":
        _append_tam_obfuscation_insertions(
            insertions,
            orig_times,
            counts,
            max_load_time=float(max_load_time),
            strategy=str(tam_obfuscation_strategy),
            slot_jitter=float(tam_slot_jitter),
            cluster_ratio=float(tam_cluster_ratio),
            local_run_max=int(tam_local_run_max),
            rng=rng,
        )
    elif str(coordinate) == "multi_view":
        if str(multi_view_mode) == "split":
            renderer_stats.update(
                _append_multi_view_insertions(
                    insertions,
                    clean.astype(np.float32),
                    counts,
                    coordinate_length=max(int(clean.size), 1),
                    max_load_time=float(max_load_time),
                    trace_index_share=float(multi_view_df_share),
                    burst_share=float(multi_view_awf_share),
                    tam_share=float(multi_view_rf_share),
                    trace_strategy="burst_aware",
                    tam_strategy=str(tam_obfuscation_strategy),
                    tam_slot_jitter=float(tam_slot_jitter),
                    tam_cluster_ratio=float(tam_cluster_ratio),
                    local_run_max=int(tam_local_run_max),
                    rng=rng,
                )
            )
            renderer_stats["multi_view_mode"] = "split"
        elif str(multi_view_mode) == "fused":
            renderer_stats.update(
                _append_multi_view_fused_insertions(
                    insertions,
                    clean.astype(np.float32),
                    counts,
                    coordinate_length=max(int(clean.size), 1),
                    max_load_time=float(max_load_time),
                    df_share=float(multi_view_df_share),
                    awf_share=float(multi_view_awf_share),
                    rf_share=float(multi_view_rf_share),
                    local_run_max=int(tam_local_run_max),
                    rng=rng,
                )
            )
        else:
            raise ValueError("Unsupported multi_view_mode={!r}; expected 'fused' or 'split'.".format(multi_view_mode))
    else:
        for patch in range(patch_num):
            start_idx = int(index_edges[patch])
            end_idx = int(index_edges[patch + 1])
            slot_hi = max(end_idx, start_idx + 1)
            for channel, sign in [(0, 1.0), (1, -1.0)]:
                count = int(max(counts[channel, patch], 0))
                if count <= 0:
                    continue
                if strategy == "random_jitter_in_patch":
                    slots = rng.integers(start_idx, slot_hi + 1, size=count)
                elif strategy == "burst_aware":
                    left_count = count // 2
                    right_count = count - left_count
                    slots = np.concatenate(
                        [
                            np.full(left_count, start_idx, dtype=np.int64),
                            np.full(right_count, slot_hi, dtype=np.int64),
                        ]
                    )
                else:
                    slots = np.linspace(start_idx, slot_hi, count + 2, dtype=np.float64)[1:-1]
                    slots = np.clip(np.rint(slots).astype(np.int64), 0, clean.size)
                for slot in np.clip(slots.astype(np.int64), 0, clean.size):
                    insertions[int(slot)].append(float(sign * dummy_time(int(slot))))

    output: list[float] = []
    output_is_original: list[bool] = []
    for index, value in enumerate(clean.astype(float).tolist()):
        row_insertions = sorted(insertions[index], key=lambda value: abs(value))
        output.extend(row_insertions)
        output_is_original.extend([False] * len(row_insertions))
        output.append(float(value))
        output_is_original.append(True)
    tail_insertions = sorted(insertions[-1], key=lambda value: abs(value))
    output.extend(tail_insertions)
    output_is_original.extend([False] * len(tail_insertions))
    result = np.zeros(int(max_trace_length), dtype=np.float32)
    original_retained = 0
    visible_dummy = 0
    if output:
        arr = np.asarray(output[: int(max_trace_length)], dtype=np.float32)
        result[: arr.size] = arr
        mask = np.asarray(output_is_original[: arr.size], dtype=bool)
        original_retained = int(mask.sum())
        visible_dummy = int(arr.size - original_retained)
    if return_stats:
        return result, {
            "original_count": int(clean.size),
            "visible_count": int(nonzero_trace(result).size),
            "visible_dummy_count": int(visible_dummy),
            "original_retained_count": int(original_retained),
            "target_dummy_count": int(template.actual_n_pad),
            "overflow": int(clean.size + int(template.actual_n_pad) > int(max_trace_length)),
            "render_coordinate": str(coordinate),
            **renderer_stats,
        }
    return result


def render_batch(
    clean_raw: np.ndarray,
    templates: Sequence[PaddingTemplate],
    *,
    seed: int = 0,
    max_trace_length: int = 5000,
    strategy: str = "uniform_in_patch",
    coordinate: str = "rf_tam",
    max_load_time: float = 80.0,
    tam_obfuscation_strategy: str = "hybrid_clustered",
    tam_slot_jitter: float = 0.03,
    tam_cluster_ratio: float = 0.70,
    tam_local_run_max: int = 8,
    tam_preserve_real_timestamps: bool = True,
    multi_view_mode: str = "fused",
    multi_view_df_share: float = 0.40,
    multi_view_awf_share: float = 0.30,
    multi_view_rf_share: float = 0.30,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    defended = np.zeros((len(templates), int(max_trace_length)), dtype=np.float32)
    bandwidth = np.zeros(len(templates), dtype=np.float32)
    latency = np.zeros(len(templates), dtype=np.float32)
    overflow = np.zeros(len(templates), dtype=np.float32)
    original_retained = np.zeros(len(templates), dtype=np.float32)
    for index, (trace, template) in enumerate(zip(clean_raw, templates)):
        rendered, row_stats = render_trace(
            trace,
            template,
            seed=int(seed) + index,
            max_trace_length=int(max_trace_length),
            strategy=strategy,
            coordinate=str(coordinate),
            max_load_time=float(max_load_time),
            tam_obfuscation_strategy=str(tam_obfuscation_strategy),
            tam_slot_jitter=float(tam_slot_jitter),
            tam_cluster_ratio=float(tam_cluster_ratio),
            tam_local_run_max=int(tam_local_run_max),
            tam_preserve_real_timestamps=bool(tam_preserve_real_timestamps),
            multi_view_mode=str(multi_view_mode),
            multi_view_df_share=float(multi_view_df_share),
            multi_view_awf_share=float(multi_view_awf_share),
            multi_view_rf_share=float(multi_view_rf_share),
            return_stats=True,
        )
        defended[index] = rendered
        clean_count = max(int(nonzero_trace(trace).size), 1)
        bandwidth[index] = int(row_stats["visible_dummy_count"]) / float(clean_count)
        overflow[index] = float(row_stats["overflow"])
        original_retained[index] = int(row_stats["original_retained_count"]) / float(clean_count)
    return defended, {
        "bandwidth": bandwidth,
        "latency_overhead": latency,
        "overflow": overflow,
        "original_retained_ratio": original_retained,
    }


def aggregate_template_stats(stats: dict[str, np.ndarray], templates: Sequence[PaddingTemplate]) -> dict[str, float]:
    violations = [float(template.metadata.get("allowed_violation_rate", 0.0)) for template in templates]
    entropies = [float(template.metadata.get("template_entropy", 0.0)) for template in templates]
    return {
        "bandwidth_overhead": float(np.mean(stats.get("bandwidth", np.asarray([0.0])))),
        "delay_overhead": float(np.mean(stats.get("latency_overhead", np.asarray([0.0])))),
        "clip_rate": float(np.mean(stats.get("overflow", np.asarray([0.0])))),
        "original_retained_ratio": float(np.mean(stats.get("original_retained_ratio", np.asarray([1.0])))),
        "allowed_mask_violation_rate": float(np.mean(violations)) if violations else 0.0,
        "template_entropy": float(np.mean(entropies)) if entropies else 0.0,
        "generated_traces": int(len(templates)),
    }


def render_trace_variable(
    clean_trace: np.ndarray,
    template: PaddingTemplate,
    *,
    seed: int = 0,
    strategy: str = "uniform_in_patch",
    coordinate: str = "rf_tam",
    coordinate_length: int = 5000,
    max_load_time: float = 80.0,
    tam_obfuscation_strategy: str = "hybrid_clustered",
    tam_slot_jitter: float = 0.03,
    tam_cluster_ratio: float = 0.70,
    tam_local_run_max: int = 8,
    tam_preserve_real_timestamps: bool = True,
    multi_view_mode: str = "fused",
    multi_view_df_share: float = 0.40,
    multi_view_awf_share: float = 0.30,
    multi_view_rf_share: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Insert dummies without cropping; the returned origin mask proves real-packet retention."""

    if str(coordinate) in {"tam_obfuscation", "multi_view"} and not bool(tam_preserve_real_timestamps):
        raise ValueError(f"{coordinate} renderer requires tam_preserve_real_timestamps=True.")
    rng = np.random.default_rng(int(seed))
    clean = nonzero_trace(clean_trace).astype(np.float32)
    counts = np.asarray(template.counts, dtype=np.int32)
    patch_num = int(counts.shape[1])
    orig_times = np.abs(clean).astype(np.float32)
    index_edges = np.rint(np.linspace(0, int(coordinate_length), patch_num + 1)).astype(np.int64)
    index_edges = np.clip(index_edges, 0, clean.size)
    insertions: list[list[float]] = [[] for _ in range(clean.size + 1)]
    renderer_stats: dict[str, Any] = {}

    def dummy_time(slot: int) -> float:
        if clean.size == 0:
            return 1e-6
        if slot <= 0:
            return max(float(orig_times[0]) * 0.5, 1e-6)
        if slot >= clean.size:
            return max(float(orig_times[-1]) + 1e-6, 1e-6)
        left, right = float(orig_times[slot - 1]), float(orig_times[slot])
        return max((left + right) * 0.5 if right >= left else right, 1e-6)

    if str(coordinate) == "rf_tam":
        _append_tam_insertions(
            insertions,
            orig_times,
            counts,
            max_load_time=float(max_load_time),
            strategy=strategy,
            rng=rng,
        )
    elif str(coordinate) == "tam_obfuscation":
        _append_tam_obfuscation_insertions(
            insertions,
            orig_times,
            counts,
            max_load_time=float(max_load_time),
            strategy=str(tam_obfuscation_strategy),
            slot_jitter=float(tam_slot_jitter),
            cluster_ratio=float(tam_cluster_ratio),
            local_run_max=int(tam_local_run_max),
            rng=rng,
        )
    elif str(coordinate) == "multi_view":
        if str(multi_view_mode) == "split":
            renderer_stats.update(
                _append_multi_view_insertions(
                    insertions,
                    clean.astype(np.float32),
                    counts,
                    coordinate_length=int(coordinate_length),
                    max_load_time=float(max_load_time),
                    trace_index_share=float(multi_view_df_share),
                    burst_share=float(multi_view_awf_share),
                    tam_share=float(multi_view_rf_share),
                    trace_strategy="burst_aware",
                    tam_strategy=str(tam_obfuscation_strategy),
                    tam_slot_jitter=float(tam_slot_jitter),
                    tam_cluster_ratio=float(tam_cluster_ratio),
                    local_run_max=int(tam_local_run_max),
                    rng=rng,
                )
            )
            renderer_stats["multi_view_mode"] = "split"
        elif str(multi_view_mode) == "fused":
            renderer_stats.update(
                _append_multi_view_fused_insertions(
                    insertions,
                    clean.astype(np.float32),
                    counts,
                    coordinate_length=int(coordinate_length),
                    max_load_time=float(max_load_time),
                    df_share=float(multi_view_df_share),
                    awf_share=float(multi_view_awf_share),
                    rf_share=float(multi_view_rf_share),
                    local_run_max=int(tam_local_run_max),
                    rng=rng,
                )
            )
        else:
            raise ValueError("Unsupported multi_view_mode={!r}; expected 'fused' or 'split'.".format(multi_view_mode))
    else:
        for patch in range(patch_num):
            start_idx = int(index_edges[patch])
            end_idx = int(index_edges[patch + 1])
            slot_hi = max(end_idx, start_idx + 1)
            for channel, sign in ((0, 1.0), (1, -1.0)):
                count = int(max(counts[channel, patch], 0))
                if count <= 0:
                    continue
                if strategy == "random_jitter_in_patch":
                    slots = rng.integers(start_idx, slot_hi + 1, size=count)
                elif strategy == "burst_aware":
                    left_count = count // 2
                    slots = np.concatenate(
                        [np.full(left_count, start_idx, dtype=np.int64), np.full(count - left_count, slot_hi, dtype=np.int64)]
                    )
                else:
                    slots = np.linspace(start_idx, slot_hi, count + 2, dtype=np.float64)[1:-1]
                    slots = np.rint(slots).astype(np.int64)
                for slot in np.clip(slots, 0, clean.size):
                    insertions[int(slot)].append(float(sign * dummy_time(int(slot))))

    output: list[float] = []
    origin: list[bool] = []
    for index, value in enumerate(clean.astype(float).tolist()):
        row_insertions = sorted(insertions[index], key=lambda value: abs(value))
        output.extend(row_insertions)
        origin.extend([False] * len(row_insertions))
        output.append(float(value))
        origin.append(True)
    tail_insertions = sorted(insertions[-1], key=lambda value: abs(value))
    output.extend(tail_insertions)
    origin.extend([False] * len(tail_insertions))
    trace = np.asarray(output, dtype=np.float32)
    origin_mask = np.asarray(origin, dtype=bool)
    return trace, origin_mask, {
        "original_count": int(clean.size),
        "original_retained_count": int(origin_mask.sum()),
        "dummy_count": int((~origin_mask).sum()),
        "raw_count": int(trace.size),
        "render_coordinate": str(coordinate),
        **renderer_stats,
    }


def render_batch_variable(
    clean_raw: np.ndarray,
    templates: Sequence[PaddingTemplate],
    *,
    seeds: Sequence[int] | None = None,
    seed: int = 0,
    strategy: str = "uniform_in_patch",
    coordinate: str = "rf_tam",
    coordinate_length: int = 5000,
    max_load_time: float = 80.0,
    tam_obfuscation_strategy: str = "hybrid_clustered",
    tam_slot_jitter: float = 0.03,
    tam_cluster_ratio: float = 0.70,
    tam_local_run_max: int = 8,
    tam_preserve_real_timestamps: bool = True,
    multi_view_mode: str = "fused",
    multi_view_df_share: float = 0.40,
    multi_view_awf_share: float = 0.30,
    multi_view_rf_share: float = 0.30,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]:
    traces: list[np.ndarray] = []
    origins: list[np.ndarray] = []
    raw_bandwidth, raw_retention, raw_lengths = [], [], []
    for index, (clean, template) in enumerate(zip(clean_raw, templates)):
        row_seed = int(seeds[index]) if seeds is not None else int(seed) + index
        trace, origin, stats = render_trace_variable(
            clean,
            template,
            seed=row_seed,
            strategy=strategy,
            coordinate=str(coordinate),
            coordinate_length=int(coordinate_length),
            max_load_time=float(max_load_time),
            tam_obfuscation_strategy=str(tam_obfuscation_strategy),
            tam_slot_jitter=float(tam_slot_jitter),
            tam_cluster_ratio=float(tam_cluster_ratio),
            tam_local_run_max=int(tam_local_run_max),
            tam_preserve_real_timestamps=bool(tam_preserve_real_timestamps),
            multi_view_mode=str(multi_view_mode),
            multi_view_df_share=float(multi_view_df_share),
            multi_view_awf_share=float(multi_view_awf_share),
            multi_view_rf_share=float(multi_view_rf_share),
        )
        traces.append(trace)
        origins.append(origin)
        raw_bandwidth.append(stats["dummy_count"] / max(stats["original_count"], 1))
        raw_retention.append(stats["original_retained_count"] / max(stats["original_count"], 1))
        raw_lengths.append(stats["raw_count"])
    return traces, origins, {
        "raw_bandwidth": np.asarray(raw_bandwidth, dtype=np.float32),
        "raw_real_packet_retention": np.asarray(raw_retention, dtype=np.float32),
        "raw_lengths": np.asarray(raw_lengths, dtype=np.int32),
    }


def save_ragged_npz(
    path: str | Path,
    traces: Sequence[np.ndarray],
    origins: Sequence[np.ndarray],
    *,
    y: np.ndarray,
    **arrays: np.ndarray,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.asarray([len(row) for row in traces], dtype=np.int64)
    offsets = np.concatenate([np.asarray([0], dtype=np.int64), np.cumsum(lengths, dtype=np.int64)])
    flat = np.concatenate([np.asarray(row, dtype=np.float32) for row in traces]) if len(traces) else np.asarray([], dtype=np.float32)
    origin_flat = np.concatenate([np.asarray(row, dtype=np.uint8) for row in origins]) if len(origins) else np.asarray([], dtype=np.uint8)
    np.savez_compressed(target, flat=flat, offsets=offsets, origin_flat=origin_flat, y=np.asarray(y, dtype=np.int64), **arrays)


def load_ragged_npz(path: str | Path) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, np.ndarray]]:
    with np.load(Path(path), allow_pickle=False) as payload:
        flat = np.asarray(payload["flat"], dtype=np.float32)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        origin_flat = np.asarray(payload["origin_flat"], dtype=np.uint8)
        traces = [flat[offsets[index] : offsets[index + 1]].copy() for index in range(len(offsets) - 1)]
        origins = [origin_flat[offsets[index] : offsets[index + 1]].astype(bool, copy=True) for index in range(len(offsets) - 1)]
        metadata = {key: np.asarray(payload[key]) for key in payload.files if key not in {"flat", "offsets", "origin_flat"}}
    return traces, origins, metadata


def crop_ragged_for_attacker(
    traces: Sequence[np.ndarray],
    origins: Sequence[np.ndarray],
    max_trace_length: int = 5000,
) -> tuple[np.ndarray, dict[str, float]]:
    output = np.zeros((len(traces), int(max_trace_length)), dtype=np.float32)
    retentions, visible_bandwidth, clipped = [], [], []
    for index, (trace, origin) in enumerate(zip(traces, origins)):
        take = min(len(trace), int(max_trace_length))
        output[index, :take] = np.asarray(trace[:take], dtype=np.float32)
        original_total = max(int(np.asarray(origin, dtype=bool).sum()), 1)
        retained = int(np.asarray(origin[:take], dtype=bool).sum())
        visible_dummy = int(take - retained)
        retentions.append(retained / original_total)
        visible_bandwidth.append(visible_dummy / original_total)
        clipped.append(float(len(trace) > int(max_trace_length)))
    return output, {
        "attacker_input_real_packet_retention": float(np.mean(retentions)) if retentions else 1.0,
        "visible_dummy_overhead": float(np.mean(visible_bandwidth)) if visible_bandwidth else 0.0,
        "clip_rate": float(np.mean(clipped)) if clipped else 0.0,
    }

