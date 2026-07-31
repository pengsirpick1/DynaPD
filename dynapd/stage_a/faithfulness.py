"""Faithfulness tests for Stage A keypoint masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .mask_ops import entropy_np, js_divergence_np
from .modeling import StageAAttacker


METHODS = ("dynamask", "random", "random_block", "magnitude", "early")


@dataclass
class FaithfulnessMetrics:
    accuracy: float
    flip_rate: float
    js_div: float
    top1_drop: float
    entropy_gain: float
    top1_preservation: float


def hard_top_ratio_masks(scores: np.ndarray, ratio: float) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected scores [N, 2, W], got {values.shape}")
    flat = values.reshape(values.shape[0], -1)
    keep = max(1, min(flat.shape[1], int(round(flat.shape[1] * float(ratio)))))
    out = np.zeros_like(flat, dtype=np.float32)
    order = np.argsort(-flat, axis=1, kind="mergesort")
    for row in range(flat.shape[0]):
        out[row, order[row, :keep]] = 1.0
    return out.reshape(values.shape)


def random_ratio_masks(shape: tuple[int, int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    n, channels, width = shape
    total = channels * width
    keep = max(1, min(total, int(round(total * float(ratio)))))
    out = np.zeros((n, total), dtype=np.float32)
    for row in range(n):
        out[row, rng.choice(total, size=keep, replace=False)] = 1.0
    return out.reshape(shape)


def random_block_ratio_masks(shape: tuple[int, int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    n, channels, width = shape
    total = channels * width
    keep = max(1, min(total, int(round(total * float(ratio)))))
    block_width = max(1, min(width, int(np.ceil(keep / float(channels)))))
    out = np.zeros((n, channels, width), dtype=np.float32)
    for row in range(n):
        start = int(rng.integers(0, max(1, width - block_width + 1)))
        candidates = []
        for slot in range(start, min(width, start + block_width)):
            for direction in range(channels):
                candidates.append(direction * width + slot)
        candidates = np.asarray(candidates, dtype=np.int64)
        if candidates.size >= keep:
            chosen = rng.choice(candidates, size=keep, replace=False)
        else:
            remaining = np.setdiff1d(np.arange(total, dtype=np.int64), candidates, assume_unique=False)
            extra = rng.choice(remaining, size=keep - candidates.size, replace=False)
            chosen = np.concatenate([candidates, extra])
        out.reshape(n, total)[row, chosen] = 1.0
    return out


def earliest_ratio_masks(shape: tuple[int, int, int], ratio: float) -> np.ndarray:
    n, channels, width = shape
    total = channels * width
    keep = max(1, min(total, int(round(total * float(ratio)))))
    time_major = np.asarray([direction * width + slot for slot in range(width) for direction in range(channels)], dtype=np.int64)
    flat = np.zeros((n, total), dtype=np.float32)
    flat[:, time_major[:keep]] = 1.0
    return flat.reshape(shape)


def method_ratio_masks(
    method: str,
    *,
    soft_mask: np.ndarray,
    tam: np.ndarray,
    ratio: float,
    rng: np.random.Generator,
) -> np.ndarray:
    name = str(method).lower()
    if name == "dynamask":
        return hard_top_ratio_masks(soft_mask, float(ratio))
    if name == "random":
        return random_ratio_masks(soft_mask.shape, float(ratio), rng)
    if name == "random_block":
        return random_block_ratio_masks(soft_mask.shape, float(ratio), rng)
    if name == "magnitude":
        return hard_top_ratio_masks(np.abs(np.asarray(tam, dtype=np.float32)), float(ratio))
    if name == "early":
        return earliest_ratio_masks(soft_mask.shape, float(ratio))
    raise ValueError(f"Unknown faithfulness baseline method={method!r}")


def apply_deletion(tam: np.ndarray, baseline: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    return ((1.0 - hard_mask) * tam + hard_mask * baseline).astype(np.float32)


def apply_keep_only(tam: np.ndarray, baseline: np.ndarray, hard_mask: np.ndarray) -> np.ndarray:
    return (hard_mask * tam + (1.0 - hard_mask) * baseline).astype(np.float32)


@torch.no_grad()
def predict_probabilities(
    attacker: StageAAttacker,
    tam: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    rows = []
    values = np.asarray(tam, dtype=np.float32)
    for start in range(0, len(values), max(1, int(batch_size))):
        end = min(start + max(1, int(batch_size)), len(values))
        xb = torch.as_tensor(values[start:end], dtype=torch.float32, device=device)
        probs = torch.softmax(attacker.logits(xb), dim=1)
        rows.append(probs.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0).astype(np.float32)


def score_probabilities(
    original_prob: np.ndarray,
    evaluated_prob: np.ndarray,
    labels: np.ndarray,
) -> FaithfulnessMetrics:
    sample = sample_probability_metrics(original_prob, evaluated_prob, labels)
    return FaithfulnessMetrics(
        accuracy=float(np.mean(sample["correct"])),
        flip_rate=float(np.mean(sample["flip"])),
        js_div=float(np.mean(sample["js_div"])),
        top1_drop=float(np.mean(sample["top1_drop"])),
        entropy_gain=float(np.mean(sample["entropy_gain"])),
        top1_preservation=float(np.mean(sample["top1_preservation"])),
    )


def sample_probability_metrics(
    original_prob: np.ndarray,
    evaluated_prob: np.ndarray,
    labels: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return per-sample faithfulness metrics used by clustering audits."""
    original = np.asarray(original_prob, dtype=np.float32)
    evaluated = np.asarray(evaluated_prob, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    original_pred = original.argmax(axis=1)
    evaluated_pred = evaluated.argmax(axis=1)
    original_top1 = original[np.arange(len(original)), original_pred]
    evaluated_original_top1 = evaluated[np.arange(len(original)), original_pred]
    drop = original_top1 - evaluated_original_top1
    return {
        "correct": (evaluated_pred == y).astype(np.float32),
        "flip": (evaluated_pred != original_pred).astype(np.float32),
        "js_div": js_divergence_np(original, evaluated).astype(np.float32),
        "top1_drop": drop.astype(np.float32),
        "entropy_gain": (entropy_np(evaluated) - entropy_np(original)).astype(np.float32),
        "top1_preservation": (evaluated_original_top1 / np.maximum(original_top1, 1e-8)).astype(np.float32),
        "original_pred": original_pred.astype(np.int64),
        "evaluated_pred": evaluated_pred.astype(np.int64),
    }


def aopc(rows: list[dict], method: str, metric: str = "top1_drop") -> float:
    values = [float(row[metric]) for row in rows if str(row.get("method")) == str(method) and str(row.get("mode")) == "necessity"]
    if not values:
        return 0.0
    return float(np.mean(values))
