"""Mask operations and scalar metrics for Stage A."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def local_average_baseline_np(tam: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    values = np.asarray(tam, dtype=np.float32)
    if values.ndim == 2:
        values = values[None, :, :]
        squeeze = True
    else:
        squeeze = False
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected TAM [N, 2, W] or [2, W], got {values.shape}")
    kernel = max(1, int(kernel_size))
    pad = kernel // 2
    padded = np.pad(values, ((0, 0), (0, 0), (pad, pad)), mode="edge")
    out = np.empty_like(values, dtype=np.float32)
    for offset in range(values.shape[2]):
        out[:, :, offset] = padded[:, :, offset : offset + kernel].mean(axis=2)
    return out[0] if squeeze else out


def local_average_baseline_torch(tam: torch.Tensor, kernel_size: int = 9) -> torch.Tensor:
    kernel = max(1, int(kernel_size))
    if kernel == 1:
        return tam.detach().clone()
    pad_left = kernel // 2
    pad_right = kernel - 1 - pad_left
    padded = F.pad(tam, (pad_left, pad_right), mode="replicate")
    weight = torch.ones((2, 1, kernel), dtype=tam.dtype, device=tam.device) / float(kernel)
    return F.conv1d(padded, weight, groups=2).detach()


def deletion_tam(tam: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return ((1.0 - mask) * tam + mask * baseline).astype(np.float32)


def keep_keypoint_tam(tam: np.ndarray, baseline: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return (mask * tam + (1.0 - mask) * baseline).astype(np.float32)


def entropy_np(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    return -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=-1)


def js_divergence_np(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    left = np.asarray(p, dtype=np.float64)
    right = np.asarray(q, dtype=np.float64)
    midpoint = 0.5 * (left + right)
    kl_left = (left * (np.log(np.clip(left, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0)))).sum(axis=-1)
    kl_right = (right * (np.log(np.clip(right, 1e-12, 1.0)) - np.log(np.clip(midpoint, 1e-12, 1.0)))).sum(axis=-1)
    return (0.5 * (kl_left + kl_right)).astype(np.float32)


def hard_topk_mask(mask: np.ndarray, keep_ratio: float) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    flat = values.reshape(values.shape[0], -1) if values.ndim == 3 else values.reshape(1, -1)
    keep = max(1, int(round(flat.shape[1] * float(keep_ratio))))
    result = np.zeros_like(flat, dtype=np.float32)
    order = np.argsort(-flat, axis=1, kind="mergesort")
    for row in range(flat.shape[0]):
        result[row, order[row, :keep]] = 1.0
    result = result.reshape(values.shape if values.ndim == 3 else (1, *values.shape))
    return result if values.ndim == 3 else result[0]


def topk_overlap(a: np.ndarray, b: np.ndarray, keep_ratio: float) -> float:
    first = hard_topk_mask(np.asarray(a, dtype=np.float32), float(keep_ratio)).reshape(-1) > 0
    second = hard_topk_mask(np.asarray(b, dtype=np.float32), float(keep_ratio)).reshape(-1) > 0
    intersection = np.logical_and(first, second).sum()
    denominator = max(1, min(first.sum(), second.sum()))
    return float(intersection / denominator)


def topk_iou(a: np.ndarray, b: np.ndarray, keep_ratio: float) -> float:
    first = hard_topk_mask(np.asarray(a, dtype=np.float32), float(keep_ratio)).reshape(-1) > 0
    second = hard_topk_mask(np.asarray(b, dtype=np.float32), float(keep_ratio)).reshape(-1) > 0
    union = np.logical_or(first, second).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(first, second).sum() / union)
