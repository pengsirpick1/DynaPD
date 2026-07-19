"""Condition and policy helpers for DMMPv3."""

from __future__ import annotations

import numpy as np
import torch


def normalize_map(values: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if mask is not None:
        arr = arr * np.asarray(mask, dtype=np.float32)
    arr = np.maximum(arr, 0.0)
    peak = float(arr.max()) if arr.size else 0.0
    if peak > 1e-8:
        arr = arr / peak
    return arr.astype(np.float32)


def encoder_feature(condition, topk_mask: np.ndarray, s_cell: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(condition.vector, dtype=np.float32).reshape(-1),
            np.asarray(topk_mask, dtype=np.float32).reshape(-1),
            np.asarray(s_cell, dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def analytic_leakage(condition, s_cell: np.ndarray, topk_mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(condition.allowed_mask, dtype=np.float32)
    cond_saliency = np.asarray(getattr(condition, "saliency", np.zeros_like(mask)), dtype=np.float32)
    if cond_saliency.shape != mask.shape:
        cond_saliency = np.resize(cond_saliency, mask.shape)
    return normalize_map(0.65 * np.asarray(s_cell, dtype=np.float32) + 0.25 * topk_mask + 0.10 * cond_saliency, mask)


def diffusion_condition(c_global: np.ndarray, c_leakage: np.ndarray, preference: np.ndarray, budget: float) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(c_global, dtype=np.float32).reshape(-1),
            np.asarray(c_leakage, dtype=np.float32).reshape(-1),
            np.asarray(preference, dtype=np.float32).reshape(-1),
            np.asarray([float(budget)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def make_prior_logits(
    s_cell: np.ndarray,
    preference: np.ndarray,
    allowed_mask: np.ndarray,
    *,
    rng: np.random.Generator,
    alpha_leak: float = 1.50,
    alpha_pref: float = 0.15,
    alpha_noise: float = 0.0,
) -> np.ndarray:
    mask = np.asarray(allowed_mask, dtype=np.float32)
    leak = normalize_map(s_cell, mask)
    pref = normalize_map(preference, mask)
    noise = rng.normal(0.0, 1.0, size=mask.shape).astype(np.float32) * mask
    logits = float(alpha_leak) * leak + float(alpha_pref) * pref + float(alpha_noise) * noise
    logits = np.where(mask > 0, logits, -6.0)
    valid = mask > 0
    if np.any(valid):
        logits[valid] -= float(logits[valid].mean())
    return np.clip(logits, -8.0, 8.0).astype(np.float32)


@torch.no_grad()
def sample_policy_logits(diffusion, condition: torch.Tensor, budget: torch.Tensor, *, sampling_steps: int = 20, generator=None) -> torch.Tensor:
    diffusion.eval()
    return diffusion.ddim_sample(condition, budget, sampler_steps=int(sampling_steps), generator=generator)

