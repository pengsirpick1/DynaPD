"""Deletion-style DynaMask optimization over TAM keypoint maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .mask_ops import local_average_baseline_torch
from .modeling import StageAAttacker


@dataclass
class DynMaskConfig:
    steps: int = 300
    learning_rate: float = 0.05
    lambda_l1: float = 1e-3
    lambda_tv: float = 1e-2
    target_keep_ratio: float = 0.10
    lambda_area: float = 0.0
    baseline_kernel: int = 9
    init_value: float = 0.5
    log_every: int = 50


@dataclass
class DynMaskResult:
    tam: np.ndarray
    mask: np.ndarray
    tam_base: np.ndarray
    tam_masked: np.ndarray
    tam_keypoint_only: np.ndarray
    pred_logits: np.ndarray
    pred_prob: np.ndarray
    masked_logits: np.ndarray
    masked_prob: np.ndarray
    pred_label: np.ndarray
    masked_pred_label: np.ndarray
    js_div: np.ndarray
    entropy_gain: np.ndarray
    top1_drop: np.ndarray
    objective_history: list[dict[str, float]]


def _js_divergence_torch(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    midpoint = 0.5 * (p + q)
    kl_left = (p * (p.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=1)
    kl_right = (q * (q.clamp_min(1e-8).log() - midpoint.clamp_min(1e-8).log())).sum(dim=1)
    return 0.5 * (kl_left + kl_right)


def _entropy_torch(p: torch.Tensor) -> torch.Tensor:
    return -(p * p.clamp_min(1e-8).log()).sum(dim=1)


def _logit_from_init(value: float) -> float:
    clipped = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return float(np.log(clipped / (1.0 - clipped)))


def optimize_deletion_masks(
    tam: np.ndarray,
    attacker: StageAAttacker,
    cfg: DynMaskConfig,
    *,
    device: torch.device | str,
    progress: bool = False,
) -> DynMaskResult:
    """Learn masks where high values are locations to smooth/delete."""
    device = torch.device(device)
    x = torch.as_tensor(np.asarray(tam, dtype=np.float32), dtype=torch.float32, device=device)
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.ndim != 3 or x.shape[1] != 2:
        raise ValueError(f"Expected TAM [B, 2, W] or [2, W], got {tuple(x.shape)}")
    attacker.freeze()
    baseline = local_average_baseline_torch(x, int(cfg.baseline_kernel))
    with torch.no_grad():
        original_logits = attacker.logits(x)
        original_prob = torch.softmax(original_logits, dim=1)

    init_logit = _logit_from_init(float(cfg.init_value))
    mask_logits = torch.full_like(x, init_logit, requires_grad=True)
    optimizer = torch.optim.Adam([mask_logits], lr=float(cfg.learning_rate))
    history: list[dict[str, float]] = []
    for step in range(1, int(cfg.steps) + 1):
        mask = torch.sigmoid(mask_logits)
        tam_masked = (1.0 - mask) * x + mask * baseline
        masked_logits = attacker.logits(tam_masked)
        masked_prob = torch.softmax(masked_logits, dim=1)
        js = _js_divergence_torch(original_prob.detach(), masked_prob)
        l1 = mask.mean(dim=(1, 2))
        tv = torch.abs(mask[:, :, 1:] - mask[:, :, :-1]).mean(dim=(1, 2)) if mask.shape[-1] > 1 else torch.zeros_like(l1)
        area = (l1 - float(cfg.target_keep_ratio)).pow(2)
        loss = (-js + float(cfg.lambda_l1) * l1 + float(cfg.lambda_tv) * tv + float(cfg.lambda_area) * area).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step == int(cfg.steps) or (int(cfg.log_every) > 0 and step % int(cfg.log_every) == 0):
            row = {
                "step": float(step),
                "loss": float(loss.detach().cpu()),
                "js_div": float(js.mean().detach().cpu()),
                "mask_mean": float(l1.mean().detach().cpu()),
                "tv": float(tv.mean().detach().cpu()),
            }
            history.append(row)
            if progress:
                print(
                    f"[Stage A DynaMask] step={step}/{cfg.steps} "
                    f"loss={row['loss']:.6f} js={row['js_div']:.6f} mask_mean={row['mask_mean']:.4f}",
                    flush=True,
                )

    with torch.no_grad():
        mask = torch.sigmoid(mask_logits)
        tam_masked = (1.0 - mask) * x + mask * baseline
        tam_keypoint_only = mask * x + (1.0 - mask) * baseline
        masked_logits = attacker.logits(tam_masked)
        masked_prob = torch.softmax(masked_logits, dim=1)
        js = _js_divergence_torch(original_prob, masked_prob)
        entropy_gain = _entropy_torch(masked_prob) - _entropy_torch(original_prob)
        pred_label = original_prob.argmax(dim=1)
        masked_pred_label = masked_prob.argmax(dim=1)
        top1_before = original_prob.gather(1, pred_label.reshape(-1, 1)).reshape(-1)
        top1_after = masked_prob.gather(1, pred_label.reshape(-1, 1)).reshape(-1)
        top1_drop = top1_before - top1_after

    return DynMaskResult(
        tam=x.detach().cpu().numpy().astype(np.float32),
        mask=mask.detach().cpu().numpy().astype(np.float32),
        tam_base=baseline.detach().cpu().numpy().astype(np.float32),
        tam_masked=tam_masked.detach().cpu().numpy().astype(np.float32),
        tam_keypoint_only=tam_keypoint_only.detach().cpu().numpy().astype(np.float32),
        pred_logits=original_logits.detach().cpu().numpy().astype(np.float32),
        pred_prob=original_prob.detach().cpu().numpy().astype(np.float32),
        masked_logits=masked_logits.detach().cpu().numpy().astype(np.float32),
        masked_prob=masked_prob.detach().cpu().numpy().astype(np.float32),
        pred_label=pred_label.detach().cpu().numpy().astype(np.int64),
        masked_pred_label=masked_pred_label.detach().cpu().numpy().astype(np.int64),
        js_div=js.detach().cpu().numpy().astype(np.float32),
        entropy_gain=entropy_gain.detach().cpu().numpy().astype(np.float32),
        top1_drop=top1_drop.detach().cpu().numpy().astype(np.float32),
        objective_history=history,
    )
