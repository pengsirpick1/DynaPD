"""Torch losses for target-policy diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mask_like(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if mask.shape != target.shape:
        mask = mask.reshape_as(target)
    return (mask > 0).to(dtype=target.dtype, device=target.device)


def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_t = _mask_like(mask, pred)
    loss = F.smooth_l1_loss(pred, target.to(pred), reduction="none")
    return (loss * mask_t).sum() / mask_t.sum().clamp_min(1.0)


def masked_softmax_torch(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_t = _mask_like(mask, logits)
    masked = logits.masked_fill(mask_t <= 0, -1.0e9)
    probs = torch.softmax(masked.reshape(masked.shape[0], -1), dim=1).reshape_as(logits)
    probs = probs * mask_t
    return probs / probs.sum(dim=tuple(range(1, probs.ndim)), keepdim=True).clamp_min(1.0e-8)


def allocation_kl_loss(target_allocation: torch.Tensor, pred_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    pred = masked_softmax_torch(pred_logits, mask)
    target = target_allocation.to(pred).clamp_min(1.0e-8)
    target = target / target.sum(dim=tuple(range(1, target.ndim)), keepdim=True).clamp_min(1.0e-8)
    loss = target * (torch.log(target) - torch.log(pred.clamp_min(1.0e-8)))
    return loss.sum(dim=tuple(range(1, loss.ndim))).mean()


def symmetric_kl_from_logits(first_logits: torch.Tensor, second_logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    first = masked_softmax_torch(first_logits, mask).clamp_min(1.0e-8)
    second = masked_softmax_torch(second_logits, mask).clamp_min(1.0e-8)
    kl_ab = (first * (torch.log(first) - torch.log(second))).sum(dim=tuple(range(1, first.ndim)))
    kl_ba = (second * (torch.log(second) - torch.log(first))).sum(dim=tuple(range(1, first.ndim)))
    return (0.5 * (kl_ab + kl_ba)).mean()


def categorical_kl_loss(target_weights: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    target = target_weights.to(logits).clamp_min(1.0e-8)
    target = target / target.sum(dim=1, keepdim=True).clamp_min(1.0e-8)
    log_probs = torch.log_softmax(logits, dim=1)
    return (target * (torch.log(target) - log_probs)).sum(dim=1).mean()
