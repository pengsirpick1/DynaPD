"""Losses for conditional diffusion purifier training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(predicted_x0: torch.Tensor, clean: torch.Tensor, mode: str = "smooth_l1") -> torch.Tensor:
    name = str(mode).lower()
    if name in {"l1", "mae"}:
        return F.l1_loss(predicted_x0, clean)
    if name in {"mse", "l2"}:
        return F.mse_loss(predicted_x0, clean)
    if name in {"smooth_l1", "huber"}:
        return F.smooth_l1_loss(predicted_x0, clean)
    raise ValueError(f"Unsupported reconstruction loss {mode!r}")


def purifier_losses(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    predicted_x0: torch.Tensor,
    clean: torch.Tensor,
    *,
    lambda_rec: float,
    reconstruction_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    diff = F.mse_loss(predicted_noise, target_noise)
    rec = reconstruction_loss(predicted_x0, clean, reconstruction_mode)
    total = diff + float(lambda_rec) * rec
    return total, {"loss": total, "diffusion_loss": diff, "reconstruction_loss": rec}
