"""Forward and reverse diffusion utilities for purifier training."""

from __future__ import annotations

import torch
from torch import nn


class TrafficDiffusion(nn.Module):
    def __init__(
        self,
        diffusion_steps: int = 32,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        x0_clip: float = 1.0,
    ):
        super().__init__()
        self.diffusion_steps = int(diffusion_steps)
        self.x0_clip = float(x0_clip)
        betas = torch.linspace(float(beta_start), float(beta_end), self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def q_sample(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alpha_bar[timesteps].reshape(-1, *([1] * (x0.ndim - 1)))
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[timesteps].reshape(-1, *([1] * (x0.ndim - 1)))
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise

    def predict_x0_from_noise(self, x_t: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alpha_bar[timesteps].reshape(-1, *([1] * (x_t.ndim - 1)))
        sqrt_one_minus = self.sqrt_one_minus_alpha_bar[timesteps].reshape(-1, *([1] * (x_t.ndim - 1)))
        return (x_t - sqrt_one_minus * predicted_noise) / sqrt_alpha.clamp_min(1.0e-8)

    @torch.no_grad()
    def ddim_sample(
        self,
        predict_noise,
        shape: tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        sampling_steps: int = 8,
        eta: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if generator is None:
            x = torch.randn(shape, device=device, dtype=dtype)
        else:
            x = torch.randn(shape, device=device, dtype=dtype, generator=generator)
        requested = max(1, min(int(sampling_steps), self.diffusion_steps))
        timesteps = torch.linspace(self.diffusion_steps - 1, 0, requested, device=device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        for index, timestep_value in enumerate(timesteps.tolist()):
            timestep = torch.full((shape[0],), int(timestep_value), dtype=torch.long, device=device)
            predicted_noise = predict_noise(x, timestep)
            alpha_t = self.alpha_bar[int(timestep_value)]
            x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t).clamp_min(1.0e-8)
            x0 = x0.clamp(-self.x0_clip, self.x0_clip)
            next_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            if next_timestep < 0:
                x = x0
                break
            alpha_next = self.alpha_bar[next_timestep]
            sigma = float(eta) * torch.sqrt(
                torch.clamp((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next), min=0.0)
            )
            direction = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma.pow(2), min=0.0)) * predicted_noise
            if float(eta) > 0:
                stochastic = torch.randn(x.shape, device=device, dtype=dtype, generator=generator)
            else:
                stochastic = torch.zeros_like(x)
            x = torch.sqrt(alpha_next) * x0 + direction + sigma * stochastic
        return x
