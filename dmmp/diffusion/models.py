"""Neural models used by DMMPv3."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class TopKLeakageEncoder(nn.Module):
    def __init__(self, input_dim: int, patch_num: int = 200, hidden_dim: int = 384, global_dim: int | None = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.patch_num = int(patch_num)
        self.hidden_dim = int(hidden_dim)
        self.global_dim = int(global_dim or hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.global_head = nn.Linear(self.hidden_dim, self.global_dim)
        self.leakage_head = nn.Linear(self.hidden_dim, 2 * self.patch_num)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(features.float())
        return self.global_head(h), self.leakage_head(h).reshape(-1, 2, self.patch_num)


@dataclass
class EncoderLossWeights:
    struct: float = 0.05
    fusion: float = 0.05
    smooth: float = 0.02


def _normalize_cell_map(values: torch.Tensor) -> torch.Tensor:
    flat = values.float().reshape(values.shape[0], -1)
    mean = flat.mean(dim=1, keepdim=True)
    std = flat.std(dim=1, keepdim=True).clamp_min(1e-6)
    return ((flat - mean) / std).reshape_as(values)


def leakage_encoder_loss(
    c_leakage: torch.Tensor,
    target_s_cell: torch.Tensor,
    topk_mask: torch.Tensor,
    weights: EncoderLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or EncoderLossWeights()
    target = _normalize_cell_map(target_s_cell)
    pred = _normalize_cell_map(c_leakage)
    leak = F.mse_loss(pred, target)
    mask = topk_mask.float()
    struct = F.mse_loss(torch.sigmoid(c_leakage) * mask, torch.sigmoid(target_s_cell) * mask)
    fusion = F.mse_loss(torch.sigmoid(c_leakage).mean(dim=1), torch.sigmoid(target_s_cell).mean(dim=1))
    smooth = F.l1_loss(c_leakage[:, :, 1:], c_leakage[:, :, :-1]) if c_leakage.shape[-1] > 1 else leak * 0.0
    total = leak + weights.struct * struct + weights.fusion * fusion + weights.smooth * smooth
    return total, {
        "loss": float(total.detach().cpu()),
        "leak": float(leak.detach().cpu()),
        "struct": float(struct.detach().cpu()),
        "fusion": float(fusion.detach().cpu()),
        "smooth": float(smooth.detach().cpu()),
    }


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    if half == 0:
        return timesteps.float().reshape(-1, 1)
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device, dtype=torch.float32))
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        h = self.norm(x + conditioning)
        h = F.silu(self.fc1(h))
        return x + self.dropout(self.fc2(h))


class LightConditionalDenoiser(nn.Module):
    def __init__(self, template_dim: int, condition_dim: int, hidden_dim: int = 384, time_dim: int = 128, num_blocks: int = 4):
        super().__init__()
        self.template_dim = int(template_dim)
        self.condition_dim = int(condition_dim)
        self.hidden_dim = int(hidden_dim)
        self.time_dim = int(time_dim)
        self.noisy_projection = nn.Linear(self.template_dim, self.hidden_dim)
        self.condition_projection = nn.Sequential(nn.Linear(self.condition_dim, self.hidden_dim), nn.LayerNorm(self.hidden_dim), nn.SiLU())
        self.time_projection = nn.Sequential(nn.Linear(self.time_dim, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.bandwidth_projection = nn.Sequential(nn.Linear(1, self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim))
        self.blocks = nn.ModuleList([ConditionalResidualBlock(self.hidden_dim) for _ in range(int(num_blocks))])
        self.output = nn.Sequential(nn.LayerNorm(self.hidden_dim), nn.SiLU(), nn.Linear(self.hidden_dim, self.template_dim))

    def forward(
        self,
        noisy_template: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        bandwidth: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        time_embedding = sinusoidal_timestep_embedding(timesteps, self.time_dim)
        conditioning = (
            self.condition_projection(condition)
            + self.time_projection(time_embedding)
            + self.bandwidth_projection(bandwidth.reshape(-1, 1).float())
        )
        h = self.noisy_projection(noisy_template)
        for block in self.blocks:
            h = block(h, conditioning)
        predicted = self.output(h)
        return (predicted, h) if return_hidden else predicted


class LightPaddingDiffusion(nn.Module):
    def __init__(
        self,
        denoiser: LightConditionalDenoiser,
        diffusion_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        x0_clip: float = 12.0,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.diffusion_steps = int(diffusion_steps)
        self.x0_clip = float(x0_clip)
        betas = torch.linspace(float(beta_start), float(beta_end), self.diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.register_buffer("sqrt_alpha_cumprod", torch.sqrt(alpha_cumprod))
        self.register_buffer("sqrt_one_minus_alpha_cumprod", torch.sqrt(1.0 - alpha_cumprod))

    @property
    def template_dim(self) -> int:
        return self.denoiser.template_dim

    def q_sample(self, clean_logits: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean_logits)
        sqrt_alpha = self.sqrt_alpha_cumprod[timesteps].reshape(-1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alpha_cumprod[timesteps].reshape(-1, 1)
        return sqrt_alpha * clean_logits + sqrt_one_minus * noise, noise

    def training_loss(
        self,
        clean_logits: torch.Tensor,
        condition: torch.Tensor,
        bandwidth: torch.Tensor,
        timesteps: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if timesteps is None:
            timesteps = torch.randint(0, self.diffusion_steps, (clean_logits.shape[0],), device=clean_logits.device, dtype=torch.long)
        noisy, target_noise = self.q_sample(clean_logits, timesteps, noise=noise)
        predicted = self.denoiser(noisy, timesteps, condition, bandwidth)
        return F.mse_loss(predicted, target_noise)

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        bandwidth: torch.Tensor,
        sampler_steps: int = 20,
        eta: float = 0.0,
        initial_noise: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch = condition.shape[0]
        if initial_noise is None:
            x = torch.randn(batch, self.template_dim, device=condition.device, dtype=condition.dtype, generator=generator)
        else:
            x = initial_noise.to(device=condition.device, dtype=condition.dtype)
        requested_steps = max(1, min(int(sampler_steps), self.diffusion_steps))
        timesteps = torch.linspace(self.diffusion_steps - 1, 0, requested_steps, device=condition.device).round().long()
        timesteps = torch.unique_consecutive(timesteps)
        for index, timestep in enumerate(timesteps.tolist()):
            t = torch.full((batch,), int(timestep), device=condition.device, dtype=torch.long)
            predicted_noise = self.denoiser(x, t, condition, bandwidth)
            alpha_t = self.alpha_cumprod[int(timestep)]
            predicted_x0 = (x - torch.sqrt(1.0 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)
            predicted_x0 = predicted_x0.clamp(-self.x0_clip, self.x0_clip)
            next_timestep = int(timesteps[index + 1].item()) if index + 1 < len(timesteps) else -1
            if next_timestep < 0:
                x = predicted_x0
                break
            alpha_next = self.alpha_cumprod[next_timestep]
            sigma = float(eta) * torch.sqrt(torch.clamp((1.0 - alpha_next) / (1.0 - alpha_t) * (1.0 - alpha_t / alpha_next), min=0.0))
            direction = torch.sqrt(torch.clamp(1.0 - alpha_next - sigma.pow(2), min=0.0)) * predicted_noise
            stochastic = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) if float(eta) > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_next) * predicted_x0 + direction + sigma * stochastic
        return x - x.mean(dim=1, keepdim=True)


def build_policy_diffusion(condition_dim: int, patch_num: int = 200, hidden_dim: int = 384, diffusion_steps: int = 100) -> LightPaddingDiffusion:
    denoiser = LightConditionalDenoiser(template_dim=2 * int(patch_num), condition_dim=int(condition_dim), hidden_dim=int(hidden_dim))
    return LightPaddingDiffusion(denoiser, diffusion_steps=int(diffusion_steps))

