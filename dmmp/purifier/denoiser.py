"""Conditional denoiser for traffic diffusion."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from .condition_encoder import ConditionFeatures, resize_condition


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half = int(dim) // 2
    if half == 0:
        return timesteps.float().reshape(-1, 1)
    scale = math.log(10000.0) / max(half - 1, 1)
    frequencies = torch.exp(-scale * torch.arange(half, device=timesteps.device, dtype=torch.float32))
    angles = timesteps.float().reshape(-1, 1) * frequencies.reshape(1, -1)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if int(dim) % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class DenoiserBlock(nn.Module):
    def __init__(self, channels: int, cond_channels: int, time_dim: int, global_dim: int, *, dilation: int, dropout: float = 0.0):
        super().__init__()
        self.cond_projection = nn.Conv1d(int(cond_channels), int(channels), kernel_size=1)
        self.time_projection = nn.Linear(int(time_dim), int(channels))
        self.global_projection = nn.Linear(int(global_dim), int(channels))
        self.norm1 = nn.GroupNorm(4 if int(channels) >= 4 else 1, int(channels))
        self.conv1 = nn.Conv1d(int(channels), int(channels), kernel_size=3, padding=int(dilation), dilation=int(dilation))
        self.norm2 = nn.GroupNorm(4 if int(channels) >= 4 else 1, int(channels))
        self.dropout = nn.Dropout(float(dropout))
        self.conv2 = nn.Conv1d(int(channels), int(channels), kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor, time_emb: torch.Tensor, global_condition: torch.Tensor) -> torch.Tensor:
        cond = resize_condition(condition, int(x.shape[-1]))
        bias = self.time_projection(time_emb) + self.global_projection(global_condition)
        h = x + self.cond_projection(cond) + bias.unsqueeze(-1)
        h = self.conv1(F.silu(self.norm1(h)))
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return x + h


class ConditionalDenoiser(nn.Module):
    def __init__(
        self,
        hidden_channels: int = 32,
        condition_channels: int = 32,
        global_dim: int = 64,
        time_dim: int = 128,
        num_blocks: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.condition_channels = int(condition_channels)
        self.global_dim = int(global_dim)
        self.time_dim = int(time_dim)
        self.input_projection = nn.Conv1d(1, self.hidden_channels, kernel_size=7, padding=3)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim),
            nn.SiLU(),
            nn.Linear(self.time_dim, self.time_dim),
        )
        dilations = [1, 2, 4, 8, 4, 2, 1, 1]
        self.blocks = nn.ModuleList(
            [
                DenoiserBlock(
                    self.hidden_channels,
                    self.condition_channels,
                    self.time_dim,
                    self.global_dim,
                    dilation=dilations[index % len(dilations)],
                    dropout=dropout,
                )
                for index in range(int(num_blocks))
            ]
        )
        self.multi_projection = nn.Conv1d(self.condition_channels, self.hidden_channels, kernel_size=1)
        self.output = nn.Sequential(
            nn.GroupNorm(4 if self.hidden_channels >= 4 else 1, self.hidden_channels),
            nn.SiLU(),
            nn.Conv1d(self.hidden_channels, 1, kernel_size=7, padding=3),
        )

    def forward(self, x_t: torch.Tensor, timesteps: torch.Tensor, condition: ConditionFeatures) -> torch.Tensor:
        x = x_t.float()
        if x.ndim == 2:
            x = x.unsqueeze(1)
        h = self.input_projection(x)
        time_emb = self.time_mlp(sinusoidal_timestep_embedding(timesteps, self.time_dim))
        for index, block in enumerate(self.blocks):
            cond = condition.c_local
            if condition.c_multi and index % 2 == 1:
                cond = condition.c_multi[min(index // 2, len(condition.c_multi) - 1)]
            h = block(h, cond, time_emb, condition.c_global)
            if condition.c_multi and index == len(self.blocks) // 2:
                h = h + resize_condition(self.multi_projection(condition.c_multi[-1]), int(h.shape[-1]))
        return self.output(h).squeeze(1)
