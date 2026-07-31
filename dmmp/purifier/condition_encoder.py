"""Condition encoder that embeds only defended traffic."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class ConditionFeatures:
    c_local: torch.Tensor
    c_multi: tuple[torch.Tensor, ...]
    c_global: torch.Tensor

    def zeros_like(self) -> "ConditionFeatures":
        return ConditionFeatures(
            c_local=torch.zeros_like(self.c_local),
            c_multi=tuple(torch.zeros_like(item) for item in self.c_multi),
            c_global=torch.zeros_like(self.c_global),
        )


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, *, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        padding = int(dilation)
        self.net = nn.Sequential(
            nn.GroupNorm(4 if channels >= 4 else 1, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=int(dilation)),
            nn.GroupNorm(4 if channels >= 4 else 1, channels),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TrafficConditionEncoder(nn.Module):
    """Input Projection -> residual blocks -> multi-scale downsampling -> global pooling."""

    def __init__(self, condition_channels: int = 32, global_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        channels = int(condition_channels)
        self.condition_channels = channels
        self.global_dim = int(global_dim or channels * 2)
        self.input_projection = nn.Conv1d(1, channels, kernel_size=7, padding=3)
        self.local_blocks = nn.Sequential(
            ConvResidualBlock(channels, dilation=1, dropout=dropout),
            ConvResidualBlock(channels, dilation=2, dropout=dropout),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1),
            ConvResidualBlock(channels, dilation=1, dropout=dropout),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1),
            ConvResidualBlock(channels, dilation=1, dropout=dropout),
        )
        self.global_head = nn.Sequential(
            nn.Linear(channels * 2, self.global_dim),
            nn.SiLU(),
            nn.Linear(self.global_dim, self.global_dim),
        )

    def forward(self, defended: torch.Tensor) -> ConditionFeatures:
        x = defended.float()
        if x.ndim == 2:
            x = x.unsqueeze(1)
        local = self.local_blocks(self.input_projection(x))
        mid = self.down1(local)
        coarse = self.down2(mid)
        pooled = torch.cat([coarse.mean(dim=-1), coarse.amax(dim=-1)], dim=1)
        global_condition = self.global_head(pooled)
        return ConditionFeatures(c_local=local, c_multi=(mid, coarse), c_global=global_condition)


class LabelConditionEncoder(nn.Module):
    """Embeds a website label into local, multi-scale, and global condition features."""

    def __init__(self, num_classes: int = 95, condition_channels: int = 32, global_dim: int | None = None, dropout: float = 0.0):
        super().__init__()
        channels = int(condition_channels)
        self.condition_channels = channels
        self.global_dim = int(global_dim or channels * 2)
        self.local_embedding = nn.Embedding(int(num_classes), channels)
        self.global_head = nn.Sequential(
            nn.Embedding(int(num_classes), self.global_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.global_dim, self.global_dim),
        )

    def forward(self, labels: torch.Tensor, *, length: int, dtype: torch.dtype | None = None) -> ConditionFeatures:
        label = labels.long().reshape(-1)
        local_vec = self.local_embedding(label)
        if dtype is not None:
            local_vec = local_vec.to(dtype=dtype)
        local = local_vec.unsqueeze(-1).expand(-1, -1, int(length)).contiguous()
        mid = local_vec.unsqueeze(-1).expand(-1, -1, max(1, int(length) // 2)).contiguous()
        coarse = local_vec.unsqueeze(-1).expand(-1, -1, max(1, int(length) // 4)).contiguous()
        global_condition = self.global_head(label)
        if dtype is not None:
            global_condition = global_condition.to(dtype=dtype)
        return ConditionFeatures(c_local=local, c_multi=(mid, coarse), c_global=global_condition)


def resize_condition(condition: torch.Tensor, length: int) -> torch.Tensor:
    if int(condition.shape[-1]) == int(length):
        return condition
    return F.interpolate(condition, size=int(length), mode="linear", align_corners=False)
