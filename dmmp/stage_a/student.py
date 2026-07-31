"""Teacher-student mask predictor scaffold for future Stage A scaling."""

from __future__ import annotations

import torch
from torch import nn


class ResidualMaskBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = int(kernel_size) // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=int(kernel_size), padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size=int(kernel_size), padding=padding),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.net(x))


class TamMaskPredictor(nn.Module):
    """Small 1D CNN mapping ``[B, 2, W]`` TAM to ``[B, 2, W]`` mask."""

    def __init__(self, hidden: int = 64, blocks: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, int(hidden), kernel_size=5, padding=2),
            nn.BatchNorm1d(int(hidden)),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualMaskBlock(int(hidden)) for _ in range(int(blocks))])
        self.head = nn.Conv1d(int(hidden), 2, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.blocks(self.stem(x))))
