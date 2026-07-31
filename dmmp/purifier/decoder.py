"""Output projection, range constraints, and representation legalization."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class TrafficOutputDecoder(nn.Module):
    """Keeps differentiable continuous x0 separate from saved legalized traces."""

    legalization_version = "fixed_length_v2_configurable_tail_policy"

    def __init__(self, *, value_scale: float = 80.0, value_clip: float = 80.0, zero_threshold: float = 0.03):
        super().__init__()
        self.value_scale = float(value_scale)
        self.value_clip = float(value_clip)
        self.zero_threshold = float(zero_threshold)

    def range_constraint(self, normalized_x0: torch.Tensor) -> torch.Tensor:
        limit = float(self.value_clip) / max(float(self.value_scale), 1.0e-8)
        return normalized_x0.clamp(-limit, limit)

    def to_traffic_units(self, normalized_x0: torch.Tensor) -> torch.Tensor:
        return self.range_constraint(normalized_x0) * float(self.value_scale)

    @torch.no_grad()
    def legalize_tensor(
        self,
        normalized_x0: torch.Tensor,
        *,
        output_length: torch.Tensor | None = None,
        defended_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        values = self.to_traffic_units(normalized_x0).detach().float().cpu()
        values = torch.nan_to_num(values, nan=0.0, posinf=self.value_clip, neginf=-self.value_clip)
        values = values.clamp(-self.value_clip, self.value_clip)
        if self.zero_threshold > 0:
            values = torch.where(values.abs() < self.zero_threshold, torch.zeros_like(values), values)
        # Backward compatible alias for old callers; new purifier generation
        # should pass output_length only when deliberately choosing a tail policy.
        if output_length is None:
            output_length = defended_length
        if output_length is not None:
            lengths = output_length.detach().cpu().long().clamp(min=0, max=values.shape[1])
            positions = torch.arange(values.shape[1]).reshape(1, -1)
            mask = positions < lengths.reshape(-1, 1)
            values = torch.where(mask, values, torch.zeros_like(values))
        return values.contiguous()

    @torch.no_grad()
    def legalize_numpy(
        self,
        normalized_x0: torch.Tensor,
        *,
        output_length: torch.Tensor | None = None,
        defended_length: torch.Tensor | None = None,
    ) -> np.ndarray:
        return self.legalize_tensor(
            normalized_x0,
            output_length=output_length,
            defended_length=defended_length,
        ).numpy().astype(np.float32, copy=False)
