"""V4 leakage encoder with executable, ranking, structure, and global supervision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class V4LeakageEncoder(nn.Module):
    def __init__(self, input_dim: int, structure_dim: int, patch_num: int = 200, hidden_dim: int = 384):
        super().__init__()
        self.input_dim = int(input_dim)
        self.structure_dim = int(structure_dim)
        self.patch_num = int(patch_num)
        self.hidden_dim = int(hidden_dim)
        self.backbone = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.global_head = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.leakage_head = nn.Linear(self.hidden_dim, 2 * self.patch_num)
        self.structure_decoder = nn.Linear(self.hidden_dim, self.structure_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features.float())
        c_global = self.global_head(hidden)
        return {
            "c_global": c_global,
            "c_leakage": self.leakage_head(hidden).reshape(-1, 2, self.patch_num),
            "structure": self.structure_decoder(c_global),
        }


class ProfileEncoder(nn.Module):
    def __init__(self, output_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, output_dim), nn.LayerNorm(output_dim), nn.SiLU(), nn.Linear(output_dim, output_dim))

    def forward(self, profile_mask: torch.Tensor) -> torch.Tensor:
        return self.net(profile_mask.float())


class VisitPreferenceEncoder(nn.Module):
    def __init__(self, patch_num: int = 200, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(patch_num) + 10, output_dim),
            nn.LayerNorm(output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, mixed_preference: torch.Tensor, selected_mask: torch.Tensor, primitive_weights: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            [mixed_preference.reshape(len(mixed_preference), -1), selected_mask.float(), primitive_weights.float()],
            dim=1,
        )
        return self.net(values)


class CompositionalConditionEncoder(nn.Module):
    """Projects leakage, executable candidate, profile, and visit conditions before fusion."""

    def __init__(self, patch_num: int = 200, hidden_dim: int = 384, output_dim: int = 384):
        super().__init__()
        patch_dim = 2 * int(patch_num)
        branch_dim = 128
        self.global_projection = nn.Sequential(nn.Linear(int(hidden_dim), branch_dim), nn.LayerNorm(branch_dim), nn.SiLU())
        self.leakage_projection = nn.Sequential(nn.Linear(patch_dim, branch_dim), nn.LayerNorm(branch_dim), nn.SiLU())
        self.candidate_projection = nn.Sequential(nn.Linear(patch_dim, branch_dim), nn.LayerNorm(branch_dim), nn.SiLU())
        self.profile_encoder = ProfileEncoder(96)
        self.visit_encoder = VisitPreferenceEncoder(int(patch_num), 128)
        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 4 + 96, int(output_dim)),
            nn.LayerNorm(int(output_dim)),
            nn.SiLU(),
            nn.Linear(int(output_dim), int(output_dim)),
        )
        self.output_dim = int(output_dim)

    def forward(
        self,
        c_global: torch.Tensor,
        c_leakage: torch.Tensor,
        candidate_mask: torch.Tensor,
        mixed_preference: torch.Tensor,
        primitive_weights: torch.Tensor,
        selected_mask: torch.Tensor,
        profile_mask: torch.Tensor,
    ) -> torch.Tensor:
        branches = [
            self.global_projection(c_global),
            self.leakage_projection(c_leakage.reshape(len(c_leakage), -1)),
            self.candidate_projection(candidate_mask.reshape(len(candidate_mask), -1)),
            self.visit_encoder(mixed_preference, selected_mask, primitive_weights),
            self.profile_encoder(profile_mask),
        ]
        return self.fusion(torch.cat(branches, dim=1))


def deterministic_global_target(structure: np.ndarray, hidden_dim: int) -> np.ndarray:
    values = np.asarray(structure, dtype=np.float32)
    result = np.zeros((len(values), int(hidden_dim)), dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, int(hidden_dim), dtype=np.float32)
    for index, row in enumerate(values):
        source = np.linspace(0.0, 1.0, len(row), dtype=np.float32) if len(row) > 1 else np.asarray([0.0], dtype=np.float32)
        interpolated = np.interp(target_grid, source, row if len(row) else np.asarray([0.0], dtype=np.float32)).astype(np.float32)
        result[index] = (interpolated - interpolated.mean()) / max(float(interpolated.std()), 1e-4)
    return result


@dataclass
class V4EncoderLossWeights:
    rank: float = 0.10
    struct: float = 0.10
    global_: float = 0.05
    fusion: float = 0.05
    smooth: float = 0.02


def v4_encoder_loss(
    outputs: dict[str, torch.Tensor],
    utility_target: torch.Tensor,
    candidate_mask: torch.Tensor,
    allowed_mask: torch.Tensor,
    structure_target: torch.Tensor,
    global_target: torch.Tensor,
    weights: V4EncoderLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = weights or V4EncoderLossWeights()
    pred = F.softplus(outputs["c_leakage"]) * allowed_mask
    target = utility_target * allowed_mask
    exec_loss = F.mse_loss(pred, target)
    rank_loss = 1.0 - F.cosine_similarity(pred.reshape(len(pred), -1), target.reshape(len(target), -1), dim=1).mean()
    struct_loss = F.mse_loss(outputs["structure"], structure_target)
    normalized_global = (outputs["c_global"] - outputs["c_global"].mean(dim=1, keepdim=True)) / outputs["c_global"].std(dim=1, keepdim=True).clamp_min(1e-4)
    global_loss = F.mse_loss(normalized_global, global_target)
    fusion_loss = F.mse_loss(pred.mean(dim=1), target.mean(dim=1))
    smooth_mask = allowed_mask[:, :, 1:] * allowed_mask[:, :, :-1]
    smooth_loss = ((pred[:, :, 1:] - pred[:, :, :-1]).abs() * smooth_mask).sum() / smooth_mask.sum().clamp_min(1.0)
    candidate_alignment = F.mse_loss(torch.clamp(pred, 0.0, 1.0) * candidate_mask, target * candidate_mask)
    total = (
        exec_loss
        + weights.rank * rank_loss
        + weights.struct * (struct_loss + candidate_alignment)
        + weights.global_ * global_loss
        + weights.fusion * fusion_loss
        + weights.smooth * smooth_loss
    )
    return total, {
        "loss": float(total.detach().cpu()),
        "exec": float(exec_loss.detach().cpu()),
        "rank": float(rank_loss.detach().cpu()),
        "struct": float(struct_loss.detach().cpu()),
        "global": float(global_loss.detach().cpu()),
        "fusion": float(fusion_loss.detach().cpu()),
        "smooth": float(smooth_loss.detach().cpu()),
    }

