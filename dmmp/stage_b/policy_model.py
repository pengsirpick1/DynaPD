"""Neural policy for scoring candidate Stage B actions."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from dmmp.stage_b.policy_data import ACTION_FEATURE_NAMES, STATE_FEATURE_NAMES


@dataclass(frozen=True)
class PolicyModelConfig:
    width: int = 1800
    state_channels: int = 4
    action_channels: int = 2
    state_feature_dim: int = len(STATE_FEATURE_NAMES)
    action_feature_dim: int = len(ACTION_FEATURE_NAMES)
    hidden_dim: int = 192
    dropout: float = 0.10


class ConvSequenceEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(int(in_channels), 32, kernel_size=9, stride=2, padding=4),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 96, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(96),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(96, int(hidden_dim)),
            nn.GELU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class StateEncoder(nn.Module):
    def __init__(self, config: PolicyModelConfig) -> None:
        super().__init__()
        self.sequence = ConvSequenceEncoder(config.state_channels, config.hidden_dim, config.dropout)
        self.features = nn.Sequential(
            nn.Linear(config.state_feature_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.out = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim),
            nn.GELU(),
        )

    def forward(self, state_tensor: torch.Tensor, state_features: torch.Tensor) -> torch.Tensor:
        seq = self.sequence(state_tensor)
        feat = self.features(state_features)
        return self.out(torch.cat([seq, feat], dim=-1))


class ActionEncoder(nn.Module):
    def __init__(self, config: PolicyModelConfig) -> None:
        super().__init__()
        self.sequence = ConvSequenceEncoder(config.action_channels, config.hidden_dim, config.dropout)
        self.features = nn.Sequential(
            nn.Linear(config.action_feature_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
        )
        self.out = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim),
            nn.GELU(),
        )

    def forward(self, action_counts: torch.Tensor, action_features: torch.Tensor) -> torch.Tensor:
        batch, candidates, channels, width = action_counts.shape
        seq = self.sequence(action_counts.reshape(batch * candidates, channels, width))
        feat = self.features(action_features.reshape(batch * candidates, action_features.shape[-1]))
        encoded = self.out(torch.cat([seq, feat], dim=-1))
        return encoded.reshape(batch, candidates, -1)


class CandidateScoringPolicy(nn.Module):
    """Score candidate actions and predict whether the current state should stop."""

    def __init__(self, config: PolicyModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or PolicyModelConfig()
        hidden = int(self.config.hidden_dim)
        self.state_encoder = StateEncoder(self.config)
        self.action_encoder = ActionEncoder(self.config)
        self.score_head = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.stop_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(float(self.config.dropout)),
            nn.Linear(hidden // 2, 1),
        )

    def forward(
        self,
        *,
        state_tensor: torch.Tensor,
        state_features: torch.Tensor,
        action_counts: torch.Tensor,
        action_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        state_z = self.state_encoder(state_tensor, state_features)
        action_z = self.action_encoder(action_counts, action_features)
        state_rep = state_z[:, None, :].expand(-1, action_z.shape[1], -1)
        pair = torch.cat([state_rep, action_z, state_rep * action_z], dim=-1)
        scores = self.score_head(pair).squeeze(-1)
        if candidate_mask is not None:
            scores = scores.masked_fill(~candidate_mask.bool(), -1e9)
        stop_logit = self.stop_head(state_z).squeeze(-1)
        return {"scores": scores, "stop_logit": stop_logit, "state_embedding": state_z, "action_embedding": action_z}
