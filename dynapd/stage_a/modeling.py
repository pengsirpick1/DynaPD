"""Frozen attacker adapters for Stage A TAM masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..evaluation.attack_models import make_attack_model


def _checkpoint_state(payload) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state", "state_dict", "model_states"):
            state = payload.get(key)
            if isinstance(state, dict):
                if key == "model_states":
                    raise ValueError("Ensemble checkpoints need a concrete attacker state, not model_states")
                return state
    if isinstance(payload, dict) and all(torch.is_tensor(value) for value in payload.values()):
        return payload
    raise ValueError("Checkpoint does not contain a recognized model state dict")


def _checkpoint_classes(payload, num_classes: int | None) -> np.ndarray:
    if isinstance(payload, dict) and "classes" in payload:
        return np.asarray(payload["classes"], dtype=np.int64)
    if int(num_classes or 0) <= 0:
        raise ValueError("num_classes is required when checkpoint does not contain classes")
    return np.arange(int(num_classes), dtype=np.int64)


def _resize_mass(tam: torch.Tensor, width: int) -> torch.Tensor:
    if tam.shape[-1] == int(width):
        return tam
    resized = F.interpolate(tam, size=int(width), mode="linear", align_corners=False)
    return resized * (float(tam.shape[-1]) / float(width))


@dataclass
class StageAAttacker:
    model: nn.Module
    attacker: str
    classes: np.ndarray
    max_trace_length: int = 5000
    rf_num_slots: int = 1800
    df_tam_adapter: str = "signed_balance"

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def logits(self, tam: torch.Tensor) -> torch.Tensor:
        name = self.attacker.lower()
        if name == "rf":
            values = _resize_mass(tam, int(self.rf_num_slots))
            return self.model(values)
        if name == "df":
            signed = tam[:, 0, :] - tam[:, 1, :]
            if self.df_tam_adapter == "signed_balance":
                signed = torch.tanh(signed)
            elif self.df_tam_adapter == "raw_balance":
                signed = signed.clamp(-1.0, 1.0)
            else:
                raise ValueError(f"Unknown df_tam_adapter={self.df_tam_adapter!r}")
            values = F.interpolate(
                signed.unsqueeze(1),
                size=int(self.max_trace_length),
                mode="linear",
                align_corners=False,
            )
            return self.model(values.clamp(-1.0, 1.0))
        raise ValueError(f"Unsupported attacker={self.attacker!r}")


def load_stage_a_attacker(
    checkpoint_path: str | Path,
    *,
    attacker: str,
    num_classes: int | None = None,
    device: torch.device | str = "cpu",
    max_trace_length: int = 5000,
    rf_num_slots: int = 1800,
    df_architecture: str = "project",
    df_tam_adapter: str = "signed_balance",
) -> StageAAttacker:
    device = torch.device(device)
    payload = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    classes = _checkpoint_classes(payload, num_classes)
    model = make_attack_model(
        str(attacker).upper(),
        len(classes),
        max_trace_length=int(max_trace_length),
        df_architecture=str(df_architecture),
    ).to(device)
    model.load_state_dict(_checkpoint_state(payload), strict=True)
    adapter = StageAAttacker(
        model=model,
        attacker=str(attacker).lower(),
        classes=classes,
        max_trace_length=int(max_trace_length),
        rf_num_slots=int(rf_num_slots),
        df_tam_adapter=str(df_tam_adapter),
    )
    adapter.freeze()
    return adapter
