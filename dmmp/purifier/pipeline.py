"""End-to-end conditional diffusion purifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .condition_encoder import ConditionFeatures, LabelConditionEncoder, TrafficConditionEncoder
from .decoder import TrafficOutputDecoder
from .denoiser import ConditionalDenoiser
from .diffusion import TrafficDiffusion
from .losses import purifier_losses


class ConditionalTrafficPurifier(nn.Module):
    """Learns p_theta(x_clean | x_defended) without classifier inputs."""

    def __init__(
        self,
        *,
        seq_length: int = 10000,
        hidden_channels: int = 32,
        condition_channels: int = 32,
        num_classes: int = 95,
        time_dim: int = 128,
        num_denoiser_blocks: int = 6,
        diffusion_steps: int = 32,
        beta_start: float = 1.0e-4,
        beta_end: float = 2.0e-2,
        value_scale: float = 80.0,
        value_clip: float = 80.0,
        zero_threshold: float = 0.03,
        dropout: float = 0.0,
        condition_mode: str = "conditioned",
        condition_source: str = "defended",
    ):
        super().__init__()
        self.seq_length = int(seq_length)
        self.condition_mode = str(condition_mode)
        self.condition_source = str(condition_source).strip().lower()
        if self.condition_source not in {"defended", "clean", "label"}:
            raise ValueError("condition_source must be one of: defended, clean, label")
        self.num_classes = int(num_classes)
        self.condition_channels = int(condition_channels)
        self.global_dim = int(condition_channels) * 2
        self.use_condition_encoder = self.condition_mode not in {"unconditional", "zero"}
        if self.use_condition_encoder and self.condition_source == "label":
            self.condition_encoder = LabelConditionEncoder(
                num_classes=int(num_classes),
                condition_channels=int(condition_channels),
                global_dim=self.global_dim,
                dropout=dropout,
            )
        elif self.use_condition_encoder:
            self.condition_encoder = TrafficConditionEncoder(condition_channels=int(condition_channels), global_dim=self.global_dim, dropout=dropout)
        else:
            self.condition_encoder = None
        self.denoiser = ConditionalDenoiser(
            hidden_channels=int(hidden_channels),
            condition_channels=int(condition_channels),
            global_dim=self.global_dim,
            time_dim=int(time_dim),
            num_blocks=int(num_denoiser_blocks),
            dropout=dropout,
        )
        self.diffusion = TrafficDiffusion(
            diffusion_steps=int(diffusion_steps),
            beta_start=float(beta_start),
            beta_end=float(beta_end),
            x0_clip=float(value_clip) / max(float(value_scale), 1.0e-8),
        )
        self.decoder = TrafficOutputDecoder(value_scale=float(value_scale), value_clip=float(value_clip), zero_threshold=float(zero_threshold))

    def _zero_condition(self, batch: int, length: int, device: torch.device, dtype: torch.dtype) -> ConditionFeatures:
        channels = int(self.condition_channels)
        return ConditionFeatures(
            c_local=torch.zeros(batch, channels, length, device=device, dtype=dtype),
            c_multi=(
                torch.zeros(batch, channels, max(1, length // 2), device=device, dtype=dtype),
                torch.zeros(batch, channels, max(1, length // 4), device=device, dtype=dtype),
            ),
            c_global=torch.zeros(batch, self.global_dim, device=device, dtype=dtype),
        )

    def encode_condition(
        self,
        defended: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        force_zero_condition: bool = False,
    ) -> ConditionFeatures:
        if force_zero_condition or self.condition_encoder is None:
            return self._zero_condition(int(defended.shape[0]), int(defended.shape[-1]), defended.device, defended.dtype)
        if self.condition_source == "label":
            if labels is None:
                raise ValueError("labels are required when condition_source='label'")
            return self.condition_encoder(labels.to(device=defended.device), length=int(defended.shape[-1]), dtype=defended.dtype)
        return self.condition_encoder(defended)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        defended: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        force_zero_condition: bool = False,
        encoded_condition: ConditionFeatures | None = None,
    ) -> torch.Tensor:
        condition = (
            encoded_condition
            if encoded_condition is not None
            else self.encode_condition(defended, labels=labels, force_zero_condition=force_zero_condition)
        )
        return self.denoiser(x_t, timesteps, condition)

    def training_losses(
        self,
        clean: torch.Tensor,
        defended: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        lambda_rec: float,
        reconstruction_mode: str = "smooth_l1",
        force_zero_condition: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch = int(clean.shape[0])
        if generator is None:
            timesteps = torch.randint(0, self.diffusion.diffusion_steps, (batch,), device=clean.device, dtype=torch.long)
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype)
        else:
            timesteps = torch.randint(0, self.diffusion.diffusion_steps, (batch,), device=clean.device, dtype=torch.long, generator=generator)
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        x_t, target_noise = self.diffusion.q_sample(clean, timesteps, noise=noise)
        predicted_noise = self.predict_noise(x_t, timesteps, defended, labels=labels, force_zero_condition=force_zero_condition)
        predicted_x0 = self.diffusion.predict_x0_from_noise(x_t, timesteps, predicted_noise)
        return purifier_losses(
            predicted_noise,
            target_noise,
            predicted_x0,
            clean,
            lambda_rec=float(lambda_rec),
            reconstruction_mode=reconstruction_mode,
        )

    @torch.no_grad()
    def sample(
        self,
        defended: torch.Tensor,
        *,
        labels: torch.Tensor | None = None,
        sampling_steps: int,
        force_zero_condition: bool = False,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        condition = self.encode_condition(defended, labels=labels, force_zero_condition=force_zero_condition)

        def _predict(x_t: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return self.predict_noise(x_t, timesteps, defended, labels=labels, encoded_condition=condition)

        return self.diffusion.ddim_sample(
            _predict,
            tuple(defended.shape),
            device=defended.device,
            dtype=defended.dtype,
            sampling_steps=int(sampling_steps),
            generator=generator,
        )

    def model_config(self) -> dict[str, Any]:
        return {
            "seq_length": self.seq_length,
            "hidden_channels": self.denoiser.hidden_channels,
            "condition_channels": self.condition_channels,
            "num_classes": self.num_classes,
            "time_dim": self.denoiser.time_dim,
            "num_denoiser_blocks": len(self.denoiser.blocks),
            "diffusion_steps": self.diffusion.diffusion_steps,
            "beta_start": float(self.diffusion.betas[0].detach().cpu()),
            "beta_end": float(self.diffusion.betas[-1].detach().cpu()),
            "value_scale": self.decoder.value_scale,
            "value_clip": self.decoder.value_clip,
            "zero_threshold": self.decoder.zero_threshold,
            "condition_mode": self.condition_mode,
            "condition_source": self.condition_source,
        }


def build_purifier(cfg) -> ConditionalTrafficPurifier:
    return ConditionalTrafficPurifier(
        seq_length=int(cfg.seq_length),
        hidden_channels=int(cfg.hidden_channels),
        condition_channels=int(cfg.condition_channels),
        num_classes=int(getattr(cfg, "num_classes", 95)),
        time_dim=int(cfg.time_dim),
        num_denoiser_blocks=int(cfg.num_denoiser_blocks),
        diffusion_steps=int(cfg.diffusion_steps),
        beta_start=float(cfg.beta_start),
        beta_end=float(cfg.beta_end),
        value_scale=float(cfg.value_scale),
        value_clip=float(cfg.value_clip),
        zero_threshold=float(cfg.zero_threshold),
        dropout=float(cfg.dropout),
        condition_mode=str(cfg.condition_mode),
        condition_source=str(getattr(cfg, "condition_source", "defended")),
    )


def grad_norm_for(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float()
        total += float(torch.sum(value * value).cpu())
    return total**0.5


def gradient_report(model: ConditionalTrafficPurifier) -> dict[str, Any]:
    unused: list[str] = []
    nan_count = 0
    inf_count = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            unused.append(name)
            continue
        grad = parameter.grad.detach()
        nan_count += int(torch.isnan(grad).sum().cpu())
        inf_count += int(torch.isinf(grad).sum().cpu())
    return {
        "condition_encoder_grad_norm": grad_norm_for(model.condition_encoder),
        "denoiser_grad_norm": grad_norm_for(model.denoiser),
        "decoder_grad_norm": grad_norm_for(model.decoder),
        "unused_parameter_names": unused,
        "nan_gradient_count": int(nan_count),
        "inf_gradient_count": int(inf_count),
        "label_in_forward_signature": False,
        "uses_attack_classifier_parameters": False,
    }
