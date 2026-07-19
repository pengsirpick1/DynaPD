"""Configuration loading for target-policy x0* experiments.

The project intentionally keeps dependencies small.  This module accepts a
YAML-like flat config file and uses PyYAML when it is installed, falling back to
a narrow parser for lists, booleans, ints, and floats.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if text.startswith("[") and text.endswith("]"):
        body = text[1:-1].strip()
        if not body:
            return []
        return [_parse_scalar(part.strip()) for part in body.split(",")]
    if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
        return text[1:-1]
    try:
        if any(mark in text for mark in [".", "e", "E"]):
            return float(text)
        return int(text)
    except ValueError:
        return text


def _load_yaml_like(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config must be a mapping: {path}")
        return dict(payload)
    except ModuleNotFoundError:
        result: dict[str, Any] = {}
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = _parse_scalar(value.strip())
        return result


@dataclass
class TargetPolicyConfig:
    prefix_length: int = 500
    strategy_horizon: int = 200
    budgets: tuple[float, ...] = (0.09, 0.18, 0.30)
    max_budget: float = 0.30
    family_top_k: int = 3
    selected_family_count_choices: tuple[int, ...] = (1, 2)
    primitive_top_k: int = 3
    selected_primitive_count_choices: tuple[int, ...] = (1, 2)
    primitive_dirichlet_alpha: float = 0.7
    primitive_noise_scale: float = 0.05
    completion_set_size: int = 4
    completion_mean_weight: float = 0.5
    completion_min_weight: float = 0.5
    num_candidates: int = 48
    target_count: int = 8
    quality_target_count: int = 4
    diverse_target_count: int = 4
    score_entropy_weight: float = 0.30
    score_pseudo_weight: float = 0.30
    score_margin_weight: float = 0.20
    score_max_weight: float = 0.20
    robust_min_weight: float = 0.25
    attacker_gap_weight: float = 0.25
    min_df_gain: float = 0.0
    min_rf_gain: float = 0.0
    max_dummy_per_slot: int = 8
    max_consecutive_dummy_run: int = 24
    max_tail_extension_ratio: float = 0.10
    max_local_dummy_density: float = 0.80
    density_penalty_weight: float = 0.10
    tail_penalty_weight: float = 0.10
    direction_penalty_weight: float = 0.05
    burst_penalty_weight: float = 0.05
    allocation_l1_weight: float = 0.5
    allocation_cosine_weight: float = 0.5
    clr_epsilon: float = 1.0e-6
    diffusion_steps: int = 100
    sampling_steps: int = 20
    beta_schedule: str = "linear"
    hidden_dim: int = 384
    num_layers: int = 3
    batch_size: str | int = "auto"
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    epochs: int = 10
    gradient_clip: float = 1.0
    use_amp: bool = True
    lambda_eps: float = 1.0
    lambda_x0: float = 0.10
    lambda_alloc: float = 0.10
    lambda_effect: float = 0.03
    lambda_family: float = 0.05
    lambda_primitive: float = 0.05
    lambda_struct: float = 0.03
    lambda_fusion: float = 0.01
    lambda_smooth: float = 0.01
    target_sampling_temperature: float = 1.0
    target_uniform_ratio: float = 0.40
    seed: int = 0
    num_workers: int = 0
    cache_size: int = 1024

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TargetPolicyConfig":
        known = {field.name: field for field in fields(cls)}
        values: dict[str, Any] = {}
        for key, value in payload.items():
            if key not in known:
                continue
            if key in {
                "budgets",
                "selected_family_count_choices",
                "selected_primitive_count_choices",
            }:
                values[key] = tuple(value if isinstance(value, list) else [value])
            else:
                values[key] = value
        return cls(**values)


def load_target_policy_config(path: str | Path | None = None) -> TargetPolicyConfig:
    if path is None:
        return TargetPolicyConfig()
    return TargetPolicyConfig.from_mapping(_load_yaml_like(Path(path)))
