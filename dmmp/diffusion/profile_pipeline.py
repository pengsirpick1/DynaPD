"""DMMPv3 V4 executable, user-specific, guided conditional diffusion pipeline."""

from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn

from ..evaluation.attack_models import build_df_input, build_rf_tam_input
from ..evaluation.attacks import train_df_model
from ..guidance.candidate_scorer import (
    build_candidate_features,
    load_candidate_components,
    run_candidate_stage,
    soft_topk_mask,
)
from ..constraints.combination_catalogue import PRIMITIVES, catalogue_payload
from ..encoders.condition_encoders import CompositionalConditionEncoder, V4LeakageEncoder, v4_encoder_loss
from ..utils.config import AttackConfig, DefenseConfig, parse_csv_strings
from ..data import choose_stratified_subset, load_cw_data
from ..guidance.diffusion_guidance import (
    continuous_refine_logits,
    defense_guidance_loss,
    defense_target_risk,
    differentiable_ddim_sample,
    enforce_defense_risk_guard,
    guided_ddim_sample,
    policy_diversity_loss,
    soft_allocation,
)
from ..diffusion.models import build_policy_diffusion
from ..projection.padding import (
    crop_ragged_for_attacker,
    normalized_template_entropy,
    project_policy_to_template,
    render_batch_variable,
    renderer_options_from_config,
    save_ragged_npz,
)
from ..diffusion.policy import make_prior_logits
from ..constraints.preferences import PreferencePool
from ..encoders.prefix import extract_prefix_condition, nonzero_trace
from ..guidance.strong_surrogates import (
    build_attack_context,
    ensemble_metrics_from_rendered,
    ensemble_utility_maps,
    resolve_guidance_positions,
    strong_global_targets,
)
from ..constraints.user_profiles import (
    UserDefenseProfile,
    generate_profile_splits,
    load_profiles,
    profile_overlap,
    save_profiles,
    select_visit,
)
from ..utils import as_serializable_config, log, resolve_device, set_seed, write_csv, write_json


def _stage_begin(name: str, cfg: DefenseConfig, detail: str = "") -> float:
    suffix = f" | {detail}" if detail else ""
    log(f"========== {name} START{suffix} ==========", cfg.progress)
    return time.perf_counter()


def _stage_end(name: str, cfg: DefenseConfig, started_at: float, detail: str = "") -> None:
    suffix = f" | {detail}" if detail else ""
    log(f"========== {name} DONE in {time.perf_counter() - started_at:.1f}s{suffix} ==========", cfg.progress)


def _global_view_vector(view_profile: dict) -> np.ndarray:
    by_name = {row["view"]: float(row["view_score"]) for row in view_profile["rows"]}
    names = ("V_raw", "V_count", "V_interval", "V_burst", "V_rate", "V_cumul", "V_patch")
    return np.asarray([by_name.get(name, 0.0) for name in names], dtype=np.float32)


def _condition_for_trace(trace: np.ndarray, cfg: DefenseConfig):
    return extract_prefix_condition(
        trace,
        int(cfg.prefix_n),
        int(cfg.patch_num),
        max_trace_length=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        early_fraction=float(cfg.early_fraction),
    )


def _normalize_torch_map(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = torch.relu(values) * mask
    peak = result.reshape(len(result), -1).max(dim=1).values.reshape(-1, 1, 1).clamp_min(1e-8)
    return result / peak


def _repeat_count_for_namespace(cfg: DefenseConfig, visit_namespace: str) -> int:
    if str(visit_namespace).startswith("pareto-"):
        return max(1, int(cfg.stage3_repeats))
    return max(1, int(cfg.deployment_repeats))


def _read_json_if_exists(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _require_artifacts(run_dir: Path, stage: str) -> None:
    required: list[Path] = [run_dir / "run_config.json", run_dir / "split_indices.json"]
    if stage in {"2", "3"}:
        required.extend(
            [
                run_dir / "stage1_executable_condition" / "candidate_scorer_checkpoint.pt",
                run_dir / "stage1_executable_condition" / "strong_surrogate_ensemble.pt",
                run_dir / "stage1_executable_condition" / "candidate_metrics.json",
            ]
        )
    if stage == "3":
        required.extend(
            [
                run_dir / "stage2_user_diffusion" / "encoder_checkpoint.pt",
                run_dir / "stage2_user_diffusion" / "diffusion_guided_checkpoint.pt",
                run_dir / "stage2_user_diffusion" / "stage2_metrics.json",
                run_dir / "stage2_user_diffusion" / "user_profiles",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Stage {stage} requires previous DMMPv3 artifacts that are missing:\n{formatted}")


def _load_splits_for_run(run_dir: Path) -> dict[str, np.ndarray]:
    payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _prepare_profile_artifacts(run_dir: Path, cfg: DefenseConfig) -> dict[str, list[UserDefenseProfile]]:
    stage2_dir = run_dir / "stage2_user_diffusion"
    profiles_dir = stage2_dir / "user_profiles"
    if profiles_dir.exists():
        return load_profiles(profiles_dir)
    stage2_dir.mkdir(parents=True, exist_ok=True)
    profiles = generate_profile_splits(cfg)
    save_profiles(profiles_dir, profiles)
    write_json(stage2_dir / "combination_catalogue.json", catalogue_payload())
    overlaps = []
    for source in profiles["train"]:
        for target in profiles["test"]:
            overlaps.append({"source": source.profile_id, "target": target.profile_id, **profile_overlap(source, target)})
    write_json(stage2_dir / "profile_overlaps.json", {"rows": overlaps})
    return profiles


def _direction_target_incoming_share(clean_trace: np.ndarray, cfg: DefenseConfig) -> float | None:
    mode = str(cfg.direction_target).strip().lower()
    if mode == "none" or float(cfg.direction_correction_strength) <= 0.0:
        return None
    if mode == "balanced":
        return 0.5
    elif mode == "incoming":
        return 1.0
    else:
        prefix = nonzero_trace(clean_trace)[: int(cfg.prefix_n)]
        if prefix.size:
            target = float(np.mean(prefix < 0))
        else:
            target = 0.5
        return float(np.clip(max(target, float(cfg.min_incoming_dummy_share)), 0.0, 1.0))


def _prefix_only_rows(rows: Sequence[np.ndarray], cfg: DefenseConfig) -> np.ndarray:
    width = max(int(cfg.prefix_n), 1)
    result = np.zeros((len(rows), width), dtype=np.float32)
    for row_index, row in enumerate(rows):
        prefix = nonzero_trace(np.asarray(row, dtype=np.float32))[:width]
        result[row_index, : len(prefix)] = prefix
    return result


def _large_dataset_metric_indices(count: int, cfg: DefenseConfig, *, salt: int) -> np.ndarray:
    """Bound diagnostic allocations for large mixed/adaptive defended pools."""
    count = int(count)
    if count <= 0:
        return np.asarray([], dtype=np.int64)
    max_samples = max(
        1,
        int(getattr(cfg, "probe_samples", 0) or 0),
        int(getattr(cfg, "pareto_samples", 0) or 0),
        int(getattr(cfg, "stage3_fixed_probe_samples", 0) or 0),
    )
    if count <= max_samples:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(int(cfg.seed) + int(salt))
    return np.sort(rng.choice(count, size=max_samples, replace=False).astype(np.int64))


def _select_sequence_rows(rows: Sequence[np.ndarray], indices: np.ndarray) -> list[np.ndarray]:
    return [rows[int(index)] for index in np.asarray(indices, dtype=np.int64).tolist()]


def _ragged_adapter_stats(traces: Sequence[np.ndarray], origins: Sequence[np.ndarray], max_trace_length: int) -> dict[str, float]:
    retentions: list[float] = []
    visible_bandwidth: list[float] = []
    clipped: list[float] = []
    limit = int(max_trace_length)
    for trace, origin in zip(traces, origins):
        origin_bool = np.asarray(origin, dtype=bool)
        take = min(len(trace), limit)
        original_total = max(int(origin_bool.sum()), 1)
        retained = int(origin_bool[:take].sum())
        visible_dummy = int(take - retained)
        retentions.append(retained / original_total)
        visible_bandwidth.append(visible_dummy / original_total)
        clipped.append(float(len(trace) > limit))
    return {
        "attacker_input_real_packet_retention": float(np.mean(retentions)) if retentions else 1.0,
        "visible_dummy_overhead": float(np.mean(visible_bandwidth)) if visible_bandwidth else 0.0,
        "clip_rate": float(np.mean(clipped)) if clipped else 0.0,
    }


@torch.no_grad()
def _policy_prefix_label_free_metrics(
    policy_logits: np.ndarray,
    candidate_masks: np.ndarray,
    prefix_rows: np.ndarray,
    target_counts: np.ndarray,
    surrogate_bundle,
    cfg: DefenseConfig,
    device: torch.device,
) -> dict[str, float]:
    if len(policy_logits) == 0:
        return {
            "prefix_policy_label_free_attack_pressure": 1.0,
            "prefix_policy_label_free_row_worst_pressure": 1.0,
            "prefix_policy_label_free_mean_entropy": 0.0,
            "prefix_policy_label_free_worst_max_confidence": 1.0,
            "prefix_policy_label_free_worst_margin": 1.0,
        }
    totals = {
        name: {"entropy": 0.0, "max_confidence": 0.0, "margin": 0.0, "pressure": 0.0}
        for name in surrogate_bundle.attacker_names
    }
    row_worst_pressures = []
    batch_size = max(1, int(cfg.surrogate_gradient_batch_size))
    for start in range(0, len(policy_logits), batch_size):
        end = min(start + batch_size, len(policy_logits))
        logits_t = torch.as_tensor(policy_logits[start:end], dtype=torch.float32, device=device)
        mask_t = torch.as_tensor(candidate_masks[start:end], dtype=torch.float32, device=device)
        count_t = torch.as_tensor(target_counts[start:end], dtype=torch.float32, device=device)
        allocation = soft_allocation(logits_t, mask_t, count_t)
        attack_context = build_attack_context(prefix_rows[start:end], cfg, device)
        attack_logits = surrogate_bundle.logits_from_allocation(allocation, attack_context)
        batch_pressures = []
        for name, values in attack_logits.items():
            probabilities = torch.softmax(values, dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1) / math.log(max(values.shape[1], 2))
            top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
            margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
            confidence = probabilities.max(dim=1).values
            pressure = confidence + 0.50 * margin - 0.50 * entropy
            totals[name]["entropy"] += float(entropy.sum().cpu())
            totals[name]["max_confidence"] += float(confidence.sum().cpu())
            totals[name]["margin"] += float(margin.sum().cpu())
            totals[name]["pressure"] += float(pressure.sum().cpu())
            batch_pressures.append(pressure)
        if batch_pressures:
            row_worst_pressures.append(torch.stack(batch_pressures, dim=1).max(dim=1).values.cpu().numpy())
    result: dict[str, float] = {}
    attacker_pressures = []
    entropies = []
    confidences = []
    margins = []
    denominator = max(len(policy_logits), 1)
    for name, values in totals.items():
        entropy = values["entropy"] / denominator
        confidence = values["max_confidence"] / denominator
        margin = values["margin"] / denominator
        pressure = values["pressure"] / denominator
        result[f"prefix_policy_{name}_entropy"] = float(entropy)
        result[f"prefix_policy_{name}_max_confidence"] = float(confidence)
        result[f"prefix_policy_{name}_margin"] = float(margin)
        result[f"prefix_policy_{name}_label_free_pressure"] = float(pressure)
        attacker_pressures.append(float(pressure))
        entropies.append(float(entropy))
        confidences.append(float(confidence))
        margins.append(float(margin))
    row_worst = np.concatenate(row_worst_pressures, axis=0) if row_worst_pressures else np.zeros((0,), dtype=np.float32)
    result["prefix_policy_label_free_attack_pressure"] = float(max(attacker_pressures)) if attacker_pressures else 1.0
    result["prefix_policy_label_free_row_worst_pressure"] = float(np.mean(row_worst)) if len(row_worst) else 1.0
    result["prefix_policy_label_free_mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0
    result["prefix_policy_label_free_worst_max_confidence"] = float(max(confidences)) if confidences else 1.0
    result["prefix_policy_label_free_worst_margin"] = float(max(margins)) if margins else 1.0
    return result


def _template_summary(templates, cfg: DefenseConfig) -> dict[str, float]:
    if not templates:
        return {
            "template_entropy": 0.0,
            "dummy_outgoing_share": 0.0,
            "dummy_incoming_share": 0.0,
            "nonzero_policy_cells": 0.0,
            "v1_mode_usage_entropy": 0.0,
        }
    mode_counts = Counter(str(template.metadata.get("v1_mode", "")) for template in templates)
    mode_probabilities = np.asarray([value / len(templates) for value in mode_counts.values()], dtype=np.float64)
    mode_entropy = -float(np.sum(mode_probabilities * np.log(np.maximum(mode_probabilities, 1e-12))))
    mode_pool = _v1_mode_pool(cfg)
    if len(mode_pool) > 1:
        mode_entropy /= math.log(len(mode_pool))
    entropies: list[float] = []
    outgoing_shares: list[float] = []
    incoming_shares: list[float] = []
    nonzero_cells: list[float] = []
    for template in templates:
        counts = np.asarray(template.counts, dtype=np.float32)
        total = max(float(counts.sum()), 1.0)
        entropies.append(float(normalized_template_entropy(counts)))
        outgoing_shares.append(float(counts[0, :].sum() / total))
        incoming_shares.append(float(counts[1, :].sum() / total))
        nonzero_cells.append(float((counts.reshape(-1) > 0).sum()))
    return {
        "template_entropy": float(np.mean(entropies)),
        "dummy_outgoing_share": float(np.mean(outgoing_shares)),
        "dummy_incoming_share": float(np.mean(incoming_shares)),
        "nonzero_policy_cells": float(np.mean(nonzero_cells)),
        "v1_mode_usage_entropy": float(mode_entropy),
    }


def _row_normalize_tam(values: np.ndarray) -> np.ndarray:
    rows = np.asarray(values, dtype=np.float32).reshape(len(values), -1)
    totals = rows.sum(axis=1, keepdims=True)
    return rows / np.maximum(totals, 1e-8)


def _tam_perturbation_metrics(clean_rows: Sequence[np.ndarray], traces: Sequence[np.ndarray], cfg: DefenseConfig) -> dict[str, float]:
    if not clean_rows or not traces:
        return {
            "tam_raw_l1_shift_per_clean_packet": 0.0,
            "tam_distribution_l1_shift": 0.0,
            "tam_cosine_distance": 0.0,
            "tam_outgoing_distribution_l1_shift": 0.0,
            "tam_incoming_distribution_l1_shift": 0.0,
            "tam_added_count_ratio": 0.0,
        }
    clean_tam = build_rf_tam_input(
        np.asarray(clean_rows, dtype=np.float32),
        max_len=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        num_slots=int(cfg.surrogate_rf_num_slots),
    )
    defended_tam = _stage3_ragged_rf_tam(
        traces,
        int(cfg.surrogate_rf_num_slots),
        float(cfg.surrogate_rf_max_load_time),
    )
    clean_flat = clean_tam.reshape(len(clean_tam), -1).astype(np.float32)
    defended_flat = defended_tam.reshape(len(defended_tam), -1).astype(np.float32)
    clean_totals = np.maximum(clean_flat.sum(axis=1), 1.0)
    raw_l1 = np.abs(defended_flat - clean_flat).sum(axis=1) / clean_totals
    clean_prob = _row_normalize_tam(clean_tam)
    defended_prob = _row_normalize_tam(defended_tam)
    distribution_l1 = 0.5 * np.abs(defended_prob - clean_prob).sum(axis=1)
    dot = np.sum(clean_prob * defended_prob, axis=1)
    norm = np.linalg.norm(clean_prob, axis=1) * np.linalg.norm(defended_prob, axis=1)
    cosine_distance = 1.0 - dot / np.maximum(norm, 1e-8)
    direction_l1 = []
    for channel in range(2):
        clean_channel = clean_tam[:, channel, :]
        defended_channel = defended_tam[:, channel, :]
        clean_channel_prob = clean_channel / np.maximum(clean_channel.sum(axis=1, keepdims=True), 1e-8)
        defended_channel_prob = defended_channel / np.maximum(defended_channel.sum(axis=1, keepdims=True), 1e-8)
        direction_l1.append(0.5 * np.abs(defended_channel_prob - clean_channel_prob).sum(axis=1))
    added_ratio = np.maximum(defended_flat.sum(axis=1) - clean_flat.sum(axis=1), 0.0) / clean_totals
    return {
        "tam_raw_l1_shift_per_clean_packet": float(np.mean(raw_l1)),
        "tam_distribution_l1_shift": float(np.mean(distribution_l1)),
        "tam_cosine_distance": float(np.mean(cosine_distance)),
        "tam_outgoing_distribution_l1_shift": float(np.mean(direction_l1[0])),
        "tam_incoming_distribution_l1_shift": float(np.mean(direction_l1[1])),
        "tam_added_count_ratio": float(np.mean(added_ratio)),
    }


_EARLY_V1_MODE_POOL = ("early_uniform", "early_burst", "early_balanced", "early_directional", "early_saliency")
_LEGACY_DIRECT_V1_MODE_POOL = (
    "gap-adaptive-padding",
    "burst-obfuscation",
    "direction-regularization",
    "rate-smoothing",
    "public-prototype-shaping",
)


def _v1_mode_pool(cfg: DefenseConfig | str) -> tuple[str, ...]:
    value = str(getattr(cfg, "v1_mode_pool", cfg)).strip().lower().replace("-", "_")
    if value in {"legacy", "legacy_direct", "legacy_five_pool", "legacy_five_pool_direct"}:
        return _LEGACY_DIRECT_V1_MODE_POOL
    return _EARLY_V1_MODE_POOL


def _normalize_numpy_map(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0) * np.asarray(mask, dtype=np.float32)
    total = float(result.sum())
    if total <= 1e-8:
        result = np.asarray(mask, dtype=np.float32).copy()
        total = float(result.sum())
    return result / max(total, 1e-8)


def _v1_mode_prior(condition, candidate_mask: np.ndarray, mode: str) -> np.ndarray:
    mask = np.asarray(candidate_mask, dtype=np.float32) * np.asarray(condition.allowed_mask, dtype=np.float32)
    if mode == "early_uniform":
        values = mask
    elif mode == "early_burst":
        values = mask * (0.25 + np.asarray(condition.burst_saliency, dtype=np.float32).reshape(1, -1))
    elif mode == "early_balanced":
        values = mask.copy()
        direction_totals = values.sum(axis=1, keepdims=True)
        values = values / np.maximum(direction_totals, 1e-8)
    elif mode == "early_directional":
        out_ratio = float(condition.metadata.get("out_ratio", 0.5))
        in_ratio = float(condition.metadata.get("in_ratio", 0.5))
        direction_weight = np.asarray([1.5, 0.5] if out_ratio < in_ratio else [0.5, 1.5], dtype=np.float32)
        values = mask * direction_weight.reshape(2, 1)
    elif mode == "early_saliency":
        values = mask * (0.20 + np.asarray(condition.saliency, dtype=np.float32))
    elif mode == "gap-adaptive-padding":
        gap = _normalize_numpy_map(np.asarray(condition.gap_saliency, dtype=np.float32).reshape(1, -1), mask[:1])
        values = mask * (0.35 + 2.00 * gap)
    elif mode == "burst-obfuscation":
        burst = _normalize_numpy_map(np.asarray(condition.burst_saliency, dtype=np.float32).reshape(1, -1), mask[:1])
        values = mask * (0.35 + 2.00 * burst)
    elif mode == "direction-regularization":
        out_ratio = float(condition.metadata.get("out_ratio", 0.5))
        in_ratio = float(condition.metadata.get("in_ratio", 0.5))
        weak_out = in_ratio > out_ratio
        direction_weight = np.asarray([1.55 if weak_out else 0.65, 0.65 if weak_out else 1.55], dtype=np.float32)
        values = mask * direction_weight.reshape(2, 1)
    elif mode == "rate-smoothing":
        rate = _normalize_numpy_map(np.asarray(condition.rate_saliency, dtype=np.float32).reshape(1, -1), mask[:1])
        low_rate = _normalize_numpy_map((1.0 - rate) * mask[:1], mask[:1])
        values = mask * (0.35 + 1.65 * low_rate + 0.35 * np.asarray(condition.saliency, dtype=np.float32))
    elif mode == "public-prototype-shaping":
        prototype = _normalize_numpy_map(np.asarray(condition.public_prototype, dtype=np.float32).reshape(1, -1), mask[:1])
        values = mask * (0.35 + 1.90 * prototype)
    else:
        values = mask
    return _normalize_numpy_map(values, mask)


def _mixed_preference(
    condition,
    executable_utility: np.ndarray,
    candidate_mask: np.ndarray,
    visit,
    pool: PreferencePool,
    v1_mode_weight: float,
    mode_pool: Sequence[str],
) -> np.ndarray:
    maps = pool.compute_all(condition, candidate_mask, executable_utility, candidate_mask)
    mixed = np.zeros_like(candidate_mask, dtype=np.float32)
    for primitive, weight in zip(PRIMITIVES, visit.primitive_weights.tolist()):
        if float(weight) > 0:
            mixed += float(weight) * np.asarray(maps[primitive], dtype=np.float32)
    mask = np.asarray(candidate_mask, dtype=np.float32) * np.asarray(condition.allowed_mask, dtype=np.float32)
    mixed = _normalize_numpy_map(mixed, mask)
    mode_pool = tuple(mode_pool)
    mode = mode_pool[int(visit.combination_index) % len(mode_pool)]
    mode_prior = _v1_mode_prior(condition, mask, mode)
    weight = float(np.clip(v1_mode_weight, 0.0, 1.0))
    return _normalize_numpy_map((1.0 - weight) * mixed + weight * mode_prior, mask).astype(np.float32)


def _prefix_hidden_alignment_loss(
    prefix_features: torch.Tensor,
    denoiser_hidden: torch.Tensor,
    prefix_projector: nn.Module,
    hidden_projector: nn.Module,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    prefix_z = nn.functional.normalize(prefix_projector(prefix_features), dim=1)
    hidden_z = nn.functional.normalize(hidden_projector(denoiser_hidden), dim=1)
    cosine = (prefix_z * hidden_z).sum(dim=1).mean()
    if len(prefix_z) < 2:
        return 1.0 - cosine, cosine
    logits = (prefix_z @ hidden_z.t()) / max(float(temperature), 1e-6)
    targets = torch.arange(len(prefix_z), dtype=torch.long, device=prefix_z.device)
    loss = 0.5 * (nn.functional.cross_entropy(logits, targets) + nn.functional.cross_entropy(logits.t(), targets))
    return loss, cosine


def _backward_defense_first(
    hard_loss: torch.Tensor,
    soft_loss: torch.Tensor,
    parameters: Sequence[torch.nn.Parameter],
    *,
    hard_weight: float,
    soft_scale: float,
) -> dict[str, float]:
    hard_grads = torch.autograd.grad(hard_loss, parameters, retain_graph=True, allow_unused=True)
    soft_grads = torch.autograd.grad(soft_loss, parameters, allow_unused=True)
    shared = [(hard, soft) for hard, soft in zip(hard_grads, soft_grads) if hard is not None and soft is not None]
    if shared:
        dot = sum((hard * soft).sum() for hard, soft in shared)
        hard_norm = sum((hard * hard).sum() for hard, _ in shared).clamp_min(1e-12)
        projection = torch.clamp(-dot / hard_norm, min=0.0)
        hard_norm_value = float(torch.sqrt(hard_norm).detach().cpu())
        dot_value = float(dot.detach().cpu())
    else:
        projection = hard_loss.new_tensor(0.0)
        hard_norm_value = 0.0
        dot_value = 0.0
    for parameter, hard, soft in zip(parameters, hard_grads, soft_grads):
        hard_part = torch.zeros_like(parameter) if hard is None else hard
        soft_part = torch.zeros_like(parameter) if soft is None else soft
        if hard is not None and soft is not None:
            soft_part = soft_part + projection * hard_part
        parameter.grad = float(hard_weight) * hard_part + float(soft_scale) * soft_part
    return {
        "hard_soft_gradient_dot": dot_value,
        "hard_gradient_norm": hard_norm_value,
        "soft_projection_coefficient": float(projection.detach().cpu()),
    }


def _candidate_context(
    raw: np.ndarray,
    cfg: DefenseConfig,
    scorer,
    selected_views: Sequence[str],
    scorer_mean: np.ndarray,
    scorer_scale: np.ndarray,
    surrogate_bundle,
    labels: np.ndarray,
    device: torch.device,
    *,
    include_exact_utility: bool,
) -> dict[str, Any]:
    candidate_features, allowed_masks, structures = build_candidate_features(raw, cfg, selected_views)
    normalized = np.clip((candidate_features - scorer_mean) / scorer_scale, -8.0, 8.0).astype(np.float32)
    with torch.no_grad():
        scores = scorer(
            torch.as_tensor(normalized, dtype=torch.float32, device=device),
            torch.as_tensor(allowed_masks, dtype=torch.float32, device=device),
        )
        candidates = soft_topk_mask(
            scores,
            torch.as_tensor(allowed_masks, dtype=torch.float32, device=device),
            int(cfg.candidate_topk),
            float(cfg.candidate_temperature),
            hard=not bool(cfg.candidate_soft_topk),
        )
    predicted_utility = scores.detach().cpu().numpy().astype(np.float32)
    if include_exact_utility:
        utility = ensemble_utility_maps(raw, labels, allowed_masks, surrogate_bundle, cfg, device)
    else:
        peak = predicted_utility.reshape(len(predicted_utility), -1).max(axis=1).reshape(-1, 1, 1)
        utility = predicted_utility / np.maximum(peak, 1e-8)
    return {
        "candidate_features": candidate_features.astype(np.float32),
        "allowed_masks": allowed_masks.astype(np.float32),
        "structures": structures.astype(np.float32),
        "utility": utility.astype(np.float32),
        "candidate_masks": candidates.detach().cpu().numpy().astype(np.float32),
        "conditions": [_condition_for_trace(trace, cfg) for trace in raw],
    }


def _encoder_input(context: dict[str, Any], view_vector: np.ndarray) -> np.ndarray:
    repeated = np.repeat(np.asarray(view_vector, dtype=np.float32).reshape(1, -1), len(context["candidate_features"]), axis=0)
    return np.concatenate(
        [context["candidate_features"], context["utility"].reshape(len(repeated), -1), context["candidate_masks"].reshape(len(repeated), -1), repeated],
        axis=1,
    ).astype(np.float32)


def train_v4_encoder(
    raw: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    run_dir: Path,
    cfg: DefenseConfig,
    device: torch.device,
):
    log("[Stage 2/3] Loading strong DF/RF ensemble and building encoder supervision...", cfg.progress)
    scorer, selected_views, scorer_mean, scorer_scale, surrogate_bundle = load_candidate_components(run_dir, cfg, device)
    stage_dir = run_dir / "stage2_user_diffusion"
    checkpoint_path = stage_dir / "encoder_checkpoint.pt"
    if checkpoint_path.is_file():
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        ecfg = payload["config"]
        selected_indices = np.asarray(payload["train_indices"], dtype=np.int64)
        train_raw = raw[selected_indices]
        train_y = labels[selected_indices]
        context = _candidate_context(
            train_raw,
            cfg,
            scorer,
            selected_views,
            scorer_mean,
            scorer_scale,
            surrogate_bundle,
            train_y,
            device,
            include_exact_utility=True,
        )
        model = V4LeakageEncoder(int(ecfg["input_dim"]), int(ecfg["structure_dim"]), int(ecfg["patch_num"]), int(ecfg["hidden_dim"])).to(device)
        model.load_state_dict(payload["model_state"])
        model.eval()
        log(
            f"[Stage 2/3] Reusing encoder checkpoint: {checkpoint_path}; samples={len(selected_indices)}",
            cfg.progress,
        )
        return model, payload, context, surrogate_bundle
    count = min(len(train_indices), int(cfg.encoder_train_samples))
    local_indices = choose_stratified_subset(labels[train_indices], count, int(cfg.seed) + 401)
    selected_indices = train_indices[local_indices]
    train_raw = raw[selected_indices]
    train_y = labels[selected_indices]
    log(
        f"[Stage 2/3] Encoder samples selected: {len(train_raw)} traces; selected leakage views={','.join(selected_views)}",
        cfg.progress,
    )
    context = _candidate_context(
        train_raw,
        cfg,
        scorer,
        selected_views,
        scorer_mean,
        scorer_scale,
        surrogate_bundle,
        train_y,
        device,
        include_exact_utility=True,
    )
    view_profile = json.loads((run_dir / "stage1_executable_condition" / "view_profile.json").read_text(encoding="utf-8"))
    view_vector = _global_view_vector(view_profile)
    raw_features = _encoder_input(context, view_vector)
    feature_mean = raw_features.mean(axis=0, keepdims=True).astype(np.float32)
    feature_scale = (raw_features.std(axis=0, keepdims=True) + 1e-4).astype(np.float32)
    features = np.clip((raw_features - feature_mean) / feature_scale, -8.0, 8.0).astype(np.float32)
    structure_mean = context["structures"].mean(axis=0, keepdims=True).astype(np.float32)
    structure_scale = (context["structures"].std(axis=0, keepdims=True) + 1e-4).astype(np.float32)
    structures = np.clip((context["structures"] - structure_mean) / structure_scale, -8.0, 8.0).astype(np.float32)
    log(f"[Stage 2/3] Computing strong-attacker global targets for {len(train_raw)} traces...", cfg.progress)
    global_targets = strong_global_targets(train_raw, surrogate_bundle, cfg, device, int(cfg.hidden_dim))
    model = V4LeakageEncoder(features.shape[1], structures.shape[1], int(cfg.patch_num), int(cfg.hidden_dim)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.encoder_lr))
    rng = np.random.default_rng(int(cfg.seed) + 402)
    batch_size = min(int(cfg.batch_size), len(features))
    last: dict[str, float] = {"loss": 0.0}
    for epoch in range(1, int(cfg.encoder_epochs) + 1):
        order = rng.permutation(len(features))
        model.train()
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.as_tensor(features[idx], dtype=torch.float32, device=device)
            utility = torch.as_tensor(context["utility"][idx], dtype=torch.float32, device=device)
            candidate = torch.as_tensor(context["candidate_masks"][idx], dtype=torch.float32, device=device)
            allowed = torch.as_tensor(context["allowed_masks"][idx], dtype=torch.float32, device=device)
            structure = torch.as_tensor(structures[idx], dtype=torch.float32, device=device)
            global_target = torch.as_tensor(global_targets[idx], dtype=torch.float32, device=device)
            loss, last = v4_encoder_loss(model(xb), utility, candidate, allowed, structure, global_target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        log(
            f"{str(cfg.version).upper()} Stage 2 encoder: epoch {epoch}/{cfg.encoder_epochs}, "
            f"loss={last['loss']:.6f}, global={last['global']:.6f}",
            cfg.progress,
        )
    model.eval()
    stage_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": {
            "input_dim": int(features.shape[1]),
            "structure_dim": int(structures.shape[1]),
            "patch_num": int(cfg.patch_num),
            "hidden_dim": int(cfg.hidden_dim),
            "selected_views": list(selected_views),
            "feature_mean": feature_mean,
            "feature_scale": feature_scale,
            "structure_mean": structure_mean,
            "structure_scale": structure_scale,
            "view_vector": view_vector,
        },
        "train_indices": selected_indices.astype(np.int64),
        "metrics": {**last, "samples": int(len(features))},
    }
    torch.save(payload, stage_dir / "encoder_checkpoint.pt")
    log(
        f"[Stage 2/3] Encoder checkpoint saved: {stage_dir / 'encoder_checkpoint.pt'}; "
        f"loss={float(last.get('loss', 0.0)):.6f}",
        cfg.progress,
    )
    return model, payload, context, surrogate_bundle


def _visit_conditions(
    indices: np.ndarray,
    trace_ids: np.ndarray,
    contexts: dict[str, Any],
    profiles: Sequence[UserDefenseProfile],
    cfg: DefenseConfig,
    rng: np.random.Generator,
    visit_nonce: str,
) -> dict[str, np.ndarray]:
    pool = PreferencePool(patch_num=int(cfg.patch_num))
    mode_pool = _v1_mode_pool(cfg)
    preferences, profile_masks, selected_masks, weights, profile_ids, combo_indices, mode_names = [], [], [], [], [], [], []
    for row, global_index in enumerate(indices.tolist()):
        profile = profiles[int(rng.integers(0, len(profiles)))]
        visit = select_visit(profile, visit_nonce, str(trace_ids[global_index]), str(cfg.visit_selector))
        preference = _mixed_preference(
            contexts["conditions"][row],
            contexts["utility"][row],
            contexts["candidate_masks"][row],
            visit,
            pool,
            float(cfg.v1_mode_prior_weight),
            mode_pool,
        )
        preferences.append(preference)
        profile_masks.append(np.asarray(profile.profile_mask_20d, dtype=np.float32))
        selected_masks.append(visit.selected_primitive_mask)
        weights.append(visit.primitive_weights)
        profile_ids.append(profile.profile_id)
        combo_indices.append(visit.combination_index)
        mode_names.append(mode_pool[int(visit.combination_index) % len(mode_pool)])
    return {
        "preferences": np.stack(preferences).astype(np.float32),
        "profile_masks": np.stack(profile_masks).astype(np.float32),
        "selected_masks": np.stack(selected_masks).astype(np.float32),
        "weights": np.stack(weights).astype(np.float32),
        "profile_ids": np.asarray(profile_ids),
        "combination_indices": np.asarray(combo_indices, dtype=np.int64),
        "v1_mode_names": np.asarray(mode_names),
    }


def _stage2_diffusion_state_payload(
    *,
    diffusion,
    condition_encoder,
    prefix_align_projector,
    hidden_align_projector,
    optimizer,
    step: int,
    metrics: dict[str, float],
    rng: np.random.Generator,
    combination_counts: Counter[int],
    mode_counts: Counter[str],
    profile_counts: Counter[str],
    cfg: DefenseConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "diffusion_state": diffusion.state_dict(),
        "condition_encoder_state": condition_encoder.state_dict(),
        "prefix_align_projector_state": prefix_align_projector.state_dict(),
        "hidden_align_projector_state": hidden_align_projector.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": int(step),
        "metrics": dict(metrics),
        "rng_state": rng.bit_generator.state,
        "torch_rng_state": torch.get_rng_state(),
        "combination_counts": dict(combination_counts),
        "mode_counts": dict(mode_counts),
        "profile_counts": dict(profile_counts),
        "config": {
            "patch_num": int(cfg.patch_num),
            "hidden_dim": int(cfg.hidden_dim),
            "diffusion_steps": int(cfg.diffusion_steps),
            "v1_mode_pool": str(cfg.v1_mode_pool),
            "guidance_label_mode": str(cfg.guidance_label_mode),
        },
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def _load_stage2_diffusion_state(
    checkpoint: Path,
    *,
    diffusion,
    condition_encoder,
    prefix_align_projector,
    hidden_align_projector,
    optimizer,
    rng: np.random.Generator,
    combination_counts: Counter[int],
    mode_counts: Counter[str],
    profile_counts: Counter[str],
    device: torch.device,
    cfg: DefenseConfig,
) -> tuple[int, dict[str, float], bool]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model_cfg = payload.get("config", {})
    compatible = (
        int(model_cfg.get("patch_num", cfg.patch_num)) == int(cfg.patch_num)
        and int(model_cfg.get("hidden_dim", cfg.hidden_dim)) == int(cfg.hidden_dim)
        and int(model_cfg.get("diffusion_steps", cfg.diffusion_steps)) == int(cfg.diffusion_steps)
        and str(model_cfg.get("guidance_label_mode", "pseudo")) == str(cfg.guidance_label_mode)
    )
    if not compatible:
        return 0, {}, False
    diffusion.load_state_dict(payload["diffusion_state"])
    condition_encoder.load_state_dict(payload["condition_encoder_state"])
    prefix_align_projector.load_state_dict(payload["prefix_align_projector_state"])
    hidden_align_projector.load_state_dict(payload["hidden_align_projector_state"])
    if "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    if "rng_state" in payload:
        rng.bit_generator.state = payload["rng_state"]
    if "torch_rng_state" in payload:
        torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
    combination_counts.update({int(key): int(value) for key, value in payload.get("combination_counts", {}).items()})
    mode_counts.update({str(key): int(value) for key, value in payload.get("mode_counts", {}).items()})
    profile_counts.update({str(key): int(value) for key, value in payload.get("profile_counts", {}).items()})
    return int(payload.get("step", 0)), dict(payload.get("metrics", {})), True


def train_v4_diffusion(
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    train_indices: np.ndarray,
    run_dir: Path,
    cfg: DefenseConfig,
    device: torch.device,
    encoder: V4LeakageEncoder,
    encoder_payload: dict,
    context: dict[str, Any],
    surrogate_bundle,
    profiles: dict[str, list[UserDefenseProfile]],
) -> dict:
    stage_dir = run_dir / "stage2_user_diffusion"
    encoder_cfg = encoder_payload["config"]
    feature_mean = np.asarray(encoder_cfg["feature_mean"], dtype=np.float32)
    feature_scale = np.asarray(encoder_cfg["feature_scale"], dtype=np.float32)
    view_vector = np.asarray(encoder_cfg["view_vector"], dtype=np.float32)
    raw_features = _encoder_input(context, view_vector)
    encoder_features = np.clip((raw_features - feature_mean) / feature_scale, -8.0, 8.0).astype(np.float32)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    surrogate_bundle.freeze()
    condition_encoder = CompositionalConditionEncoder(int(cfg.patch_num), int(cfg.hidden_dim), int(cfg.hidden_dim)).to(device)
    diffusion = build_policy_diffusion(int(cfg.hidden_dim), int(cfg.patch_num), int(cfg.hidden_dim), int(cfg.diffusion_steps)).to(device)
    align_dim = int(cfg.prefix_hidden_align_dim)
    prefix_align_projector = nn.Sequential(
        nn.Linear(int(cfg.hidden_dim), align_dim),
        nn.LayerNorm(align_dim),
    ).to(device)
    hidden_align_projector = nn.Sequential(
        nn.Linear(int(cfg.hidden_dim), align_dim),
        nn.LayerNorm(align_dim),
    ).to(device)
    optimized_parameters = (
        list(condition_encoder.parameters())
        + list(diffusion.parameters())
        + list(prefix_align_projector.parameters())
        + list(hidden_align_projector.parameters())
    )
    optimizer = torch.optim.AdamW(optimized_parameters, lr=float(cfg.diffusion_lr))
    rng = np.random.default_rng(int(cfg.seed) + 500)
    train_prefix_rows = _prefix_only_rows(raw[train_indices], cfg)
    guidance_positions, guidance_target_metrics = resolve_guidance_positions(
        train_prefix_rows,
        labels[train_indices],
        surrogate_bundle,
        cfg,
        device,
    )
    steps = max(int(cfg.diffusion_train_steps), 1)
    guidance_start = max(0, steps - max(int(cfg.guidance_train_steps), 0))
    last: dict[str, float] = {}
    warmstart_saved = False
    batch_size = min(int(cfg.batch_size), len(train_indices))
    budgets = cfg.budget_values
    combination_counts: Counter[int] = Counter()
    mode_counts: Counter[str] = Counter()
    profile_counts: Counter[str] = Counter()
    start_step = 0
    resume_checkpoint = stage_dir / "diffusion_resume_checkpoint.pt"
    warmstart_checkpoint = stage_dir / "diffusion_warmstart_checkpoint.pt"
    for checkpoint in (resume_checkpoint, warmstart_checkpoint):
        if checkpoint.is_file():
            loaded_step, loaded_metrics, loaded = _load_stage2_diffusion_state(
                checkpoint,
                diffusion=diffusion,
                condition_encoder=condition_encoder,
                prefix_align_projector=prefix_align_projector,
                hidden_align_projector=hidden_align_projector,
                optimizer=optimizer,
                rng=rng,
                combination_counts=combination_counts,
                mode_counts=mode_counts,
                profile_counts=profile_counts,
                device=device,
                cfg=cfg,
            )
            if loaded:
                start_step = min(max(int(loaded_step), 0), steps)
                last = loaded_metrics
                warmstart_saved = warmstart_checkpoint.is_file()
                log(f"[Stage 2/3] Resuming diffusion from {checkpoint}: next_step={start_step + 1}/{steps}", cfg.progress)
                break
    log(
        f"[Stage 2/3] Diffusion training begins: steps={steps}, batch_size={batch_size}, "
        f"guidance_start={guidance_start}, budgets={','.join(f'{value:.3f}' for value in budgets)}, "
        f"guidance_target={guidance_target_metrics.get('guidance_target_source')}, "
        f"target_samples={float(guidance_target_metrics.get('guidance_target_samples', 0.0)):.0f}, "
        f"pseudo_target_conf={float(guidance_target_metrics.get('guidance_pseudo_target_mean_confidence', 0.0)):.4f}, "
        f"true_target_conf={float(guidance_target_metrics.get('guidance_true_target_mean_confidence', 0.0)):.4f}, "
        f"start_step={start_step}",
        cfg.progress,
    )
    for step in range(start_step, steps):
        chosen_local = rng.choice(len(train_indices), size=batch_size, replace=len(train_indices) < batch_size)
        chosen_global = train_indices[chosen_local]
        batch_context = {key: value[chosen_local] if isinstance(value, np.ndarray) and len(value) == len(train_indices) else value for key, value in context.items()}
        batch_context["conditions"] = [context["conditions"][int(item)] for item in chosen_local.tolist()]
        visits = _visit_conditions(chosen_global, trace_ids, batch_context, profiles["train"], cfg, rng, f"train-step-{step}")
        for value in visits["combination_indices"].tolist():
            combination_counts[int(value)] += 1
        for value in visits["v1_mode_names"].tolist():
            mode_counts[str(value)] += 1
        for value in visits["profile_ids"].tolist():
            profile_counts[str(value)] += 1
        selected_budget = np.asarray(rng.choice(budgets, size=batch_size), dtype=np.float32)
        budget_trace_lengths = np.asarray([nonzero_trace(raw[index]).size for index in chosen_global], dtype=np.float32)
        target_counts = np.maximum(np.rint(budget_trace_lengths * selected_budget), 1.0).astype(np.float32)
        xb = torch.as_tensor(encoder_features[chosen_local], dtype=torch.float32, device=device)
        with torch.no_grad():
            encoded = encoder(xb)
            c_global = encoded["c_global"]
            c_leakage = torch.nn.functional.softplus(encoded["c_leakage"])
        candidate_t = torch.as_tensor(batch_context["candidate_masks"], dtype=torch.float32, device=device)
        preference_t = torch.as_tensor(visits["preferences"], dtype=torch.float32, device=device)
        preference_condition_t = preference_t
        weights_t = torch.as_tensor(visits["weights"], dtype=torch.float32, device=device)
        selected_t = torch.as_tensor(visits["selected_masks"], dtype=torch.float32, device=device)
        profile_t = torch.as_tensor(visits["profile_masks"], dtype=torch.float32, device=device)
        if not bool(cfg.condition_preference_map):
            preference_condition_t = torch.zeros_like(preference_condition_t)
        if not bool(cfg.condition_profile_mask):
            profile_t = torch.zeros_like(profile_t)
        if not bool(cfg.condition_selected_mask):
            selected_t = torch.zeros_like(selected_t)
        if not bool(cfg.condition_preference_weights):
            weights_t = torch.zeros_like(weights_t)
        condition = condition_encoder(c_global, c_leakage, candidate_t, preference_condition_t, weights_t, selected_t, profile_t)
        prior_rows = [
            make_prior_logits(
                batch_context["utility"][row],
                visits["preferences"][row],
                batch_context["candidate_masks"][row],
                rng=rng,
                alpha_leak=float(cfg.prior_leak_weight),
                alpha_pref=float(cfg.prior_preference_weight),
                alpha_noise=float(cfg.prior_noise_std),
            ).reshape(-1)
            for row in range(batch_size)
        ]
        x0 = torch.as_tensor(np.stack(prior_rows), dtype=torch.float32, device=device)
        budget_t = torch.as_tensor(selected_budget, dtype=torch.float32, device=device)
        timestep = torch.randint(0, int(cfg.diffusion_steps), (batch_size,), device=device, dtype=torch.long)
        noisy, target_noise = diffusion.q_sample(x0, timestep)
        predicted_noise, denoiser_hidden = diffusion.denoiser(noisy, timestep, condition, budget_t, return_hidden=True)
        denoise_loss = nn.functional.mse_loss(predicted_noise, target_noise)
        alignment_loss, alignment_cosine = _prefix_hidden_alignment_loss(
            c_global,
            denoiser_hidden,
            prefix_align_projector,
            hidden_align_projector,
            float(cfg.prefix_hidden_align_temperature),
        )
        alpha = diffusion.alpha_cumprod[timestep].reshape(-1, 1)
        predicted_x0 = ((noisy - torch.sqrt(1.0 - alpha) * predicted_noise) / torch.sqrt(alpha)).clamp(-diffusion.x0_clip, diffusion.x0_clip)
        predicted_map = predicted_x0.reshape(-1, 2, int(cfg.patch_num))
        target_count_t = torch.as_tensor(target_counts, dtype=torch.float32, device=device)
        allocation = soft_allocation(predicted_map, candidate_t, target_count_t)
        pref_flat = preference_t.reshape(batch_size, -1)
        alloc_flat = allocation.reshape(batch_size, -1)
        raw_preference_loss = 1.0 - nn.functional.cosine_similarity(alloc_flat, pref_flat, dim=1).mean()
        preference_loss = raw_preference_loss
        preference_gate_value = predicted_map.new_tensor(1.0)
        if bool(cfg.preference_attack_gate) and float(cfg.preference_weight) > 0:
            preference_loss = raw_preference_loss * 0.0
            preference_gate_value = predicted_map.new_tensor(0.0)
        preference_attack_risk = predicted_map.new_tensor(0.0)
        defense_attack_risk = predicted_map.new_tensor(0.0)
        outside = 1.0 - torch.as_tensor(batch_context["allowed_masks"], dtype=torch.float32, device=device)
        constraint_loss = (torch.relu(predicted_map) * outside).mean()
        shuffled_profile = torch.roll(profile_t, shifts=1, dims=0)
        shuffled_condition = condition_encoder(c_global, c_leakage, candidate_t, preference_condition_t, weights_t, selected_t, shuffled_profile)
        profile_distance = (condition - shuffled_condition).abs().mean(dim=1)
        profile_loss = nn.functional.relu(0.02 - profile_distance).mean()
        defense_loss = torch.zeros((), device=device)
        full_sample_defense_loss = torch.zeros((), device=device)
        diversity_loss = torch.zeros((), device=device)
        if step >= guidance_start and float(cfg.guidance_weight) > 0:
            guidance_count = min(batch_size, max(1, int(cfg.surrogate_gradient_batch_size)))
            guidance_rows = np.asarray(rng.choice(batch_size, size=guidance_count, replace=False), dtype=np.int64)
            guidance_t = torch.as_tensor(guidance_rows, dtype=torch.long, device=device)
            attack_context = build_attack_context(train_prefix_rows[chosen_local[guidance_rows]], cfg, device)
            guidance_pos = torch.as_tensor(guidance_positions[chosen_local[guidance_rows]], dtype=torch.long, device=device)
            defense_loss = defense_guidance_loss(
                predicted_map.index_select(0, guidance_t),
                candidate_t.index_select(0, guidance_t),
                target_count_t.index_select(0, guidance_t),
                surrogate_bundle,
                attack_context,
                guidance_pos,
                float(cfg.surrogate_robust_weight),
                float(cfg.defense_soft_utility_weight),
            )
            if bool(cfg.preference_attack_gate) and float(cfg.preference_weight) > 0:
                with torch.no_grad():
                    centered_preference = preference_t.index_select(0, guidance_t)
                    centered_preference = centered_preference - centered_preference.mean(dim=(1, 2), keepdim=True)
                    preference_attack_risk = defense_target_risk(
                        centered_preference,
                        candidate_t.index_select(0, guidance_t),
                        target_count_t.index_select(0, guidance_t),
                        surrogate_bundle,
                        attack_context,
                        guidance_pos,
                    )
                    defense_attack_risk = defense_target_risk(
                        predicted_map.index_select(0, guidance_t),
                        candidate_t.index_select(0, guidance_t),
                        target_count_t.index_select(0, guidance_t),
                        surrogate_bundle,
                        attack_context,
                        guidance_pos,
                    )
                    margin = float(cfg.preference_attack_gate_margin)
                    preference_gate_value = (preference_attack_risk <= defense_attack_risk.detach() + margin).to(dtype=predicted_map.dtype)
                preference_loss = raw_preference_loss * preference_gate_value
            full_interval = max(1, int(cfg.full_sample_guidance_interval))
            if (step + 1 - guidance_start) % full_interval == 0:
                sampled_guidance = differentiable_ddim_sample(
                    diffusion,
                    condition.index_select(0, guidance_t),
                    budget_t.index_select(0, guidance_t),
                    sampler_steps=int(cfg.full_sample_guidance_steps),
                ).reshape(-1, 2, int(cfg.patch_num))
                full_sample_defense_loss = defense_guidance_loss(
                    sampled_guidance,
                    candidate_t.index_select(0, guidance_t),
                    target_count_t.index_select(0, guidance_t),
                    surrogate_bundle,
                    attack_context,
                    guidance_pos,
                    float(cfg.surrogate_robust_weight),
                    float(cfg.defense_soft_utility_weight),
                )
                defense_loss = torch.maximum(defense_loss, full_sample_defense_loss)
            second_noisy, _ = diffusion.q_sample(x0, timestep)
            second_noise = diffusion.denoiser(second_noisy, timestep, condition, budget_t)
            second_x0 = ((second_noisy - torch.sqrt(1.0 - alpha) * second_noise) / torch.sqrt(alpha)).reshape_as(predicted_map)
            diversity_loss = policy_diversity_loss(predicted_map, second_x0, candidate_t)
        soft_loss = (
            denoise_loss
            + float(cfg.preference_weight) * preference_loss
            + float(cfg.diversity_weight) * diversity_loss
            + float(cfg.constraint_weight) * constraint_loss
            + float(cfg.profile_weight) * profile_loss
            + float(cfg.prefix_hidden_align_weight) * alignment_loss
        )
        if step >= guidance_start and float(cfg.guidance_weight) > 0:
            loss = (
                float(cfg.defense_hard_weight) * defense_loss
                + float(cfg.defense_soft_objective_scale) * soft_loss
            )
        else:
            loss = soft_loss
        optimizer.zero_grad(set_to_none=True)
        gradient_report = {
            "hard_soft_gradient_dot": 0.0,
            "hard_gradient_norm": 0.0,
            "soft_projection_coefficient": 0.0,
        }
        if step >= guidance_start and float(cfg.guidance_weight) > 0:
            gradient_report = _backward_defense_first(
                defense_loss,
                soft_loss,
                optimized_parameters,
                hard_weight=float(cfg.defense_hard_weight),
                soft_scale=float(cfg.defense_soft_objective_scale),
            )
        else:
            soft_loss.backward()
        torch.nn.utils.clip_grad_norm_(optimized_parameters, 5.0)
        optimizer.step()
        last = {
            "loss": float(loss.detach().cpu()),
            "denoise": float(denoise_loss.detach().cpu()),
            "preference": float(raw_preference_loss.detach().cpu()),
            "preference_gated": float(preference_loss.detach().cpu()),
            "preference_gate": float(preference_gate_value.detach().cpu()),
            "preference_attack_risk": float(preference_attack_risk.detach().cpu()),
            "defense_attack_risk": float(defense_attack_risk.detach().cpu()),
            "defense": float(defense_loss.detach().cpu()),
            "full_sample_defense": float(full_sample_defense_loss.detach().cpu()),
            "diversity": float(diversity_loss.detach().cpu()),
            "constraint": float(constraint_loss.detach().cpu()),
            "profile": float(profile_loss.detach().cpu()),
            "alignment": float(alignment_loss.detach().cpu()),
            "alignment_cosine": float(alignment_cosine.detach().cpu()),
            **gradient_report,
        }
        if not warmstart_saved and step + 1 >= max(guidance_start, 1):
            torch.save(
                _stage2_diffusion_state_payload(
                    diffusion=diffusion,
                    condition_encoder=condition_encoder,
                    prefix_align_projector=prefix_align_projector,
                    hidden_align_projector=hidden_align_projector,
                    optimizer=optimizer,
                    step=int(step + 1),
                    metrics=last,
                    rng=rng,
                    combination_counts=combination_counts,
                    mode_counts=mode_counts,
                    profile_counts=profile_counts,
                    cfg=cfg,
                ),
                stage_dir / "diffusion_warmstart_checkpoint.pt",
            )
            warmstart_saved = True
        if step == 0 or step + 1 == steps or (step + 1) % max(int(cfg.log_every), 1) == 0:
            log(
                f"{str(cfg.version).upper()} Stage 2 diffusion: step {step + 1}/{steps}, "
                f"denoise={last['denoise']:.6f}, hard_defense={last['defense']:.6f}, "
                f"full_sample={last['full_sample_defense']:.6f}, "
                f"align={last['alignment']:.6f}, align_cos={last['alignment_cosine']:.4f}, "
                f"preference={last['preference_gated']:.6f}/{last['preference']:.6f}, "
                f"pref_gate={last['preference_gate']:.1f}, soft_proj={last['soft_projection_coefficient']:.4f}",
                cfg.progress,
            )
            torch.save(
                _stage2_diffusion_state_payload(
                    diffusion=diffusion,
                    condition_encoder=condition_encoder,
                    prefix_align_projector=prefix_align_projector,
                    hidden_align_projector=hidden_align_projector,
                    optimizer=optimizer,
                    step=int(step + 1),
                    metrics=last,
                    rng=rng,
                    combination_counts=combination_counts,
                    mode_counts=mode_counts,
                    profile_counts=profile_counts,
                    cfg=cfg,
                ),
                resume_checkpoint,
            )
            log(f"[Stage 2/3] Resume checkpoint saved: {resume_checkpoint}; step={step + 1}", cfg.progress)
    payload = {
        "diffusion_state": diffusion.state_dict(),
        "condition_encoder_state": condition_encoder.state_dict(),
        "prefix_align_projector_state": prefix_align_projector.state_dict(),
        "hidden_align_projector_state": hidden_align_projector.state_dict(),
        "config": {
            "condition_dim": int(cfg.hidden_dim),
            "patch_num": int(cfg.patch_num),
            "hidden_dim": int(cfg.hidden_dim),
            "diffusion_steps": int(cfg.diffusion_steps),
            "sampling_steps": int(cfg.sampling_steps),
            "v1_mode_pool": str(cfg.v1_mode_pool),
            "guidance_label_mode": str(cfg.guidance_label_mode),
        },
        "metrics": {**last, **guidance_target_metrics, "steps": int(steps), "guidance_start": int(guidance_start)},
    }
    torch.save(payload, stage_dir / "diffusion_guided_checkpoint.pt")
    log(
        f"[Stage 2/3] Guided diffusion checkpoint saved: {stage_dir / 'diffusion_guided_checkpoint.pt'}; "
        f"last_loss={float(last.get('loss', 0.0)):.6f}, defense={float(last.get('defense', 0.0)):.6f}",
        cfg.progress,
    )
    write_json(
        stage_dir / "profile_statistics.json",
        {
            "train_profile_usage": dict(profile_counts),
            "combination_usage": {str(key): value for key, value in combination_counts.items()},
            "v1_mode_usage": dict(mode_counts),
            "v1_mode_pool": str(cfg.v1_mode_pool),
            "profile_combination_mode": str(cfg.profile_combination_mode),
            "profile_pair_weight_range": [float(cfg.profile_pair_weight_min), float(cfg.profile_pair_weight_max)],
            "sample_records_saved": 0,
            "requested_guidance_attackers": str(cfg.guidance_attackers),
            "actual_guidance_surrogate": "frozen_ProjectDF_ProjectRF_ensemble",
            "guidance_label_mode": str(cfg.guidance_label_mode),
            "guidance_applies_from_step": int(guidance_start),
        },
    )
    return payload


def _load_v4_models(run_dir: Path, cfg: DefenseConfig, device: torch.device):
    scorer, selected_views, scorer_mean, scorer_scale, surrogate_bundle = load_candidate_components(run_dir, cfg, device)
    encoder_payload = torch.load(run_dir / "stage2_user_diffusion" / "encoder_checkpoint.pt", map_location=device, weights_only=False)
    ecfg = encoder_payload["config"]
    encoder = V4LeakageEncoder(int(ecfg["input_dim"]), int(ecfg["structure_dim"]), int(ecfg["patch_num"]), int(ecfg["hidden_dim"])).to(device)
    encoder.load_state_dict(encoder_payload["model_state"])
    encoder.eval()
    diffusion_payload = torch.load(run_dir / "stage2_user_diffusion" / "diffusion_guided_checkpoint.pt", map_location=device, weights_only=False)
    dcfg = diffusion_payload["config"]
    condition_encoder = CompositionalConditionEncoder(int(dcfg["patch_num"]), int(dcfg["hidden_dim"]), int(dcfg["condition_dim"])).to(device)
    condition_encoder.load_state_dict(diffusion_payload["condition_encoder_state"])
    condition_encoder.eval()
    diffusion = build_policy_diffusion(int(dcfg["condition_dim"]), int(dcfg["patch_num"]), int(dcfg["hidden_dim"]), int(dcfg["diffusion_steps"])).to(device)
    diffusion.load_state_dict(diffusion_payload["diffusion_state"])
    diffusion.eval()
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    return scorer, selected_views, scorer_mean, scorer_scale, surrogate_bundle, encoder, encoder_payload, condition_encoder, diffusion, profiles


def generate_v4_ragged_dataset(
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    indices: np.ndarray,
    run_dir: Path,
    cfg: DefenseConfig,
    *,
    profile: UserDefenseProfile,
    visit_namespace: str,
    budget: float,
    keep_ratio: float,
    output_npz: str | Path | None,
    device: torch.device,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    log(
        f"[Stage 3 sampling] namespace={visit_namespace}, base_traces={len(indices)}, "
        f"budget={float(budget):.3f}, keep_ratio={float(keep_ratio):.2f}, output={output_npz or '<memory only>'}",
        cfg.progress,
    )
    (
        scorer,
        selected_views,
        scorer_mean,
        scorer_scale,
        surrogate_bundle,
        encoder,
        encoder_payload,
        condition_encoder,
        diffusion,
        _,
    ) = _load_v4_models(run_dir, cfg, device)
    selected = np.asarray(indices, dtype=np.int64)
    clean = raw[selected]
    y = labels[selected].astype(np.int64)
    clean_prefix_rows = _prefix_only_rows(clean, cfg)
    guidance_positions, guidance_target_metrics = resolve_guidance_positions(
        clean_prefix_rows,
        y,
        surrogate_bundle,
        cfg,
        device,
    )
    log(
        f"[Stage 3 sampling] models loaded; repeat_count={_repeat_count_for_namespace(cfg, visit_namespace)}, "
        f"guidance_target={guidance_target_metrics.get('guidance_target_source')}, "
        f"target_samples={float(guidance_target_metrics.get('guidance_target_samples', 0.0)):.0f}, "
        f"pseudo_conf={float(guidance_target_metrics.get('guidance_pseudo_target_mean_confidence', 0.0)):.4f}, "
        f"true_conf={float(guidance_target_metrics.get('guidance_true_target_mean_confidence', 0.0)):.4f}",
        cfg.progress,
    )
    context = _candidate_context(
        clean,
        cfg,
        scorer,
        selected_views,
        scorer_mean,
        scorer_scale,
        surrogate_bundle,
        y,
        device,
        include_exact_utility=False,
    )
    ecfg = encoder_payload["config"]
    raw_encoder_features = _encoder_input(context, np.asarray(ecfg["view_vector"], dtype=np.float32))
    encoder_features = np.clip(
        (raw_encoder_features - np.asarray(ecfg["feature_mean"], dtype=np.float32)) / np.asarray(ecfg["feature_scale"], dtype=np.float32),
        -8.0,
        8.0,
    ).astype(np.float32)
    pool = PreferencePool(patch_num=int(cfg.patch_num))
    mode_pool = _v1_mode_pool(cfg)
    repeat_count = _repeat_count_for_namespace(cfg, visit_namespace)
    templates = []
    renderer_seeds, combo_indices, visit_weights, selected_masks, v1_mode_names = [], [], [], [], []
    template_clean_rows, generated_clean_indices, generated_y = [], [], []
    all_policy_logits, all_candidates = [], []
    per_repeat_budget = max(1, int(cfg.surrogate_gradient_batch_size) // max(int(repeat_count), 1))
    batch_size = min(int(cfg.batch_size), per_repeat_budget, len(selected)) if len(selected) else 1
    refinement_reports = []
    with torch.no_grad():
        encoded_all = encoder(torch.as_tensor(encoder_features, dtype=torch.float32, device=device))
    for start in range(0, len(selected), batch_size):
        end = min(start + batch_size, len(selected))
        preferences, profile_masks, visit_selected, weights, diffusion_seeds, base_offsets = [], [], [], [], [], []
        for base_offset, local in enumerate(range(start, end)):
            for repeat_index in range(repeat_count):
                visit = select_visit(
                    profile,
                    f"{visit_namespace}:r{repeat_index}:{local}",
                    str(trace_ids[selected[local]]),
                    str(cfg.visit_selector),
                )
                preferences.append(
                    _mixed_preference(
                        context["conditions"][local],
                        context["utility"][local],
                        context["candidate_masks"][local],
                        visit,
                        pool,
                        float(cfg.v1_mode_prior_weight),
                        mode_pool,
                    )
                )
                profile_masks.append(np.asarray(profile.profile_mask_20d, dtype=np.float32))
                visit_selected.append(visit.selected_primitive_mask)
                weights.append(visit.primitive_weights)
                diffusion_seeds.append(visit.diffusion_seed)
                renderer_seeds.append(visit.renderer_seed)
                combo_indices.append(visit.combination_index)
                v1_mode_names.append(mode_pool[int(visit.combination_index) % len(mode_pool)])
                visit_weights.append(visit.primitive_weights)
                selected_masks.append(visit.selected_primitive_mask)
                base_offsets.append(base_offset)
                template_clean_rows.append(clean[local])
                generated_clean_indices.append(int(selected[local]))
                generated_y.append(int(labels[selected[local]]))
        row_count = len(base_offsets)
        base_index_t = torch.as_tensor(base_offsets, dtype=torch.long, device=device)
        batch_clean = clean[start:end]
        batch_prefix = clean_prefix_rows[start:end]
        batch_conditions = context["conditions"][start:end]
        batch_candidate_np = context["candidate_masks"][start:end][np.asarray(base_offsets, dtype=np.int64)]
        batch_utility_np = context["utility"][start:end][np.asarray(base_offsets, dtype=np.int64)]
        batch_clean_rows = [batch_clean[int(offset)] for offset in base_offsets]
        batch_prefix_rows = batch_prefix[np.asarray(base_offsets, dtype=np.int64)]
        batch_guidance_pos = torch.as_tensor(
            [int(guidance_positions[start + int(offset)]) for offset in base_offsets],
            dtype=torch.long,
            device=device,
        )
        candidate_t = torch.as_tensor(batch_candidate_np, dtype=torch.float32, device=device)
        preference_t = torch.as_tensor(np.stack(preferences), dtype=torch.float32, device=device)
        preference_condition_t = preference_t
        profile_t = torch.as_tensor(np.stack(profile_masks), dtype=torch.float32, device=device)
        selected_t = torch.as_tensor(np.stack(visit_selected), dtype=torch.float32, device=device)
        weights_t = torch.as_tensor(np.stack(weights), dtype=torch.float32, device=device)
        if not bool(cfg.condition_preference_map):
            preference_condition_t = torch.zeros_like(preference_condition_t)
        if not bool(cfg.condition_profile_mask):
            profile_t = torch.zeros_like(profile_t)
        if not bool(cfg.condition_selected_mask):
            selected_t = torch.zeros_like(selected_t)
        if not bool(cfg.condition_preference_weights):
            weights_t = torch.zeros_like(weights_t)
        c_global = encoded_all["c_global"][start:end].index_select(0, base_index_t)
        c_leakage = torch.nn.functional.softplus(encoded_all["c_leakage"][start:end].index_select(0, base_index_t))
        condition_t = condition_encoder(c_global, c_leakage, candidate_t, preference_condition_t, weights_t, selected_t, profile_t)
        budget_trace_lengths = np.asarray([nonzero_trace(row).size for row in batch_clean_rows], dtype=np.float32)
        target_counts = torch.as_tensor(np.maximum(np.rint(budget_trace_lengths * float(budget)), 1.0), dtype=torch.float32, device=device)
        attack_context = build_attack_context(batch_prefix_rows, cfg, device)
        initial_rows = []
        for seed_value in diffusion_seeds:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed_value))
            initial_rows.append(torch.randn(2 * int(cfg.patch_num), generator=generator, device=device))
        initial_noise = torch.stack(initial_rows)
        if str(cfg.policy_generator) == "heuristic_prior_direct":
            rng = np.random.default_rng(int(cfg.seed) + start + 8100)
            sampled = torch.as_tensor(
                np.stack(
                    [
                        make_prior_logits(
                            batch_utility_np[row],
                            preferences[row],
                            batch_candidate_np[row],
                            rng=rng,
                            alpha_leak=float(cfg.prior_leak_weight),
                            alpha_pref=float(cfg.prior_preference_weight),
                            alpha_noise=float(cfg.prior_noise_std),
                        ).reshape(-1)
                        for row in range(row_count)
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
        else:
            batch_generator = torch.Generator(device=device)
            batch_generator.manual_seed(int(diffusion_seeds[0]))
            sampled = guided_ddim_sample(
                diffusion,
                condition_t,
                torch.full((row_count,), float(budget), dtype=torch.float32, device=device),
                candidate_t,
                target_counts,
                surrogate_bundle,
                attack_context,
                batch_guidance_pos,
                sampler_steps=int(cfg.sampling_steps),
                guidance_weight=float(cfg.guidance_weight),
                guidance_last_steps=int(cfg.guidance_last_steps),
                robust_weight=float(cfg.surrogate_robust_weight),
                soft_utility_weight=float(cfg.defense_soft_utility_weight),
                risk_tolerance=float(cfg.defense_risk_tolerance),
                generator=batch_generator,
                initial_noise=initial_noise,
            )
        sampled_map = sampled.reshape(-1, 2, int(cfg.patch_num))
        weighted_logits = (
            sampled_map
            + float(cfg.prior_leak_weight) * _normalize_torch_map(c_leakage, candidate_t)
            + float(cfg.prior_preference_weight) * _normalize_torch_map(preference_t, candidate_t)
        )
        if str(cfg.refine_method) == "continuous":
            refined, report = continuous_refine_logits(
                weighted_logits,
                candidate_t,
                target_counts,
                surrogate_bundle,
                attack_context,
                batch_guidance_pos,
                keep_ratio=float(keep_ratio),
                steps=int(cfg.refine_steps),
                robust_weight=float(cfg.surrogate_robust_weight),
                soft_utility_weight=float(cfg.defense_soft_utility_weight),
                risk_tolerance=float(cfg.defense_risk_tolerance),
            )
        else:
            refined, report = weighted_logits, {
                "keep_ratio": float(keep_ratio),
                "gate_mean": 1.0,
                "utility_before": 0.0,
                "utility_after": 0.0,
                "risk_before": 0.0,
                "risk_after": 0.0,
                "accepted_fraction": 1.0,
            }
        guarded_counts = target_counts * float(np.clip(keep_ratio, 0.0, 1.0))
        refined, preference_guard = enforce_defense_risk_guard(
            sampled_map,
            refined,
            candidate_t,
            guarded_counts,
            surrogate_bundle,
            attack_context,
            batch_guidance_pos,
            risk_tolerance=float(cfg.defense_risk_tolerance),
        )
        report.update(preference_guard)
        refinement_reports.append(report)
        refined_np = refined.detach().cpu().numpy()
        candidate_np = candidate_t.detach().cpu().numpy()
        for row, base_offset in enumerate(base_offsets):
            local = start + int(base_offset)
            projected_condition = replace(batch_conditions[int(base_offset)], allowed_mask=candidate_np[row])
            template = project_policy_to_template(
                refined_np[row],
                projected_condition,
                batch_clean_rows[row],
                float(budget) * float(keep_ratio),
                c_leakage=None,
                preference=None,
                method=f"{str(cfg.version).lower()}_{cfg.policy_generator}_guided_continuous",
                metadata={
                    "profile_id": profile.profile_id,
                    "combination_index": int(combo_indices[len(templates)]),
                    "v1_mode": str(v1_mode_names[len(templates)]),
                    "repeat_index": int(row % repeat_count),
                    "deployment_repeats": int(repeat_count),
                },
                logit_temperature=float(cfg.policy_logit_temperature),
                logit_noise_std=float(cfg.policy_logit_noise_std),
                rng=np.random.default_rng(int(renderer_seeds[len(templates)])),
                direction_target_incoming_share=_direction_target_incoming_share(batch_clean_rows[row], cfg),
                direction_correction_strength=float(cfg.direction_correction_strength),
                tam_flatten_strength=float(cfg.tam_flatten_strength),
                tam_flatten_floor=float(cfg.tam_flatten_floor),
                max_load_time=float(cfg.surrogate_rf_max_load_time),
            )
            templates.append(template)
            all_policy_logits.append(refined_np[row])
            all_candidates.append(candidate_np[row])
        progress_interval = max(int(cfg.log_every), batch_size)
        if end == batch_size or end == len(selected) or end % progress_interval < batch_size:
            log(
                f"{str(cfg.version).upper()} fresh sampling [{profile.profile_id}]: {end}/{len(selected)} "
                f"base traces, repeats={repeat_count}, generated={len(templates)}",
                cfg.progress,
            )
    traces, origins, raw_stats = render_batch_variable(
        template_clean_rows,
        templates,
        seeds=renderer_seeds,
        coordinate_length=int(cfg.max_trace_length),
        **renderer_options_from_config(cfg),
    )
    adapter_stats = _ragged_adapter_stats(traces, origins, int(cfg.max_trace_length))
    generated_y_np = np.asarray(generated_y, dtype=np.int64)
    generated_clean_indices_np = np.asarray(generated_clean_indices, dtype=np.int64)
    metric_indices = _large_dataset_metric_indices(len(templates), cfg, salt=1201)
    metric_clean_rows = _select_sequence_rows(template_clean_rows, metric_indices)
    metric_traces = _select_sequence_rows(traces, metric_indices)
    metric_y_np = generated_y_np[metric_indices] if len(metric_indices) else np.asarray([], dtype=np.int64)
    policy_logits_np = (
        np.asarray([all_policy_logits[int(index)] for index in metric_indices], dtype=np.float32)
        if len(metric_indices)
        else np.zeros((0, 2, int(cfg.patch_num)), dtype=np.float32)
    )
    candidate_mask_np = (
        np.asarray([all_candidates[int(index)] for index in metric_indices], dtype=np.float32)
        if len(metric_indices)
        else np.zeros((0, 2, int(cfg.patch_num)), dtype=np.float32)
    )
    prefix_rows = _prefix_only_rows(metric_clean_rows, cfg)
    prefix_lengths = np.asarray([np.count_nonzero(row) for row in prefix_rows], dtype=np.float32)
    prefix_target_counts = np.maximum(np.rint(prefix_lengths * float(budget) * float(keep_ratio)), 1.0).astype(np.float32)
    prefix_policy_metrics = _policy_prefix_label_free_metrics(
        policy_logits_np,
        candidate_mask_np,
        prefix_rows,
        prefix_target_counts,
        surrogate_bundle,
        cfg,
        device,
    )
    surrogate_metrics = ensemble_metrics_from_rendered(metric_traces, metric_y_np, surrogate_bundle, cfg, device)
    template_metrics = _template_summary(templates, cfg)
    tam_metrics = _tam_perturbation_metrics(metric_clean_rows, metric_traces, cfg)
    metrics = {
        "profile_id": profile.profile_id,
        "profile_split": profile.split,
        "visit_namespace": str(visit_namespace),
        "budget": float(budget),
        "keep_ratio": float(keep_ratio),
        "deployment_repeats": int(repeat_count),
        "diagnostic_metric_samples": int(len(metric_indices)),
        "policy_generator": str(cfg.policy_generator),
        **guidance_target_metrics,
        "generated_traces": int(len(traces)),
        "raw_bandwidth_overhead": float(np.mean(raw_stats["raw_bandwidth"])) if len(traces) else 0.0,
        "raw_real_packet_retention": float(np.mean(raw_stats["raw_real_packet_retention"])) if len(traces) else 1.0,
        "df_input_real_packet_retention": float(adapter_stats["attacker_input_real_packet_retention"]),
        "rf_input_real_packet_retention": 1.0,
        "visible_dummy_overhead": float(adapter_stats["visible_dummy_overhead"]),
        "clip_rate": float(adapter_stats["clip_rate"]),
        "delay_overhead": 0.0,
        "allowed_mask_violation_rate": float(
            np.mean([template.metadata.get("allowed_violation_rate", 0.0) for template in templates])
        ) if templates else 0.0,
        **template_metrics,
        **tam_metrics,
        **prefix_policy_metrics,
        **surrogate_metrics,
        "surrogate_defended_accuracy": float(surrogate_metrics.get("surrogate_ensemble_worst_accuracy", 0.0)),
        "refinement_utility_before": float(np.mean([row["utility_before"] for row in refinement_reports])) if refinement_reports else 0.0,
        "refinement_utility_after": float(np.mean([row["utility_after"] for row in refinement_reports])) if refinement_reports else 0.0,
        "refinement_risk_before": float(np.mean([row.get("risk_before", 0.0) for row in refinement_reports])) if refinement_reports else 0.0,
        "refinement_risk_after": float(np.mean([row.get("risk_after", 0.0) for row in refinement_reports])) if refinement_reports else 0.0,
        "refinement_accepted_fraction": float(np.mean([row.get("accepted_fraction", 0.0) for row in refinement_reports])) if refinement_reports else 0.0,
        "preference_guard_risk_before": float(np.mean([row.get("preference_guard_risk_before", 0.0) for row in refinement_reports])) if refinement_reports else 0.0,
        "preference_guard_risk_after": float(np.mean([row.get("preference_guard_risk_after", 0.0) for row in refinement_reports])) if refinement_reports else 0.0,
        "preference_guard_accepted_fraction": float(np.mean([row.get("preference_guard_accepted_fraction", 0.0) for row in refinement_reports])) if refinement_reports else 0.0,
    }
    log(
        f"[{str(cfg.version).upper()} defense surrogate] namespace={visit_namespace}, "
        f"budget={float(budget):.3f}, keep={float(keep_ratio):.2f}, traces={len(traces)}, "
        f"DF_acc={metrics.get('surrogate_df_accuracy', float('nan')):.4f}, "
        f"RF_acc={metrics.get('surrogate_rf_accuracy', float('nan')):.4f}, "
        f"worst_acc={metrics['surrogate_defended_accuracy']:.4f}, "
        f"overhead={metrics['visible_dummy_overhead']:.4f}, "
        f"incoming_share={metrics.get('dummy_incoming_share', 0.0):.4f}, "
        f"tam_l1={metrics.get('tam_distribution_l1_shift', 0.0):.4f}, "
        f"tam_cos={metrics.get('tam_cosine_distance', 0.0):.4f}, "
        f"mode_entropy={metrics.get('v1_mode_usage_entropy', 0.0):.3f}, "
        f"risk={metrics['preference_guard_risk_before']:.4f}->{metrics['preference_guard_risk_after']:.4f}, "
        f"preference_accept={metrics['preference_guard_accepted_fraction']:.3f}, "
        f"utility={metrics['refinement_utility_before']:.4f}->{metrics['refinement_utility_after']:.4f}",
        cfg.progress,
    )
    if output_npz is not None:
        output_path = Path(output_npz)
        save_ragged_npz(
            output_path,
            traces,
            origins,
            y=generated_y_np,
            clean_index=generated_clean_indices_np,
            combination_index=np.asarray(combo_indices, dtype=np.int64),
            primitive_weights=np.asarray(visit_weights, dtype=np.float32),
            selected_primitive_mask=np.asarray(selected_masks, dtype=np.float32),
            profile_mask=np.repeat(np.asarray(profile.profile_mask_20d, dtype=np.float32).reshape(1, -1), len(templates), axis=0),
            budget=np.full(len(templates), float(budget), dtype=np.float32),
            keep_ratio=np.full(len(templates), float(keep_ratio), dtype=np.float32),
        )
        write_json(output_path.with_name(output_path.stem + "_metrics.json"), metrics)
        stat_indices = _large_dataset_metric_indices(len(templates), cfg, salt=1301)
        sampled_policy_logits = (
            np.asarray([all_policy_logits[int(index)] for index in stat_indices], dtype=np.float32)
            if len(stat_indices)
            else np.zeros((0, 2, int(cfg.patch_num)), dtype=np.float32)
        )
        sampled_candidates = (
            np.asarray([all_candidates[int(index)] for index in stat_indices], dtype=np.float32)
            if len(stat_indices)
            else np.zeros((0, 2, int(cfg.patch_num)), dtype=np.float32)
        )
        sampled_dummy_counts = (
            np.asarray([templates[int(index)].counts for index in stat_indices], dtype=np.int32)
            if len(stat_indices)
            else np.zeros((0, 2, int(cfg.patch_num)), dtype=np.int32)
        )
        np.savez_compressed(
            output_path.with_name(output_path.stem + "_policy_statistics.npz"),
            policy_logits=sampled_policy_logits,
            candidate_mask=sampled_candidates,
            dummy_counts=sampled_dummy_counts,
            sampled_indices=stat_indices,
            sampled_from_count=np.asarray([len(templates)], dtype=np.int64),
        )
        log(f"[Stage 3 sampling] defended traces saved: {output_path}", cfg.progress)
    return traces, origins, metrics


def run_v4_stage2(
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    run_dir: Path,
    cfg: DefenseConfig,
    device: torch.device,
    profiles: dict[str, list[UserDefenseProfile]],
) -> dict:
    log(
        f"[Stage 2/3] Stage 2 starts: train_split={len(splits['train'])}, "
        f"generation_split={cfg.generation_split}, output={run_dir / 'stage2_user_diffusion'}",
        cfg.progress,
    )
    encoder, encoder_payload, context, surrogate_bundle = train_v4_encoder(raw, labels, splits["train"], run_dir, cfg, device)
    diffusion_payload = train_v4_diffusion(
        raw,
        labels,
        trace_ids,
        np.asarray(encoder_payload["train_indices"], dtype=np.int64),
        run_dir,
        cfg,
        device,
        encoder,
        encoder_payload,
        context,
        surrogate_bundle,
        profiles,
    )
    generation_indices = splits.get(str(cfg.generation_split), splits["test"])
    if int(cfg.max_generation_traces) > 0:
        local = choose_stratified_subset(labels[generation_indices], int(cfg.max_generation_traces), int(cfg.seed) + 601)
        generation_indices = generation_indices[local]
    target_profile = profiles["test"][0]
    log(
        f"[Stage 2/3] Generating reference defended traces: split={cfg.generation_split}, "
        f"traces={len(generation_indices)}, target_profile={target_profile.profile_id}",
        cfg.progress,
    )
    _, _, generation_metrics = generate_v4_ragged_dataset(
        raw,
        labels,
        trace_ids,
        generation_indices,
        run_dir,
        cfg,
        profile=target_profile,
        visit_namespace="stage2-reference",
        budget=float(cfg.budget_values[-1]),
        keep_ratio=1.0,
        output_npz=run_dir / "stage2_user_diffusion" / "reference_target_user_test.npz",
        device=device,
    )
    metrics = {"encoder": encoder_payload["metrics"], "diffusion": diffusion_payload["metrics"], "reference_generation": generation_metrics}
    write_json(run_dir / "stage2_user_diffusion" / "stage2_metrics.json", metrics)
    (run_dir / "stage2_user_diffusion" / "summary_zh.md").write_text(
        "\n".join(
            [
                f"# {str(cfg.version).upper()} Stage 2: User-specific compositional conditional diffusion",
                "",
                f"- train profiles: {len(profiles['train'])}",
                f"- validation profiles: {len(profiles['validation'])}",
                f"- unseen test profiles: {len(profiles['test'])}",
                f"- final denoising loss: {diffusion_payload['metrics'].get('denoise', 0.0):.6f}",
                f"- defense guidance loss: {diffusion_payload['metrics'].get('defense', 0.0):.6f}",
                f"- reference target profile: {target_profile.profile_id}",
                f"- raw real-packet retention: {generation_metrics['raw_real_packet_retention']:.6f}",
                f"- visible dummy overhead: {generation_metrics['visible_dummy_overhead']:.6f}",
                "- final policy source: DDIM reverse sample (unless policy_generator=heuristic_prior_direct baseline)",
                f"- deployment guidance target: {generation_metrics.get('guidance_target_source', 'frozen_surrogate_observed_prefix_pseudo_label')}",
            ]
        ),
        encoding="utf-8",
    )
    log(
        f"[Stage 2/3] Stage 2 metrics saved: {run_dir / 'stage2_user_diffusion' / 'stage2_metrics.json'}; "
        f"reference_overhead={generation_metrics['visible_dummy_overhead']:.4f}, "
        f"reference_pressure={generation_metrics.get('prefix_policy_label_free_attack_pressure', 0.0):.4f}",
        cfg.progress,
    )
    return metrics


def _stage3_clean_attack_input(kind: str, rows: np.ndarray, cfg: DefenseConfig, attack_cfg: AttackConfig) -> np.ndarray:
    if kind == "df":
        return build_df_input(rows, int(cfg.max_trace_length))
    return build_rf_tam_input(
        rows,
        max_len=int(cfg.max_trace_length),
        max_load_time=float(attack_cfg.max_load_time),
        num_slots=int(attack_cfg.rf_tam_num_slots),
    )


def _stage3_ragged_rf_tam(traces: Sequence[np.ndarray], slots: int, max_load_time: float) -> np.ndarray:
    tam = np.zeros((len(traces), 2, int(slots)), dtype=np.float32)
    scale = float(int(slots) - 1) / max(float(max_load_time), 1e-6)
    for row_index, trace in enumerate(traces):
        nonzero = np.asarray(trace, dtype=np.float32)
        outgoing = nonzero[nonzero > 0]
        incoming = -nonzero[nonzero < 0]
        if outgoing.size:
            bins = np.floor(outgoing * scale).astype(np.int64)
            bins[outgoing >= float(max_load_time)] = int(slots) - 1
            np.add.at(tam[row_index, 0], np.clip(bins, 0, int(slots) - 1), 1.0)
        if incoming.size:
            bins = np.floor(incoming * scale).astype(np.int64)
            bins[incoming >= float(max_load_time)] = int(slots) - 1
            np.add.at(tam[row_index, 1], np.clip(bins, 0, int(slots) - 1), 1.0)
    return tam


def _stage3_defended_attack_input(
    kind: str,
    traces: Sequence[np.ndarray],
    origins: Sequence[np.ndarray],
    cfg: DefenseConfig,
    attack_cfg: AttackConfig,
) -> np.ndarray:
    if kind == "df":
        padded, _ = crop_ragged_for_attacker(traces, origins, int(cfg.max_trace_length))
        return build_df_input(padded, int(cfg.max_trace_length))
    return _stage3_ragged_rf_tam(traces, int(attack_cfg.rf_tam_num_slots), float(attack_cfg.max_load_time))


def _stage3_probe_attack_config(run_dir: Path, cfg: DefenseConfig) -> AttackConfig:
    return AttackConfig(
        run_dir=str(run_dir),
        attackers=str(cfg.stage3_fixed_probe_attackers),
        device=str(cfg.device),
        clean_df_epochs=int(cfg.stage3_fixed_probe_epochs),
        clean_df_patience=max(1, min(2, int(cfg.stage3_fixed_probe_epochs))),
        clean_df_lr=float(cfg.surrogate_lr),
        df_batch_size=int(cfg.surrogate_batch_size),
        df_architecture=str(cfg.surrogate_df_architecture),
        rf_tam_num_slots=int(cfg.surrogate_rf_num_slots),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        progress=bool(cfg.progress),
    )


def _build_stage3_fixed_probe_models(
    raw: np.ndarray,
    labels: np.ndarray,
    splits: dict[str, np.ndarray],
    run_dir: Path,
    cfg: DefenseConfig,
    device: torch.device,
) -> tuple[AttackConfig, dict[str, tuple[nn.Module, np.ndarray, float]]]:
    attack_cfg = _stage3_probe_attack_config(run_dir, cfg)
    train_count = min(len(splits["train"]), int(cfg.stage3_fixed_probe_train_samples))
    val_count = min(len(splits["val"]), int(cfg.stage3_fixed_probe_val_samples))
    train_local = choose_stratified_subset(labels[splits["train"]], train_count, int(cfg.seed) + 751)
    val_local = choose_stratified_subset(labels[splits["val"]], val_count, int(cfg.seed) + 752)
    train_idx = np.asarray(splits["train"], dtype=np.int64)[train_local]
    val_idx = np.asarray(splits["val"], dtype=np.int64)[val_local]
    models: dict[str, tuple[nn.Module, np.ndarray, float]] = {}
    for kind in parse_csv_strings(cfg.stage3_fixed_probe_attackers):
        normalized = "rf" if str(kind).lower().endswith("rf") or str(kind).lower() == "rf" else "df"
        if normalized in models:
            continue
        train_x = _stage3_clean_attack_input(normalized, raw[train_idx], cfg, attack_cfg)
        val_x = _stage3_clean_attack_input(normalized, raw[val_idx], cfg, attack_cfg)
        model, classes, best_val = train_df_model(
            train_x,
            labels[train_idx],
            val_x,
            labels[val_idx],
            attacker_kind=normalized.upper(),
            defense_cfg=cfg,
            attack_cfg=attack_cfg,
            initial_state=None,
            epochs=int(cfg.stage3_fixed_probe_epochs),
            patience=max(1, min(2, int(cfg.stage3_fixed_probe_epochs))),
            lr=float(cfg.surrogate_lr),
            batch_size=int(cfg.surrogate_batch_size),
            device=device,
            seed=int(cfg.seed) + (760 if normalized == "df" else 770),
            progress=bool(cfg.progress),
        )
        models[normalized] = (model, classes, float(best_val))
        log(
            f"Stage 3 fixed {normalized.upper()} probe: clean_val_accuracy={best_val:.4f}, "
            f"reliable={int(float(best_val) >= float(cfg.stage3_fixed_probe_min_clean_accuracy))}",
            cfg.progress,
        )
    return attack_cfg, models


def _evaluate_stage3_fixed_probe(
    traces: Sequence[np.ndarray],
    origins: Sequence[np.ndarray],
    y: np.ndarray,
    cfg: DefenseConfig,
    attack_cfg: AttackConfig,
    models: dict[str, tuple[nn.Module, np.ndarray, float]],
    device: torch.device,
) -> dict[str, float]:
    selected = np.arange(len(y), dtype=np.int64)
    if int(cfg.stage3_fixed_probe_samples) > 0 and len(selected) > int(cfg.stage3_fixed_probe_samples):
        rng = np.random.default_rng(int(cfg.seed) + 753)
        selected = np.sort(rng.choice(selected, size=int(cfg.stage3_fixed_probe_samples), replace=False)).astype(np.int64)
    chosen_traces = [traces[int(index)] for index in selected]
    chosen_origins = [origins[int(index)] for index in selected]
    chosen_y = np.asarray(y, dtype=np.int64)[selected]
    min_clean_accuracy = float(cfg.stage3_fixed_probe_min_clean_accuracy)
    result: dict[str, float] = {
        "fixed_probe_samples": int(len(chosen_y)),
        "fixed_probe_min_clean_accuracy": min_clean_accuracy,
    }
    raw_accuracies = []
    reliable_accuracies = []
    pressures = []
    reliable_pressures = []
    reliable_entropies = []
    reliable_confidences = []
    reliable_margins = []
    for kind, (model, classes, best_val) in models.items():
        x = _stage3_defended_attack_input(kind, chosen_traces, chosen_origins, cfg, attack_cfg)
        class_to_pos = {int(label): pos for pos, label in enumerate(classes.tolist())}
        positions = torch.as_tensor([class_to_pos[int(label)] for label in chosen_y], dtype=torch.long, device=device)
        logits_rows = []
        batch_size = max(1, int(attack_cfg.df_batch_size))
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
                logits_rows.append(model(batch))
            logits = torch.cat(logits_rows, dim=0) if logits_rows else torch.zeros((0, len(classes)), dtype=torch.float32, device=device)
            probabilities = torch.softmax(logits, dim=1)
            entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1) / math.log(max(logits.shape[1], 2))
            top2 = probabilities.topk(k=min(2, probabilities.shape[1]), dim=1).values
            margin = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else torch.zeros_like(top2[:, 0]))
            accuracy = float((logits.argmax(dim=1) == positions).float().mean().cpu()) if len(chosen_y) else 0.0
            mean_entropy = float(entropy.mean().cpu()) if len(chosen_y) else 0.0
            mean_confidence = float(probabilities.max(dim=1).values.mean().cpu()) if len(chosen_y) else 0.0
            mean_margin = float(margin.mean().cpu()) if len(chosen_y) else 0.0
        pressure = float(mean_confidence + 0.50 * mean_margin - 0.50 * mean_entropy)
        result[f"fixed_probe_{kind}_accuracy"] = accuracy
        result[f"fixed_probe_{kind}_entropy"] = mean_entropy
        result[f"fixed_probe_{kind}_max_confidence"] = mean_confidence
        result[f"fixed_probe_{kind}_margin"] = mean_margin
        result[f"fixed_probe_{kind}_label_free_pressure"] = pressure
        result[f"fixed_probe_{kind}_best_val_accuracy"] = float(best_val)
        reliable = float(best_val) >= min_clean_accuracy
        result[f"fixed_probe_{kind}_reliable"] = int(reliable)
        raw_accuracies.append(accuracy)
        pressures.append(pressure)
        if reliable:
            reliable_accuracies.append(accuracy)
            reliable_pressures.append(pressure)
            reliable_entropies.append(mean_entropy)
            reliable_confidences.append(mean_confidence)
            reliable_margins.append(mean_margin)
    reliable_worst = float(max(reliable_accuracies)) if reliable_accuracies else 0.0
    result["fixed_probe_raw_worst_accuracy"] = float(max(raw_accuracies)) if raw_accuracies else 0.0
    result["fixed_probe_reliable_count"] = int(len(reliable_accuracies))
    result["fixed_probe_reliable_worst_accuracy"] = reliable_worst
    result["fixed_probe_worst_accuracy"] = reliable_worst
    result["fixed_probe_mean_accuracy"] = float(np.mean(reliable_accuracies)) if reliable_accuracies else 0.0
    result["fixed_probe_raw_worst_label_free_pressure"] = float(max(pressures)) if pressures else 0.0
    result["fixed_probe_reliable_worst_label_free_pressure"] = float(max(reliable_pressures)) if reliable_pressures else 0.0
    result["fixed_probe_reliable_mean_entropy"] = float(np.mean(reliable_entropies)) if reliable_entropies else 0.0
    result["fixed_probe_reliable_worst_max_confidence"] = float(max(reliable_confidences)) if reliable_confidences else 0.0
    result["fixed_probe_reliable_worst_margin"] = float(max(reliable_margins)) if reliable_margins else 0.0
    return result


def run_v4_stage3(
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    splits: dict[str, np.ndarray],
    run_dir: Path,
    cfg: DefenseConfig,
    device: torch.device,
    profiles: dict[str, list[UserDefenseProfile]],
) -> dict:
    stage_dir = run_dir / "stage3_guided_refinement"
    stage_dir.mkdir(parents=True, exist_ok=True)
    val_indices = splits["val"]
    if int(cfg.pareto_samples) > 0 and len(val_indices) > int(cfg.pareto_samples):
        rng = np.random.default_rng(int(cfg.seed) + 701)
        val_indices = np.sort(rng.choice(np.asarray(val_indices, dtype=np.int64), size=int(cfg.pareto_samples), replace=False)).astype(np.int64)
    profile = profiles["validation"][0]
    log(
        f"[Stage 3/3] Stage 3 starts: pareto_samples={len(val_indices)}, "
        f"budgets={cfg.pareto_budgets}, keep_ratios={cfg.refine_keep_ratios}, "
        f"profile={profile.profile_id}, output={stage_dir}",
        cfg.progress,
    )
    probe_attack_cfg = None
    probe_models: dict[str, tuple[nn.Module, np.ndarray, float]] = {}
    if int(cfg.stage3_fixed_probe_samples) > 0:
        log(
            f"[Stage 3/3] Training quick fixed probe models for Pareto selection: "
            f"attackers={cfg.stage3_fixed_probe_attackers}",
            cfg.progress,
        )
        probe_attack_cfg, probe_models = _build_stage3_fixed_probe_models(raw, labels, splits, run_dir, cfg, device)
    rows = []
    for budget in cfg.pareto_budget_values:
        for keep_ratio in cfg.refine_keep_ratio_values:
            log(
                f"[Stage 3/3] Candidate run begins: budget={float(budget):.3f}, "
                f"keep_ratio={float(keep_ratio):.2f}",
                cfg.progress,
            )
            traces, origins, metrics = generate_v4_ragged_dataset(
                raw,
                labels,
                trace_ids,
                val_indices,
                run_dir,
                cfg,
                profile=profile,
                visit_namespace=f"pareto-b{budget:.2f}-k{keep_ratio:.2f}",
                budget=float(budget),
                keep_ratio=float(keep_ratio),
                output_npz=stage_dir / f"pareto_b{budget:.2f}_k{keep_ratio:.2f}.npz",
                device=device,
            )
            prefix_policy_pressure = float(metrics.get("prefix_policy_label_free_attack_pressure", 1.0))
            rendered_surrogate_pressure = float(metrics.get("surrogate_label_free_attack_pressure", 1.0))
            rendered_rf_pressure = float(metrics.get("surrogate_rf_label_free_pressure", rendered_surrogate_pressure))
            diagnostic_surrogate_accuracy = float(metrics["surrogate_defended_accuracy"])
            rendered_rf_accuracy = float(metrics.get("surrogate_rf_accuracy", diagnostic_surrogate_accuracy))
            probe_metrics: dict[str, float] = {}
            probe_penalty = 0.0
            selection_attack_pressure = max(prefix_policy_pressure, rendered_surrogate_pressure, rendered_rf_pressure)
            selection_diagnostic_attack_accuracy = diagnostic_surrogate_accuracy
            selection_fixed_probe_accuracy = 0.0
            selection_fixed_probe_reliable = 0
            selection_fixed_rf_probe_accuracy = 0.0
            selection_fixed_rf_probe_reliable = 0
            selection_budget_proxy = float(budget) * float(keep_ratio)
            dummy_incoming_share = float(metrics.get("dummy_incoming_share", 0.0))
            tam_incoming_l1_shift = float(metrics.get("tam_incoming_distribution_l1_shift", 0.0))
            incoming_share_shortfall = max(0.0, float(cfg.stage3_min_dummy_incoming_share) - dummy_incoming_share)
            incoming_share_excess = max(0.0, dummy_incoming_share - float(cfg.stage3_max_dummy_incoming_share))
            tam_incoming_shortfall = max(0.0, float(cfg.stage3_min_tam_incoming_l1_shift) - tam_incoming_l1_shift)
            incoming_metric_penalty = float(cfg.stage3_incoming_metric_weight) * (
                incoming_share_shortfall + incoming_share_excess + tam_incoming_shortfall
            )
            if probe_models and probe_attack_cfg is not None:
                repeat_count = max(1, int(metrics.get("deployment_repeats", 1)))
                probe_y = np.repeat(labels[val_indices].astype(np.int64), repeat_count)
                probe_metrics = _evaluate_stage3_fixed_probe(traces, origins, probe_y, cfg, probe_attack_cfg, probe_models, device)
                if int(probe_metrics.get("fixed_probe_reliable_count", 0)) > 0:
                    probe_accuracy = float(probe_metrics["fixed_probe_reliable_worst_accuracy"])
                    selection_fixed_probe_accuracy = probe_accuracy
                    selection_fixed_probe_reliable = 1
                    selection_diagnostic_attack_accuracy = max(diagnostic_surrogate_accuracy, probe_accuracy)
                    probe_penalty = float(cfg.stage3_fixed_probe_weight) * max(
                        0.0,
                        probe_accuracy - float(cfg.stage3_max_reliable_fixed_probe_accuracy),
                    )
                if int(probe_metrics.get("fixed_probe_rf_reliable", 0)) > 0:
                    selection_fixed_rf_probe_accuracy = float(probe_metrics.get("fixed_probe_rf_accuracy", 0.0))
                    selection_fixed_rf_probe_reliable = 1
                    selection_diagnostic_attack_accuracy = max(selection_diagnostic_attack_accuracy, selection_fixed_rf_probe_accuracy)
                    probe_penalty = max(
                        probe_penalty,
                        float(cfg.stage3_fixed_probe_weight)
                        * max(0.0, selection_fixed_rf_probe_accuracy - float(cfg.stage3_max_reliable_fixed_probe_accuracy)),
                    )
            rendered_rf_accuracy_penalty = max(0.0, rendered_rf_accuracy - float(cfg.stage3_max_rendered_rf_accuracy))
            diagnostic_accuracy_penalty = max(0.0, selection_diagnostic_attack_accuracy - float(cfg.stage3_max_attack_accuracy))
            score = (
                -selection_attack_pressure
                - 0.20 * rendered_rf_pressure
                - 0.25 * rendered_rf_accuracy_penalty
                - 0.25 * diagnostic_accuracy_penalty
                - 0.15 * selection_budget_proxy
                + 0.05 * float(metrics.get("v1_mode_usage_entropy", 0.0))
                - probe_penalty
                - incoming_metric_penalty
            )
            probe_status = "disabled"
            if probe_metrics:
                probe_status = (
                    f"reliable={int(probe_metrics.get('fixed_probe_reliable_count', 0))}, "
                    f"raw_pressure={float(probe_metrics.get('fixed_probe_raw_worst_label_free_pressure', 0.0)):.4f}, "
                    f"trusted_pressure={float(probe_metrics.get('fixed_probe_reliable_worst_label_free_pressure', 0.0)):.4f}, "
                    f"diagnostic_acc={float(probe_metrics.get('fixed_probe_reliable_worst_accuracy', 0.0)):.4f}"
                )
            log(
                f"[Stage 3 candidate] budget={float(budget):.3f}, keep={float(keep_ratio):.2f}, "
                f"surrogate_DF={float(metrics.get('surrogate_df_accuracy', 0.0)):.4f}, "
                f"surrogate_RF={float(metrics.get('surrogate_rf_accuracy', 0.0)):.4f}, "
                f"selection_pressure={selection_attack_pressure:.4f}, "
                f"rendered_pressure={rendered_surrogate_pressure:.4f}, "
                f"rf_pressure={rendered_rf_pressure:.4f}, "
                f"rf_acc={rendered_rf_accuracy:.4f}, "
                f"incoming={dummy_incoming_share:.4f}, tam_in_l1={tam_incoming_l1_shift:.4f}, "
                f"diagnostic_acc={selection_diagnostic_attack_accuracy:.4f}, probe=({probe_status}), score={score:.4f}",
                cfg.progress,
            )
            rows.append(
                {
                    **metrics,
                    **probe_metrics,
                    "selection_label_free_surrogate_pressure": float(prefix_policy_pressure),
                    "selection_rendered_label_free_surrogate_pressure": float(rendered_surrogate_pressure),
                    "selection_rf_label_free_surrogate_pressure": float(rendered_rf_pressure),
                    "selection_rendered_rf_accuracy": float(rendered_rf_accuracy),
                    "selection_fixed_probe_reliable": int(selection_fixed_probe_reliable),
                    "selection_fixed_probe_reliable_accuracy": float(selection_fixed_probe_accuracy),
                    "selection_fixed_rf_probe_reliable": int(selection_fixed_rf_probe_reliable),
                    "selection_fixed_rf_probe_reliable_accuracy": float(selection_fixed_rf_probe_accuracy),
                    "selection_budget_proxy": float(selection_budget_proxy),
                    "selection_score_fixed_probe": float(-probe_penalty),
                    "selection_score_fixed_probe_penalty": float(probe_penalty),
                    "selection_score_rendered_rf_accuracy_penalty": float(-0.25 * rendered_rf_accuracy_penalty),
                    "selection_score_diagnostic_accuracy_penalty": float(-0.25 * diagnostic_accuracy_penalty),
                    "selection_dummy_incoming_share_gate": float(cfg.stage3_min_dummy_incoming_share),
                    "selection_dummy_incoming_share_max_gate": float(cfg.stage3_max_dummy_incoming_share),
                    "selection_dummy_incoming_share_shortfall": float(incoming_share_shortfall),
                    "selection_dummy_incoming_share_excess": float(incoming_share_excess),
                    "selection_tam_incoming_l1_shift_gate": float(cfg.stage3_min_tam_incoming_l1_shift),
                    "selection_tam_incoming_l1_shift_shortfall": float(tam_incoming_shortfall),
                    "selection_score_incoming_metric_penalty": float(incoming_metric_penalty),
                    "selection_attack_pressure": float(selection_attack_pressure),
                    "selection_attack_accuracy": float(selection_diagnostic_attack_accuracy),
                    "selection_score": float(score),
                }
            )
    target_budget = float(cfg.budget_values[-1])
    candidates = [row for row in rows if abs(float(row["budget"]) - target_budget) < 1e-8] or rows
    max_attack_pressure = float(cfg.stage3_max_label_free_attack_pressure)
    max_attack_accuracy = float(cfg.stage3_max_attack_accuracy)
    max_rendered_rf_accuracy = float(cfg.stage3_max_rendered_rf_accuracy)
    max_fixed_probe_accuracy = float(cfg.stage3_max_reliable_fixed_probe_accuracy)
    min_dummy_incoming_share = float(cfg.stage3_min_dummy_incoming_share)
    max_dummy_incoming_share = float(cfg.stage3_max_dummy_incoming_share)
    min_tam_incoming_l1_shift = float(cfg.stage3_min_tam_incoming_l1_shift)
    use_diagnostic_accuracy_gate = bool(cfg.stage3_use_diagnostic_accuracy_gate)
    for row in rows:
        pressure_passed = int(float(row["selection_attack_pressure"]) <= max_attack_pressure)
        diagnostic_accuracy_passed = int(float(row["selection_attack_accuracy"]) <= max_attack_accuracy)
        rendered_rf_accuracy_passed = int(float(row.get("selection_rendered_rf_accuracy", 0.0)) <= max_rendered_rf_accuracy)
        dummy_incoming_share_passed = int(float(row.get("dummy_incoming_share", 0.0)) >= min_dummy_incoming_share)
        dummy_incoming_share_max_passed = int(float(row.get("dummy_incoming_share", 0.0)) <= max_dummy_incoming_share)
        tam_incoming_l1_shift_passed = int(
            float(row.get("tam_incoming_distribution_l1_shift", 0.0)) >= min_tam_incoming_l1_shift
        )
        if int(row.get("selection_fixed_rf_probe_reliable", 0)) > 0:
            fixed_probe_accuracy_passed = int(
                float(row.get("selection_fixed_rf_probe_reliable_accuracy", 1.0)) <= max_fixed_probe_accuracy
            )
        elif int(row.get("selection_fixed_probe_reliable", 0)) > 0:
            fixed_probe_accuracy_passed = int(float(row.get("selection_fixed_probe_reliable_accuracy", 1.0)) <= max_fixed_probe_accuracy)
        else:
            fixed_probe_accuracy_passed = 1
        row["selection_quality_gate"] = float(max_attack_pressure)
        row["selection_quality_gate_passed"] = pressure_passed
        row["selection_label_free_gate"] = float(max_attack_pressure)
        row["selection_label_free_gate_passed"] = pressure_passed
        row["selection_diagnostic_attack_accuracy_gate"] = float(max_attack_accuracy)
        row["selection_diagnostic_attack_accuracy_gate_passed"] = diagnostic_accuracy_passed
        row["selection_rendered_rf_accuracy_gate"] = float(max_rendered_rf_accuracy)
        row["selection_rendered_rf_accuracy_gate_passed"] = rendered_rf_accuracy_passed
        row["selection_reliable_fixed_probe_accuracy_gate"] = float(max_fixed_probe_accuracy)
        row["selection_reliable_fixed_probe_accuracy_gate_passed"] = fixed_probe_accuracy_passed
        row["selection_dummy_incoming_share_gate"] = float(min_dummy_incoming_share)
        row["selection_dummy_incoming_share_gate_passed"] = dummy_incoming_share_passed
        row["selection_dummy_incoming_share_max_gate"] = float(max_dummy_incoming_share)
        row["selection_dummy_incoming_share_max_gate_passed"] = dummy_incoming_share_max_passed
        row["selection_tam_incoming_l1_shift_gate"] = float(min_tam_incoming_l1_shift)
        row["selection_tam_incoming_l1_shift_gate_passed"] = tam_incoming_l1_shift_passed
        row["selection_uses_diagnostic_accuracy_gate"] = int(use_diagnostic_accuracy_gate)
        row["selection_all_quality_gates_passed"] = int(
            bool(pressure_passed)
            and bool(dummy_incoming_share_passed)
            and bool(dummy_incoming_share_max_passed)
            and bool(tam_incoming_l1_shift_passed)
            and (
                not use_diagnostic_accuracy_gate
                or (bool(diagnostic_accuracy_passed) and bool(rendered_rf_accuracy_passed) and bool(fixed_probe_accuracy_passed))
            )
        )
    quality_candidates = [row for row in candidates if int(row.get("selection_all_quality_gates_passed", 0)) > 0]
    selection_pool = quality_candidates or candidates
    best_attack_pressure = min((float(row["selection_attack_pressure"]) for row in selection_pool), default=1.0)
    guard_limit = best_attack_pressure + max(0.0, float(cfg.stage3_accuracy_guard_margin))
    guarded_candidates = [row for row in selection_pool if float(row["selection_attack_pressure"]) <= guard_limit + 1e-12]
    guarded_ids = {id(row) for row in guarded_candidates}
    for row in rows:
        row["selection_guard_best_pressure"] = float(best_attack_pressure)
        row["selection_guard_limit"] = float(guard_limit)
        row["selection_guard_eligible"] = int(id(row) in guarded_ids)
    selected = max(guarded_candidates, key=lambda row: float(row["selection_score"])) if guarded_candidates else {}
    if selected:
        selected["selection_policy_valid"] = int(bool(quality_candidates))
        selected["selection_used_quality_fallback"] = int(not quality_candidates)
        if not quality_candidates:
            log(
                f"[Stage 3/3] WARNING: no candidate passed the Stage 3 quality gates "
                f"(pressure<={max_attack_pressure:.3f}, diagnostic_acc<={max_attack_accuracy:.3f}, "
                f"rendered_rf_acc<={max_rendered_rf_accuracy:.3f}, reliable_probe_acc<={max_fixed_probe_accuracy:.3f}, "
                f"incoming_share={min_dummy_incoming_share:.3f}..{max_dummy_incoming_share:.3f}, "
                f"tam_in_l1>={min_tam_incoming_l1_shift:.3f}); "
                f"saving best fallback at pressure={selected['selection_attack_pressure']:.3f}, "
                f"diagnostic_acc={selected['selection_attack_accuracy']:.3f} for diagnosis only.",
                True,
            )
        log(
            f"[Stage 3 selected] budget={float(selected['budget']):.3f}, "
            f"keep={float(selected['keep_ratio']):.2f}, "
            f"label_free_pressure={float(selected['selection_attack_pressure']):.4f}, "
            f"diagnostic_acc={float(selected.get('selection_attack_accuracy', 0.0)):.4f}, "
            f"rf_acc={float(selected.get('selection_rendered_rf_accuracy', 0.0)):.4f}, "
            f"quality_gate<={max_attack_pressure:.3f}, valid={int(selected['selection_policy_valid'])}, "
            f"score={float(selected['selection_score']):.4f}",
            True,
        )
    write_csv(stage_dir / "pareto_results.csv", rows)
    write_json(stage_dir / "selected_policy.json", selected)
    write_json(stage_dir / "stage3_metrics.json", {"pareto_rows": rows, "selected": selected})
    (stage_dir / "summary_zh.md").write_text(
        "\n".join(
            [
                f"# {str(cfg.version).upper()} Stage 3: Budget-performance-guided diffusion refinement",
                "",
                f"- Pareto samples: {len(val_indices)}",
                f"- budgets: {cfg.pareto_budgets}",
                f"- keep ratios: {cfg.refine_keep_ratios}",
                f"- selected budget: {selected.get('budget', 0.0):.4f}",
                f"- selected keep ratio: {selected.get('keep_ratio', 1.0):.4f}",
                f"- selected visible bandwidth: {selected.get('visible_dummy_overhead', 0.0):.6f}",
                f"- selected raw retention: {selected.get('raw_real_packet_retention', 1.0):.6f}",
                f"- selected dummy incoming share: {selected.get('dummy_incoming_share', 0.0):.6f}",
                f"- selected template entropy: {selected.get('template_entropy', 0.0):.6f}",
                f"- selected label-free attack pressure: {selected.get('selection_attack_pressure', 0.0):.6f}",
                f"- selected rendered label-free pressure: {selected.get('selection_rendered_label_free_surrogate_pressure', 0.0):.6f}",
                f"- selected RF label-free pressure: {selected.get('selection_rf_label_free_surrogate_pressure', 0.0):.6f}",
                f"- selected rendered RF accuracy: {selected.get('selection_rendered_rf_accuracy', 0.0):.6f}",
                f"- selected diagnostic conservative attack acc: {selected.get('selection_attack_accuracy', 0.0):.6f}",
                f"- selected policy valid: {selected.get('selection_policy_valid', 0)}",
                f"- label-free pressure quality gate: {selected.get('selection_quality_gate', max_attack_pressure):.6f}",
                f"- diagnostic accuracy gate: {selected.get('selection_diagnostic_attack_accuracy_gate', max_attack_accuracy):.6f}",
                f"- rendered RF accuracy gate: {selected.get('selection_rendered_rf_accuracy_gate', max_rendered_rf_accuracy):.6f}",
                f"- reliable fixed probe accuracy gate: {selected.get('selection_reliable_fixed_probe_accuracy_gate', max_fixed_probe_accuracy):.6f}",
                f"- all quality gates passed: {selected.get('selection_all_quality_gates_passed', 0)}",
                f"- reliable fixed probes: {selected.get('fixed_probe_reliable_count', 0)}",
                f"- selected reliable fixed probe worst acc: {selected.get('fixed_probe_reliable_worst_accuracy', 0.0):.6f}",
                f"- selected reliable fixed RF probe acc: {selected.get('selection_fixed_rf_probe_reliable_accuracy', 0.0):.6f}",
                f"- selected reliable fixed RF probe available: {selected.get('selection_fixed_rf_probe_reliable', 0)}",
                f"- dummy incoming share gate/pass: {selected.get('selection_dummy_incoming_share_gate', min_dummy_incoming_share):.6f} / {selected.get('selection_dummy_incoming_share_gate_passed', 0)}",
                f"- dummy incoming share max gate/pass: {selected.get('selection_dummy_incoming_share_max_gate', max_dummy_incoming_share):.6f} / {selected.get('selection_dummy_incoming_share_max_gate_passed', 0)}",
                f"- TAM incoming L1 shift gate/pass: {selected.get('selection_tam_incoming_l1_shift_gate', min_tam_incoming_l1_shift):.6f} / {selected.get('selection_tam_incoming_l1_shift_gate_passed', 0)}",
                f"- TAM distribution L1 shift: {selected.get('tam_distribution_l1_shift', 0.0):.6f}",
                f"- TAM cosine distance: {selected.get('tam_cosine_distance', 0.0):.6f}",
                f"- TAM incoming distribution L1 shift: {selected.get('tam_incoming_distribution_l1_shift', 0.0):.6f}",
                f"- TAM outgoing distribution L1 shift: {selected.get('tam_outgoing_distribution_l1_shift', 0.0):.6f}",
            ]
        ),
        encoding="utf-8",
    )
    if selected and not int(selected.get("selection_policy_valid", 0)):
        write_json(stage_dir / "best_diagnostic_fallback.json", selected)
        if bool(cfg.stage3_require_quality_gate):
            raise RuntimeError(
                f"Stage 3 hard defense gate failed: selected pressure="
                f"{float(selected['selection_attack_pressure']):.4f} "
                f"(gate {float(cfg.stage3_max_label_free_attack_pressure):.4f}), "
                f"diagnostic_acc={float(selected.get('selection_attack_accuracy', 0.0)):.4f} "
                f"(gate {float(cfg.stage3_max_attack_accuracy):.4f}), "
                f"rendered_rf_acc={float(selected.get('selection_rendered_rf_accuracy', 0.0)):.4f} "
                f"(gate {float(cfg.stage3_max_rendered_rf_accuracy):.4f}), "
                f"fixed_probe_acc={float(selected.get('selection_fixed_probe_reliable_accuracy', 0.0)):.4f} "
                f"(gate {float(cfg.stage3_max_reliable_fixed_probe_accuracy):.4f}). "
                f"fixed_rf_probe_acc={float(selected.get('selection_fixed_rf_probe_reliable_accuracy', 0.0)):.4f}. "
                "Artifacts were saved for diagnosis."
            )
    log(
        f"[Stage 3/3] Stage 3 metrics saved: {stage_dir / 'stage3_metrics.json'}; "
        f"selected_policy={stage_dir / 'selected_policy.json'}",
        cfg.progress,
    )
    return {"pareto_rows": rows, "selected": selected}


def _write_v4_final_summary(run_dir: Path, cfg: DefenseConfig, data_source: str, stage1: dict, stage2: dict, stage3: dict) -> None:
    selected = stage3.get("selected", {})
    title = "DMMPv3 用户特异组合条件扩散防御"
    if str(cfg.version).lower() != "v3":
        title = f"DMMPv3 {str(cfg.version).upper()} 用户特异组合条件扩散防御"
    lines = [
        f"# {title}",
        "",
        f"- dataset: {data_source}",
        f"- seed: {cfg.seed}",
        f"- prefix_n: {cfg.prefix_n}",
        f"- main budget: {cfg.budgets}",
        f"- train/val/test profiles: {cfg.num_train_profiles}/{cfg.num_val_profiles}/{cfg.num_test_profiles}",
        f"- profile combination mode: {cfg.profile_combination_mode}",
        f"- V1 mode pool: {cfg.v1_mode_pool}",
        f"- legacy active pair/triple count: {cfg.active_pair_count}/{cfg.active_triple_count}",
        f"- fixed pair raw weight range: [{cfg.profile_pair_weight_min:.4f}, {cfg.profile_pair_weight_max:.4f}]",
        f"- visit selector: {cfg.visit_selector}",
        f"- Dirichlet alpha: {cfg.dirichlet_alpha} (legacy_pool only)",
        "",
        "## Stage 1",
        "",
        f"- selected views: {', '.join(stage1.get('selected_views', []))}",
        f"- exact/predicted utility Spearman: {stage1.get('spearman', 0.0):.6f}",
        f"- Top-K overlap: {stage1.get('topk_overlap', 0.0):.6f}",
        f"- allowed-region violation: {stage1.get('allowed_region_violation', 0.0):.8f}",
        "",
        "## Stage 2",
        "",
        f"- diffusion checkpoint: {run_dir / 'stage2_user_diffusion' / 'diffusion_guided_checkpoint.pt'}",
        f"- denoising loss: {stage2.get('diffusion', {}).get('denoise', 0.0):.6f}",
        f"- strong guidance attackers: {cfg.guidance_attackers}",
        f"- guidance label mode: {cfg.guidance_label_mode} (pseudo keeps label-free frozen-surrogate targets; true uses true site labels for Stage 2/3 guidance targets).",
        f"- defense-first hard weight / soft scale: {cfg.defense_hard_weight:.4f} / {cfg.defense_soft_objective_scale:.4f}",
        f"- attack-first prior leak/preference/noise weights: {cfg.prior_leak_weight:.4f} / {cfg.prior_preference_weight:.4f} / {cfg.prior_noise_std:.4f}",
        f"- preference auxiliary weight / attack gate margin: {cfg.preference_weight:.4f} / {cfg.preference_attack_gate_margin:.4f}",
        f"- condition profile/selected/preference map/preference weights: "
        f"{cfg.condition_profile_mask}/{cfg.condition_selected_mask}/{cfg.condition_preference_map}/{cfg.condition_preference_weights}",
        f"- direction correction target/strength/min incoming: {cfg.direction_target} / {cfg.direction_correction_strength:.4f} / {cfg.min_incoming_dummy_share:.4f}",
        f"- V1 prefix-hidden alignment weight: {cfg.prefix_hidden_align_weight:.4f}",
        f"- V1 mode-prior weight: {cfg.v1_mode_prior_weight:.4f}",
        f"- differentiable full-DDIM guidance: every {cfg.full_sample_guidance_interval} steps, {cfg.full_sample_guidance_steps} DDIM steps",
        f"- TAM flatten strength/floor: {cfg.tam_flatten_strength:.4f} / {cfg.tam_flatten_floor:.4f}",
        "",
        "## Stage 3",
        "",
        f"- selected keep ratio: {selected.get('keep_ratio', 1.0):.4f}",
        f"- raw bandwidth: {selected.get('raw_bandwidth_overhead', 0.0):.6f}",
        f"- visible bandwidth: {selected.get('visible_dummy_overhead', 0.0):.6f}",
        f"- raw real-packet retention: {selected.get('raw_real_packet_retention', 1.0):.6f}",
        f"- DF input retention: {selected.get('df_input_real_packet_retention', 1.0):.6f}",
        f"- clip rate: {selected.get('clip_rate', 0.0):.6f}",
        f"- deployment repeats: {selected.get('deployment_repeats', 1)}",
        f"- dummy incoming share: {selected.get('dummy_incoming_share', 0.0):.6f}",
        f"- template entropy: {selected.get('template_entropy', 0.0):.6f}",
        f"- V1 mode usage entropy: {selected.get('v1_mode_usage_entropy', 0.0):.6f}",
        f"- preference guard risk: {selected.get('preference_guard_risk_before', 0.0):.6f} -> {selected.get('preference_guard_risk_after', 0.0):.6f}",
        f"- preference accepted fraction: {selected.get('preference_guard_accepted_fraction', 0.0):.6f}",
        f"- label-free attack pressure: {selected.get('selection_attack_pressure', 0.0):.6f}",
        f"- diagnostic conservative attack accuracy: {selected.get('selection_attack_accuracy', 0.0):.6f}",
        f"- selected policy valid: {selected.get('selection_policy_valid', 0)}",
        f"- label-free pressure quality gate: {selected.get('selection_quality_gate', cfg.stage3_max_label_free_attack_pressure):.6f}",
        f"- reliable fixed probes: {selected.get('fixed_probe_reliable_count', 0)}",
        f"- reliable fixed probe worst accuracy: {selected.get('fixed_probe_reliable_worst_accuracy', 0.0):.6f}",
        f"- dummy incoming share gate/pass: {selected.get('selection_dummy_incoming_share_gate', cfg.stage3_min_dummy_incoming_share):.6f} / {selected.get('selection_dummy_incoming_share_gate_passed', 0)}",
        f"- dummy incoming share max gate/pass: {selected.get('selection_dummy_incoming_share_max_gate', cfg.stage3_max_dummy_incoming_share):.6f} / {selected.get('selection_dummy_incoming_share_max_gate_passed', 0)}",
        f"- TAM incoming L1 shift gate/pass: {selected.get('selection_tam_incoming_l1_shift_gate', cfg.stage3_min_tam_incoming_l1_shift):.6f} / {selected.get('selection_tam_incoming_l1_shift_gate_passed', 0)}",
        f"- selection score: {selected.get('selection_score', 0.0):.6f}",
        "",
        "## Reproducibility",
        "",
        "The exact CLI configuration is stored in run_config.json. Sample-level preference JSON is disabled by default.",
    ]
    (run_dir / "final_summary_zh.md").write_text("\n".join(lines), encoding="utf-8")


def run_v4_pipeline(cfg: DefenseConfig) -> Path:
    set_seed(int(cfg.seed))
    device = resolve_device(str(cfg.device))
    version_tag = str(cfg.version).lower()
    display_name = "DMMPv3" if version_tag == "v3" else f"DMMPv3 {version_tag.upper()}"
    requested_stage = str(cfg.stage).strip().lower()
    if requested_stage not in {"1", "2", "3", "all"}:
        raise ValueError(f"Unsupported stage {cfg.stage!r}; expected one of 1, 2, 3, all")
    if requested_stage in {"2", "3"} and not str(cfg.run_name).strip():
        raise ValueError(f"--stage {requested_stage} requires --run_name so DMMPv3 can reuse the existing run directory")
    tag_prefix = "" if version_tag == "v3" else f"{version_tag}_"
    run_name = cfg.run_name or f"dmmpv3_{tag_prefix}strong_surrogate_seed{cfg.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if version_tag != "v3" and version_tag not in run_name.lower():
        run_name = f"{run_name}_{version_tag}"
    run_dir = Path(cfg.output_dir) / run_name
    cli_device = str(cfg.device)
    cli_progress = bool(cfg.progress)
    saved_splits = None
    if requested_stage in {"2", "3"}:
        _require_artifacts(run_dir, requested_stage)
        saved_splits = _load_splits_for_run(run_dir)
        saved_config = _read_json_if_exists(run_dir / "run_config.json")
        saved_values = {
            name: saved_config[name]
            for name in DefenseConfig.__dataclass_fields__
            if name in saved_config and name != "profile_secret"
        }
        cfg = DefenseConfig(**saved_values)
        cfg.stage = requested_stage
        cfg.run_name = run_name
        cfg.output_dir = str(run_dir.parent)
        cfg.device = cli_device
        cfg.progress = cli_progress
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"{display_name} refuses to overwrite a non-empty result directory: {run_dir}")
    log(
        f"[{display_name}] Process started; stage={requested_stage}; device={device}; loading CW from {Path(cfg.data_root).resolve()}...",
        cfg.progress,
    )
    raw, labels, trace_ids, splits, data_source = load_cw_data(cfg)
    if saved_splits is not None:
        splits = saved_splits
    run_dir.mkdir(parents=True, exist_ok=True)
    log(
        f"[{display_name}] CW loaded: samples={len(labels)}, classes={len(np.unique(labels))}, "
        f"train/val/test={len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}",
        cfg.progress,
    )
    if requested_stage in {"1", "all"}:
        config_payload = as_serializable_config(cfg)
        config_payload["profile_secret"] = "<redacted; derived profile keys are stored in profiles_private.json>"
        write_json(
            run_dir / "run_config.json",
            {
                **config_payload,
                "version": version_tag,
                "implementation": "defence/DMMPv3/strong-DF-RF-guided",
                "device_resolved": str(device),
                "data_source": data_source,
                "num_samples": int(len(labels)),
                "num_classes": int(len(np.unique(labels))),
                "split_sizes": {key: int(len(value)) for key, value in splits.items()},
            },
        )
        write_json(run_dir / "split_indices.json", {key: value.astype(int).tolist() for key, value in splits.items()})
        source_scan = Path(__file__).resolve().parents[2] / "docs" / "project_map.md"
        if source_scan.is_file():
            shutil.copy2(source_scan, run_dir / "project_map.md")
    profiles = _prepare_profile_artifacts(run_dir, cfg)
    log(f"{display_name} run start: {run_dir}; stage={requested_stage}; samples={len(labels)}, profiles={len(profiles['train'])}/{len(profiles['validation'])}/{len(profiles['test'])}", cfg.progress)
    stage1_metrics = _read_json_if_exists(run_dir / "stage1_executable_condition" / "candidate_metrics.json")
    stage2_metrics = _read_json_if_exists(run_dir / "stage2_user_diffusion" / "stage2_metrics.json")
    stage3_metrics = _read_json_if_exists(run_dir / "stage3_guided_refinement" / "stage3_metrics.json")
    if requested_stage in {"1", "all"}:
        stage_timer = _stage_begin(
            "Stage 1/3 Prefix Leakage + Executable Candidate Cells",
            cfg,
            f"train={len(splits['train'])}, val={len(splits['val'])}",
        )
        stage1_metrics = run_candidate_stage(raw, labels, splits["train"], splits["val"], run_dir, cfg, device)
        _stage_end(
            "Stage 1/3 Prefix Leakage + Executable Candidate Cells",
            cfg,
            stage_timer,
            f"spearman={float(stage1_metrics.get('spearman', 0.0)):.4f}, topk_overlap={float(stage1_metrics.get('topk_overlap', 0.0)):.4f}",
        )
    if requested_stage in {"2", "all"}:
        if requested_stage == "2":
            _require_artifacts(run_dir, "2")
        stage_timer = _stage_begin(
            "Stage 2/3 Encoder + Guided Diffusion",
            cfg,
            f"steps={cfg.diffusion_train_steps}, encoder_epochs={cfg.encoder_epochs}",
        )
        stage2_metrics = run_v4_stage2(raw, labels, trace_ids, splits, run_dir, cfg, device, profiles)
        _stage_end(
            "Stage 2/3 Encoder + Guided Diffusion",
            cfg,
            stage_timer,
            f"denoise={float(stage2_metrics.get('diffusion', {}).get('denoise', 0.0)):.4f}, "
            f"reference_overhead={float(stage2_metrics.get('reference_generation', {}).get('visible_dummy_overhead', 0.0)):.4f}",
        )
    if requested_stage in {"3", "all"}:
        if requested_stage == "3":
            _require_artifacts(run_dir, "3")
        stage_timer = _stage_begin(
            "Stage 3/3 Guided Refinement + Selection",
            cfg,
            f"pareto_budgets={cfg.pareto_budgets}, keep_ratios={cfg.refine_keep_ratios}",
        )
        stage3_metrics = run_v4_stage3(raw, labels, trace_ids, splits, run_dir, cfg, device, profiles)
        selected = stage3_metrics.get("selected", {})
        _stage_end(
            "Stage 3/3 Guided Refinement + Selection",
            cfg,
            stage_timer,
            f"selected_budget={float(selected.get('budget', 0.0)):.3f}, "
            f"pressure={float(selected.get('selection_attack_pressure', 0.0)):.4f}, "
            f"diagnostic_acc={float(selected.get('selection_attack_accuracy', 0.0)):.4f}",
        )
    if requested_stage in {"3", "all"}:
        _write_v4_final_summary(run_dir, cfg, data_source, stage1_metrics, stage2_metrics, stage3_metrics)
    log(f"[done] {display_name} results saved to: {run_dir}", cfg.progress)
    return run_dir

