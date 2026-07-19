"""High-level target strategy families built on low-level primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..encoders.prefix import PrefixCondition
from .low_level_primitives import PRIMITIVES

FAMILIES = (
    "residual_leakage_suppression",
    "local_structure_camouflage",
    "global_to_local_regularization",
    "anonymous_profile_collision",
    "realistic_stochastic_morphing",
)

FAMILY_TO_PRIMITIVE_PRIOR: dict[str, np.ndarray] = {
    "residual_leakage_suppression": np.asarray([0.30, 0.10, 0.25, 0.15, 0.20], dtype=np.float32),
    "local_structure_camouflage": np.asarray([0.35, 0.25, 0.20, 0.15, 0.05], dtype=np.float32),
    "global_to_local_regularization": np.asarray([0.15, 0.15, 0.30, 0.10, 0.30], dtype=np.float32),
    "anonymous_profile_collision": np.asarray([0.10, 0.25, 0.20, 0.20, 0.25], dtype=np.float32),
    "realistic_stochastic_morphing": np.asarray([0.15, 0.20, 0.20, 0.15, 0.30], dtype=np.float32),
}


@dataclass(frozen=True)
class FamilySelection:
    family_indices: np.ndarray
    family_weights: np.ndarray
    primitive_indices: np.ndarray
    primitive_weights: np.ndarray


def _normalize(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = float(arr.sum())
    if total <= 1.0e-12:
        return np.full(arr.shape, 1.0 / max(arr.size, 1), dtype=np.float32)
    return (arr / total).astype(np.float32)


def score_families(condition: PrefixCondition, allowed_mask: np.ndarray, residual_feedback: np.ndarray | None = None) -> np.ndarray:
    mask_coverage = float(np.asarray(allowed_mask, dtype=np.float32).mean())
    leakage = float(np.asarray(condition.saliency, dtype=np.float32).mean())
    gap = float(np.asarray(condition.gap_saliency, dtype=np.float32).mean())
    rate_var = float(np.asarray(condition.rate_saliency, dtype=np.float32).var())
    burst = float(np.asarray(condition.burst_saliency, dtype=np.float32).max(initial=0.0))
    residual = 0.0 if residual_feedback is None else float(np.maximum(residual_feedback, 0.0).mean())
    scores = np.asarray(
        [
            0.45 * leakage + 0.35 * residual + 0.20 * mask_coverage,
            0.45 * burst + 0.35 * gap + 0.20 * rate_var,
            0.35 * mask_coverage + 0.35 * (1.0 - min(leakage, 1.0)) + 0.30 * rate_var,
            0.35 * gap + 0.30 * mask_coverage + 0.35 * leakage,
            0.30 * mask_coverage + 0.30 * gap + 0.40 * rate_var,
        ],
        dtype=np.float32,
    )
    return scores


def sparse_topk_dirichlet(
    scores: np.ndarray,
    *,
    top_k: int,
    count_choices: tuple[int, ...],
    alpha: float,
    noise_scale: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(scores, dtype=np.float64)
    if float(noise_scale) > 0:
        raw = raw + rng.normal(0.0, float(noise_scale), size=raw.shape)
    top = np.argsort(-raw, kind="mergesort")[: max(1, min(int(top_k), raw.size))]
    count = int(rng.choice(np.asarray(count_choices, dtype=np.int64)))
    count = max(1, min(count, top.size))
    selected = np.asarray(rng.choice(top, size=count, replace=False), dtype=np.int64)
    weights = np.zeros(raw.size, dtype=np.float32)
    local = rng.dirichlet(np.full(count, max(float(alpha), 1.0e-3), dtype=np.float64)).astype(np.float32)
    weights[selected] = local
    return selected, _normalize(weights)


def select_family_and_primitives(
    condition: PrefixCondition,
    allowed_mask: np.ndarray,
    *,
    family_top_k: int,
    selected_family_count_choices: tuple[int, ...],
    primitive_top_k: int,
    selected_primitive_count_choices: tuple[int, ...],
    primitive_dirichlet_alpha: float,
    primitive_noise_scale: float,
    rng: np.random.Generator,
) -> FamilySelection:
    family_scores = score_families(condition, allowed_mask)
    family_indices, family_weights = sparse_topk_dirichlet(
        family_scores,
        top_k=family_top_k,
        count_choices=selected_family_count_choices,
        alpha=primitive_dirichlet_alpha,
        noise_scale=primitive_noise_scale,
        rng=rng,
    )
    primitive_scores = np.zeros(len(PRIMITIVES), dtype=np.float32)
    for family_index, weight in enumerate(family_weights.tolist()):
        if weight <= 0:
            continue
        primitive_scores += float(weight) * FAMILY_TO_PRIMITIVE_PRIOR[FAMILIES[family_index]]
    primitive_indices, primitive_weights = sparse_topk_dirichlet(
        primitive_scores,
        top_k=primitive_top_k,
        count_choices=selected_primitive_count_choices,
        alpha=primitive_dirichlet_alpha,
        noise_scale=primitive_noise_scale,
        rng=rng,
    )
    return FamilySelection(
        family_indices=family_indices.astype(np.int64),
        family_weights=family_weights.astype(np.float32),
        primitive_indices=primitive_indices.astype(np.int64),
        primitive_weights=primitive_weights.astype(np.float32),
    )
