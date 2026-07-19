"""Candidate x0* policy generation from prefix-only information."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..encoders.prefix import PrefixCondition, extract_prefix_condition, nonzero_trace
from .config import TargetPolicyConfig
from .constraint_checker import ConstraintReport, check_counts
from .low_level_primitives import PRIMITIVES, compute_primitive_utilities
from .representation import counts_to_x0_star, largest_remainder_rounding, normalize_positive
from .strategy_families import FamilySelection, select_family_and_primitives


@dataclass
class CandidatePolicy:
    x0_star: np.ndarray
    allocation: np.ndarray
    counts: np.ndarray
    allowed_mask: np.ndarray
    budget_ratio: float
    budget_count: int
    family_weights: np.ndarray
    primitive_weights: np.ndarray
    family_indices: np.ndarray
    primitive_indices: np.ndarray
    effect_map: np.ndarray
    action_rank: np.ndarray
    marginal_gain: np.ndarray
    quality_score: float
    proxy_score_df: float
    proxy_score_rf: float
    proxy_score_attack: float
    latency_cost: float
    fallback_flag: bool
    constraint_report: ConstraintReport
    construction_seed: int
    selection_score_attack: float | None = None
    selection_score_df: float | None = None
    selection_score_rf: float | None = None
    teacher_scored: bool = False
    teacher_score_source: str = "heuristic_proxy"
    teacher_score_attack: float | None = None
    teacher_score_df: float | None = None
    teacher_score_rf: float | None = None
    teacher_score_components: dict[str, float] = field(default_factory=dict)


def _mix_class_condition_prior(
    utility: np.ndarray,
    allowed_mask: np.ndarray,
    class_condition_prior: np.ndarray | None,
    class_condition_weight: float,
) -> np.ndarray:
    weight = float(np.clip(class_condition_weight, 0.0, 1.0))
    base = normalize_positive(utility, allowed_mask)
    if class_condition_prior is None or weight <= 0.0:
        return base
    prior = np.asarray(class_condition_prior, dtype=np.float32)
    if prior.shape != base.shape:
        prior = np.resize(prior, base.shape).astype(np.float32)
    prior = normalize_positive(prior, allowed_mask)
    if float(prior.sum()) <= 1.0e-8:
        return base
    return normalize_positive((1.0 - weight) * base + weight * prior, allowed_mask)


def _utility_from_selection(
    condition: PrefixCondition,
    allowed_mask: np.ndarray,
    selection: FamilySelection,
    *,
    class_condition_prior: np.ndarray | None = None,
    class_condition_weight: float = 0.0,
) -> np.ndarray:
    primitive_maps = compute_primitive_utilities(condition, allowed_mask).as_dict()
    utility = np.zeros_like(allowed_mask, dtype=np.float32)
    for primitive_name, weight in zip(PRIMITIVES, selection.primitive_weights.tolist()):
        if float(weight) > 0:
            utility += float(weight) * primitive_maps[primitive_name]
    return _mix_class_condition_prior(
        utility,
        allowed_mask,
        class_condition_prior,
        class_condition_weight,
    )


def _ranked_path(utility: np.ndarray, target_count: int, allowed_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_utility = np.asarray(utility, dtype=np.float64).reshape(-1)
    flat_mask = np.asarray(allowed_mask, dtype=np.float32).reshape(-1) > 0
    order = np.argsort(-flat_utility, kind="mergesort")
    order = np.asarray([idx for idx in order.tolist() if flat_mask[int(idx)]], dtype=np.int64)
    if order.size == 0:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    chosen = np.resize(order, max(0, int(target_count)))
    patches = np.stack(np.unravel_index(chosen, np.asarray(utility).shape), axis=1).astype(np.int64)
    gains = flat_utility[chosen].astype(np.float32)
    return patches, gains


def _proxy_quality(
    utility: np.ndarray,
    counts: np.ndarray,
    report: ConstraintReport,
    cfg: TargetPolicyConfig,
) -> tuple[float, float, float, float]:
    allocation = normalize_positive(counts, np.ones_like(counts, dtype=np.float32))
    focus = float((allocation * utility).sum())
    entropy = 0.0
    positive = allocation[allocation > 0]
    if positive.size > 1:
        entropy = float(-(positive * np.log(positive + 1.0e-12)).sum() / np.log(allocation.size))
    density_reward = max(0.0, 1.0 - float(report.max_local_density))
    tail_reward = max(0.0, 1.0 - float(report.tail_extension_ratio))
    pseudo_drop_proxy = focus
    margin_drop_proxy = 0.5 * density_reward + 0.5 * tail_reward
    max_drop_proxy = focus
    score_df = (
        float(cfg.score_entropy_weight) * entropy
        + float(cfg.score_pseudo_weight) * pseudo_drop_proxy
        + float(cfg.score_margin_weight) * margin_drop_proxy
        + float(cfg.score_max_weight) * max_drop_proxy
        - float(cfg.tail_penalty_weight) * float(report.tail_extension_ratio)
    )
    score_rf = (
        float(cfg.score_entropy_weight) * entropy
        + float(cfg.score_pseudo_weight) * pseudo_drop_proxy
        + float(cfg.score_margin_weight) * density_reward
        + float(cfg.score_max_weight) * max_drop_proxy
        - float(cfg.density_penalty_weight) * float(report.max_local_density)
    )
    score_attack = (
        0.5 * (score_df + score_rf)
        + float(cfg.robust_min_weight) * min(score_df, score_rf)
        - float(cfg.attacker_gap_weight) * abs(score_df - score_rf)
    )
    gate_penalty = 0.0
    if score_df < float(cfg.min_df_gain):
        gate_penalty += float(cfg.min_df_gain) - score_df
    if score_rf < float(cfg.min_rf_gain):
        gate_penalty += float(cfg.min_rf_gain) - score_rf
    legality_penalty = 5.0 * float(not report.valid) + 1.0 * float(not report.deployable)
    quality = score_attack - gate_penalty - legality_penalty
    return float(quality), float(score_df), float(score_rf), float(score_attack)


def generate_candidates_for_trace(
    clean_trace: np.ndarray,
    *,
    cfg: TargetPolicyConfig,
    prefix_condition: PrefixCondition | None = None,
    clean_index: int = -1,
    class_condition_prior: np.ndarray | None = None,
    class_condition_weight: float = 0.0,
    rng: np.random.Generator,
) -> list[CandidatePolicy]:
    del clean_index
    condition = prefix_condition or extract_prefix_condition(
        clean_trace,
        prefix_n=int(cfg.prefix_length),
        patch_num=int(cfg.strategy_horizon),
    )
    allowed_mask = np.asarray(condition.allowed_mask, dtype=np.float32)
    clean_count = int(nonzero_trace(clean_trace).size)
    candidates: list[CandidatePolicy] = []
    for _ in range(max(1, int(cfg.num_candidates))):
        construction_seed = int(rng.integers(0, 2**31 - 1))
        local_rng = np.random.default_rng(construction_seed)
        selection = select_family_and_primitives(
            condition,
            allowed_mask,
            family_top_k=int(cfg.family_top_k),
            selected_family_count_choices=tuple(int(v) for v in cfg.selected_family_count_choices),
            primitive_top_k=int(cfg.primitive_top_k),
            selected_primitive_count_choices=tuple(int(v) for v in cfg.selected_primitive_count_choices),
            primitive_dirichlet_alpha=float(cfg.primitive_dirichlet_alpha),
            primitive_noise_scale=float(cfg.primitive_noise_scale),
            rng=local_rng,
        )
        utility = _utility_from_selection(
            condition,
            allowed_mask,
            selection,
            class_condition_prior=class_condition_prior,
            class_condition_weight=float(class_condition_weight),
        )
        max_count = max(1, int(round(clean_count * float(cfg.max_budget)))) if clean_count > 0 else 0
        action_rank, marginal_gain = _ranked_path(utility, max_count, allowed_mask)
        for budget in cfg.budgets:
            target_count = max(1, int(round(clean_count * float(budget)))) if clean_count > 0 and budget > 0 else 0
            counts = largest_remainder_rounding(utility, target_count, allowed_mask)
            report = check_counts(
                counts,
                allowed_mask,
                target_count,
                max_dummy_per_slot=int(cfg.max_dummy_per_slot),
                max_consecutive_dummy_run=int(cfg.max_consecutive_dummy_run),
                max_tail_extension_ratio=float(cfg.max_tail_extension_ratio),
                max_local_dummy_density=float(cfg.max_local_dummy_density),
            )
            x0_star, allocation = counts_to_x0_star(counts, allowed_mask, clr_epsilon=float(cfg.clr_epsilon))
            quality, score_df, score_rf, score_attack = _proxy_quality(utility, counts, report, cfg)
            candidates.append(
                CandidatePolicy(
                    x0_star=x0_star,
                    allocation=allocation,
                    counts=counts.astype(np.int32),
                    allowed_mask=allowed_mask.astype(np.float32),
                    budget_ratio=float(budget),
                    budget_count=int(target_count),
                    family_weights=selection.family_weights.astype(np.float32),
                    primitive_weights=selection.primitive_weights.astype(np.float32),
                    family_indices=selection.family_indices.astype(np.int64),
                    primitive_indices=selection.primitive_indices.astype(np.int64),
                    effect_map=utility.astype(np.float32),
                    action_rank=action_rank.astype(np.int64),
                    marginal_gain=marginal_gain.astype(np.float32),
                    quality_score=float(quality),
                    proxy_score_df=float(score_df),
                    proxy_score_rf=float(score_rf),
                    proxy_score_attack=float(score_attack),
                    latency_cost=float(report.tail_extension_ratio),
                    fallback_flag=not bool(report.deployable),
                    constraint_report=report,
                    construction_seed=construction_seed,
                    selection_score_attack=float(score_attack),
                    selection_score_df=float(score_df),
                    selection_score_rf=float(score_rf),
                )
            )
    return candidates
