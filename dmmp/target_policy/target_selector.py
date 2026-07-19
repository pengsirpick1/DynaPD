"""Quality and diversity selection for target policies."""

from __future__ import annotations

import numpy as np

from .candidate_generator import CandidatePolicy


def _selection_score(candidate: CandidatePolicy, name: str, fallback: float) -> float:
    value = getattr(candidate, name, None)
    if value is None:
        return float(fallback)
    return float(value)


def allocation_distance(
    first: np.ndarray,
    second: np.ndarray,
    *,
    l1_weight: float = 0.5,
    cosine_weight: float = 0.5,
) -> float:
    a = np.asarray(first, dtype=np.float64).reshape(-1)
    b = np.asarray(second, dtype=np.float64).reshape(-1)
    l1 = float(np.abs(a - b).sum())
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1.0e-12)
    cosine = 1.0 - float(np.dot(a, b) / denom)
    return float(float(l1_weight) * l1 + float(cosine_weight) * cosine)


def select_targets(
    candidates: list[CandidatePolicy],
    *,
    target_count: int,
    quality_target_count: int,
    diverse_target_count: int,
    allocation_l1_weight: float = 0.5,
    allocation_cosine_weight: float = 0.5,
) -> tuple[list[CandidatePolicy], int]:
    deployable = [candidate for candidate in candidates if candidate.constraint_report.valid and candidate.constraint_report.deployable]
    legal = [candidate for candidate in candidates if candidate.constraint_report.valid]
    pool = deployable if deployable else legal if legal else list(candidates)
    ordered = sorted(
        pool,
        key=lambda item: (
            _selection_score(item, "selection_score_attack", item.quality_score),
            float(
                min(
                    _selection_score(item, "selection_score_df", item.proxy_score_df),
                    _selection_score(item, "selection_score_rf", item.proxy_score_rf),
                )
            ),
            -float(
                abs(
                    _selection_score(item, "selection_score_df", item.proxy_score_df)
                    - _selection_score(item, "selection_score_rf", item.proxy_score_rf)
                )
            ),
            -float(item.latency_cost),
        ),
        reverse=True,
    )
    selected: list[CandidatePolicy] = ordered[: max(0, min(int(quality_target_count), int(target_count), len(ordered)))]
    remaining = [candidate for candidate in ordered if id(candidate) not in {id(item) for item in selected}]
    while remaining and len(selected) < int(target_count) and len(selected) < int(quality_target_count) + int(diverse_target_count):
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best_index = max(
            range(len(remaining)),
            key=lambda idx: min(
                allocation_distance(
                    remaining[idx].allocation,
                    chosen.allocation,
                    l1_weight=float(allocation_l1_weight),
                    cosine_weight=float(allocation_cosine_weight),
                )
                for chosen in selected
            ),
        )
        selected.append(remaining.pop(int(best_index)))
    if len(selected) < int(target_count):
        selected.extend(remaining[: int(target_count) - len(selected)])
    fallback_count = max(0, int(target_count) - len(deployable)) if len(deployable) < int(target_count) else 0
    return selected[: int(target_count)], int(fallback_count)
