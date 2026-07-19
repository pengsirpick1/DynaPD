"""Label-free attack-score formulas used by offline target-pool builders."""

from __future__ import annotations

import numpy as np


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr - float(np.max(arr))
    exp = np.exp(np.clip(arr, -60.0, 60.0))
    return (exp / max(float(exp.sum()), 1.0e-12)).astype(np.float64)


def normalized_entropy(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=np.float64)
    p = p / max(float(p.sum()), 1.0e-12)
    if p.size <= 1:
        return 0.0
    return float(-(p * np.log(p + 1.0e-12)).sum() / np.log(p.size))


def _top_margin(probabilities: np.ndarray) -> float:
    p = np.sort(np.asarray(probabilities, dtype=np.float64))
    if p.size <= 1:
        return 0.0
    return float(p[-1] - p[-2])


def score_from_probabilities(
    clean_probabilities: np.ndarray,
    defended_probabilities: np.ndarray,
    *,
    entropy_weight: float = 0.30,
    pseudo_weight: float = 0.30,
    margin_weight: float = 0.20,
    max_weight: float = 0.20,
) -> dict[str, float | int]:
    p_clean = np.asarray(clean_probabilities, dtype=np.float64)
    p_def = np.asarray(defended_probabilities, dtype=np.float64)
    p_clean = p_clean / max(float(p_clean.sum()), 1.0e-12)
    p_def = p_def / max(float(p_def.sum()), 1.0e-12)
    pseudo = int(np.argmax(p_clean))
    entropy_gain = normalized_entropy(p_def) - normalized_entropy(p_clean)
    pseudo_drop = float(p_clean[pseudo] - p_def[pseudo])
    margin_drop = _top_margin(p_clean) - _top_margin(p_def)
    max_conf_drop = float(np.max(p_clean) - np.max(p_def))
    score = (
        float(entropy_weight) * entropy_gain
        + float(pseudo_weight) * pseudo_drop
        + float(margin_weight) * margin_drop
        + float(max_weight) * max_conf_drop
    )
    return {
        "score": float(score),
        "pseudo_label": pseudo,
        "entropy_gain": float(entropy_gain),
        "pseudo_conf_drop": float(pseudo_drop),
        "margin_drop": float(margin_drop),
        "max_conf_drop": float(max_conf_drop),
    }


def score_from_logits(clean_logits: np.ndarray, defended_logits: np.ndarray, **weights) -> dict[str, float | int]:
    return score_from_probabilities(_softmax(clean_logits), _softmax(defended_logits), **weights)


def combine_df_rf_scores(
    score_df: float,
    score_rf: float,
    *,
    robust_min_weight: float = 0.25,
    attacker_gap_weight: float = 0.25,
) -> float:
    mean = 0.5 * (float(score_df) + float(score_rf))
    minimum = min(float(score_df), float(score_rf))
    gap = abs(float(score_df) - float(score_rf))
    return float(mean + float(robust_min_weight) * minimum - float(attacker_gap_weight) * gap)
