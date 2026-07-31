"""Label-free objectives and post-hoc metrics for Stage B."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dynapd.stage_a.mask_ops import entropy_np, js_divergence_np


@dataclass(frozen=True)
class ObjectiveWeights:
    confidence: float = 0.30
    margin: float = 0.50
    entropy: float = 0.20


def top1_margin(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape[-1] < 2:
        return np.zeros(probs.shape[:-1], dtype=np.float32)
    ordered = np.sort(probs, axis=-1)
    return (ordered[..., -1] - ordered[..., -2]).astype(np.float32)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float64)
    denom = max(float(np.log(max(probs.shape[-1], 2))), 1e-8)
    return (entropy_np(probs) / denom).astype(np.float32)


def uncertainty_utility(probabilities: np.ndarray, weights: ObjectiveWeights | None = None) -> np.ndarray:
    """Label-free uncertainty utility; higher means less confident attacker output."""
    probs = np.asarray(probabilities, dtype=np.float64)
    w = weights or ObjectiveWeights()
    confidence_term = 1.0 - np.max(probs, axis=-1)
    margin_term = 1.0 - top1_margin(probs)
    entropy_term = normalized_entropy(probs)
    return (
        float(w.confidence) * confidence_term
        + float(w.margin) * margin_term
        + float(w.entropy) * entropy_term
    ).astype(np.float32)


def objective_delta(
    reference_prob: np.ndarray,
    evaluated_prob: np.ndarray,
    weights: ObjectiveWeights | None = None,
) -> np.ndarray:
    """Label-free utility gain from reference to evaluated probabilities."""
    reference = np.asarray(reference_prob, dtype=np.float32)
    evaluated = np.asarray(evaluated_prob, dtype=np.float32)
    if reference.ndim == 1:
        reference = reference.reshape(1, -1)
    if evaluated.ndim == 1:
        evaluated = evaluated.reshape(1, -1)
    if reference.shape[0] == 1 and evaluated.shape[0] > 1:
        reference = np.repeat(reference, evaluated.shape[0], axis=0)
    if reference.shape != evaluated.shape:
        raise ValueError(f"Probability shapes must match, got {reference.shape} and {evaluated.shape}")
    w = weights or ObjectiveWeights()
    reference_pred = reference.argmax(axis=1)
    confidence_drop = reference[np.arange(len(reference)), reference_pred] - evaluated[np.arange(len(reference)), reference_pred]
    margin_drop = top1_margin(reference) - top1_margin(evaluated)
    entropy_gain = normalized_entropy(evaluated) - normalized_entropy(reference)
    return (
        float(w.confidence) * confidence_drop
        + float(w.margin) * margin_drop
        + float(w.entropy) * entropy_gain
    ).astype(np.float32)


def original_class_margin(probabilities: np.ndarray, original_class: np.ndarray) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=np.float32)
    y0 = np.asarray(original_class, dtype=np.int64).reshape(-1)
    if probs.ndim == 1:
        probs = probs.reshape(1, -1)
    if y0.size == 1 and probs.shape[0] > 1:
        y0 = np.repeat(y0, probs.shape[0])
    masked = probs.copy()
    masked[np.arange(len(probs)), y0] = -np.inf
    other = np.max(masked, axis=1)
    return (probs[np.arange(len(probs)), y0] - other).astype(np.float32)


def original_class_utility(
    original_prob: np.ndarray,
    evaluated_prob: np.ndarray,
    weights: ObjectiveWeights | None = None,
) -> np.ndarray:
    """Utility anchored to the attacker's original predicted class, not the current top-1."""
    original = np.asarray(original_prob, dtype=np.float32)
    evaluated = np.asarray(evaluated_prob, dtype=np.float32)
    if original.ndim == 1:
        original = original.reshape(1, -1)
    if evaluated.ndim == 1:
        evaluated = evaluated.reshape(1, -1)
    if original.shape[0] == 1 and evaluated.shape[0] > 1:
        original = np.repeat(original, evaluated.shape[0], axis=0)
    if original.shape != evaluated.shape:
        raise ValueError(f"Probability shapes must match, got {original.shape} and {evaluated.shape}")
    w = weights or ObjectiveWeights()
    y0 = original.argmax(axis=1)
    original_p = original[np.arange(len(original)), y0]
    evaluated_p = evaluated[np.arange(len(evaluated)), y0]
    original_margin = original_class_margin(original, y0)
    evaluated_margin = original_class_margin(evaluated, y0)
    return (
        float(w.confidence) * (original_p - evaluated_p)
        + float(w.margin) * (original_margin - evaluated_margin)
        + float(w.entropy) * (normalized_entropy(evaluated) - normalized_entropy(original))
    ).astype(np.float32)


def original_class_objective_delta(
    original_prob: np.ndarray,
    reference_prob: np.ndarray,
    evaluated_prob: np.ndarray,
    weights: ObjectiveWeights | None = None,
) -> np.ndarray:
    """Marginal utility gain with the original predicted class frozen."""
    before = original_class_utility(original_prob, reference_prob, weights)
    after = original_class_utility(original_prob, evaluated_prob, weights)
    return (after - before).astype(np.float32)


def probability_metrics(original_prob: np.ndarray, evaluated_prob: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    """Return metrics relative to the original prediction distribution."""
    original = np.asarray(original_prob, dtype=np.float32)
    evaluated = np.asarray(evaluated_prob, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    original_pred = original.argmax(axis=1)
    evaluated_pred = evaluated.argmax(axis=1)
    original_top1 = original[np.arange(len(original)), original_pred]
    evaluated_original_top1 = evaluated[np.arange(len(original)), original_pred]
    original_margin = top1_margin(original)
    evaluated_margin = top1_margin(evaluated)
    y0_margin = original_class_margin(evaluated, original_pred)
    original_y0_margin = original_class_margin(original, original_pred)
    return {
        "accuracy": (evaluated_pred == y).astype(np.float32),
        "flip": (evaluated_pred != original_pred).astype(np.float32),
        "js_div": js_divergence_np(original, evaluated).astype(np.float32),
        "top1_drop": (original_top1 - evaluated_original_top1).astype(np.float32),
        "original_top1_drop": (original_top1 - evaluated_original_top1).astype(np.float32),
        "original_class_probability": evaluated_original_top1.astype(np.float32),
        "original_class_margin": y0_margin.astype(np.float32),
        "original_class_margin_drop": (original_y0_margin - y0_margin).astype(np.float32),
        "max_confidence_drop": (np.max(original, axis=1) - np.max(evaluated, axis=1)).astype(np.float32),
        "margin_drop": (original_margin - evaluated_margin).astype(np.float32),
        "current_top1_confidence": np.max(evaluated, axis=1).astype(np.float32),
        "current_top1_margin": evaluated_margin.astype(np.float32),
        "entropy_gain": (entropy_np(evaluated) - entropy_np(original)).astype(np.float32),
        "normalized_entropy_gain": (normalized_entropy(evaluated) - normalized_entropy(original)).astype(np.float32),
        "original_pred": original_pred.astype(np.int64),
        "evaluated_pred": evaluated_pred.astype(np.int64),
        "utility": uncertainty_utility(evaluated).astype(np.float32),
        "original_class_utility": original_class_utility(original, evaluated).astype(np.float32),
    }
