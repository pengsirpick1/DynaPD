"""Projection, rounding, and padding template helpers."""

from .padding import (
    PaddingTemplate,
    aggregate_template_stats,
    crop_ragged_for_attacker,
    exact_round_probabilities,
    load_ragged_npz,
    normalized_template_entropy,
    project_policy_to_template,
    refine_counts,
    render_batch,
    render_batch_variable,
    render_trace,
    render_trace_variable,
    save_ragged_npz,
    target_padding_count,
)

__all__ = [
    "PaddingTemplate",
    "aggregate_template_stats",
    "crop_ragged_for_attacker",
    "exact_round_probabilities",
    "load_ragged_npz",
    "normalized_template_entropy",
    "project_policy_to_template",
    "refine_counts",
    "render_batch",
    "render_batch_variable",
    "render_trace",
    "render_trace_variable",
    "save_ragged_npz",
    "target_padding_count",
]
