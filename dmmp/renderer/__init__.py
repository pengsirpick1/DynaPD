"""Renderer-facing exports for defended trace materialization."""

from ..projection.padding import (
    crop_ragged_for_attacker,
    load_ragged_npz,
    render_batch,
    render_batch_variable,
    render_trace,
    render_trace_variable,
    save_ragged_npz,
)

__all__ = [
    "crop_ragged_for_attacker",
    "load_ragged_npz",
    "render_batch",
    "render_batch_variable",
    "render_trace",
    "render_trace_variable",
    "save_ragged_npz",
]
