# -*- coding: utf-8 -*-
"""Run the E2b RF+DF+AWF multi-surrogate Stage B teacher oracle audit.

This keeps the existing RF keypoints and stratified Top128 candidate space, but
scores each candidate with RF, WFlib DF, and the third WFlib surrogate slot.
For E2b, pass the AWF checkpoint to ``--varcnn_checkpoint``; the slot name stays
``varcnn`` for compatibility with the shared oracle/reporting code.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import re
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WFLIB_ROOT = ROOT / "wflib_copy"
if str(WFLIB_ROOT) not in sys.path:
    sys.path.insert(0, str(WFLIB_ROOT))

from dynapd.stage_a.faithfulness import predict_probabilities
from dynapd.stage_a.modeling import load_stage_a_attacker
from dynapd.stage_b.expanded_generator import (
    action_cost,
    action_identity,
    generate_compact_action_descriptors,
    generate_expanded_actions,
    materialize_candidate_descriptors,
)
from dynapd.stage_b.objectives import probability_metrics
from dynapd.utils import resolve_device, set_seed
from dynapd.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR
from scripts.stage_b_run_b2e_diverse_search import (
    DEFAULT_ARCHIVE,
    MethodConfig,
    _action_dummy_count,
    _identity,
    _method_config,
    _parse_csv_floats,
    _parse_csv_strings,
    _prefilter_actions,
    _resource_fields,
    _runtime_args,
    _selection_scores,
)
from scripts.stage_b_run_dual_actuator import (
    EvalState,
    _apply_delay,
    _default_checkpoint,
    _fast_refresh_mask,
    _initial_state,
    _load_archive,
    _load_raw_rows,
    _predict_one,
    _render_dummy,
    _render_dummy_batch,
    _timing_add,
)
from WFlib import models as wflib_models


MODEL_NAMES = ("rf", "df", "varcnn")


@dataclass
class EnsembleResult:
    state: EvalState
    final_probs: dict[str, np.ndarray]
    original_probs: dict[str, np.ndarray]
    stop_reason: str
    accepted_action_count: int
    candidate_step_count: int
    candidate_eval_count: int
    selected_positive_models: list[int]
    selected_worst_gains: list[float]
    selected_mean_gains: list[float]
    audit_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--rf_checkpoint", default="")
    parser.add_argument("--df_checkpoint", default="wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth")
    parser.add_argument("--varcnn_checkpoint", default="wflib_copy/checkpoints/CW/VarCNN/dynapd_clean_seed0.pth")
    parser.add_argument("--protocols", default="bidirectional_cooperative")
    parser.add_argument("--methods", default="rf_only,mean,worst,blend_a0.5,blend_a0.7")
    parser.add_argument("--dummy_budgets", default="0.10")
    parser.add_argument("--max_samples", type=int, default=512)
    parser.add_argument("--sample_start", type=int, default=0)
    parser.add_argument("--sample_end", type=int, default=0)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--cost_lambda", type=float, default=0.05)
    parser.add_argument("--robust_min_positive_models", type=int, default=2)
    parser.add_argument("--robust_epsilon", type=float, default=0.0)
    parser.add_argument("--random_epsilon", type=float, default=0.0)
    parser.add_argument("--norm_epsilon", type=float, default=1e-3)
    parser.add_argument("--topk_overlap_k", type=int, default=10)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--profile_candidate_generation", action="store_true")
    parser.add_argument("--compact_candidate_generation", action="store_true")
    parser.add_argument("--deferred_materialize_oversample", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=0)
    parser.add_argument("--materialization_batch_size", type=int, default=64)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_score_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="renderer")
    parser.add_argument("--export_defended_npz", default="")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b_ensemble_oracle_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _resolve_path(path: str) -> str:
    p = Path(path)
    return str(p if p.is_absolute() else ROOT / p)


def _load_wflib_model(model_name: str, checkpoint: str, num_classes: int, device: torch.device) -> torch.nn.Module:
    model = getattr(wflib_models, model_name)(int(num_classes))
    state = torch.load(_resolve_path(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _align_rows(rows: list[np.ndarray] | np.ndarray, max_len: int) -> np.ndarray:
    if isinstance(rows, np.ndarray) and rows.ndim == 2:
        arr = np.asarray(rows, dtype=np.float32)
        out = np.zeros((arr.shape[0], int(max_len)), dtype=np.float32)
        n = min(arr.shape[1], int(max_len))
        if n:
            out[:, :n] = arr[:, :n]
        return out
    else:
        items = [np.asarray(row, dtype=np.float32).reshape(-1) for row in rows]
        out = np.zeros((len(items), int(max_len)), dtype=np.float32)
        for idx, item in enumerate(items):
            n = min(item.size, int(max_len))
            if n:
                out[idx, :n] = item[:n]
        return out


def _wflib_features(traces: list[np.ndarray] | np.ndarray, feature: str, seq_len: int) -> torch.Tensor:
    x = _align_rows(traces, int(seq_len))
    if str(feature) == "DIR":
        return torch.as_tensor(np.sign(x)[:, None, :], dtype=torch.float32)
    if str(feature) == "DT2":
        x_dir = np.sign(x)
        x_time = np.abs(x)
        x_time = np.diff(x_time, axis=1)
        x_time[x_time < 0] = 0.0
        x_time = _align_rows(x_time, int(seq_len))
        return torch.as_tensor(np.stack([x_dir, x_time], axis=1), dtype=torch.float32)
    raise ValueError(f"Unsupported WFlib feature: {feature}")


def _predict_wflib(
    model: torch.nn.Module,
    traces: list[np.ndarray] | np.ndarray,
    *,
    feature: str,
    device: torch.device,
    batch_size: int,
    seq_len: int,
) -> np.ndarray:
    x = _wflib_features(traces, feature, int(seq_len))
    rows: list[np.ndarray] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, int(x.shape[0]), step):
            xb = x[start : start + step].to(device)
            logits, _feat = model(xb)
            rows.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 0), dtype=np.float32)


def _reference_margin(probs: np.ndarray, ref_class: int) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    y = int(ref_class)
    ref = values[:, y]
    if values.shape[1] <= 1:
        other = np.zeros(values.shape[0], dtype=np.float32)
    else:
        other_values = np.concatenate([values[:, :y], values[:, y + 1 :]], axis=1)
        other = np.max(other_values, axis=1)
    return (ref - other).astype(np.float32)


def _model_margins(probs: dict[str, np.ndarray], ref_classes: dict[str, int]) -> dict[str, float]:
    return {name: float(_reference_margin(prob, ref_classes[name])[0]) for name, prob in probs.items()}


def _normalize_gains(
    gains: dict[str, np.ndarray],
    *,
    current_probs: dict[str, np.ndarray],
    ref_classes: dict[str, int],
    epsilon: float,
) -> dict[str, np.ndarray]:
    normalized: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        current_margin = float(_reference_margin(current_probs[name], ref_classes[name])[0])
        denom = max(abs(current_margin), float(epsilon))
        normalized[name] = (np.asarray(gains[name], dtype=np.float32) / denom).astype(np.float32)
    return normalized


def _current_normalized_margins(
    current_probs: dict[str, np.ndarray],
    original_probs: dict[str, np.ndarray],
    ref_classes: dict[str, int],
    *,
    epsilon: float,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for name in MODEL_NAMES:
        current_margin = float(_reference_margin(current_probs[name], ref_classes[name])[0])
        original_margin = float(_reference_margin(original_probs[name], ref_classes[name])[0])
        out[name] = float(current_margin / max(abs(original_margin), float(epsilon)))
    return out


def _score_gain_basis(method: str, gains: dict[str, np.ndarray], normalized_gains: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    value = str(method)
    if value.startswith("norm_") or value == "dynamic_worst":
        return normalized_gains
    return gains


def _prob_preds(probs: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(np.argmax(prob)) for name, prob in probs.items()}


def _evaluate_candidate_batch(
    *,
    state: EvalState,
    actions: list[Any],
    current_probs: dict[str, np.ndarray],
    ref_classes: dict[str, int],
    rf_attacker: Any,
    df_model: torch.nn.Module,
    varcnn_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[np.ndarray]]:
    if not actions:
        empty = np.zeros(0, dtype=np.float32)
        return {name: empty for name in MODEL_NAMES}, {name: np.zeros((0, 0), dtype=np.float32) for name in MODEL_NAMES}, []
    trial_counts = [np.asarray(state.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32) for action in actions]
    traces, tams, _stats = _render_dummy_batch(base_trace=state.trace, counts_list=trial_counts, args=args)
    rf_probs = predict_probabilities(rf_attacker, tams, device=device, batch_size=int(args.batch_size))
    df_probs = _predict_wflib(df_model, traces, feature="DIR", device=device, batch_size=int(args.batch_size), seq_len=int(args.max_trace_length))
    varcnn_probs = _predict_wflib(varcnn_model, traces, feature="DIR", device=device, batch_size=int(args.batch_size), seq_len=3000)
    candidate_probs = {"rf": rf_probs, "df": df_probs, "varcnn": varcnn_probs}
    gains: dict[str, np.ndarray] = {}
    for name in MODEL_NAMES:
        current_margin = _reference_margin(current_probs[name], ref_classes[name])[0]
        candidate_margin = _reference_margin(candidate_probs[name], ref_classes[name])
        gains[name] = (float(current_margin) - candidate_margin).astype(np.float32)
    return gains, candidate_probs, traces


def _method_score(
    method: str,
    gains: dict[str, np.ndarray],
    actions: list[Any],
    clean_total: float,
    args: argparse.Namespace,
    *,
    target_model: str | None = None,
) -> np.ndarray:
    stacked = np.stack([np.asarray(gains[name], dtype=np.float32) for name in MODEL_NAMES], axis=1)
    costs = np.asarray([action_cost(action, clean_total) for action in actions], dtype=np.float32)
    value = str(method)
    if value == "rf_only":
        return stacked[:, 0].astype(np.float32)
    if value == "mean":
        return np.mean(stacked, axis=1).astype(np.float32)
    if value == "worst":
        return np.min(stacked, axis=1).astype(np.float32)
    if value.startswith("blend_a"):
        alpha = float(value.replace("blend_a", ""))
        return (alpha * np.mean(stacked, axis=1) + (1.0 - alpha) * np.min(stacked, axis=1) - float(args.cost_lambda) * costs).astype(np.float32)
    if value == "norm_mean":
        return np.mean(stacked, axis=1).astype(np.float32)
    if value == "norm_worst":
        return np.min(stacked, axis=1).astype(np.float32)
    if value.startswith("norm_blend_a"):
        alpha = float(value.replace("norm_blend_a", ""))
        return (alpha * np.mean(stacked, axis=1) + (1.0 - alpha) * np.min(stacked, axis=1) - float(args.cost_lambda) * costs).astype(np.float32)
    if value == "norm_minmean":
        return (np.min(stacked, axis=1) + 0.25 * np.mean(stacked, axis=1) - float(args.cost_lambda) * costs).astype(np.float32)

    if value.startswith("norm_weighted_"):
        m = re.match(r"norm_weighted_r(\d+)_d(\d+)_v(\d+)", value)
        if not m:
            raise ValueError(f"Invalid norm_weighted format: {value}, expected norm_weighted_rXX_dXX_vXX")
        w_rf = int(m.group(1)) / 100.0
        w_df = int(m.group(2)) / 100.0
        w_vc = int(m.group(3)) / 100.0
        weights = np.array([w_rf, w_df, w_vc], dtype=np.float32)
        return (np.sum(stacked * weights[None, :], axis=1) - float(args.cost_lambda) * costs).astype(np.float32)
    if value == "dynamic_worst":
        if target_model is None:
            raise ValueError("dynamic_worst requires target_model")
        target_idx = MODEL_NAMES.index(str(target_model))
        return (stacked[:, target_idx] - float(args.cost_lambda) * costs).astype(np.float32)
    raise ValueError(f"Unknown ensemble oracle method: {method!r}")


def _choose_action(
    method: str,
    scores: np.ndarray,
    gains: dict[str, np.ndarray],
    args: argparse.Namespace,
    *,
    target_model: str | None = None,
) -> int:
    if scores.size == 0:
        return -1
    stacked = np.stack([np.asarray(gains[name], dtype=np.float32) for name in MODEL_NAMES], axis=1)
    order = np.argsort(-np.asarray(scores, dtype=np.float32), kind="mergesort")
    valid: list[int] = []
    for idx in order.tolist():
        score = float(scores[idx])
        if str(method) == "rf_only":
            if score > 0.0:
                valid.append(int(idx))
            continue
        if str(method) == "dynamic_worst":
            if target_model is None:
                raise ValueError("dynamic_worst requires target_model")
            target_idx = MODEL_NAMES.index(str(target_model))
            if score <= 0.0 or float(stacked[idx, target_idx]) <= 0.0:
                continue
            other = [j for j in range(len(MODEL_NAMES)) if j != target_idx]
            if min(float(stacked[idx, j]) for j in other) < -float(args.robust_epsilon):
                continue
            valid.append(int(idx))
            continue
        positive = int(np.sum(stacked[idx] > 0.0))
        worst = float(np.min(stacked[idx]))
        if score <= 0.0:
            continue
        if positive < int(args.robust_min_positive_models):
            continue
        if worst < -float(args.robust_epsilon):
            continue
        valid.append(int(idx))
    if getattr(args, "random_epsilon", 0.0) > 0:
        if valid and np.random.random() < float(args.random_epsilon):
            return int(np.random.choice(valid))
    return int(valid[0]) if valid else -1


def _topk_overlap(values_a: np.ndarray, values_b: np.ndarray, k: int) -> float:
    n = int(min(len(values_a), len(values_b), max(1, int(k))))
    if n <= 0:
        return 0.0
    a = set(np.argsort(-np.asarray(values_a, dtype=np.float32), kind="mergesort")[:n].tolist())
    b = set(np.argsort(-np.asarray(values_b, dtype=np.float32), kind="mergesort")[:n].tolist())
    return float(len(a & b) / max(len(a | b), 1))


def _candidate_audit_row(
    *,
    sample_index: int,
    sample_id: str,
    method: str,
    protocol: str,
    budget: float,
    round_index: int,
    step_index: int,
    actions: list[Any],
    gains: dict[str, np.ndarray],
    scores: np.ndarray,
    selected_index: int,
    target_model: str,
    clean_total: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stacked = np.stack([np.asarray(gains[name], dtype=np.float32) for name in MODEL_NAMES], axis=1) if actions else np.zeros((0, 3), dtype=np.float32)
    rf_best = int(np.argmax(gains["rf"])) if len(actions) else -1
    selected = selected_index if selected_index >= 0 else rf_best
    positive_counts = np.sum(stacked > 0.0, axis=1) if len(actions) else np.zeros(0, dtype=np.int32)
    costs = np.asarray([action_cost(action, clean_total) for action in actions], dtype=np.float32) if actions else np.zeros(0, dtype=np.float32)
    rf_pos = stacked[:, 0] > 0.0 if len(actions) else np.zeros(0, dtype=bool)
    df_pos = stacked[:, 1] > 0.0 if len(actions) else np.zeros(0, dtype=bool)
    var_pos = stacked[:, 2] > 0.0 if len(actions) else np.zeros(0, dtype=bool)
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(method),
        "target_model": str(target_model),
        "dummy_budget_bound": float(budget),
        "round_index": int(round_index),
        "step_index": int(step_index),
        "candidate_count": int(len(actions)),
        "rf_df_topk_jaccard": _topk_overlap(gains["rf"], gains["df"], int(args.topk_overlap_k)),
        "rf_varcnn_topk_jaccard": _topk_overlap(gains["rf"], gains["varcnn"], int(args.topk_overlap_k)),
        "df_varcnn_topk_jaccard": _topk_overlap(gains["df"], gains["varcnn"], int(args.topk_overlap_k)),
        "all_positive_rate": float(np.mean(positive_counts == 3)) if len(actions) else 0.0,
        "at_least_two_positive_rate": float(np.mean(positive_counts >= 2)) if len(actions) else 0.0,
        "rf_df_positive_rate": float(np.mean(rf_pos & df_pos)) if len(actions) else 0.0,
        "rf_varcnn_positive_rate": float(np.mean(rf_pos & var_pos)) if len(actions) else 0.0,
        "df_varcnn_positive_rate": float(np.mean(df_pos & var_pos)) if len(actions) else 0.0,
        "only_rf_positive_rate": float(np.mean(rf_pos & ~df_pos & ~var_pos)) if len(actions) else 0.0,
        "only_df_positive_rate": float(np.mean(~rf_pos & df_pos & ~var_pos)) if len(actions) else 0.0,
        "only_varcnn_positive_rate": float(np.mean(~rf_pos & ~df_pos & var_pos)) if len(actions) else 0.0,
        "df_varcnn_only_positive_rate": float(np.mean(~rf_pos & df_pos & var_pos)) if len(actions) else 0.0,
        "mixed_sign_rate": float(np.mean((positive_counts > 0) & (positive_counts < 3))) if len(actions) else 0.0,
        "rf_best_index": int(rf_best),
        "rf_best_df_positive": int(bool(len(actions) and gains["df"][rf_best] > 0.0)),
        "rf_best_varcnn_positive": int(bool(len(actions) and gains["varcnn"][rf_best] > 0.0)),
        "rf_best_all_positive": int(bool(len(actions) and np.all(stacked[rf_best] > 0.0))),
        "selected_index": int(selected_index),
        "selected_score": float(scores[selected_index]) if selected_index >= 0 else 0.0,
        "selected_rf_gain": float(gains["rf"][selected]) if selected >= 0 and len(actions) else 0.0,
        "selected_df_gain": float(gains["df"][selected]) if selected >= 0 and len(actions) else 0.0,
        "selected_varcnn_gain": float(gains["varcnn"][selected]) if selected >= 0 and len(actions) else 0.0,
        "selected_positive_models": int(positive_counts[selected]) if selected >= 0 and len(actions) else 0,
        "selected_worst_gain": float(np.min(stacked[selected])) if selected >= 0 and len(actions) else 0.0,
        "selected_mean_gain": float(np.mean(stacked[selected])) if selected >= 0 and len(actions) else 0.0,
        "selected_cost": float(costs[selected]) if selected >= 0 and len(actions) else 0.0,
    }


def _target_reached(method: str, margins: dict[str, float], target: float) -> bool:
    if str(method) == "rf_only":
        return float(margins["rf"]) <= float(target)
    return all(float(margins[name]) <= float(target) for name in MODEL_NAMES)


def _run_ensemble_controller(
    *,
    method: str,
    protocol: str,
    budget: float,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_rf_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    rf_attacker: Any,
    df_model: torch.nn.Module,
    varcnn_model: torch.nn.Module,
    device: torch.device,
    args: argparse.Namespace,
) -> EnsembleResult:
    del original_mask
    clean_total = max(float(np.asarray(original_tam, dtype=np.float32).sum()), 1.0)
    state = _initial_state(raw_trace, original_tam, original_rf_prob)
    original_probs = {
        "rf": np.asarray(original_rf_prob, dtype=np.float32),
        "df": _predict_wflib(df_model, [raw_trace], feature="DIR", device=device, batch_size=1, seq_len=int(args.max_trace_length))[0],
        "varcnn": _predict_wflib(varcnn_model, [raw_trace], feature="DIR", device=device, batch_size=1, seq_len=3000)[0],
    }
    current_probs = {name: np.asarray(prob, dtype=np.float32) for name, prob in original_probs.items()}
    ref_classes = {name: int(np.argmax(prob)) for name, prob in original_probs.items()}
    current = state
    config = _method_config("stratified_top128")
    max_dummy = int(round(float(clean_total) * float(budget)))
    used: set[tuple] = {action_identity(action) for action in current.selected_actions}
    stable_name_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(str(method)))
    rng = random.Random(int(args.seed) + int(sample_index) * 1009 + stable_name_seed)
    audit_rows: list[dict[str, Any]] = []
    selected_positive_models: list[int] = []
    selected_worst_gains: list[float] = []
    selected_mean_gains: list[float] = []
    candidate_steps = 0
    candidate_evals = 0
    stop_reason = "max_actions_reached"

    for round_index in range(max(1, int(args.rounds))):
        if int(args.max_delay) > 0:
            mask0 = _fast_refresh_mask(rf_attacker, current.tam, original_probs["rf"], device=device)
            current = _apply_delay(
                state=current,
                mask=mask0,
                protocol=str(protocol),
                delay_budget=max(1, int(round(int(args.max_delay) / max(1, int(args.rounds))))),
                args=args,
            )
            current_probs["rf"] = _predict_one(rf_attacker, current.tam, device=device, args=args)
            current_probs["df"] = _predict_wflib(df_model, [current.trace], feature="DIR", device=device, batch_size=1, seq_len=int(args.max_trace_length))[0]
            current_probs["varcnn"] = _predict_wflib(varcnn_model, [current.trace], feature="DIR", device=device, batch_size=1, seq_len=3000)[0]
            current.prob = current_probs["rf"]
        if _target_reached(str(method), _model_margins(current_probs, ref_classes), float(args.margin_target)):
            stop_reason = "target_reached" if str(method) == "rf_only" else "all_models_target_reached"
            break

        for step_index in range(int(args.max_dummy_steps)):
            if _target_reached(str(method), _model_margins(current_probs, ref_classes), float(args.margin_target)):
                stop_reason = "target_reached" if str(method) == "rf_only" else "all_models_target_reached"
                break
            remaining_dummy = max_dummy - int(np.asarray(current.dummy_counts, dtype=np.int32).sum())
            if remaining_dummy <= 0:
                stop_reason = "bandwidth_10pct_reached"
                break
            mask = _fast_refresh_mask(rf_attacker, current.tam, original_probs["rf"], device=device)
            diagnostics: dict[str, Any] = {"profile_detail": False}
            if bool(getattr(args, "compact_candidate_generation", False)):
                raw_descriptors = generate_compact_action_descriptors(
                    tam=current.tam,
                    soft_mask=mask,
                    sample_index=int(sample_index),
                    sample_id=str(sample_id),
                    true_label=int(label),
                    protocol=str(protocol),
                    clean_total=float(clean_total),
                    ratio=float(args.ratio),
                    max_windows=int(args.max_windows),
                    max_action_budget=float(args.max_action_budget),
                    max_actions=int(config.raw_pool),
                    candidate_batch_size=int(getattr(args, "candidate_batch_size", 0)),
                    candidate_device=str(getattr(args, "candidate_score_device", getattr(args, "candidate_device", "auto"))),
                    diagnostics=diagnostics,
                )
                raw_descriptors = [
                    action
                    for action in raw_descriptors
                    if _identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy
                ]
                materialize_k = min(int(config.eval_k), len(raw_descriptors))
                descriptor_config = MethodConfig(
                    name=str(config.name),
                    prefilter=str(config.prefilter),
                    raw_pool=int(config.raw_pool),
                    eval_k=int(materialize_k),
                    objective=str(config.objective),
                )
                descriptor_selected = _prefilter_actions(
                    raw_descriptors,
                    config=descriptor_config,
                    clean_total=clean_total,
                    args=args,
                    rng=rng,
                )
                actions = []
                materialize_batch = max(1, int(getattr(args, "materialization_batch_size", 64)))
                for start_idx in range(0, len(descriptor_selected), materialize_batch):
                    actions.extend(
                        materialize_candidate_descriptors(
                            descriptor_selected[start_idx : start_idx + materialize_batch],
                            tam=current.tam,
                            clean_total=float(clean_total),
                            protocol=str(protocol),
                            max_action_budget=float(args.max_action_budget),
                            max_local_rate_peak=int(args.max_local_rate_peak),
                        )
                    )
            else:
                raw_actions = generate_expanded_actions(
                    tam=current.tam,
                    soft_mask=mask,
                    sample_index=int(sample_index),
                    sample_id=str(sample_id),
                    true_label=int(label),
                    protocol=str(protocol),
                    clean_total=float(clean_total),
                    ratio=float(args.ratio),
                    max_windows=int(args.max_windows),
                    max_action_budget=float(args.max_action_budget),
                    max_local_rate_peak=int(args.max_local_rate_peak),
                    max_actions=int(config.raw_pool),
                    max_pair_actions=0,
                    diagnostics=diagnostics,
                )
                raw_actions = [action for action in raw_actions if _identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]
                actions = _prefilter_actions(raw_actions, config=config, clean_total=clean_total, args=args, rng=rng)
            actions = [action for action in actions if _identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]
            if not actions:
                stop_reason = "candidate_pool_exhausted"
                break
            gains, _candidate_probs, _traces = _evaluate_candidate_batch(
                state=current,
                actions=actions,
                current_probs=current_probs,
                ref_classes=ref_classes,
                rf_attacker=rf_attacker,
                df_model=df_model,
                varcnn_model=varcnn_model,
                device=device,
                args=args,
            )
            candidate_steps += 1
            candidate_evals += int(len(actions))
            normalized_gains = _normalize_gains(
                gains,
                current_probs=current_probs,
                ref_classes=ref_classes,
                epsilon=float(args.norm_epsilon),
            )
            current_norm_margins = _current_normalized_margins(
                current_probs,
                original_probs,
                ref_classes,
                epsilon=float(args.norm_epsilon),
            )
            target_model = ""
            if str(method) == "dynamic_worst":
                target_model = max(MODEL_NAMES, key=lambda name: float(current_norm_margins[name]))
            score_gains = _score_gain_basis(str(method), gains, normalized_gains)
            scores = _method_score(str(method), score_gains, actions, clean_total, args, target_model=target_model or None)
            selected_idx = _choose_action(str(method), scores, score_gains, args, target_model=target_model or None)
            audit_rows.append(
                _candidate_audit_row(
                    sample_index=sample_index,
                    sample_id=sample_id,
                    method=method,
                    protocol=protocol,
                    budget=budget,
                    round_index=round_index,
                    step_index=step_index,
                    actions=actions,
                    gains=score_gains,
                    scores=scores,
                    selected_index=selected_idx,
                    target_model=target_model,
                    clean_total=clean_total,
                    args=args,
                )
            )
            if selected_idx < 0:
                stop_reason = "no_positive_single" if str(method) == "rf_only" else "no_robust_positive"
                break
            action = actions[selected_idx]
            counts = np.asarray(current.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32)
            trace, tam, stats = _render_dummy(base_trace=current.trace, counts=counts, args=args)
            rf_prob = _predict_one(rf_attacker, tam, device=device, args=args)
            df_prob = _predict_wflib(df_model, [trace], feature="DIR", device=device, batch_size=1, seq_len=int(args.max_trace_length))[0]
            varcnn_prob = _predict_wflib(varcnn_model, [trace], feature="DIR", device=device, batch_size=1, seq_len=3000)[0]
            stacked_selected = np.asarray([score_gains[name][selected_idx] for name in MODEL_NAMES], dtype=np.float32)
            selected_positive_models.append(int(np.sum(stacked_selected > 0.0)))
            selected_worst_gains.append(float(np.min(stacked_selected)))
            selected_mean_gains.append(float(np.mean(stacked_selected)))
            used.add(_identity(action))
            current = EvalState(
                trace=trace,
                tam=tam,
                prob=rf_prob,
                dummy_counts=counts,
                dummy_bandwidth=float(stats["raw_bandwidth"]),
                avg_delay=current.avg_delay,
                p95_delay=current.p95_delay,
                max_delay=current.max_delay,
                delay_values=tuple(current.delay_values),
                outgoing_delay_values=tuple(current.outgoing_delay_values),
                incoming_delay_values=tuple(current.incoming_delay_values),
                selected_actions=list(current.selected_actions) + [action],
            )
            current_probs = {"rf": rf_prob, "df": df_prob, "varcnn": varcnn_prob}
        if stop_reason in {"target_reached", "all_models_target_reached", "bandwidth_10pct_reached", "candidate_pool_exhausted", "no_positive_single", "no_robust_positive"}:
            break
    final_margins = _model_margins(current_probs, ref_classes)
    if _target_reached(str(method), final_margins, float(args.margin_target)):
        stop_reason = "target_reached" if str(method) == "rf_only" else "all_models_target_reached"
    return EnsembleResult(
        state=current,
        final_probs={name: np.asarray(prob, dtype=np.float32) for name, prob in current_probs.items()},
        original_probs=original_probs,
        stop_reason=stop_reason,
        accepted_action_count=int(len(current.selected_actions)),
        candidate_step_count=int(candidate_steps),
        candidate_eval_count=int(candidate_evals),
        selected_positive_models=selected_positive_models,
        selected_worst_gains=selected_worst_gains,
        selected_mean_gains=selected_mean_gains,
        audit_rows=audit_rows,
    )


def _sample_row(
    *,
    sample_index: int,
    sample_id: str,
    protocol: str,
    method: str,
    budget: float,
    original_tam: np.ndarray,
    label: int,
    runtime: float,
    result: EnsembleResult,
) -> dict[str, Any]:
    clean_total = max(float(np.asarray(original_tam, dtype=np.float32).sum()), 1.0)
    original_preds = _prob_preds(result.original_probs)
    final_preds = _prob_preds(result.final_probs)
    out: dict[str, Any] = {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(method),
        "dummy_budget_bound": float(budget),
        "true_label": int(label),
        "stop_reason": str(result.stop_reason),
        "runtime_sec": float(runtime),
        "candidate_step_count": int(result.candidate_step_count),
        "candidate_eval_count": int(result.candidate_eval_count),
        "accepted_action_count": int(result.accepted_action_count),
        "mean_selected_positive_models": float(np.mean(result.selected_positive_models)) if result.selected_positive_models else 0.0,
        "mean_selected_worst_gain": float(np.mean(result.selected_worst_gains)) if result.selected_worst_gains else 0.0,
        "mean_selected_mean_gain": float(np.mean(result.selected_mean_gains)) if result.selected_mean_gains else 0.0,
        **_resource_fields(result.state, clean_total),
    }
    flips = []
    accuracies = []
    for name in MODEL_NAMES:
        original_prob = result.original_probs[name]
        final_prob = result.final_probs[name]
        metrics = probability_metrics(original_prob.reshape(1, -1), final_prob.reshape(1, -1), np.asarray([int(label)], dtype=np.int64))
        acc = float(metrics["accuracy"][0])
        flip = int(final_preds[name] != original_preds[name])
        accuracies.append(acc)
        flips.append(flip)
        out[f"{name}_original_pred"] = int(original_preds[name])
        out[f"{name}_final_pred"] = int(final_preds[name])
        out[f"{name}_accuracy"] = acc
        out[f"{name}_flip"] = float(flip)
        out[f"{name}_original_class_margin"] = float(metrics["original_class_margin"][0])
        out[f"{name}_original_class_margin_drop"] = float(metrics["original_class_margin_drop"][0])
    out["mean_attack_accuracy"] = float(np.mean(accuracies))
    out["any_model_correct"] = float(int(np.max(accuracies) > 0.0))
    out["all_models_flipped"] = float(int(sum(flips) == len(MODEL_NAMES)))
    out["at_least_two_models_flipped"] = float(int(sum(flips) >= 2))
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _summarize_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["protocol"]), str(row["method"]), float(row["dummy_budget_bound"]))].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, method, budget), items in sorted(groups.items()):
        rf_accuracy = float(np.mean([_float(row, "rf_accuracy") for row in items]))
        df_accuracy = float(np.mean([_float(row, "df_accuracy") for row in items]))
        varcnn_accuracy = float(np.mean([_float(row, "varcnn_accuracy") for row in items]))
        out.append(
            {
                "protocol": protocol,
                "method": method,
                "dummy_budget_bound": float(budget),
                "samples": int(len(items)),
                "rf_accuracy": rf_accuracy,
                "df_accuracy": df_accuracy,
                "varcnn_accuracy": varcnn_accuracy,
                "mean_attack_accuracy": float(np.mean([_float(row, "mean_attack_accuracy") for row in items])),
                "worst_case_attack_accuracy": float(max(rf_accuracy, df_accuracy, varcnn_accuracy)),
                "any_model_correct_rate": float(np.mean([_float(row, "any_model_correct") for row in items])) if any("any_model_correct" in row for row in items) else float(np.mean([max(_float(row, "rf_accuracy"), _float(row, "df_accuracy"), _float(row, "varcnn_accuracy")) for row in items])),
                "rf_flip": float(np.mean([_float(row, "rf_flip") for row in items])),
                "df_flip": float(np.mean([_float(row, "df_flip") for row in items])),
                "varcnn_flip": float(np.mean([_float(row, "varcnn_flip") for row in items])),
                "all_models_flipped": float(np.mean([_float(row, "all_models_flipped") for row in items])),
                "at_least_two_models_flipped": float(np.mean([_float(row, "at_least_two_models_flipped") for row in items])),
                "mean_actual_bandwidth": float(np.mean([_float(row, "actual_dummy_bandwidth") for row in items])),
                "mean_delay_bins": float(np.mean([_float(row, "average_delay_bins") for row in items])),
                "mean_accepted_actions": float(np.mean([_float(row, "accepted_action_count") for row in items])),
                "mean_candidate_steps": float(np.mean([_float(row, "candidate_step_count") for row in items])),
                "mean_candidate_eval_count": float(np.mean([_float(row, "candidate_eval_count") for row in items])),
                "mean_selected_positive_models": float(np.mean([_float(row, "mean_selected_positive_models") for row in items])),
                "mean_selected_worst_gain": float(np.mean([_float(row, "mean_selected_worst_gain") for row in items])),
                "mean_runtime_sec": float(np.mean([_float(row, "runtime_sec") for row in items])),
                "stop_reasons": json.dumps(dict(Counter(str(row.get("stop_reason", "")) for row in items)), sort_keys=True),
            }
        )
    return out


def _summarize_audit(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["protocol"]), str(row["method"]), float(row["dummy_budget_bound"]))].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, method, budget), items in sorted(groups.items()):
        out.append(
            {
                "protocol": protocol,
                "method": method,
                "dummy_budget_bound": float(budget),
                "candidate_steps": int(len(items)),
                "mean_rf_df_topk_jaccard": float(np.mean([_float(row, "rf_df_topk_jaccard") for row in items])) if items else 0.0,
                "mean_rf_varcnn_topk_jaccard": float(np.mean([_float(row, "rf_varcnn_topk_jaccard") for row in items])) if items else 0.0,
                "mean_df_varcnn_topk_jaccard": float(np.mean([_float(row, "df_varcnn_topk_jaccard") for row in items])) if items else 0.0,
                "mean_all_positive_rate": float(np.mean([_float(row, "all_positive_rate") for row in items])) if items else 0.0,
                "mean_at_least_two_positive_rate": float(np.mean([_float(row, "at_least_two_positive_rate") for row in items])) if items else 0.0,
                "mean_rf_df_positive_rate": float(np.mean([_float(row, "rf_df_positive_rate") for row in items])) if items else 0.0,
                "mean_rf_varcnn_positive_rate": float(np.mean([_float(row, "rf_varcnn_positive_rate") for row in items])) if items else 0.0,
                "mean_df_varcnn_positive_rate": float(np.mean([_float(row, "df_varcnn_positive_rate") for row in items])) if items else 0.0,
                "mean_only_rf_positive_rate": float(np.mean([_float(row, "only_rf_positive_rate") for row in items])) if items else 0.0,
                "mean_only_df_positive_rate": float(np.mean([_float(row, "only_df_positive_rate") for row in items])) if items else 0.0,
                "mean_only_varcnn_positive_rate": float(np.mean([_float(row, "only_varcnn_positive_rate") for row in items])) if items else 0.0,
                "mean_df_varcnn_only_positive_rate": float(np.mean([_float(row, "df_varcnn_only_positive_rate") for row in items])) if items else 0.0,
                "mean_mixed_sign_rate": float(np.mean([_float(row, "mixed_sign_rate") for row in items])) if items else 0.0,
                "rf_best_df_positive_rate": float(np.mean([_float(row, "rf_best_df_positive") for row in items])) if items else 0.0,
                "rf_best_varcnn_positive_rate": float(np.mean([_float(row, "rf_best_varcnn_positive") for row in items])) if items else 0.0,
                "rf_best_all_positive_rate": float(np.mean([_float(row, "rf_best_all_positive") for row in items])) if items else 0.0,
                "mean_selected_positive_models": float(np.mean([_float(row, "selected_positive_models") for row in items])) if items else 0.0,
                "mean_selected_worst_gain": float(np.mean([_float(row, "selected_worst_gain") for row in items])) if items else 0.0,
                "mean_selected_mean_gain": float(np.mean([_float(row, "selected_mean_gain") for row in items])) if items else 0.0,
            }
        )
    return out


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    checkpoint = args.rf_checkpoint or _default_checkpoint("rf")
    protocols = _parse_csv_strings(args.protocols)
    methods = _parse_csv_strings(args.methods)
    budgets = _parse_csv_floats(args.dummy_budgets)
    archive = _load_archive(str(args.archive), int(args.max_samples))
    total = int(archive["tam"].shape[0])
    start_index = max(0, int(args.sample_start))
    end_index = int(args.sample_end) if int(args.sample_end) > 0 else total
    end_index = min(max(start_index, end_index), total)
    for key, value in list(archive.items()):
        arr = np.asarray(value)
        if arr.shape[:1] == (total,):
            archive[key] = arr[start_index:end_index]
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(start_index, end_index))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(start_index, end_index)), dtype=np.int64)
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, _runtime_args(args))

    rf_attacker = load_stage_a_attacker(
        checkpoint,
        attacker="rf",
        num_classes=int(prob.shape[1]),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    df_model = _load_wflib_model("DF", str(args.df_checkpoint), int(prob.shape[1]), device)
    varcnn_model = _load_wflib_model("AWF", str(args.varcnn_checkpoint), int(prob.shape[1]), device)

    sample_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    defended_traces: list[np.ndarray] = []
    defended_labels: list[int] = []
    defended_source_indices: list[int] = []
    defended_sample_ids: list[str] = []
    defended_methods: list[str] = []
    defended_protocols: list[str] = []
    defended_budgets: list[float] = []
    total_jobs = len(methods) * len(protocols) * len(budgets) * int(tam.shape[0])
    done = 0
    for method in methods:
        for protocol in protocols:
            for budget in budgets:
                for local_index in range(int(tam.shape[0])):
                    global_index = int(start_index + local_index)
                    done += 1
                    if bool(args.progress):
                        print(
                            f"[ensemble-oracle] {done}/{total_jobs} method={method} "
                            f"B={budget:g} sample={global_index}",
                            flush=True,
                        )
                    run_start = time.perf_counter()
                    result = _run_ensemble_controller(
                        method=str(method),
                        protocol=str(protocol),
                        budget=float(budget),
                        raw_trace=np.asarray(raw_rows[local_index], dtype=np.float32),
                        original_tam=np.asarray(tam[local_index], dtype=np.float32),
                        original_mask=np.asarray(mask[local_index], dtype=np.float32),
                        original_rf_prob=np.asarray(prob[local_index], dtype=np.float32),
                        label=int(labels[local_index]),
                        sample_index=global_index,
                        sample_id=str(sample_ids[local_index]),
                        rf_attacker=rf_attacker,
                        df_model=df_model,
                        varcnn_model=varcnn_model,
                        device=device,
                        args=args,
                    )
                    runtime = time.perf_counter() - run_start
                    if str(args.export_defended_npz):
                        defended_traces.append(np.asarray(result.state.trace, dtype=np.float32).reshape(-1))
                        defended_labels.append(int(labels[local_index]))
                        defended_source_indices.append(int(source_indices[local_index]))
                        defended_sample_ids.append(str(sample_ids[local_index]))
                        defended_methods.append(str(method))
                        defended_protocols.append(str(protocol))
                        defended_budgets.append(float(budget))
                    sample_rows.append(
                        _sample_row(
                            sample_index=global_index,
                            sample_id=str(sample_ids[local_index]),
                            protocol=str(protocol),
                            method=str(method),
                            budget=float(budget),
                            original_tam=np.asarray(tam[local_index], dtype=np.float32),
                            label=int(labels[local_index]),
                            runtime=runtime,
                            result=result,
                        )
                    )
                    audit_rows.extend(result.audit_rows)

    sample_summary = _summarize_samples(sample_rows)
    audit_summary = _summarize_audit(audit_rows)
    _write_csv(output_dir / "ensemble_oracle_sample_results.csv", sample_rows)
    _write_csv(output_dir / "ensemble_oracle_summary.csv", sample_summary)
    _write_csv(output_dir / "ensemble_oracle_candidate_audit.csv", audit_rows)
    _write_csv(output_dir / "ensemble_oracle_candidate_audit_summary.csv", audit_summary)
    export_defended_npz = ""
    if str(args.export_defended_npz):
        export_path = Path(str(args.export_defended_npz))
        if not export_path.is_absolute():
            export_path = ROOT / export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        x_def = np.sign(_align_rows(defended_traces, int(args.max_trace_length)))[:, None, :].astype(np.float32)
        y_def = np.asarray(defended_labels, dtype=np.int64)
        np.savez_compressed(
            export_path,
            X=x_def,
            y=y_def,
            source_indices=np.asarray(defended_source_indices, dtype=np.int64),
            sample_ids=np.asarray(defended_sample_ids),
            methods=np.asarray(defended_methods),
            protocols=np.asarray(defended_protocols),
            dummy_budgets=np.asarray(defended_budgets, dtype=np.float32),
        )
        export_defended_npz = str(export_path)
    manifest = {
        "archive": str(args.archive),
        "rf_checkpoint": str(Path(checkpoint).resolve()),
        "df_checkpoint": _resolve_path(str(args.df_checkpoint)),
        "varcnn_checkpoint": _resolve_path(str(args.varcnn_checkpoint)),
        "samples": int(tam.shape[0]),
        "sample_start": int(start_index),
        "sample_end": int(end_index),
        "protocols": protocols,
        "methods": methods,
        "dummy_budgets": budgets,
        "seed": int(args.seed),
        "cost_lambda": float(args.cost_lambda),
        "robust_min_positive_models": int(args.robust_min_positive_models),
        "robust_epsilon": float(args.robust_epsilon),
        "random_epsilon": float(args.random_epsilon),
        "sample_results": str(output_dir / "ensemble_oracle_sample_results.csv"),
        "summary": str(output_dir / "ensemble_oracle_summary.csv"),
        "candidate_audit": str(output_dir / "ensemble_oracle_candidate_audit.csv"),
        "candidate_audit_summary": str(output_dir / "ensemble_oracle_candidate_audit_summary.csv"),
        "export_defended_npz": export_defended_npz,
    }
    (output_dir / "ensemble_oracle_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
