# -*- coding: utf-8 -*-
"""Stage B2-E: diverse candidate exposure and relaxed two-step search."""

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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynapd.stage_a.faithfulness import predict_probabilities
from dynapd.stage_a.modeling import load_stage_a_attacker
from dynapd.stage_b.expanded_generator import (
    CandidateDescriptor,
    ExpandedAction,
    action_cost,
    action_identity,
    descriptor_identity,
    generate_compact_action_descriptors,
    generate_expanded_actions,
    materialize_candidate_descriptors,
)
from dynapd.stage_b.objectives import ObjectiveWeights, original_class_margin, original_class_objective_delta, probability_metrics
from dynapd.utils import resolve_device, set_seed
from dynapd.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_dual_actuator import (
    EvalState,
    _apply_delay,
    _default_checkpoint,
    _fast_refresh_mask,
    _initial_state,
    _load_archive,
    _load_raw_rows,
    _p95,
    _predict_one,
    _render_dummy,
    _render_dummy_batch,
    _timing_add,
)


DEFAULT_ARCHIVE = (
    "results/stage_a_rf_native_w1800_n96_s60_seed0/"
    "stage_a_masks_rf/all_masks.npz"
)


@dataclass(frozen=True)
class MethodConfig:
    name: str
    prefilter: str
    raw_pool: int
    eval_k: int
    objective: str
    pair_enabled: bool = False
    pair_k: int = 24
    epsilon: float = 0.0
    tau: float = 0.0
    generator_pair_actions: int = 0


@dataclass
class SelectionResult:
    state: EvalState
    stop_reason: str
    accepted_single_count: int
    accepted_pair_count: int
    valley_pair_rescue_count: int
    rf_eval_count: int
    candidate_step_count: int
    best_single_gain_seen: float
    best_pair_gain_seen: float
    proxy_recall_values: list[float]
    true_recall_values: list[float]
    funnel_rows: list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocols", default="bidirectional_cooperative")
    parser.add_argument("--methods", default="current_v1,score_hint_top128,stratified_top64,stratified_top128,stratified_pair64_e0.01_t0,stratified_pair128_e0.01_t0")
    parser.add_argument("--dummy_budgets", default="0,0.01,0.02,0.05,0.08,0.10")
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--true_recall_pool_size", type=int, default=0)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
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
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b2e_diverse_search_{args.attacker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _runtime_args(args: argparse.Namespace) -> argparse.Namespace:
    return SimpleNamespace(
        data_root=str(args.data_root),
        attacker=str(args.attacker),
        checkpoint=str(args.checkpoint),
        device=str(args.device),
        seed=int(args.seed),
        max_trace_length=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
        delay_length=int(args.delay_length),
        delay_rho=float(args.delay_rho),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        max_action_budget=float(args.max_action_budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
        confidence_weight=float(args.confidence_weight),
        margin_weight=float(args.margin_weight),
        entropy_weight=float(args.entropy_weight),
        renderer_batch_size=int(args.renderer_batch_size),
        batch_size=int(args.batch_size),
        renderer_coordinate=str(args.renderer_coordinate),
        renderer_strategy=str(args.renderer_strategy),
        profile_candidate_generation=bool(getattr(args, "profile_candidate_generation", False)),
        compact_candidate_generation=bool(getattr(args, "compact_candidate_generation", False)),
        deferred_materialize_oversample=int(getattr(args, "deferred_materialize_oversample", 1)),
        candidate_batch_size=int(getattr(args, "candidate_batch_size", 0)),
        materialization_batch_size=int(getattr(args, "materialization_batch_size", 64)),
        candidate_device=str(getattr(args, "candidate_device", "auto")),
        candidate_score_device=str(getattr(args, "candidate_score_device", "auto")),
        candidate_eval_device=str(getattr(args, "candidate_eval_device", "auto")),
        candidate_eval_mode=str(getattr(args, "candidate_eval_mode", "renderer")),
        progress=bool(args.progress),
    )


def _method_config(name: str) -> MethodConfig:
    value = str(name).strip()
    if value == "current_v1":
        return MethodConfig(value, "score_hint", raw_pool=64, eval_k=8, objective="efficiency")
    if value == "score_hint_top32":
        return MethodConfig(value, "score_hint", raw_pool=32, eval_k=32, objective="absolute")
    if value == "score_hint_top128":
        return MethodConfig(value, "score_hint", raw_pool=128, eval_k=128, objective="absolute")
    if value == "stratified_top64":
        return MethodConfig(value, "stratified", raw_pool=1000, eval_k=64, objective="absolute")
    if value == "stratified_top128":
        return MethodConfig(value, "stratified", raw_pool=1000, eval_k=128, objective="absolute")
    if value.startswith("stratified_top") and value.replace("stratified_top", "").isdigit():
        size = max(1, int(value.replace("stratified_top", "")))
        return MethodConfig(value, "stratified", raw_pool=max(1000, size * 4), eval_k=size, objective="absolute")
    if value.startswith("stratified_pair"):
        # Examples: stratified_pair64_e0.01_t0, stratified_pair128_e0.02_t0.005.
        stem = value.replace("stratified_pair", "")
        size = 64
        epsilon = 0.01
        tau = 0.0
        parts = [part for part in stem.split("_") if part]
        if parts and parts[0].isdigit():
            size = int(parts[0])
        for part in parts[1:]:
            if part.startswith("e"):
                epsilon = float(part[1:]) / 1000.0 if part[1:].isdigit() else float(part[1:])
            if part.startswith("t"):
                tau = float(part[1:]) / 1000.0 if part[1:].isdigit() else float(part[1:])
        return MethodConfig(value, "stratified", raw_pool=1000, eval_k=size, objective="absolute", pair_enabled=True, pair_k=min(32, max(16, size // 4)), epsilon=epsilon, tau=tau, generator_pair_actions=24)
    raise ValueError(f"Unknown B2-E method: {name!r}")


def _margin(prob: np.ndarray, original_prob: np.ndarray) -> float:
    y0 = int(np.argmax(original_prob))
    return float(original_class_margin(np.asarray(prob, dtype=np.float32).reshape(1, -1), np.asarray([y0], dtype=np.int64))[0])


def _action_dummy_count(action: ExpandedAction) -> int:
    dummy = int(getattr(action, "dummy_count", 0))
    return dummy if dummy > 0 else int(np.asarray(action.counts, dtype=np.int32).sum())


def _action_nonzero_bins(action: ExpandedAction) -> int:
    nonzero = int(getattr(action, "nonzero_bin_count", 0))
    return nonzero if nonzero > 0 else int(np.count_nonzero(np.asarray(action.counts, dtype=np.int32)))


def _action_direction_bucket(action: ExpandedAction) -> str:
    out_count = int(getattr(action, "outgoing_dummy_count", 0))
    inc_count = int(getattr(action, "incoming_dummy_count", 0))
    if out_count <= 0 and inc_count <= 0:
        counts = np.asarray(action.counts, dtype=np.int32)
        out_count = int(counts[0].sum())
        inc_count = int(counts[1].sum())
    out = out_count > 0
    inc = inc_count > 0
    if out and inc:
        return "bidirectional"
    if inc:
        return "incoming"
    return "outgoing"


def _estimated_abs_gain(action: ExpandedAction, clean_total: float) -> float:
    # In V1 score_hint is roughly mask/type/dose divided by sqrt(cost).  Multiplying
    # back by sqrt(cost) keeps higher-dose actions visible during prefiltering.
    return float(action.score_hint) * float(np.sqrt(max(action_cost(action, clean_total), 1e-8)))


def _identity(action: ExpandedAction | CandidateDescriptor) -> tuple:
    cached = getattr(action, "_descriptor_identity", None)
    if cached:
        return tuple(cached)
    if isinstance(action, CandidateDescriptor):
        return descriptor_identity(action)
    return action_identity(action)


def _observe_state(observer: Any | None, event: dict[str, Any], state: EvalState) -> None:
    if observer is not None:
        observer(dict(event), state)


def _unique_extend(target: list[ExpandedAction], seen: set[tuple], items: list[ExpandedAction], limit: int) -> None:
    for action in items:
        key = _identity(action)
        if key in seen:
            continue
        target.append(action)
        seen.add(key)
        if len(target) >= int(limit):
            return


def _pareto_front(actions: list[ExpandedAction], clean_total: float, limit: int) -> list[ExpandedAction]:
    ranked = sorted(actions, key=lambda action: (action_cost(action, clean_total), -_estimated_abs_gain(action, clean_total)))
    front: list[ExpandedAction] = []
    best_gain = -float("inf")
    for action in ranked:
        gain = _estimated_abs_gain(action, clean_total)
        if gain > best_gain + 1e-12:
            front.append(action)
            best_gain = gain
        if len(front) >= int(limit):
            break
    return front


def _trim_diverse(actions: list[ExpandedAction], clean_total: float, limit: int) -> list[ExpandedAction]:
    if len(actions) <= int(limit):
        return actions
    selected: list[ExpandedAction] = []
    seen: set[tuple] = set()

    def add_bucket(items: list[ExpandedAction], n: int) -> None:
        _unique_extend(selected, seen, items, min(int(limit), len(selected) + int(n)))

    for dose in (32, 16, 8, 4, 2, 1):
        bucket = [a for a in actions if _action_dummy_count(a) == dose]
        bucket = sorted(bucket, key=lambda a: -_estimated_abs_gain(a, clean_total))
        add_bucket(bucket, 4)
    for action_type in sorted({a.action_type for a in actions}):
        bucket = sorted([a for a in actions if a.action_type == action_type], key=lambda a: -_estimated_abs_gain(a, clean_total))
        add_bucket(bucket, 4)
    multi = sorted([a for a in actions if _action_nonzero_bins(a) > 1], key=lambda a: -_estimated_abs_gain(a, clean_total))
    add_bucket(multi, 12)
    two_window = sorted([a for a in actions if a.action_type == "two_window_coordinated_insert"], key=lambda a: -_estimated_abs_gain(a, clean_total))
    add_bucket(two_window, 12)
    def composite(action: ExpandedAction) -> float:
        cost = action_cost(action, clean_total)
        dose_bonus = 0.05 * np.log1p(_action_dummy_count(action))
        bin_bonus = 0.06 if _action_nonzero_bins(action) > 1 else 0.0
        pair_bonus = 0.08 if action.action_type == "two_window_coordinated_insert" else 0.0
        return float(_estimated_abs_gain(action, clean_total) + 0.20 * float(action.score_hint) + dose_bonus + bin_bonus + pair_bonus - 0.05 * cost)
    rest = sorted(actions, key=composite, reverse=True)
    _unique_extend(selected, seen, rest, int(limit))
    return selected[: int(limit)]


def _prefilter_actions(
    actions: list[ExpandedAction],
    *,
    config: MethodConfig,
    clean_total: float,
    args: argparse.Namespace,
    rng: random.Random,
) -> list[ExpandedAction]:
    if not actions:
        return []
    ranked_score = sorted(actions, key=lambda a: (-float(a.score_hint), action_cost(a, clean_total), str(a.action_type)))
    if config.prefilter == "score_hint":
        return ranked_score[: int(config.eval_k)]
    selected: list[ExpandedAction] = []
    seen: set[tuple] = set()
    bucket_k = int(args.stratified_bucket_k)
    global_k = int(args.stratified_global_k)
    for dose in (1, 2, 4, 8, 16, 32):
        bucket = [a for a in actions if _action_dummy_count(a) == dose]
        _unique_extend(selected, seen, sorted(bucket, key=lambda a: -float(a.score_hint))[:bucket_k], 10_000)
        _unique_extend(selected, seen, sorted(bucket, key=lambda a: -_estimated_abs_gain(a, clean_total))[:bucket_k], 10_000)
    for action_type in sorted({str(a.action_type) for a in actions}):
        bucket = [a for a in actions if str(a.action_type) == action_type]
        _unique_extend(selected, seen, sorted(bucket, key=lambda a: -_estimated_abs_gain(a, clean_total))[:bucket_k], 10_000)
    for direction in ("outgoing", "incoming", "bidirectional"):
        bucket = [a for a in actions if _action_direction_bucket(a) == direction]
        _unique_extend(selected, seen, sorted(bucket, key=lambda a: -_estimated_abs_gain(a, clean_total))[:bucket_k], 10_000)
    _unique_extend(selected, seen, ranked_score[:global_k], 10_000)
    _unique_extend(selected, seen, sorted(actions, key=lambda a: -_estimated_abs_gain(a, clean_total))[:global_k], 10_000)
    _unique_extend(selected, seen, sorted(actions, key=lambda a: -(float(a.score_hint) / max(action_cost(a, clean_total), 1e-8)))[:global_k], 10_000)
    multi = [a for a in actions if _action_nonzero_bins(a) > 1]
    _unique_extend(selected, seen, sorted(multi, key=lambda a: -_estimated_abs_gain(a, clean_total))[: max(bucket_k * 2, 8)], 10_000)
    structured = [a for a in actions if str(a.action_type) != "dynamask_causal"]
    _unique_extend(selected, seen, sorted(structured, key=lambda a: -_estimated_abs_gain(a, clean_total))[: max(bucket_k * 2, 8)], 10_000)
    _unique_extend(selected, seen, _pareto_front(actions, clean_total, max(global_k, bucket_k * 2)), 10_000)
    shuffled = list(actions)
    rng.shuffle(shuffled)
    _unique_extend(selected, seen, shuffled[: max(0, int(args.random_explore_k))], 10_000)
    return _trim_diverse(selected, clean_total, int(config.eval_k))


def _candidate_torch_device(device: torch.device, args: argparse.Namespace) -> torch.device:
    requested = str(getattr(args, "candidate_eval_device", "auto")).lower()
    if requested == "auto":
        requested = str(getattr(args, "candidate_device", "auto")).lower()
    if requested == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cpu":
        return torch.device("cpu")
    return device if getattr(device, "type", "") == "cuda" else torch.device("cpu")


def _predict_probabilities_tensor(
    attacker,
    values: torch.Tensor,
    *,
    batch_size: int,
) -> torch.Tensor:
    rows: list[torch.Tensor] = []
    step = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, int(values.shape[0]), step):
            xb = values[start : start + step]
            rows.append(torch.softmax(attacker.logits(xb), dim=1))
    return torch.cat(rows, dim=0) if rows else torch.zeros((0, 0), dtype=torch.float32, device=values.device)


def _counts_chunk_tensor(
    counts_list: list[np.ndarray],
    *,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    # Materialize only the current candidate chunk; this keeps CPU RAM bounded
    # while letting the RF candidate tensor live on the chosen candidate device.
    counts_np = np.stack([np.asarray(counts, dtype=np.float32) for counts in counts_list], axis=0)
    if counts_np.shape[-1] != int(width):
        raise ValueError(f"Expected counts width {width}, got {counts_np.shape}")
    return torch.as_tensor(counts_np, dtype=torch.float32, device=device)


def _evaluate_incremental_counts_gpu_tam(
    *,
    state: EvalState,
    incremental_counts: list[np.ndarray],
    original_prob: np.ndarray,
    label: int,
    attacker,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not incremental_counts:
        return np.zeros(0, dtype=np.float32), np.zeros((0, original_prob.size), dtype=np.float32), {}
    candidate_device = _candidate_torch_device(device, args)
    chunk_size = int(getattr(args, "candidate_batch_size", 0) or getattr(args, "batch_size", 128))
    chunk_size = max(1, int(chunk_size))
    width = int(state.tam.shape[-1])
    probs_rows: list[np.ndarray] = []
    start_total = time.perf_counter()
    if candidate_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(candidate_device)
    base_tam = torch.as_tensor(np.asarray(state.tam, dtype=np.float32), dtype=torch.float32, device=candidate_device).reshape(1, 2, width)
    for start in range(0, len(incremental_counts), chunk_size):
        chunk = incremental_counts[start : start + chunk_size]
        build_start = time.perf_counter()
        counts_t = _counts_chunk_tensor(chunk, width=width, device=candidate_device)
        candidate_tams = base_tam + counts_t
        _timing_add(args, "candidate_tam_gpu_build_time_sec", time.perf_counter() - build_start)
        forward_start = time.perf_counter()
        probs_t = _predict_probabilities_tensor(attacker, candidate_tams, batch_size=chunk_size)
        if candidate_device.type == "cuda":
            torch.cuda.synchronize(candidate_device)
        _timing_add(args, "rf_forward_time_sec", time.perf_counter() - forward_start)
        probs_rows.append(probs_t.detach().cpu().numpy().astype(np.float32))
        del counts_t, candidate_tams, probs_t
    if candidate_device.type == "cuda":
        peak_mb = float(torch.cuda.max_memory_allocated(candidate_device) / (1024.0 * 1024.0))
        _timing_add(args, "candidate_gpu_peak_allocated_mb", peak_mb)
    _timing_add(args, "candidate_gpu_tam_eval_time_sec", time.perf_counter() - start_total)
    _timing_add(args, "candidate_gpu_tam_eval_count", float(len(incremental_counts)))
    probs = np.concatenate(probs_rows, axis=0).astype(np.float32)
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
    reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
    gains = original_class_objective_delta(original, reference, probs, weights)
    metrics = probability_metrics(original, probs, np.repeat(np.asarray([int(label)], dtype=np.int64), len(probs)))
    return gains.astype(np.float32), probs.astype(np.float32), metrics


def _evaluate_actions(
    *,
    state: EvalState,
    actions: list[ExpandedAction],
    original_prob: np.ndarray,
    label: int,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not actions:
        return np.zeros(0, dtype=np.float32), np.zeros((0, original_prob.size), dtype=np.float32), {}
    if str(getattr(args, "candidate_eval_mode", "renderer")) == "gpu_tam":
        incremental_counts = [np.asarray(action.counts, dtype=np.int32) for action in actions]
        return _evaluate_incremental_counts_gpu_tam(
            state=state,
            incremental_counts=incremental_counts,
            original_prob=original_prob,
            label=int(label),
            attacker=attacker,
            device=device,
            args=args,
        )
    trial_counts = [np.asarray(state.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32) for action in actions]
    _traces, tams, _stats = _render_dummy_batch(base_trace=state.trace, counts_list=trial_counts, args=args)
    start = time.perf_counter()
    probs = predict_probabilities(attacker, tams, device=device, batch_size=int(args.batch_size))
    _timing_add(args, "rf_forward_time_sec", time.perf_counter() - start)
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
    reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
    gains = original_class_objective_delta(original, reference, probs, weights)
    metrics = probability_metrics(original, probs, np.repeat(np.asarray([int(label)], dtype=np.int64), len(probs)))
    return gains.astype(np.float32), probs.astype(np.float32), metrics


def _selection_scores(gains: np.ndarray, actions: list[ExpandedAction], clean_total: float, objective: str) -> np.ndarray:
    values = np.asarray(gains, dtype=np.float32)
    costs = np.asarray([action_cost(a, clean_total) for a in actions], dtype=np.float32)
    if str(objective) == "efficiency":
        return values / np.maximum(costs, 1e-8)
    if str(objective) == "hybrid":
        eff = values / np.maximum(costs, 1e-8)
        if eff.size and float(np.max(np.abs(eff))) > 1e-8:
            eff = eff / max(float(np.max(np.abs(eff))), 1e-8)
        return values + 0.15 * eff
    return values


def _pair_candidates(
    actions: list[ExpandedAction],
    gains: np.ndarray,
    *,
    clean_total: float,
    remaining_dummy: int,
    config: MethodConfig,
) -> list[tuple[int, int, np.ndarray, int, float]]:
    if not config.pair_enabled or len(actions) < 2:
        return []
    scores = _selection_scores(gains, actions, clean_total, config.objective)
    near = [i for i, gain in enumerate(gains.tolist()) if float(gain) >= -float(config.epsilon)]
    ranked = sorted(set(near + np.argsort(-scores)[: int(config.pair_k)].tolist()), key=lambda i: -float(scores[i]))[: int(config.pair_k)]
    pairs: list[tuple[int, int, np.ndarray, int, float]] = []
    for pos, left_idx in enumerate(ranked):
        for right_idx in ranked[pos + 1 :]:
            left = actions[left_idx]
            right = actions[right_idx]
            if float(gains[left_idx]) < -float(config.epsilon) or float(gains[right_idx]) < -float(config.epsilon):
                continue
            counts = np.asarray(left.counts, dtype=np.int32) + np.asarray(right.counts, dtype=np.int32)
            dummy = int(counts.sum())
            if dummy <= 0 or dummy > int(remaining_dummy):
                continue
            cost = float(dummy / max(float(clean_total), 1.0))
            pairs.append((left_idx, right_idx, counts.astype(np.int32), dummy, cost))
    return pairs


def _evaluate_pairs(
    *,
    state: EvalState,
    pairs: list[tuple[int, int, np.ndarray, int, float]],
    original_prob: np.ndarray,
    label: int,
    attacker,
    device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    if not pairs:
        return np.zeros(0, dtype=np.float32), np.zeros((0, original_prob.size), dtype=np.float32), {}
    if str(getattr(args, "candidate_eval_mode", "renderer")) == "gpu_tam":
        incremental_counts = [np.asarray(counts, dtype=np.int32) for _li, _ri, counts, _dummy, _cost in pairs]
        return _evaluate_incremental_counts_gpu_tam(
            state=state,
            incremental_counts=incremental_counts,
            original_prob=original_prob,
            label=int(label),
            attacker=attacker,
            device=device,
            args=args,
        )
    trial_counts = [np.asarray(state.dummy_counts, dtype=np.int32) + counts for _li, _ri, counts, _dummy, _cost in pairs]
    _traces, tams, _stats = _render_dummy_batch(base_trace=state.trace, counts_list=trial_counts, args=args)
    start = time.perf_counter()
    probs = predict_probabilities(attacker, tams, device=device, batch_size=int(args.batch_size))
    _timing_add(args, "rf_forward_time_sec", time.perf_counter() - start)
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
    reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
    gains = original_class_objective_delta(original, reference, probs, weights)
    metrics = probability_metrics(original, probs, np.repeat(np.asarray([int(label)], dtype=np.int64), len(probs)))
    return gains.astype(np.float32), probs.astype(np.float32), metrics


def _funnel_rows(
    *,
    diagnostics: dict,
    sample_index: int,
    sample_id: str,
    protocol: str,
    method: str,
    budget: float,
    round_index: int,
    step_index: int,
    evaluated: list[ExpandedAction],
    accepted: list[ExpandedAction],
) -> list[dict[str, Any]]:
    rows = []
    base = {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(method),
        "dummy_budget_bound": float(budget),
        "round_index": int(round_index),
        "step_index": int(step_index),
    }

    def add_rows(kind: str, suffix: str, eval_counter: Counter, accept_counter: Counter) -> None:
        buckets = set()
        for stage in ("generated_before_filter", "after_legality", "after_score_hint", "after_max_generated_actions"):
            key = stage if not suffix else f"{stage}_{suffix}"
            buckets.update(str(item) for item in diagnostics.get(key, {}).keys())
        buckets.update(str(item) for item in eval_counter.keys())
        buckets.update(str(item) for item in accept_counter.keys())
        for bucket in sorted(buckets):
            rows.append(
                {
                    **base,
                    "bucket_kind": str(kind),
                    "bucket": str(bucket),
                    "action_type": str(bucket) if kind == "action_type" else "",
                    "generated_before_filter": int(diagnostics.get("generated_before_filter" if not suffix else f"generated_before_filter_{suffix}", {}).get(bucket, 0)),
                    "after_legality": int(diagnostics.get("after_legality" if not suffix else f"after_legality_{suffix}", {}).get(bucket, 0)),
                    "after_score_hint": int(diagnostics.get("after_score_hint" if not suffix else f"after_score_hint_{suffix}", {}).get(bucket, 0)),
                    "after_max_generated_actions": int(diagnostics.get("after_max_generated_actions" if not suffix else f"after_max_generated_actions_{suffix}", {}).get(bucket, 0)),
                    "evaluated_by_RF": int(eval_counter.get(bucket, 0)),
                    "accepted": int(accept_counter.get(bucket, 0)),
                }
            )

    add_rows("action_type", "", Counter(str(action.action_type) for action in evaluated), Counter(str(action.action_type) for action in accepted))
    add_rows("dose", "dose", Counter(f"dose_{_action_dummy_count(action)}" for action in evaluated), Counter(f"dose_{_action_dummy_count(action)}" for action in accepted))
    add_rows(
        "bin",
        "bin",
        Counter("multi_bin" if _action_nonzero_bins(action) > 1 else "single_bin" for action in evaluated),
        Counter("multi_bin" if _action_nonzero_bins(action) > 1 else "single_bin" for action in accepted),
    )
    add_rows("direction", "direction", Counter(_action_direction_bucket(action) for action in evaluated), Counter(_action_direction_bucket(action) for action in accepted))
    return rows


def _bucket_counts(actions: list[ExpandedAction]) -> dict[str, int]:
    counter = Counter()
    for action in actions:
        dummy = _action_dummy_count(action)
        counter[f"dose_{dummy}"] += 1
        counter["multi_bin" if _action_nonzero_bins(action) > 1 else "single_bin"] += 1
        counter[_action_direction_bucket(action)] += 1
        counter[str(action.action_type)] += 1
    return dict(counter)


def _select_dummy_b2e(
    *,
    state: EvalState,
    mask: np.ndarray,
    protocol: str,
    budget: float,
    clean_total: float,
    original_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    config: MethodConfig,
    round_index: int,
    attacker,
    device,
    args: argparse.Namespace,
    rng: random.Random,
    state_observer: Any | None = None,
    teacher_observer: Any | None = None,
) -> SelectionResult:
    current = state
    used: set[tuple] = {_identity(action) for action in current.selected_actions}
    max_dummy = int(np.floor(float(clean_total) * float(budget) + 1e-9))
    accepted_single = 0
    accepted_pair = 0
    valley_rescue = 0
    rf_eval_count = 0
    candidate_step_count = 0
    best_single_seen = 0.0
    best_pair_seen = 0.0
    proxy_recalls: list[float] = []
    true_recalls: list[float] = []
    funnel_rows: list[dict[str, Any]] = []
    stop_reason = "max_actions_reached"

    for step_index in range(int(args.max_dummy_steps)):
        if _margin(current.prob, original_prob) <= float(args.margin_target):
            stop_reason = "target_reached"
            break
        remaining_dummy = max_dummy - int(np.asarray(current.dummy_counts).sum())
        if remaining_dummy <= 0:
            if teacher_observer is not None:
                teacher_observer(
                    {
                        "event_type": "stop",
                        "stop_reason": "bandwidth_10pct_reached",
                        "round_index": int(round_index),
                        "step_index": int(step_index),
                        "budget": float(budget),
                        "remaining_dummy": int(remaining_dummy),
                        "clean_total": float(clean_total),
                        "candidate_actions": [],
                        "candidate_gains": np.zeros(0, dtype=np.float32),
                        "candidate_scores": np.zeros(0, dtype=np.float32),
                        "selected_index": -1,
                        "selected_kind": "",
                        "selected_gain": 0.0,
                        "pre_state": current,
                        "next_state": current,
                        "mask": np.asarray(mask, dtype=np.float32),
                        "original_prob": np.asarray(original_prob, dtype=np.float32),
                        "label": int(label),
                    }
                )
            stop_reason = "bandwidth_10pct_reached"
            break
        diagnostics: dict[str, Any] = {"profile_detail": bool(getattr(args, "profile_candidate_generation", False))}
        start = time.perf_counter()
        proxy_pool: list[ExpandedAction | CandidateDescriptor]
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
            descriptor_filter_start = time.perf_counter()
            raw_descriptors = [
                action
                for action in raw_descriptors
                if _identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy
            ]
            _timing_add(args, "compact_descriptor_used_budget_filter_time_sec", time.perf_counter() - descriptor_filter_start)
            proxy_pool = list(raw_descriptors)
            materialize_k = max(int(config.eval_k), int(config.eval_k) * max(1, int(getattr(args, "deferred_materialize_oversample", 1))))
            materialize_k = min(max(0, int(materialize_k)), len(raw_descriptors))
            descriptor_config = MethodConfig(
                name=str(config.name),
                prefilter=str(config.prefilter),
                raw_pool=int(config.raw_pool),
                eval_k=int(materialize_k),
                objective=str(config.objective),
                pair_enabled=False,
                pair_k=int(config.pair_k),
                epsilon=float(config.epsilon),
                tau=float(config.tau),
                generator_pair_actions=0,
            )
            descriptor_selected = _prefilter_actions(
                raw_descriptors,
                config=descriptor_config,
                clean_total=clean_total,
                args=args,
                rng=rng,
            )
            materialize_start = time.perf_counter()
            raw_actions = []
            materialize_batch = max(1, int(getattr(args, "materialization_batch_size", 64)))
            for materialize_start_idx in range(0, len(descriptor_selected), materialize_batch):
                raw_actions.extend(
                    materialize_candidate_descriptors(
                        descriptor_selected[materialize_start_idx : materialize_start_idx + materialize_batch],
                        tam=current.tam,
                        clean_total=float(clean_total),
                        protocol=str(protocol),
                        max_action_budget=float(args.max_action_budget),
                        max_local_rate_peak=int(args.max_local_rate_peak),
                    )
                )
            _timing_add(args, "deferred_materialization_time_sec", time.perf_counter() - materialize_start)
            _timing_add(args, "compact_descriptor_count", float(len(raw_descriptors)))
            _timing_add(args, "deferred_action_objects_built", float(len(raw_actions)))
            _timing_add(args, "deferred_dense_counts_built", float(len(raw_actions)))
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
                max_pair_actions=int(config.generator_pair_actions),
                diagnostics=diagnostics,
            )
            proxy_pool = list(raw_actions)
        for timing_key, timing_value in dict(diagnostics.get("timing_sec", {})).items():
            _timing_add(args, str(timing_key), float(timing_value))
        filter_start = time.perf_counter()
        raw_actions = [action for action in raw_actions if _identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]
        _timing_add(args, "candidate_used_budget_filter_time_sec", time.perf_counter() - filter_start)
        _timing_add(args, "candidate_generation_time_sec", time.perf_counter() - start)
        if not raw_actions:
            if teacher_observer is not None:
                teacher_observer(
                    {
                        "event_type": "stop",
                        "stop_reason": "candidate_pool_exhausted",
                        "round_index": int(round_index),
                        "step_index": int(step_index),
                        "budget": float(budget),
                        "remaining_dummy": int(remaining_dummy),
                        "clean_total": float(clean_total),
                        "candidate_actions": [],
                        "candidate_gains": np.zeros(0, dtype=np.float32),
                        "candidate_scores": np.zeros(0, dtype=np.float32),
                        "selected_index": -1,
                        "selected_kind": "",
                        "selected_gain": 0.0,
                        "pre_state": current,
                        "next_state": current,
                        "mask": np.asarray(mask, dtype=np.float32),
                        "original_prob": np.asarray(original_prob, dtype=np.float32),
                        "label": int(label),
                    }
                )
            stop_reason = "candidate_pool_exhausted"
            break
        start = time.perf_counter()
        actions = _prefilter_actions(raw_actions, config=config, clean_total=clean_total, args=args, rng=rng)
        filter_start = time.perf_counter()
        actions = [action for action in actions if _identity(action) not in used and 0 < _action_dummy_count(action) <= remaining_dummy]
        _timing_add(args, "candidate_selected_used_budget_filter_time_sec", time.perf_counter() - filter_start)
        _timing_add(args, "candidate_prefilter_time_sec", time.perf_counter() - start)
        if not actions:
            if teacher_observer is not None:
                teacher_observer(
                    {
                        "event_type": "stop",
                        "stop_reason": "candidate_pool_exhausted",
                        "round_index": int(round_index),
                        "step_index": int(step_index),
                        "budget": float(budget),
                        "remaining_dummy": int(remaining_dummy),
                        "clean_total": float(clean_total),
                        "candidate_actions": [],
                        "candidate_gains": np.zeros(0, dtype=np.float32),
                        "candidate_scores": np.zeros(0, dtype=np.float32),
                        "selected_index": -1,
                        "selected_kind": "",
                        "selected_gain": 0.0,
                        "pre_state": current,
                        "next_state": current,
                        "mask": np.asarray(mask, dtype=np.float32),
                        "original_prob": np.asarray(original_prob, dtype=np.float32),
                        "label": int(label),
                    }
                )
            stop_reason = "candidate_pool_exhausted"
            break
        raw_best_proxy = max((_estimated_abs_gain(action, clean_total) for action in proxy_pool), default=0.0)
        selected_best_proxy = max((_estimated_abs_gain(action, clean_total) for action in actions), default=0.0)
        proxy_recalls.append(float(selected_best_proxy / max(raw_best_proxy, 1e-8)))
        candidate_step_count += 1
        gains, _probs, _metrics = _evaluate_actions(
            state=current,
            actions=actions,
            original_prob=original_prob,
            label=label,
            attacker=attacker,
            device=device,
            args=args,
        )
        rf_eval_count += len(actions)
        scores = _selection_scores(gains, actions, clean_total, config.objective)
        best_idx = int(np.argmax(scores)) if len(scores) else -1
        best_gain = float(gains[best_idx]) if best_idx >= 0 else 0.0
        best_single_seen = max(best_single_seen, max([float(item) for item in gains.tolist()], default=0.0))

        if int(args.true_recall_pool_size) > 0:
            recall_pool = sorted(raw_actions, key=lambda action: -_estimated_abs_gain(action, clean_total))[: int(args.true_recall_pool_size)]
            recall_gains, _rp, _rm = _evaluate_actions(
                state=current,
                actions=recall_pool,
                original_prob=original_prob,
                label=label,
                attacker=attacker,
                device=device,
                args=args,
            )
            rf_eval_count += len(recall_pool)
            raw_true = max([float(item) for item in recall_gains.tolist()], default=0.0)
            selected_true = max([float(item) for item in gains.tolist()], default=0.0)
            true_recalls.append(float(selected_true / max(raw_true, 1e-8)) if raw_true > 0 else 1.0)

        selected_kind = ""
        selected_actions: list[ExpandedAction] = []
        selected_counts: np.ndarray | None = None
        selected_pair_indices: tuple[int, int] | None = None
        selected_gain = 0.0
        pair_gain = 0.0
        pair_choice: tuple[int, int, np.ndarray, int, float] | None = None
        if config.pair_enabled:
            start = time.perf_counter()
            pairs = _pair_candidates(
                actions,
                gains,
                clean_total=clean_total,
                remaining_dummy=remaining_dummy,
                config=config,
            )
            _timing_add(args, "candidate_pair_generation_time_sec", time.perf_counter() - start)
            pair_gains, _pair_probs, _pair_metrics = _evaluate_pairs(
                state=current,
                pairs=pairs,
                original_prob=original_prob,
                label=label,
                attacker=attacker,
                device=device,
                args=args,
            )
            rf_eval_count += len(pairs)
            if len(pair_gains):
                best_pair_seen = max(best_pair_seen, max(float(item) for item in pair_gains.tolist()))
                pair_scores = _selection_scores(pair_gains, [actions[left] for left, _right, _counts, _dummy, _cost in pairs], clean_total, config.objective)
                pair_scores = np.asarray(pair_scores, dtype=np.float32)
                for idx, gain in enumerate(pair_gains.tolist()):
                    if float(gain) <= float(config.tau):
                        pair_scores[idx] = -np.inf
                pair_idx = int(np.argmax(pair_scores))
                if np.isfinite(pair_scores[pair_idx]) and float(pair_gains[pair_idx]) > float(config.tau):
                    pair_choice = pairs[pair_idx]
                    pair_gain = float(pair_gains[pair_idx])

        if pair_choice is not None and (best_gain <= 0.0 or pair_gain > best_gain):
            left_idx, right_idx, counts, _dummy, _cost = pair_choice
            selected_kind = "pair"
            selected_actions = [actions[left_idx], actions[right_idx]]
            selected_counts = np.asarray(current.dummy_counts, dtype=np.int32) + counts.astype(np.int32)
            selected_pair_indices = (int(left_idx), int(right_idx))
            selected_gain = pair_gain
        elif best_idx >= 0 and best_gain > 0.0:
            selected_kind = "single"
            selected_actions = [actions[best_idx]]
            selected_counts = np.asarray(current.dummy_counts, dtype=np.int32) + np.asarray(actions[best_idx].counts, dtype=np.int32)
            selected_gain = best_gain
        else:
            if teacher_observer is not None:
                teacher_observer(
                    {
                        "event_type": "stop",
                        "stop_reason": "no_positive_pair" if config.pair_enabled else "no_positive_single",
                        "round_index": int(round_index),
                        "step_index": int(step_index),
                        "budget": float(budget),
                        "remaining_dummy": int(remaining_dummy),
                        "clean_total": float(clean_total),
                        "candidate_actions": list(actions),
                        "candidate_gains": np.asarray(gains, dtype=np.float32),
                        "candidate_scores": np.asarray(scores, dtype=np.float32),
                        "selected_index": -1,
                        "selected_kind": "",
                        "selected_gain": 0.0,
                        "pre_state": current,
                        "next_state": current,
                        "mask": np.asarray(mask, dtype=np.float32),
                        "original_prob": np.asarray(original_prob, dtype=np.float32),
                        "label": int(label),
                    }
                )
            funnel_rows.extend(
                _funnel_rows(
                    diagnostics=diagnostics,
                    sample_index=sample_index,
                    sample_id=sample_id,
                    protocol=protocol,
                    method=config.name,
                    budget=budget,
                    round_index=round_index,
                    step_index=step_index,
                    evaluated=actions,
                    accepted=[],
                )
            )
            stop_reason = "no_positive_pair" if config.pair_enabled else "no_positive_single"
            break

        pre_state = current
        previous_dummy_count = int(np.asarray(pre_state.dummy_counts, dtype=np.int32).sum())
        trace, tam, stats = _render_dummy(base_trace=pre_state.trace, counts=selected_counts, args=args)
        prob = _predict_one(attacker, tam, device=device, args=args)
        for action in selected_actions:
            used.add(_identity(action))
        if selected_kind == "pair":
            accepted_pair += 1
            if selected_pair_indices is not None and all(float(gains[idx]) <= 0.0 for idx in selected_pair_indices):
                valley_rescue += 1
        else:
            accepted_single += 1
        current = EvalState(
            trace=trace,
            tam=tam,
            prob=prob,
            dummy_counts=np.asarray(selected_counts, dtype=np.int32),
            dummy_bandwidth=float(stats["raw_bandwidth"]),
            avg_delay=current.avg_delay,
            p95_delay=current.p95_delay,
            max_delay=current.max_delay,
            delay_values=tuple(current.delay_values),
            outgoing_delay_values=tuple(current.outgoing_delay_values),
            incoming_delay_values=tuple(current.incoming_delay_values),
            selected_actions=list(current.selected_actions) + list(selected_actions),
        )
        if teacher_observer is not None:
            teacher_observer(
                {
                    "event_type": "action",
                    "stop_reason": "",
                    "round_index": int(round_index),
                    "step_index": int(step_index),
                    "budget": float(budget),
                    "remaining_dummy": int(remaining_dummy),
                    "clean_total": float(clean_total),
                    "candidate_actions": list(actions),
                    "candidate_gains": np.asarray(gains, dtype=np.float32),
                    "candidate_scores": np.asarray(scores, dtype=np.float32),
                    "selected_index": int(best_idx) if selected_kind == "single" else -1,
                    "selected_kind": str(selected_kind),
                    "selected_gain": float(selected_gain),
                    "pre_state": pre_state,
                    "next_state": current,
                    "mask": np.asarray(mask, dtype=np.float32),
                    "original_prob": np.asarray(original_prob, dtype=np.float32),
                    "label": int(label),
                    "selected_action_count": int(len(selected_actions)),
                }
            )
        _observe_state(
            state_observer,
            {
                "event_type": "dummy",
                "round_index": int(round_index),
                "step_index": int(step_index),
                "selected_kind": str(selected_kind),
                "selected_gain": float(selected_gain),
                "added_dummy": int(np.asarray(selected_counts, dtype=np.int32).sum() - previous_dummy_count),
                "cumulative_dummy": int(np.asarray(selected_counts, dtype=np.int32).sum()),
                "selected_action_count": int(len(selected_actions)),
            },
            current,
        )
        funnel_rows.extend(
            _funnel_rows(
                diagnostics=diagnostics,
                sample_index=sample_index,
                sample_id=sample_id,
                protocol=protocol,
                method=config.name,
                budget=budget,
                round_index=round_index,
                step_index=step_index,
                evaluated=actions,
                accepted=selected_actions,
            )
        )
        if selected_gain <= 0.0:
            stop_reason = "no_positive_pair" if config.pair_enabled else "no_positive_single"
            break
    else:
        if _margin(current.prob, original_prob) <= float(args.margin_target):
            stop_reason = "target_reached"
        elif int(np.asarray(current.dummy_counts).sum()) >= max_dummy:
            stop_reason = "bandwidth_10pct_reached"
        else:
            stop_reason = "max_actions_reached"

    return SelectionResult(
        state=current,
        stop_reason=stop_reason,
        accepted_single_count=accepted_single,
        accepted_pair_count=accepted_pair,
        valley_pair_rescue_count=valley_rescue,
        rf_eval_count=rf_eval_count,
        candidate_step_count=candidate_step_count,
        best_single_gain_seen=float(best_single_seen),
        best_pair_gain_seen=float(best_pair_seen),
        proxy_recall_values=proxy_recalls,
        true_recall_values=true_recalls,
        funnel_rows=funnel_rows,
    )


def _run_controller(
    *,
    config: MethodConfig,
    protocol: str,
    budget: float,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    sample_index: int,
    sample_id: str,
    attacker,
    device,
    args: argparse.Namespace,
    state_observer: Any | None = None,
    teacher_observer: Any | None = None,
) -> tuple[EvalState, dict[str, Any], list[dict[str, Any]]]:
    clean_total = max(float(original_tam.sum()), 1.0)
    state = _initial_state(raw_trace, original_tam, original_prob)
    aggregate = {
        "stop_reason": "max_rounds_reached",
        "accepted_single_count": 0,
        "accepted_pair_count": 0,
        "valley_pair_rescue_count": 0,
        "rf_eval_count": 0,
        "candidate_step_count": 0,
        "best_single_gain_seen": 0.0,
        "best_pair_gain_seen": 0.0,
        "proxy_best_gain_recall_values": [],
        "true_best_gain_recall_values": [],
    }
    all_funnel_rows: list[dict[str, Any]] = []
    rounds = max(1, int(args.rounds))
    stable_name_seed = sum((idx + 1) * ord(ch) for idx, ch in enumerate(config.name))
    rng = random.Random(int(args.seed) + int(sample_index) * 1009 + stable_name_seed)
    for round_index in range(rounds):
        stop_mask = np.asarray(original_mask, dtype=np.float32)
        if int(args.max_delay) > 0:
            start = time.perf_counter()
            mask0 = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
            _timing_add(args, "keypoint_refresh_time_sec", time.perf_counter() - start)
            stop_mask = np.asarray(mask0, dtype=np.float32)
            state = _apply_delay(
                state=state,
                mask=mask0,
                protocol=str(protocol),
                delay_budget=max(1, int(round(int(args.max_delay) / rounds))),
                args=args,
            )
            state.prob = _predict_one(attacker, state.tam, device=device, args=args)
            _observe_state(
                state_observer,
                {
                    "event_type": "delay",
                    "round_index": int(round_index),
                    "step_index": -1,
                    "selected_kind": "",
                    "selected_gain": 0.0,
                    "added_dummy": 0,
                    "cumulative_dummy": int(np.asarray(state.dummy_counts, dtype=np.int32).sum()),
                    "selected_action_count": 0,
                },
                state,
            )
        if _margin(state.prob, original_prob) <= float(args.margin_target):
            if teacher_observer is not None:
                teacher_observer(
                    {
                        "event_type": "stop",
                        "stop_reason": "target_reached",
                        "round_index": int(round_index),
                        "step_index": -1,
                        "budget": float(budget),
                        "remaining_dummy": int(max(0, int(np.floor(float(clean_total) * float(budget) + 1e-9)) - int(np.asarray(state.dummy_counts, dtype=np.int32).sum()))),
                        "clean_total": float(clean_total),
                        "candidate_actions": [],
                        "candidate_gains": np.zeros(0, dtype=np.float32),
                        "candidate_scores": np.zeros(0, dtype=np.float32),
                        "selected_index": -1,
                        "selected_kind": "",
                        "selected_gain": 0.0,
                        "pre_state": state,
                        "next_state": state,
                        "mask": np.asarray(stop_mask, dtype=np.float32),
                        "original_prob": np.asarray(original_prob, dtype=np.float32),
                        "label": int(label),
                    }
                )
            aggregate["stop_reason"] = "target_reached"
            break
        if float(budget) <= 0.0:
            if teacher_observer is not None:
                teacher_observer(
                    {
                        "event_type": "stop",
                        "stop_reason": "no_dummy_budget",
                        "round_index": int(round_index),
                        "step_index": -1,
                        "budget": float(budget),
                        "remaining_dummy": 0,
                        "clean_total": float(clean_total),
                        "candidate_actions": [],
                        "candidate_gains": np.zeros(0, dtype=np.float32),
                        "candidate_scores": np.zeros(0, dtype=np.float32),
                        "selected_index": -1,
                        "selected_kind": "",
                        "selected_gain": 0.0,
                        "pre_state": state,
                        "next_state": state,
                        "mask": np.asarray(stop_mask, dtype=np.float32),
                        "original_prob": np.asarray(original_prob, dtype=np.float32),
                        "label": int(label),
                    }
                )
                aggregate["stop_reason"] = "no_dummy_budget"
            continue
        start = time.perf_counter()
        mask = _fast_refresh_mask(attacker, state.tam, original_prob, device=device)
        _timing_add(args, "keypoint_refresh_time_sec", time.perf_counter() - start)
        result = _select_dummy_b2e(
            state=state,
            mask=mask,
            protocol=protocol,
            budget=budget,
            clean_total=clean_total,
            original_prob=original_prob,
            label=label,
            sample_index=sample_index,
            sample_id=sample_id,
            config=config,
            round_index=round_index,
            attacker=attacker,
            device=device,
            args=args,
            rng=rng,
            state_observer=state_observer,
            teacher_observer=teacher_observer,
        )
        state = result.state
        aggregate["accepted_single_count"] += result.accepted_single_count
        aggregate["accepted_pair_count"] += result.accepted_pair_count
        aggregate["valley_pair_rescue_count"] += result.valley_pair_rescue_count
        aggregate["rf_eval_count"] += result.rf_eval_count
        aggregate["candidate_step_count"] += result.candidate_step_count
        aggregate["best_single_gain_seen"] = max(float(aggregate["best_single_gain_seen"]), float(result.best_single_gain_seen))
        aggregate["best_pair_gain_seen"] = max(float(aggregate["best_pair_gain_seen"]), float(result.best_pair_gain_seen))
        aggregate["proxy_best_gain_recall_values"].extend(result.proxy_recall_values)
        aggregate["true_best_gain_recall_values"].extend(result.true_recall_values)
        all_funnel_rows.extend(result.funnel_rows)
        aggregate["stop_reason"] = result.stop_reason
        if result.stop_reason in {"target_reached", "bandwidth_10pct_reached", "candidate_pool_exhausted", "no_positive_single", "no_positive_pair"}:
            if _margin(state.prob, original_prob) <= float(args.margin_target):
                aggregate["stop_reason"] = "target_reached"
                break
    if _margin(state.prob, original_prob) <= float(args.margin_target):
        aggregate["stop_reason"] = "target_reached"
    return state, aggregate, all_funnel_rows


def _resource_fields(state: EvalState, clean_total: float) -> dict[str, Any]:
    clean = max(float(clean_total), 1.0)
    counts = np.asarray(state.dummy_counts, dtype=np.int32)
    total_dummy = int(counts.sum())
    defended = clean + float(total_dummy)
    accepted = list(state.selected_actions)
    dose_counter = Counter(str(_action_dummy_count(action)) for action in accepted)
    type_counter = Counter(str(action.action_type) for action in accepted)
    return {
        "clean_packet_count": float(clean),
        "dummy_packet_count": int(total_dummy),
        "defended_packet_count": float(defended),
        "actual_dummy_bandwidth": float(total_dummy / clean),
        "dummy_overhead": float(total_dummy / clean),
        "total_overhead": float((defended - clean) / clean),
        "outgoing_dummy_packet_count": int(counts[0].sum()),
        "incoming_dummy_packet_count": int(counts[1].sum()),
        "average_delay_bins": float(state.avg_delay),
        "p95_delay_bins": float(state.p95_delay),
        "maximum_delay_bins": int(state.max_delay),
        "delay_packet_count": int(len(state.delay_values)),
        "outgoing_delay_packet_count": int(len(state.outgoing_delay_values)),
        "incoming_delay_packet_count": int(len(state.incoming_delay_values)),
        "accepted_action_count": int(len(accepted)),
        "accepted_action_dose_distribution": json.dumps(dict(sorted(dose_counter.items())), sort_keys=True),
        "accepted_action_type_distribution": json.dumps(dict(sorted(type_counter.items())), sort_keys=True),
        "multi_bin_action_rate": float(np.mean([_action_nonzero_bins(action) > 1 for action in accepted])) if accepted else 0.0,
        "accepted_multi_dummy_rate": float(np.mean([_action_dummy_count(action) > 1 for action in accepted])) if accepted else 0.0,
    }


def _sample_row(
    *,
    sample_index: int,
    sample_id: str,
    protocol: str,
    config: MethodConfig,
    budget: float,
    margin_target: float,
    original_prob: np.ndarray,
    state: EvalState,
    label: int,
    clean_total: float,
    runtime: float,
    aggregate: dict[str, Any],
) -> dict[str, Any]:
    metrics = probability_metrics(original_prob.reshape(1, -1), state.prob.reshape(1, -1), np.asarray([int(label)], dtype=np.int64))
    recalls = [float(item) for item in aggregate.get("proxy_best_gain_recall_values", [])]
    true_recalls = [float(item) for item in aggregate.get("true_best_gain_recall_values", [])]
    resource = _resource_fields(state, clean_total)
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(config.name),
        "prefilter": str(config.prefilter),
        "eval_k": int(config.eval_k),
        "raw_pool": int(config.raw_pool),
        "pair_enabled": int(config.pair_enabled),
        "pair_k": int(config.pair_k),
        "epsilon": float(config.epsilon),
        "tau": float(config.tau),
        "margin_target": float(margin_target),
        "dummy_budget_bound": float(budget),
        "max_delay_budget": 64,
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "target_margin_success": int(float(metrics["original_class_margin"][0]) <= float(margin_target)),
        "original_pred": int(metrics["original_pred"][0]),
        "final_pred": int(metrics["evaluated_pred"][0]),
        "true_label": int(label),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "accepted_single_count": int(aggregate.get("accepted_single_count", 0)),
        "accepted_pair_count": int(aggregate.get("accepted_pair_count", 0)),
        "valley_pair_rescue_count": int(aggregate.get("valley_pair_rescue_count", 0)),
        "pair_acceptance_rate": float(aggregate.get("accepted_pair_count", 0) / max(float(aggregate.get("candidate_step_count", 0)), 1.0)),
        "candidate_step_count": int(aggregate.get("candidate_step_count", 0)),
        "rf_eval_count": int(aggregate.get("rf_eval_count", 0)),
        "best_single_gain_seen": float(aggregate.get("best_single_gain_seen", 0.0)),
        "best_pair_gain_seen": float(aggregate.get("best_pair_gain_seen", 0.0)),
        "candidate_best_gain_recall_at_k": float(np.mean(recalls)) if recalls else "",
        "true_best_gain_recall_at_k": float(np.mean(true_recalls)) if true_recalls else "",
        "stop_reason": str(aggregate.get("stop_reason", "")),
        "runtime_sec": float(runtime),
        **resource,
    }


def _percentile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _float(row: dict[str, Any], key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["protocol"]), str(row["method"]), float(row["dummy_budget_bound"]))].append(row)
    out: list[dict[str, Any]] = []
    for (protocol, method, budget), items in sorted(groups.items()):
        bw = [_float(row, "actual_dummy_bandwidth") for row in items]
        runtimes = [_float(row, "runtime_sec") for row in items]
        out.append(
            {
                "protocol": protocol,
                "method": method,
                "dummy_budget_bound": float(budget),
                "samples": int(len(items)),
                "accuracy": float(np.mean([_float(row, "accuracy") for row in items])),
                "flip": float(np.mean([_float(row, "flip") for row in items])),
                "margin_success_rate": float(np.mean([_float(row, "target_margin_success") for row in items])),
                "mean_actual_bandwidth": float(np.mean(bw)) if bw else 0.0,
                "median_actual_bandwidth": _percentile(bw, 50),
                "p90_actual_bandwidth": _percentile(bw, 90),
                "p95_actual_bandwidth": _percentile(bw, 95),
                "max_actual_bandwidth": max(bw) if bw else 0.0,
                "mean_dummy_count": float(np.mean([_float(row, "dummy_packet_count") for row in items])),
                "mean_accepted_actions": float(np.mean([_float(row, "accepted_action_count") for row in items])),
                "mean_pair_acceptance_rate": float(np.mean([_float(row, "pair_acceptance_rate") for row in items])),
                "valley_pair_rescue_count": int(sum(_float(row, "valley_pair_rescue_count") for row in items)),
                "mean_multi_bin_action_rate": float(np.mean([_float(row, "multi_bin_action_rate") for row in items])),
                "mean_multi_dummy_action_rate": float(np.mean([_float(row, "accepted_multi_dummy_rate") for row in items])),
                "mean_rf_eval_count": float(np.mean([_float(row, "rf_eval_count") for row in items])),
                "mean_best_single_gain_seen": float(np.mean([_float(row, "best_single_gain_seen") for row in items])),
                "mean_best_pair_gain_seen": float(np.mean([_float(row, "best_pair_gain_seen") for row in items])),
                "mean_candidate_best_gain_recall_at_k": float(np.mean([_float(row, "candidate_best_gain_recall_at_k") for row in items if row.get("candidate_best_gain_recall_at_k") != ""])) if any(row.get("candidate_best_gain_recall_at_k") != "" for row in items) else "",
                "mean_true_best_gain_recall_at_k": float(np.mean([_float(row, "true_best_gain_recall_at_k") for row in items if row.get("true_best_gain_recall_at_k") != ""])) if any(row.get("true_best_gain_recall_at_k") != "" for row in items) else "",
                "mean_runtime_sec": float(np.mean(runtimes)) if runtimes else 0.0,
                "p95_runtime_sec": _percentile(runtimes, 95),
            }
        )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
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


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    run_args = _runtime_args(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    protocols = _parse_csv_strings(args.protocols)
    budgets = _parse_csv_floats(args.dummy_budgets)
    configs = [_method_config(name) for name in _parse_csv_strings(args.methods)]
    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(args.data_root, source_indices, run_args)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(prob.shape[1]),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    sample_rows: list[dict[str, Any]] = []
    funnel_rows: list[dict[str, Any]] = []
    total_jobs = len(configs) * len(protocols) * len(budgets) * int(tam.shape[0])
    done = 0
    for config in configs:
        for protocol in protocols:
            for budget in budgets:
                for sample_index in range(int(tam.shape[0])):
                    done += 1
                    if args.progress:
                        print(
                            f"[b2e] {done}/{total_jobs} method={config.name} protocol={protocol} "
                            f"B={budget:g} sample={sample_index}",
                            flush=True,
                        )
                    start = time.perf_counter()
                    state, aggregate, rows = _run_controller(
                        config=config,
                        protocol=str(protocol),
                        budget=float(budget),
                        raw_trace=np.asarray(raw_rows[sample_index], dtype=np.float32),
                        original_tam=np.asarray(tam[sample_index], dtype=np.float32),
                        original_mask=np.asarray(mask[sample_index], dtype=np.float32),
                        original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                        label=int(labels[sample_index]),
                        sample_index=int(sample_index),
                        sample_id=str(sample_ids[sample_index]),
                        attacker=attacker,
                        device=device,
                        args=args,
                    )
                    runtime = time.perf_counter() - start
                    clean_total = max(float(np.asarray(tam[sample_index], dtype=np.float32).sum()), 1.0)
                    sample_rows.append(
                        _sample_row(
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            protocol=str(protocol),
                            config=config,
                            budget=float(budget),
                            margin_target=float(args.margin_target),
                            original_prob=np.asarray(prob[sample_index], dtype=np.float32),
                            state=state,
                            label=int(labels[sample_index]),
                            clean_total=clean_total,
                            runtime=runtime,
                            aggregate=aggregate,
                        )
                    )
                    funnel_rows.extend(rows)
    summary_rows = _summarize(sample_rows)
    _write_csv(output_dir / "b2e_sample_results.csv", sample_rows)
    _write_csv(output_dir / "b2e_summary.csv", summary_rows)
    _write_csv(output_dir / "b2e_candidate_funnel.csv", funnel_rows)
    manifest = {
        "archive": str(args.archive),
        "checkpoint": str(Path(checkpoint).resolve()),
        "samples": int(tam.shape[0]),
        "protocols": protocols,
        "dummy_budgets": budgets,
        "max_delay": int(args.max_delay),
        "margin_target": float(args.margin_target),
        "methods": [config.__dict__ for config in configs],
        "sample_results": str(output_dir / "b2e_sample_results.csv"),
        "summary": str(output_dir / "b2e_summary.csv"),
        "candidate_funnel": str(output_dir / "b2e_candidate_funnel.csv"),
    }
    (output_dir / "b2e_run.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
