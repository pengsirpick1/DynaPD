"""Run Stage B2-D keypoint-guided dual-actuator dynamic controller."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynapd.data import load_cw_data
from dynapd.evaluation.attack_models import build_rf_tam_input, crop_or_pad_2d
from dynapd.projection.padding import PaddingTemplate, render_batch_variable
from dynapd.stage_a.faithfulness import predict_probabilities
from dynapd.stage_a.modeling import load_stage_a_attacker
from dynapd.stage_b.expanded_generator import ExpandedAction, action_cost, action_identity, generate_expanded_actions
from dynapd.stage_b.objectives import ObjectiveWeights, original_class_objective_delta, original_class_utility, probability_metrics
from dynapd.stage_b.smoothing import causal_delay_trace, keypoint_windows, trace_to_tam
from dynapd.utils import resolve_device, set_seed, write_csv, write_json
from dynapd.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, parse_csv_floats, parse_csv_ints, parse_csv_strings


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"
PROTOCOLS = ("client_only", "bidirectional_cooperative")
METHODS = (
    "delay_only",
    "dummy_only",
    "static_hybrid_no_refresh",
    "delay_then_dummy_fast_refresh",
    "delay_then_dummy_oracle_refresh",
    "dummy_then_delay_fast_refresh",
    "alternating2_fast_refresh",
    "alternating3_fast_refresh",
)


def _timing_add(args: argparse.Namespace, key: str, seconds: float) -> None:
    sink = getattr(args, "timing_accumulator", None)
    if isinstance(sink, dict):
        sink[key] = float(sink.get(key, 0.0)) + float(seconds)


@dataclass
class EvalState:
    trace: np.ndarray
    tam: np.ndarray
    prob: np.ndarray
    dummy_counts: np.ndarray
    dummy_bandwidth: float
    avg_delay: float
    p95_delay: float
    max_delay: int
    delay_values: tuple[int, ...]
    outgoing_delay_values: tuple[int, ...]
    incoming_delay_values: tuple[int, ...]
    selected_actions: list[ExpandedAction]


def _default_checkpoint(attacker: str) -> str:
    return DEFAULT_FIXED_DF if str(attacker).lower() == "df" else DEFAULT_FIXED_RF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocols", default="client_only,bidirectional_cooperative")
    parser.add_argument("--methods", default="delay_only,dummy_only,static_hybrid_no_refresh,delay_then_dummy_fast_refresh,delay_then_dummy_oracle_refresh,dummy_then_delay_fast_refresh,alternating2_fast_refresh,alternating3_fast_refresh")
    parser.add_argument("--dummy_budgets", default="0.02,0.05,0.08,0.10")
    parser.add_argument("--max_delays", default="4,8,16,32")
    parser.add_argument("--delay_length", type=int, default=32)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_candidates", type=int, default=24)
    parser.add_argument("--max_generated_actions", type=int, default=128)
    parser.add_argument("--max_pair_actions", type=int, default=24)
    parser.add_argument("--max_action_budget", type=float, default=0.035)
    parser.add_argument("--max_local_rate_peak", type=int, default=16)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--refresh_stride", type=int, default=32)
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
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b2d_dual_actuator_{args.attacker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_archive(path: str, max_samples: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    if int(max_samples) > 0:
        original_n = int(payload["tam"].shape[0])
        n = min(int(max_samples), original_n)
        for key, value in list(payload.items()):
            arr = np.asarray(value)
            if arr.shape[:1] == (original_n,):
                payload[key] = arr[:n]
    return payload


def _load_raw_rows(data_root: str, source_indices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    cfg = SimpleNamespace(data_root=str(data_root), seed=int(args.seed), val_ratio=0.10, test_ratio=0.10, max_samples=0, max_classes=0)
    raw, _labels, _trace_ids, _splits, _source = load_cw_data(cfg)
    return np.asarray(raw[np.asarray(source_indices, dtype=np.int64)], dtype=np.float32)


def _parse_protocols(value: str) -> list[str]:
    protocols = parse_csv_strings(value)
    invalid = sorted(set(protocols) - set(PROTOCOLS))
    if invalid:
        raise ValueError(f"Unknown protocols: {invalid}")
    return protocols


def _parse_methods(value: str) -> list[str]:
    methods = parse_csv_strings(value)
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    return methods


def _delay_policy(protocol: str) -> str:
    return "outgoing_only" if str(protocol) == "client_only" else "bidirectional"


def _p95(values: tuple[int, ...] | list[int] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float32)
    return float(np.percentile(arr, 95)) if arr.size else 0.0


def _render_dummy(
    *,
    base_trace: np.ndarray,
    counts: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict]:
    template = PaddingTemplate(
        counts=np.asarray(counts, dtype=np.int32),
        target_n_pad=int(np.asarray(counts).sum()),
        actual_n_pad=int(np.asarray(counts).sum()),
        target_bandwidth=0.0,
        metadata={"method": "stage_b2d_dual_actuator"},
    )
    start = time.perf_counter()
    traces, _origins, stats = render_batch_variable(
        crop_or_pad_2d(base_trace, int(args.max_trace_length)).astype(np.float32),
        [template],
        seed=int(args.seed),
        strategy=str(args.renderer_strategy),
        coordinate=str(args.renderer_coordinate),
        coordinate_length=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
    )
    _timing_add(args, "renderer_time_sec", time.perf_counter() - start)
    start = time.perf_counter()
    padded = crop_or_pad_2d(traces[0], int(args.max_trace_length)).astype(np.float32)
    tam = build_rf_tam_input(padded, max_len=int(args.max_trace_length), max_load_time=float(args.max_load_time), num_slots=int(args.rf_num_slots))[0]
    _timing_add(args, "tam_rebuild_time_sec", time.perf_counter() - start)
    return traces[0].astype(np.float32), tam.astype(np.float32), {
        "raw_bandwidth": float(stats["raw_bandwidth"][0]),
        "raw_real_packet_retention": float(stats["raw_real_packet_retention"][0]),
        "raw_length": int(stats["raw_lengths"][0]),
    }


def _render_dummy_batch(
    *,
    base_trace: np.ndarray,
    counts_list: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[list[np.ndarray], np.ndarray, list[dict]]:
    if not counts_list:
        return [], np.zeros((0, 2, int(args.rf_num_slots)), dtype=np.float32), []
    clean_batch = np.repeat(crop_or_pad_2d(base_trace, int(args.max_trace_length)).astype(np.float32), len(counts_list), axis=0)
    templates = [
        PaddingTemplate(
            counts=np.asarray(counts, dtype=np.int32),
            target_n_pad=int(np.asarray(counts).sum()),
            actual_n_pad=int(np.asarray(counts).sum()),
            target_bandwidth=0.0,
            metadata={"method": "stage_b2d_dual_actuator"},
        )
        for counts in counts_list
    ]
    start = time.perf_counter()
    traces, _origins, stats = render_batch_variable(
        clean_batch,
        templates,
        seed=int(args.seed),
        strategy=str(args.renderer_strategy),
        coordinate=str(args.renderer_coordinate),
        coordinate_length=int(args.max_trace_length),
        max_load_time=float(args.max_load_time),
    )
    _timing_add(args, "renderer_time_sec", time.perf_counter() - start)
    start = time.perf_counter()
    padded = np.vstack([crop_or_pad_2d(trace, int(args.max_trace_length))[0] for trace in traces]).astype(np.float32)
    tam = build_rf_tam_input(padded, max_len=int(args.max_trace_length), max_load_time=float(args.max_load_time), num_slots=int(args.rf_num_slots))
    _timing_add(args, "tam_rebuild_time_sec", time.perf_counter() - start)
    stats_rows = [
        {
            "raw_bandwidth": float(stats["raw_bandwidth"][idx]),
            "raw_real_packet_retention": float(stats["raw_real_packet_retention"][idx]),
            "raw_length": int(stats["raw_lengths"][idx]),
        }
        for idx in range(len(counts_list))
    ]
    return [np.asarray(trace, dtype=np.float32) for trace in traces], tam.astype(np.float32), stats_rows


def _predict_one(attacker, tam: np.ndarray, *, device, args: argparse.Namespace) -> np.ndarray:
    start = time.perf_counter()
    prob = predict_probabilities(attacker, tam.reshape(1, *tam.shape), device=device, batch_size=int(args.batch_size))[0]
    _timing_add(args, "rf_forward_time_sec", time.perf_counter() - start)
    return prob


def _fast_refresh_mask(attacker, tam: np.ndarray, original_prob: np.ndarray, *, device) -> np.ndarray:
    y0 = int(np.argmax(original_prob))
    values = torch.as_tensor(tam.reshape(1, *tam.shape), dtype=torch.float32, device=device)
    values.requires_grad_(True)
    logits = attacker.logits(values)
    probs = torch.softmax(logits, dim=1)
    other = torch.cat([probs[:, :y0], probs[:, y0 + 1 :]], dim=1).max(dim=1).values
    objective = probs[:, y0] - other
    objective.backward()
    grad = values.grad.detach().abs().cpu().numpy()[0].astype(np.float32)
    mag = np.asarray(tam, dtype=np.float32)
    mag = mag / max(float(mag.max()), 1e-6)
    grad = grad / max(float(grad.max()), 1e-6)
    return (0.70 * grad + 0.30 * mag).astype(np.float32)


def _oracle_refresh_mask(attacker, tam: np.ndarray, original_prob: np.ndarray, *, device, args: argparse.Namespace) -> np.ndarray:
    # Coarse occlusion refresh: a light-weight DynaMask proxy for oracle relocalization.
    width = int(tam.shape[1])
    stride = max(4, int(args.refresh_stride))
    window = max(stride, int(args.delay_length))
    candidates: list[tuple[int, int, int]] = []
    for direction in range(2):
        for start in range(0, width, stride):
            end = min(width, start + window)
            if end > start:
                candidates.append((direction, start, end))
    trials = np.repeat(tam.reshape(1, *tam.shape), len(candidates), axis=0)
    for idx, (direction, start, end) in enumerate(candidates):
        trials[idx, direction, start:end] = 0.0
    probs = predict_probabilities(attacker, trials, device=device, batch_size=int(args.batch_size))
    utility = original_class_utility(np.repeat(original_prob.reshape(1, -1), len(probs), axis=0), probs, ObjectiveWeights(0.40, 0.40, 0.20))
    mask = np.zeros_like(tam, dtype=np.float32)
    for score, (direction, start, end) in zip(utility.tolist(), candidates):
        if float(score) > 0:
            mask[direction, start:end] = np.maximum(mask[direction, start:end], float(score))
    if float(mask.max()) <= 1e-8:
        return _fast_refresh_mask(attacker, tam, original_prob, device=device)
    return (mask / float(mask.max())).astype(np.float32)


def _refresh_mask(mode: str, attacker, tam: np.ndarray, original_mask: np.ndarray, original_prob: np.ndarray, *, device, args: argparse.Namespace) -> np.ndarray:
    if str(mode) == "no_refresh":
        return np.asarray(original_mask, dtype=np.float32)
    if str(mode) == "fast_refresh":
        return _fast_refresh_mask(attacker, tam, original_prob, device=device)
    if str(mode) == "oracle_refresh":
        return _oracle_refresh_mask(attacker, tam, original_prob, device=device, args=args)
    raise ValueError(f"Unknown refresh mode={mode!r}")


def _apply_delay(
    *,
    state: EvalState,
    mask: np.ndarray,
    protocol: str,
    delay_budget: int,
    args: argparse.Namespace,
) -> EvalState:
    start = time.perf_counter()
    windows = keypoint_windows(mask, ratio=float(args.ratio), max_windows=int(args.max_windows), sample_index=0)
    trace, _avg_delay, _max_delay, delay_stats = causal_delay_trace(
        state.trace,
        windows,
        width=int(args.rf_num_slots),
        length=int(args.delay_length),
        rho=float(args.delay_rho),
        max_delay=int(delay_budget),
        max_load_time=float(args.max_load_time),
        direction_policy=_delay_policy(str(protocol)),
        return_stats=True,
    )
    _timing_add(args, "delay_trace_time_sec", time.perf_counter() - start)
    start = time.perf_counter()
    tam = trace_to_tam(trace, width=int(args.rf_num_slots), max_load_time=float(args.max_load_time))
    _timing_add(args, "tam_rebuild_time_sec", time.perf_counter() - start)
    delay_values = tuple(state.delay_values) + tuple(int(item) for item in delay_stats["delay_values"])
    outgoing_values = tuple(state.outgoing_delay_values) + tuple(int(item) for item in delay_stats["outgoing_delay_values"])
    incoming_values = tuple(state.incoming_delay_values) + tuple(int(item) for item in delay_stats["incoming_delay_values"])
    return EvalState(
        trace=trace,
        tam=tam,
        prob=state.prob,
        dummy_counts=state.dummy_counts,
        dummy_bandwidth=state.dummy_bandwidth,
        avg_delay=float(np.mean(delay_values)) if delay_values else 0.0,
        p95_delay=_p95(delay_values),
        max_delay=max(delay_values) if delay_values else 0,
        delay_values=delay_values,
        outgoing_delay_values=outgoing_values,
        incoming_delay_values=incoming_values,
        selected_actions=list(state.selected_actions),
    )


def _select_dummy(
    *,
    state: EvalState,
    mask: np.ndarray,
    protocol: str,
    budget: float,
    clean_total: float,
    original_prob: np.ndarray,
    label: int,
    attacker,
    device,
    args: argparse.Namespace,
) -> EvalState:
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    current = state
    used: set[tuple] = {action_identity(action) for action in current.selected_actions}
    max_dummy = int(round(float(clean_total) * float(budget)))
    for _step in range(int(args.max_dummy_steps)):
        remaining_dummy = max_dummy - int(current.dummy_counts.sum())
        if remaining_dummy <= 0:
            break
        actions = generate_expanded_actions(
            tam=current.tam,
            soft_mask=mask,
            sample_index=0,
            sample_id="current",
            true_label=int(label),
            protocol=str(protocol),
            clean_total=float(clean_total),
            ratio=float(args.ratio),
            max_windows=int(args.max_windows),
            max_action_budget=float(args.max_action_budget),
            max_local_rate_peak=int(args.max_local_rate_peak),
            max_actions=int(args.max_generated_actions),
            max_pair_actions=int(args.max_pair_actions),
        )
        actions = sorted(actions, key=lambda action: (-float(action.score_hint), action_cost(action, clean_total)))[: int(args.max_candidates)]
        trial_actions, trial_counts = [], []
        for action in actions:
            ident = action_identity(action)
            if ident in used:
                continue
            if int(action.counts.sum()) <= 0 or int(action.counts.sum()) > remaining_dummy:
                continue
            trial_actions.append(action)
            trial_counts.append(current.dummy_counts + np.asarray(action.counts, dtype=np.int32))
        if not trial_actions:
            break
        _traces, tams, _stats = _render_dummy_batch(base_trace=current.trace, counts_list=trial_counts, args=args)
        probs = predict_probabilities(attacker, tams, device=device, batch_size=int(args.batch_size))
        reference = np.repeat(current.prob.reshape(1, -1), len(probs), axis=0)
        original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
        gains = original_class_objective_delta(original, reference, probs, weights)
        eff = gains / np.maximum(np.asarray([a.counts.sum() / max(clean_total, 1.0) for a in trial_actions], dtype=np.float32), 1e-8)
        idx = int(np.argmax(eff))
        if float(gains[idx]) <= 0.0:
            break
        action = trial_actions[idx]
        trace, tam, stats = _render_dummy(base_trace=current.trace, counts=trial_counts[idx], args=args)
        prob = _predict_one(attacker, tam, device=device, args=args)
        used.add(action_identity(action))
        current = EvalState(
            trace=trace,
            tam=tam,
            prob=prob,
            dummy_counts=trial_counts[idx].astype(np.int32),
            dummy_bandwidth=float(stats["raw_bandwidth"]),
            avg_delay=current.avg_delay,
            p95_delay=current.p95_delay,
            max_delay=current.max_delay,
            delay_values=tuple(current.delay_values),
            outgoing_delay_values=tuple(current.outgoing_delay_values),
            incoming_delay_values=tuple(current.incoming_delay_values),
            selected_actions=list(current.selected_actions) + [action],
        )
    return current


def _initial_state(raw_trace: np.ndarray, tam: np.ndarray, prob: np.ndarray) -> EvalState:
    return EvalState(
        trace=np.asarray(raw_trace, dtype=np.float32),
        tam=np.asarray(tam, dtype=np.float32),
        prob=np.asarray(prob, dtype=np.float32),
        dummy_counts=np.zeros_like(tam, dtype=np.int32),
        dummy_bandwidth=0.0,
        avg_delay=0.0,
        p95_delay=0.0,
        max_delay=0,
        delay_values=(),
        outgoing_delay_values=(),
        incoming_delay_values=(),
        selected_actions=[],
    )


def _run_method(
    *,
    method: str,
    protocol: str,
    budget: float,
    delay_budget: int,
    raw_trace: np.ndarray,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    attacker,
    device,
    args: argparse.Namespace,
) -> EvalState:
    clean_total = max(float(original_tam.sum()), 1.0)
    state = _initial_state(raw_trace, original_tam, original_prob)
    if method == "delay_only":
        mask = _refresh_mask("no_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        state = _apply_delay(state=state, mask=mask, protocol=protocol, delay_budget=delay_budget, args=args)
        state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        return state
    if method == "dummy_only":
        mask = _refresh_mask("fast_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        return _select_dummy(state=state, mask=mask, protocol=protocol, budget=budget, clean_total=clean_total, original_prob=original_prob, label=label, attacker=attacker, device=device, args=args)
    if method == "static_hybrid_no_refresh":
        mask = _refresh_mask("no_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        state = _apply_delay(state=state, mask=mask, protocol=protocol, delay_budget=delay_budget, args=args)
        state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        return _select_dummy(state=state, mask=mask, protocol=protocol, budget=budget, clean_total=clean_total, original_prob=original_prob, label=label, attacker=attacker, device=device, args=args)
    if method == "delay_then_dummy_fast_refresh":
        mask0 = _refresh_mask("no_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        state = _apply_delay(state=state, mask=mask0, protocol=protocol, delay_budget=delay_budget, args=args)
        state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        mask1 = _refresh_mask("fast_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        return _select_dummy(state=state, mask=mask1, protocol=protocol, budget=budget, clean_total=clean_total, original_prob=original_prob, label=label, attacker=attacker, device=device, args=args)
    if method == "delay_then_dummy_oracle_refresh":
        mask0 = _refresh_mask("no_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        state = _apply_delay(state=state, mask=mask0, protocol=protocol, delay_budget=delay_budget, args=args)
        state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        mask1 = _refresh_mask("oracle_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        return _select_dummy(state=state, mask=mask1, protocol=protocol, budget=budget, clean_total=clean_total, original_prob=original_prob, label=label, attacker=attacker, device=device, args=args)
    if method == "dummy_then_delay_fast_refresh":
        mask0 = _refresh_mask("fast_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        state = _select_dummy(state=state, mask=mask0, protocol=protocol, budget=budget, clean_total=clean_total, original_prob=original_prob, label=label, attacker=attacker, device=device, args=args)
        mask1 = _refresh_mask("fast_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
        state = _apply_delay(state=state, mask=mask1, protocol=protocol, delay_budget=delay_budget, args=args)
        state.prob = _predict_one(attacker, state.tam, device=device, args=args)
        return state
    if method in {"alternating2_fast_refresh", "alternating3_fast_refresh"}:
        rounds = 2 if method.startswith("alternating2") else 3
        for round_index in range(rounds):
            mask = _refresh_mask("fast_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
            state = _apply_delay(state=state, mask=mask, protocol=protocol, delay_budget=max(1, int(round(delay_budget / rounds))), args=args)
            state.prob = _predict_one(attacker, state.tam, device=device, args=args)
            mask = _refresh_mask("fast_refresh", attacker, state.tam, original_mask, original_prob, device=device, args=args)
            round_budget = float(budget) * float(round_index + 1) / float(rounds)
            state = _select_dummy(state=state, mask=mask, protocol=protocol, budget=round_budget, clean_total=clean_total, original_prob=original_prob, label=label, attacker=attacker, device=device, args=args)
        return state
    raise ValueError(f"Unknown method={method!r}")


def _cache_key(sample_index: int, protocol: str, method: str, budget: float, delay_budget: int) -> tuple:
    if str(method) == "delay_only":
        return int(sample_index), str(protocol), str(method), None, int(delay_budget)
    if str(method) == "dummy_only":
        return int(sample_index), str(protocol), str(method), round(float(budget), 8), None
    return int(sample_index), str(protocol), str(method), round(float(budget), 8), int(delay_budget)


def _mask_stats(before: np.ndarray, after: np.ndarray) -> tuple[float, float]:
    a = np.asarray(before, dtype=np.float32).reshape(-1)
    b = np.asarray(after, dtype=np.float32).reshape(-1)
    if float(a.sum()) <= 1e-8 or float(b.sum()) <= 1e-8:
        return 0.0, 0.0
    top = max(1, int(round(0.10 * a.size)))
    ia = set(np.argsort(-a)[:top].tolist())
    ib = set(np.argsort(-b)[:top].tolist())
    overlap = len(ia & ib) / max(len(ia | ib), 1)
    width = int(np.asarray(before).shape[-1])

    def centroid(mask: np.ndarray) -> float:
        flat = np.asarray(mask, dtype=np.float32).reshape(-1)
        order = np.argsort(-flat)[:top]
        weights = flat[order]
        slots = np.asarray(order % max(width, 1), dtype=np.float32)
        if float(weights.sum()) <= 1e-8:
            return float(slots.mean()) if slots.size else 0.0
        return float(np.average(slots, weights=weights))

    ca = centroid(before)
    cb = centroid(after)
    return float(overlap), float(cb - ca)


def _row(
    *,
    sample_index: int,
    sample_id: str,
    protocol: str,
    method: str,
    budget: float,
    delay_budget: int,
    original_prob: np.ndarray,
    state: EvalState,
    label: int,
    original_mask: np.ndarray,
    refresh_mask: np.ndarray,
    clean_total: float,
    cached_result: int,
    runtime: float,
) -> dict:
    metrics = probability_metrics(original_prob.reshape(1, -1), state.prob.reshape(1, -1), np.asarray([label], dtype=np.int64))
    overlap, displacement = _mask_stats(original_mask, refresh_mask)
    dummy_counts = np.asarray(state.dummy_counts, dtype=np.int32)
    outgoing_dummy = int(dummy_counts[0].sum())
    incoming_dummy = int(dummy_counts[1].sum())
    total_dummy = int(dummy_counts.sum())
    outgoing_delay = int(len(state.outgoing_delay_values))
    incoming_delay = int(len(state.incoming_delay_values))
    total_delay = int(len(state.delay_values))
    clean = max(float(clean_total), 1.0)
    defended_packet_count = float(clean + total_dummy)
    dummy_overhead = float(total_dummy / clean)
    total_overhead = float((defended_packet_count - clean) / clean)
    legal = int(protocol != "client_only" or (incoming_delay == 0 and incoming_dummy == 0))
    dummy_displacements = [abs(float(action.insert_center) - float(action.affected_center)) for action in state.selected_actions]
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "protocol": str(protocol),
        "method": str(method),
        "dummy_budget": float(budget),
        "max_delay_budget": int(delay_budget),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "original_top1_drop": float(metrics["original_top1_drop"][0]),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "original_class_utility": float(metrics["original_class_utility"][0]),
        "dummy_bandwidth": float(dummy_overhead),
        "renderer_dummy_bandwidth": float(state.dummy_bandwidth),
        "clean_packet_count": float(clean),
        "dummy_packet_count": int(total_dummy),
        "defended_packet_count": float(defended_packet_count),
        "dummy_overhead": float(dummy_overhead),
        "total_overhead": float(total_overhead),
        "bandwidth_audit_error": float(abs(dummy_overhead - total_overhead)),
        "outgoing_dummy_packet_count": int(outgoing_dummy),
        "incoming_dummy_packet_count": int(incoming_dummy),
        "outgoing_dummy_fraction_of_clean": float(outgoing_dummy / clean),
        "incoming_dummy_fraction_of_clean": float(incoming_dummy / clean),
        "average_delay_bins": float(state.avg_delay),
        "p95_delay_bins": float(state.p95_delay),
        "maximum_delay_bins": int(state.max_delay),
        "delay_packet_count": int(total_delay),
        "outgoing_delay_packet_count": int(outgoing_delay),
        "incoming_delay_packet_count": int(incoming_delay),
        "outgoing_delay_fraction_of_clean": float(outgoing_delay / clean),
        "incoming_delay_fraction_of_clean": float(incoming_delay / clean),
        "outgoing_average_delay_bins": float(np.mean(state.outgoing_delay_values)) if outgoing_delay else 0.0,
        "incoming_average_delay_bins": float(np.mean(state.incoming_delay_values)) if incoming_delay else 0.0,
        "outgoing_p95_delay_bins": _p95(state.outgoing_delay_values),
        "incoming_p95_delay_bins": _p95(state.incoming_delay_values),
        "outgoing_max_delay_bins": int(max(state.outgoing_delay_values)) if outgoing_delay else 0,
        "incoming_max_delay_bins": int(max(state.incoming_delay_values)) if incoming_delay else 0,
        "action_count": int(len(state.selected_actions)),
        "client_only_legal": legal,
        "delay_direction_policy": _delay_policy(protocol),
        "old_new_mask_overlap": float(overlap),
        "keypoint_displacement": float(displacement),
        "selected_dummy_position_mean": float(np.mean([action.insert_center for action in state.selected_actions])) if state.selected_actions else 0.0,
        "selected_dummy_position_displacement": float(np.mean(dummy_displacements)) if dummy_displacements else 0.0,
        "cached_result": int(cached_result),
        "runtime_sec": float(runtime),
    }


def _summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str, float, int], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["protocol"], row["method"], float(row["dummy_budget"]), int(row["max_delay_budget"])), []).append(row)
    metric_keys = [
        "accuracy", "flip", "original_top1_drop", "original_class_probability", "original_class_margin",
        "original_class_margin_drop", "normalized_entropy_gain", "js_div", "original_class_utility",
        "dummy_bandwidth", "renderer_dummy_bandwidth", "clean_packet_count", "dummy_packet_count",
        "defended_packet_count", "dummy_overhead", "total_overhead", "bandwidth_audit_error",
        "outgoing_dummy_packet_count", "incoming_dummy_packet_count",
        "outgoing_dummy_fraction_of_clean", "incoming_dummy_fraction_of_clean",
        "average_delay_bins", "p95_delay_bins", "maximum_delay_bins", "delay_packet_count",
        "outgoing_delay_packet_count", "incoming_delay_packet_count",
        "outgoing_delay_fraction_of_clean", "incoming_delay_fraction_of_clean",
        "outgoing_average_delay_bins", "incoming_average_delay_bins",
        "outgoing_p95_delay_bins", "incoming_p95_delay_bins", "outgoing_max_delay_bins", "incoming_max_delay_bins",
        "action_count",
        "client_only_legal", "old_new_mask_overlap", "keypoint_displacement", "selected_dummy_position_mean", "runtime_sec",
        "selected_dummy_position_displacement", "cached_result",
    ]
    out = []
    for (protocol, method, budget, delay), matched in sorted(groups.items()):
        row = {"protocol": protocol, "method": method, "dummy_budget": budget, "max_delay_budget": delay, "samples": len(matched)}
        for key in metric_keys:
            row[key] = float(np.mean([float(item[key]) for item in matched]))
        out.append(row)
    # Aggregate synergy against delay_only and dummy_only with matching protocol/budget/delay.
    index = {(r["protocol"], r["method"], r["dummy_budget"], r["max_delay_budget"]): r for r in out}
    for row in out:
        delay = index.get((row["protocol"], "delay_only", row["dummy_budget"], row["max_delay_budget"]))
        dummy = index.get((row["protocol"], "dummy_only", row["dummy_budget"], row["max_delay_budget"]))
        if delay and dummy:
            row["synergy_utility"] = float(row["original_class_utility"] - delay["original_class_utility"] - dummy["original_class_utility"])
        else:
            row["synergy_utility"] = 0.0
    return out


def main() -> None:
    import time

    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    protocols = _parse_protocols(args.protocols)
    methods = _parse_methods(args.methods)
    budgets = parse_csv_floats(args.dummy_budgets)
    delays = parse_csv_ints(args.max_delays)
    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(args.data_root, source_indices, args)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=prob.shape[1],
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    rows: list[dict] = []
    state_cache: dict[tuple, tuple[EvalState, float]] = {}
    total = len(protocols) * len(methods) * len(budgets) * len(delays)
    cursor = 0
    for protocol in protocols:
        for budget in budgets:
            for delay in delays:
                for method in methods:
                    cursor += 1
                    if args.progress:
                        print(f"[stage_b2d] config {cursor}/{total} {protocol} {method} B={budget:g} D={delay}", flush=True)
                    for sample_index in range(tam.shape[0]):
                        key = _cache_key(sample_index, protocol, method, float(budget), int(delay))
                        cached = int(key in state_cache)
                        if cached:
                            state, _cached_runtime = state_cache[key]
                            runtime = 0.0
                        else:
                            start = time.perf_counter()
                            state = _run_method(
                                method=method,
                                protocol=protocol,
                                budget=float(budget),
                                delay_budget=int(delay),
                                raw_trace=raw_rows[sample_index],
                                original_tam=tam[sample_index],
                                original_mask=mask[sample_index],
                                original_prob=prob[sample_index],
                                label=int(labels[sample_index]),
                                attacker=attacker,
                                device=device,
                                args=args,
                            )
                            runtime = float(time.perf_counter() - start)
                            state_cache[key] = (state, runtime)
                        refresh = _fast_refresh_mask(attacker, state.tam, prob[sample_index], device=device)
                        rows.append(
                            _row(
                                sample_index=sample_index,
                                sample_id=sample_ids[sample_index],
                                protocol=protocol,
                                method=method,
                                budget=float(budget),
                                delay_budget=int(delay),
                                original_prob=prob[sample_index],
                                state=state,
                                label=int(labels[sample_index]),
                                original_mask=mask[sample_index],
                                refresh_mask=refresh,
                                clean_total=float(tam[sample_index].sum()),
                                cached_result=cached,
                                runtime=runtime,
                            )
                        )
    summary = _summarize(rows)
    write_csv(output_dir / "dual_sample_results.csv", rows)
    write_csv(output_dir / "dual_summary.csv", summary)
    write_json(
        output_dir / "dual_oracle_summary.json",
        {
            "archive": str(args.archive),
            "checkpoint": str(checkpoint),
            "samples": int(tam.shape[0]),
            "protocols": protocols,
            "methods": methods,
            "dummy_budgets": budgets,
            "max_delays": delays,
            "objective": {
                "confidence": float(args.confidence_weight),
                "margin": float(args.margin_weight),
                "entropy": float(args.entropy_weight),
                "definition": "fixed original RF prediction class: 0.4 probability drop + 0.4 original-class margin drop + 0.2 normalized entropy gain",
            },
        },
    )
    print(f"Stage B2-D dual actuator complete: {output_dir}")


if __name__ == "__main__":
    main()
