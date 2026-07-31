"""Run Stage B1 expanded dynamic causal action-generation oracle."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.data import load_cw_data
from dmmp.evaluation.attack_models import build_rf_tam_input, crop_or_pad_2d
from dmmp.projection.padding import PaddingTemplate, render_batch_variable
from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.action_selector import filter_protocol, load_action_table, pareto_filter
from dmmp.stage_b.expanded_generator import (
    ExpandedAction,
    action_cost,
    action_identity,
    convert_stage_b0_action,
    generate_expanded_actions,
)
from dmmp.stage_b.objectives import ObjectiveWeights, original_class_objective_delta, original_class_utility, probability_metrics
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, parse_csv_floats, parse_csv_strings


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"

PROTOCOLS = ("client_only", "bidirectional_cooperative")
METHODS = (
    "stage_b0_static_efficiency",
    "stage_b0_sequential_causal",
    "expanded_static",
    "expanded_dynamic_greedy",
    "expanded_dynamic_beam4",
    "expanded_dynamic_beam8",
)


@dataclass
class SearchState:
    counts: np.ndarray
    prob: np.ndarray
    tam: np.ndarray
    utility: float
    history: list[dict]
    selected: list[ExpandedAction]
    selected_ids: set[tuple]


def _default_checkpoint(attacker: str) -> str:
    return DEFAULT_FIXED_DF if str(attacker).lower() == "df" else DEFAULT_FIXED_RF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--action_table", required=True)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--budgets", default="0.02,0.05,0.10,0.15")
    parser.add_argument("--protocols", default="client_only,bidirectional_cooperative")
    parser.add_argument("--methods", default="stage_b0_static_efficiency,stage_b0_sequential_causal,expanded_static,expanded_dynamic_greedy,expanded_dynamic_beam4,expanded_dynamic_beam8")
    parser.add_argument("--stopping_mode", choices=["strict_positive", "lookahead_positive", "budget_fill"], default="lookahead_positive")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_static_candidates", type=int, default=128)
    parser.add_argument("--max_dynamic_candidates", type=int, default=48)
    parser.add_argument("--max_beam_candidates", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=24)
    parser.add_argument("--min_marginal_gain", type=float, default=0.0)
    parser.add_argument("--budget_fill_weight", type=float, default=0.015)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_generated_actions", type=int, default=256)
    parser.add_argument("--max_pair_actions", type=int, default=64)
    parser.add_argument("--max_action_budget", type=float, default=0.035)
    parser.add_argument("--max_local_rate_peak", type=int, default=16)
    parser.add_argument("--absolute_doses", default="1,2,4,8,16,32")
    parser.add_argument("--relative_doses", default="0.10,0.25,0.50,1.00")
    parser.add_argument("--confidence_weight", type=float, default=0.30)
    parser.add_argument("--margin_weight", type=float, default=0.50)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--renderer_batch_size", type=int, default=64)
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
    name = args.run_name or f"stage_b1_expanded_oracle_{args.attacker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _parse_methods(value: str) -> list[str]:
    methods = parse_csv_strings(value)
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    return methods


def _parse_protocols(value: str) -> list[str]:
    protocols = parse_csv_strings(value)
    invalid = sorted(set(protocols) - set(PROTOCOLS))
    if invalid:
        raise ValueError(f"Unknown protocols: {invalid}")
    return protocols


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
    cfg = SimpleNamespace(
        data_root=str(data_root),
        seed=int(args.seed),
        val_ratio=0.10,
        test_ratio=0.10,
        max_samples=0,
        max_classes=0,
    )
    raw, _labels, _trace_ids, _splits, _source = load_cw_data(cfg)
    return np.asarray(raw[np.asarray(source_indices, dtype=np.int64)], dtype=np.float32)


def _render_evaluate_counts(
    *,
    attacker,
    raw_trace: np.ndarray,
    counts_list: list[np.ndarray],
    args: argparse.Namespace,
    device,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    probs: list[np.ndarray] = []
    tams: list[np.ndarray] = []
    stats_rows: list[dict] = []
    if not counts_list:
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0, 2, int(args.rf_num_slots)), dtype=np.float32), []
    clean = np.asarray(raw_trace, dtype=np.float32)
    batch_size = max(1, int(args.renderer_batch_size))
    for start in range(0, len(counts_list), batch_size):
        batch_counts = counts_list[start : start + batch_size]
        clean_batch = np.repeat(clean.reshape(1, -1), len(batch_counts), axis=0)
        templates = [
            PaddingTemplate(
                counts=np.asarray(counts, dtype=np.int32),
                target_n_pad=int(np.asarray(counts).sum()),
                actual_n_pad=int(np.asarray(counts).sum()),
                target_bandwidth=0.0,
                metadata={"method": "stage_b1_expanded_oracle"},
            )
            for counts in batch_counts
        ]
        traces, _origins, stats = render_batch_variable(
            clean_batch,
            templates,
            seed=int(args.seed) + int(start),
            strategy=str(args.renderer_strategy),
            coordinate=str(args.renderer_coordinate),
            coordinate_length=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
        )
        padded = np.vstack([crop_or_pad_2d(trace, int(args.max_trace_length))[0] for trace in traces]).astype(np.float32)
        tam = build_rf_tam_input(
            padded,
            max_len=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
            num_slots=int(args.rf_num_slots),
        )
        tams.append(tam.astype(np.float32))
        probs.append(predict_probabilities(attacker, tam, device=device, batch_size=int(args.batch_size)))
        for local in range(len(batch_counts)):
            stats_rows.append(
                {
                    "raw_bandwidth": float(stats["raw_bandwidth"][local]),
                    "raw_real_packet_retention": float(stats["raw_real_packet_retention"][local]),
                    "raw_length": int(stats["raw_lengths"][local]),
                }
            )
    return np.concatenate(probs, axis=0).astype(np.float32), np.concatenate(tams, axis=0).astype(np.float32), stats_rows


def _metric_row(original_prob: np.ndarray, prob: np.ndarray, label: int) -> dict:
    metrics = probability_metrics(original_prob.reshape(1, -1), prob.reshape(1, -1), np.asarray([int(label)], dtype=np.int64))
    return {key: (float(value[0]) if np.asarray(value).dtype.kind != "i" else int(value[0])) for key, value in metrics.items()}


def _dynamic_soft_mask(original_mask: np.ndarray, original_tam: np.ndarray, current_tam: np.ndarray) -> np.ndarray:
    mask = np.asarray(original_mask, dtype=np.float32)
    cur = np.asarray(current_tam, dtype=np.float32)
    base = np.asarray(original_tam, dtype=np.float32)
    mag = cur / max(float(cur.max()), 1e-6)
    delta = np.abs(cur - base)
    delta = delta / max(float(delta.max()), 1e-6)
    soft = 0.70 * mask + 0.20 * mag + 0.10 * delta
    return soft.astype(np.float32)


def _tier_counts(actions: list[ExpandedAction]) -> dict[str, int]:
    return {tier: int(sum(1 for action in actions if action.tier == tier)) for tier in ("primary", "secondary", "exploration")}


def _action_type_summary(actions: list[ExpandedAction]) -> str:
    counts: dict[str, int] = {}
    for action in actions:
        counts[str(action.action_type)] = counts.get(str(action.action_type), 0) + 1
    return ";".join(f"{key}:{counts[key]}" for key in sorted(counts))


def _state_row(
    *,
    original_prob: np.ndarray,
    state_prob: np.ndarray,
    label: int,
    state_counts: np.ndarray,
    action: ExpandedAction | None,
    step: int,
    incremental_cost: float,
    cumulative_cost: float,
    marginal_gain: float,
    marginal_efficiency: float,
    utility_before: float,
    utility_after: float,
    selected: list[ExpandedAction],
    stats: dict,
) -> dict:
    row = _metric_row(original_prob, state_prob, int(label))
    tiers = _tier_counts(selected)
    incoming = int(sum(action.requires_incoming_capability > 0 for action in selected))
    windows = [int(action.window_id) for action in selected]
    row.update(
        {
            "step": int(step),
            "incremental_cost": float(incremental_cost),
            "cumulative_cost": float(cumulative_cost),
            "marginal_gain": float(marginal_gain),
            "marginal_efficiency": float(marginal_efficiency),
            "utility_before": float(utility_before),
            "utility_after": float(utility_after),
            "selected_action_count": int(len(selected)),
            "incoming_action_count": int(incoming),
            "incoming_action_ratio": float(incoming / max(len(selected), 1)),
            "primary_action_count": int(tiers["primary"]),
            "secondary_action_count": int(tiers["secondary"]),
            "exploration_action_count": int(tiers["exploration"]),
            "primary_action_ratio": float(tiers["primary"] / max(len(selected), 1)),
            "secondary_action_ratio": float(tiers["secondary"] / max(len(selected), 1)),
            "exploration_action_ratio": float(tiers["exploration"] / max(len(selected), 1)),
            "overlap_window_count": int(len(windows) - len(set(windows))),
            "selected_action_types": _action_type_summary(selected),
            "raw_bandwidth": float(stats.get("raw_bandwidth", cumulative_cost)),
            "raw_real_packet_retention": float(stats.get("raw_real_packet_retention", 1.0)),
            "raw_length": int(stats.get("raw_length", 0)),
            "total_dummy_packets": int(np.asarray(state_counts).sum()),
        }
    )
    if action is None:
        row.update(
            {
                "action_type": "",
                "tier": "",
                "source": "",
                "window_id": "",
                "insert_center": "",
                "dose": "",
                "direction_mode": "",
                "requires_incoming_capability": "",
                "score_hint": "",
                "parent": "",
            }
        )
    else:
        row.update(
            {
                "action_type": str(action.action_type),
                "tier": str(action.tier),
                "source": str(action.source),
                "window_id": int(action.window_id),
                "insert_center": int(action.insert_center),
                "dose": int(action.dose),
                "direction_mode": str(action.direction_mode),
                "requires_incoming_capability": int(action.requires_incoming_capability),
                "score_hint": float(action.score_hint),
                "parent": str(action.parent),
            }
        )
    return row


def _empty_state(original_prob: np.ndarray, tam: np.ndarray, label: int) -> SearchState:
    counts = np.zeros_like(tam, dtype=np.int32)
    utility = float(original_class_utility(original_prob.reshape(1, -1), original_prob.reshape(1, -1))[0])
    row = _state_row(
        original_prob=original_prob,
        state_prob=original_prob,
        label=int(label),
        state_counts=counts,
        action=None,
        step=0,
        incremental_cost=0.0,
        cumulative_cost=0.0,
        marginal_gain=0.0,
        marginal_efficiency=0.0,
        utility_before=utility,
        utility_after=utility,
        selected=[],
        stats={"raw_bandwidth": 0.0, "raw_real_packet_retention": 1.0, "raw_length": 0},
    )
    return SearchState(counts=counts, prob=original_prob.astype(np.float32), tam=tam.astype(np.float32), utility=utility, history=[row], selected=[], selected_ids=set())


def _rank_actions(actions: list[ExpandedAction], clean_total: float, limit: int) -> list[ExpandedAction]:
    ranked = sorted(actions, key=lambda action: (-float(action.score_hint), action_cost(action, clean_total), str(action.action_type)))
    if int(limit) > 0 and len(ranked) > int(limit):
        selected: list[ExpandedAction] = []
        seen = set()
        groups: dict[tuple[str, str, int], list[ExpandedAction]] = {}
        for action in ranked:
            groups.setdefault((str(action.tier), str(action.action_type), int(action.window_id)), []).append(action)
        for key in sorted(groups):
            for action in groups[key][:2]:
                ident = action_identity(action)
                if ident not in seen:
                    selected.append(action)
                    seen.add(ident)
                if len(selected) >= int(limit):
                    return selected
        for action in ranked:
            ident = action_identity(action)
            if ident in seen:
                continue
            selected.append(action)
            seen.add(ident)
            if len(selected) >= int(limit):
                return selected
        return selected
    if int(limit) > 0:
        return ranked[: int(limit)]
    return ranked


def _stage_b0_candidates(
    *,
    raw_actions,
    protocol: str,
    tam: np.ndarray,
    clean_total: float,
    num_classes: int,
    limit: int,
) -> list[ExpandedAction]:
    filtered = filter_protocol(raw_actions, str(protocol))
    filtered = pareto_filter(filtered, num_classes=int(num_classes))
    converted = [convert_stage_b0_action(action, tam, clean_total) for action in filtered]
    converted = [action for action in converted if not (str(protocol) == "client_only" and not action.is_client_only)]
    return _rank_actions(converted, clean_total, int(limit))


def _expanded_candidates(
    *,
    tam: np.ndarray,
    mask: np.ndarray,
    sample_index: int,
    sample_id: str,
    label: int,
    protocol: str,
    clean_total: float,
    args: argparse.Namespace,
    limit: int,
) -> list[ExpandedAction]:
    actions = generate_expanded_actions(
        tam=tam,
        soft_mask=mask,
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        true_label=int(label),
        protocol=str(protocol),
        clean_total=float(clean_total),
        ratio=float(args.ratio),
        max_windows=int(args.max_windows),
        absolute_doses=[int(v) for v in parse_csv_floats(args.absolute_doses)],
        relative_doses=parse_csv_floats(args.relative_doses),
        max_action_budget=float(args.max_action_budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
        max_actions=int(args.max_generated_actions),
        max_pair_actions=int(args.max_pair_actions),
    )
    return _rank_actions(actions, clean_total, int(limit))


def _candidate_set_for_method(
    *,
    method: str,
    state: SearchState,
    raw_actions,
    protocol: str,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    sample_index: int,
    sample_id: str,
    label: int,
    clean_total: float,
    num_classes: int,
    args: argparse.Namespace,
    limit: int,
) -> list[ExpandedAction]:
    if str(method).startswith("stage_b0"):
        actions = _stage_b0_candidates(
            raw_actions=raw_actions,
            protocol=str(protocol),
            tam=original_tam,
            clean_total=float(clean_total),
            num_classes=int(num_classes),
            limit=int(limit),
        )
        if str(method) == "stage_b0_sequential_causal":
            actions = [action for action in actions if action.action_type == "stage_b0_causal"]
        return actions
    mask = original_mask if str(method) in {"expanded_static"} else _dynamic_soft_mask(original_mask, original_tam, state.tam)
    generated = _expanded_candidates(
        tam=state.tam if str(method) != "expanded_static" else original_tam,
        mask=mask,
        sample_index=int(sample_index),
        sample_id=str(sample_id),
        label=int(label),
        protocol=str(protocol),
        clean_total=float(clean_total),
        args=args,
        limit=max(int(limit), int(limit) * 2),
    )
    b0_primary = _stage_b0_candidates(
        raw_actions=raw_actions,
        protocol=str(protocol),
        tam=original_tam,
        clean_total=float(clean_total),
        num_classes=int(num_classes),
        limit=max(8, int(limit) // 2),
    )
    combined = b0_primary + generated
    return _rank_actions(combined, clean_total, int(limit))


def _apply_action(
    *,
    state: SearchState,
    action: ExpandedAction,
    raw_trace: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    attacker,
    args: argparse.Namespace,
    device,
) -> SearchState | None:
    incremental = action_cost(action, clean_total)
    next_counts = state.counts + np.asarray(action.counts, dtype=np.int32)
    cumulative = float(next_counts.sum() / max(float(clean_total), 1.0))
    probs, tams, stats = _render_evaluate_counts(attacker=attacker, raw_trace=raw_trace, counts_list=[next_counts], args=args, device=device)
    next_prob = probs[0]
    gain = float(original_class_objective_delta(original_prob.reshape(1, -1), state.prob.reshape(1, -1), next_prob.reshape(1, -1), ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight)))[0])
    next_utility = float(state.utility + gain)
    selected = list(state.selected) + [action]
    row = _state_row(
        original_prob=original_prob,
        state_prob=next_prob,
        label=int(label),
        state_counts=next_counts,
        action=action,
        step=len(state.history),
        incremental_cost=incremental,
        cumulative_cost=cumulative,
        marginal_gain=gain,
        marginal_efficiency=gain / max(incremental, 1e-8),
        utility_before=state.utility,
        utility_after=next_utility,
        selected=selected,
        stats=stats[0],
    )
    next_ids = set(state.selected_ids)
    next_ids.add(action_identity(action))
    return SearchState(
        counts=next_counts.astype(np.int32),
        prob=next_prob.astype(np.float32),
        tam=tams[0].astype(np.float32),
        utility=next_utility,
        history=list(state.history) + [row],
        selected=selected,
        selected_ids=next_ids,
    )


def _run_static_sequence(
    *,
    method: str,
    initial: SearchState,
    actions: list[ExpandedAction],
    raw_trace: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    max_budget: float,
    attacker,
    args: argparse.Namespace,
    device,
) -> SearchState:
    state = initial
    for action in actions:
        ident = action_identity(action)
        if ident in state.selected_ids:
            continue
        if float(state.counts.sum() / max(clean_total, 1.0)) + action_cost(action, clean_total) > float(max_budget) + 1e-9:
            continue
        next_state = _apply_action(
            state=state,
            action=action,
            raw_trace=raw_trace,
            original_prob=original_prob,
            label=int(label),
            clean_total=float(clean_total),
            attacker=attacker,
            args=args,
            device=device,
        )
        if next_state is not None:
            state = next_state
        if len(state.selected) >= int(args.max_steps):
            break
    return state


def _valid_expansion(gain: float, next_utility: float, stopping_mode: str, min_gain: float) -> bool:
    if str(stopping_mode) == "strict_positive":
        return float(gain) > float(min_gain)
    if str(stopping_mode) == "lookahead_positive":
        return float(gain) > float(min_gain) or float(next_utility) > 0.0
    if str(stopping_mode) == "budget_fill":
        return True
    raise ValueError(f"Unknown stopping_mode={stopping_mode!r}")


def _run_dynamic_greedy(
    *,
    method: str,
    initial: SearchState,
    raw_actions,
    protocol: str,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    raw_trace: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    max_budget: float,
    sample_index: int,
    sample_id: str,
    num_classes: int,
    attacker,
    args: argparse.Namespace,
    device,
) -> SearchState:
    state = initial
    for _step in range(int(args.max_steps)):
        actions = _candidate_set_for_method(
            method=method,
            state=state,
            raw_actions=raw_actions,
            protocol=str(protocol),
            original_tam=original_tam,
            original_mask=original_mask,
            sample_index=int(sample_index),
            sample_id=str(sample_id),
            label=int(label),
            clean_total=float(clean_total),
            num_classes=int(num_classes),
            args=args,
            limit=int(args.max_dynamic_candidates),
        )
        trial_actions, trial_counts, trial_costs = [], [], []
        current_cost = float(state.counts.sum() / max(float(clean_total), 1.0))
        for action in actions:
            ident = action_identity(action)
            if ident in state.selected_ids:
                continue
            next_cost = current_cost + action_cost(action, clean_total)
            if next_cost > float(max_budget) + 1e-9:
                continue
            trial_actions.append(action)
            trial_counts.append(state.counts + action.counts)
            trial_costs.append(action_cost(action, clean_total))
        if not trial_actions:
            break
        probs, tams, stats_rows = _render_evaluate_counts(attacker=attacker, raw_trace=raw_trace, counts_list=trial_counts, args=args, device=device)
        weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
        reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
        original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
        gains = original_class_objective_delta(original, reference, probs, weights)
        next_utils = state.utility + gains
        efficiencies = gains / np.maximum(np.asarray(trial_costs, dtype=np.float32), 1e-8)
        score = efficiencies + float(args.budget_fill_weight) * np.asarray([float(count.sum() / max(clean_total, 1.0)) / max(max_budget, 1e-8) for count in trial_counts])
        order = np.argsort(-score, kind="mergesort")
        chosen = None
        for idx in order:
            if _valid_expansion(float(gains[int(idx)]), float(next_utils[int(idx)]), str(args.stopping_mode), float(args.min_marginal_gain)):
                chosen = int(idx)
                break
        if chosen is None:
            break
        action = trial_actions[chosen]
        selected = list(state.selected) + [action]
        row = _state_row(
            original_prob=original_prob,
            state_prob=probs[chosen],
            label=int(label),
            state_counts=trial_counts[chosen],
            action=action,
            step=len(state.history),
            incremental_cost=float(trial_costs[chosen]),
            cumulative_cost=float(trial_counts[chosen].sum() / max(clean_total, 1.0)),
            marginal_gain=float(gains[chosen]),
            marginal_efficiency=float(efficiencies[chosen]),
            utility_before=float(state.utility),
            utility_after=float(next_utils[chosen]),
            selected=selected,
            stats=stats_rows[chosen],
        )
        next_ids = set(state.selected_ids)
        next_ids.add(action_identity(action))
        state = SearchState(
            counts=trial_counts[chosen].astype(np.int32),
            prob=probs[chosen].astype(np.float32),
            tam=tams[chosen].astype(np.float32),
            utility=float(next_utils[chosen]),
            history=list(state.history) + [row],
            selected=selected,
            selected_ids=next_ids,
        )
    return state


def _beam_width(method: str) -> int:
    if str(method).endswith("beam8"):
        return 8
    if str(method).endswith("beam4"):
        return 4
    return 1


def _run_dynamic_beam(
    *,
    method: str,
    initial: SearchState,
    raw_actions,
    protocol: str,
    original_tam: np.ndarray,
    original_mask: np.ndarray,
    raw_trace: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    max_budget: float,
    sample_index: int,
    sample_id: str,
    num_classes: int,
    attacker,
    args: argparse.Namespace,
    device,
) -> SearchState:
    beams = [initial]
    best = initial
    beam_width = _beam_width(method)
    weights = ObjectiveWeights(float(args.confidence_weight), float(args.margin_weight), float(args.entropy_weight))
    for _step in range(int(args.max_steps)):
        expansions: list[SearchState] = []
        for state in beams:
            actions = _candidate_set_for_method(
                method="expanded_dynamic_greedy",
                state=state,
                raw_actions=raw_actions,
                protocol=str(protocol),
                original_tam=original_tam,
                original_mask=original_mask,
                sample_index=int(sample_index),
                sample_id=str(sample_id),
                label=int(label),
                clean_total=float(clean_total),
                num_classes=int(num_classes),
                args=args,
                limit=int(args.max_beam_candidates),
            )
            trial_actions, trial_counts, trial_costs = [], [], []
            current_cost = float(state.counts.sum() / max(clean_total, 1.0))
            for action in actions:
                ident = action_identity(action)
                if ident in state.selected_ids:
                    continue
                next_cost = current_cost + action_cost(action, clean_total)
                if next_cost > float(max_budget) + 1e-9:
                    continue
                trial_actions.append(action)
                trial_counts.append(state.counts + action.counts)
                trial_costs.append(action_cost(action, clean_total))
            if not trial_actions:
                continue
            probs, tams, stats_rows = _render_evaluate_counts(attacker=attacker, raw_trace=raw_trace, counts_list=trial_counts, args=args, device=device)
            reference = np.repeat(state.prob.reshape(1, -1), len(probs), axis=0)
            original = np.repeat(original_prob.reshape(1, -1), len(probs), axis=0)
            gains = original_class_objective_delta(original, reference, probs, weights)
            next_utils = state.utility + gains
            efficiencies = gains / np.maximum(np.asarray(trial_costs, dtype=np.float32), 1e-8)
            for idx, action in enumerate(trial_actions):
                if not _valid_expansion(float(gains[idx]), float(next_utils[idx]), str(args.stopping_mode), float(args.min_marginal_gain)):
                    continue
                selected = list(state.selected) + [action]
                row = _state_row(
                    original_prob=original_prob,
                    state_prob=probs[idx],
                    label=int(label),
                    state_counts=trial_counts[idx],
                    action=action,
                    step=len(state.history),
                    incremental_cost=float(trial_costs[idx]),
                    cumulative_cost=float(trial_counts[idx].sum() / max(clean_total, 1.0)),
                    marginal_gain=float(gains[idx]),
                    marginal_efficiency=float(efficiencies[idx]),
                    utility_before=float(state.utility),
                    utility_after=float(next_utils[idx]),
                    selected=selected,
                    stats=stats_rows[idx],
                )
                next_ids = set(state.selected_ids)
                next_ids.add(action_identity(action))
                expansions.append(
                    SearchState(
                        counts=trial_counts[idx].astype(np.int32),
                        prob=probs[idx].astype(np.float32),
                        tam=tams[idx].astype(np.float32),
                        utility=float(next_utils[idx]),
                        history=list(state.history) + [row],
                        selected=selected,
                        selected_ids=next_ids,
                    )
                )
        if not expansions:
            break
        def state_score(state: SearchState) -> float:
            utilization = float(state.counts.sum() / max(clean_total, 1.0)) / max(float(max_budget), 1e-8)
            return float(state.utility) + float(args.budget_fill_weight) * min(utilization, 1.0)

        expansions = sorted(expansions, key=state_score, reverse=True)
        beams = expansions[:beam_width]
        if state_score(beams[0]) > state_score(best):
            best = beams[0]
    return best


def _state_for_budget(history: list[dict], budget: float) -> dict:
    eligible = [row for row in history if float(row["cumulative_cost"]) <= float(budget) + 1e-9]
    return eligible[-1] if eligible else history[0]


def _annotate_row(row: dict, *, sample_index: int, sample_id: str, protocol: str, method: str, budget: float | None, max_budget: float, candidate_count: int) -> dict:
    out = dict(row)
    out.update(
        {
            "sample_index": int(sample_index),
            "sample_id": str(sample_id),
            "protocol": str(protocol),
            "method": str(method),
            "max_budget": float(max_budget),
            "candidate_count": int(candidate_count),
        }
    )
    if budget is not None:
        out["budget"] = float(budget)
        out["budget_utilization"] = float(out["cumulative_cost"] / max(float(budget), 1e-8))
    return out


def _summarize(sample_rows: list[dict], protocols: list[str], methods: list[str], budgets: list[float]) -> list[dict]:
    metric_keys = [
        "accuracy",
        "flip",
        "original_top1_drop",
        "top1_drop",
        "original_class_probability",
        "original_class_margin",
        "original_class_margin_drop",
        "max_confidence_drop",
        "margin_drop",
        "current_top1_confidence",
        "current_top1_margin",
        "entropy_gain",
        "normalized_entropy_gain",
        "js_div",
        "original_class_utility",
        "utility_after",
        "cumulative_cost",
        "raw_bandwidth",
        "budget_utilization",
        "selected_action_count",
        "incoming_action_ratio",
        "primary_action_ratio",
        "secondary_action_ratio",
        "exploration_action_ratio",
        "overlap_window_count",
    ]
    rows = []
    for protocol in protocols:
        for method in methods:
            for budget in budgets:
                matched = [
                    row
                    for row in sample_rows
                    if row["protocol"] == protocol and row["method"] == method and abs(float(row["budget"]) - float(budget)) < 1e-9
                ]
                if not matched:
                    continue
                summary = {"protocol": protocol, "method": method, "budget": float(budget), "samples": int(len(matched))}
                for key in metric_keys:
                    summary[key if key != "cumulative_cost" else "actual_bandwidth"] = float(np.mean([float(row.get(key, 0.0)) for row in matched]))
                rows.append(summary)
    return rows


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    budgets = sorted(parse_csv_floats(args.budgets))
    protocols = _parse_protocols(args.protocols)
    methods = _parse_methods(args.methods)
    archive = _load_archive(args.archive, int(args.max_samples))
    original_tam = np.asarray(archive["tam"], dtype=np.float32)
    original_mask = np.asarray(archive["mask"], dtype=np.float32)
    original_prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(original_tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(original_tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, args)
    action_rows = load_action_table(args.action_table)
    action_rows = [action for action in action_rows if int(action.sample_index) < int(original_tam.shape[0])]
    actions_by_sample = {}
    for action in action_rows:
        actions_by_sample.setdefault(int(action.sample_index), []).append(action)
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=original_prob.shape[1],
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(args.rf_num_slots),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    max_budget = float(max(budgets))
    filter_rows: list[dict] = []
    step_rows: list[dict] = []
    sample_rows: list[dict] = []
    for sample_index in range(original_tam.shape[0]):
        if args.progress and (sample_index == 0 or (sample_index + 1) % 10 == 0 or sample_index + 1 == original_tam.shape[0]):
            print(f"[stage_b1] sample {sample_index + 1}/{original_tam.shape[0]}", flush=True)
        clean_total = max(float(original_tam[sample_index].sum()), 1.0)
        raw_actions = actions_by_sample.get(int(sample_index), [])
        for protocol in protocols:
            b0_count = len(filter_protocol(raw_actions, str(protocol)))
            expanded_preview = _expanded_candidates(
                tam=original_tam[sample_index],
                mask=original_mask[sample_index],
                sample_index=int(sample_index),
                sample_id=str(sample_ids[sample_index]),
                label=int(labels[sample_index]),
                protocol=str(protocol),
                clean_total=float(clean_total),
                args=args,
                limit=int(args.max_static_candidates),
            )
            filter_rows.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": str(sample_ids[sample_index]),
                    "protocol": str(protocol),
                    "stage_b0_protocol_actions": int(b0_count),
                    "expanded_preview_actions": int(len(expanded_preview)),
                    "expanded_primary": int(sum(1 for action in expanded_preview if action.tier == "primary")),
                    "expanded_secondary": int(sum(1 for action in expanded_preview if action.tier == "secondary")),
                    "expanded_exploration": int(sum(1 for action in expanded_preview if action.tier == "exploration")),
                }
            )
            for method in methods:
                initial = _empty_state(original_prob[sample_index], original_tam[sample_index], int(labels[sample_index]))
                candidate_count = 0
                if method in {"stage_b0_static_efficiency", "expanded_static"}:
                    if method == "stage_b0_static_efficiency":
                        actions = _stage_b0_candidates(
                            raw_actions=raw_actions,
                            protocol=str(protocol),
                            tam=original_tam[sample_index],
                            clean_total=float(clean_total),
                            num_classes=int(original_prob.shape[1]),
                            limit=int(args.max_static_candidates),
                        )
                    else:
                        actions = _candidate_set_for_method(
                            method=str(method),
                            state=initial,
                            raw_actions=raw_actions,
                            protocol=str(protocol),
                            original_tam=original_tam[sample_index],
                            original_mask=original_mask[sample_index],
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            label=int(labels[sample_index]),
                            clean_total=float(clean_total),
                            num_classes=int(original_prob.shape[1]),
                            args=args,
                            limit=int(args.max_static_candidates),
                        )
                    candidate_count = len(actions)
                    state = _run_static_sequence(
                        method=str(method),
                        initial=initial,
                        actions=actions,
                        raw_trace=raw_rows[sample_index],
                        original_prob=original_prob[sample_index],
                        label=int(labels[sample_index]),
                        clean_total=float(clean_total),
                        max_budget=float(max_budget),
                        attacker=attacker,
                        args=args,
                        device=device,
                    )
                elif method in {"stage_b0_sequential_causal", "expanded_dynamic_greedy"}:
                    state = _run_dynamic_greedy(
                        method=str(method),
                        initial=initial,
                        raw_actions=raw_actions,
                        protocol=str(protocol),
                        original_tam=original_tam[sample_index],
                        original_mask=original_mask[sample_index],
                        raw_trace=raw_rows[sample_index],
                        original_prob=original_prob[sample_index],
                        label=int(labels[sample_index]),
                        clean_total=float(clean_total),
                        max_budget=float(max_budget),
                        sample_index=int(sample_index),
                        sample_id=str(sample_ids[sample_index]),
                        num_classes=int(original_prob.shape[1]),
                        attacker=attacker,
                        args=args,
                        device=device,
                    )
                    candidate_count = int(args.max_dynamic_candidates)
                else:
                    state = _run_dynamic_beam(
                        method=str(method),
                        initial=initial,
                        raw_actions=raw_actions,
                        protocol=str(protocol),
                        original_tam=original_tam[sample_index],
                        original_mask=original_mask[sample_index],
                        raw_trace=raw_rows[sample_index],
                        original_prob=original_prob[sample_index],
                        label=int(labels[sample_index]),
                        clean_total=float(clean_total),
                        max_budget=float(max_budget),
                        sample_index=int(sample_index),
                        sample_id=str(sample_ids[sample_index]),
                        num_classes=int(original_prob.shape[1]),
                        attacker=attacker,
                        args=args,
                        device=device,
                    )
                    candidate_count = int(args.max_beam_candidates)
                for row in state.history[1:]:
                    step_rows.append(
                        _annotate_row(
                            row,
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            protocol=str(protocol),
                            method=str(method),
                            budget=None,
                            max_budget=max_budget,
                            candidate_count=candidate_count,
                        )
                    )
                for budget in budgets:
                    row = _state_for_budget(state.history, float(budget))
                    sample_rows.append(
                        _annotate_row(
                            row,
                            sample_index=int(sample_index),
                            sample_id=str(sample_ids[sample_index]),
                            protocol=str(protocol),
                            method=str(method),
                            budget=float(budget),
                            max_budget=max_budget,
                            candidate_count=candidate_count,
                        )
                    )
    summary_rows = _summarize(sample_rows, protocols, methods, budgets)
    write_csv(output_dir / "expanded_filter_counts.csv", filter_rows)
    write_csv(output_dir / "expanded_step_results.csv", step_rows)
    write_csv(output_dir / "expanded_sample_results.csv", sample_rows)
    write_csv(output_dir / "expanded_summary.csv", summary_rows)
    write_json(
        output_dir / "expanded_oracle_summary.json",
        {
            "archive": str(args.archive),
            "action_table": str(args.action_table),
            "checkpoint": str(checkpoint),
            "attacker": str(args.attacker),
            "samples": int(original_tam.shape[0]),
            "budgets": [float(item) for item in budgets],
            "protocols": protocols,
            "methods": methods,
            "stopping_mode": str(args.stopping_mode),
            "objective": {
                "confidence": float(args.confidence_weight),
                "margin": float(args.margin_weight),
                "entropy": float(args.entropy_weight),
                "definition": "original predicted class is frozen: 0.3*(p_y0(original)-p_y0(current)) + 0.5*(original-class margin drop) + 0.2*normalized entropy gain; labels are post-hoc only.",
            },
            "generator": {
                "ratio": float(args.ratio),
                "max_windows": int(args.max_windows),
                "max_generated_actions": int(args.max_generated_actions),
                "max_pair_actions": int(args.max_pair_actions),
                "max_action_budget": float(args.max_action_budget),
                "max_local_rate_peak": int(args.max_local_rate_peak),
                "absolute_doses": str(args.absolute_doses),
                "relative_doses": str(args.relative_doses),
            },
            "outputs": {
                "filter_counts": str(output_dir / "expanded_filter_counts.csv"),
                "step_results": str(output_dir / "expanded_step_results.csv"),
                "sample_results": str(output_dir / "expanded_sample_results.csv"),
                "summary": str(output_dir / "expanded_summary.csv"),
            },
        },
    )
    print(f"Stage B1 expanded oracle complete: {output_dir}")


if __name__ == "__main__":
    main()
