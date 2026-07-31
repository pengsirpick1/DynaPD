"""Run Stage A additive intervention probing from DynaMask keypoint archives."""

from __future__ import annotations

import argparse
import sys
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
from dmmp.stage_a.additive_probe import (
    BASELINE_METHODS,
    DIRECTION_MODES,
    ActionSpec,
    CandidateWindow,
    apply_counts,
    allocate_integer,
    candidate_windows_for_sample,
    channel_cosine_similarity,
    channel_l1_gap,
    counts_for_action,
    empty_heatmaps,
    monotonic_violation_rows,
    nested_topr_audit,
    probability_metrics,
    update_sparse_best,
)
from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, parse_csv_floats, parse_csv_ints, parse_csv_strings


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"


def _default_checkpoint(attacker: str) -> str:
    return DEFAULT_FIXED_DF if str(attacker).lower() == "df" else DEFAULT_FIXED_RF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--faithfulness_sample_npz", default="")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--sanity_ratios", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--closing_kernel", type=int, default=5)
    parser.add_argument("--merge_gap", type=int, default=8)
    parser.add_argument("--min_window_length", type=int, default=1)
    parser.add_argument("--min_window_mass", type=float, default=0.0)
    parser.add_argument("--max_windows", type=int, default=6)
    parser.add_argument("--offsets", default="0,-4,-8,-16,-32")
    parser.add_argument("--doses", default="1,2,4,8,16,32")
    parser.add_argument("--direction_modes", default="out-only,in-only,both-equal,current-ratio")
    parser.add_argument("--budget_ratios", default="0.02,0.05,0.10")
    parser.add_argument("--budget_methods", default="random,random_window,early,magnitude,dynamask_same,dynamask_causal")
    parser.add_argument("--min_effective_top1_drop", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_top_actions", type=int, default=1)
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage_a_additive_probe_{args.attacker}_{stamp}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _method_values(value: str) -> list[str]:
    methods = [item.strip() for item in parse_csv_strings(value)]
    invalid = sorted(set(methods) - set(BASELINE_METHODS))
    if invalid:
        raise ValueError(f"Unknown additive budget methods: {invalid}")
    return methods


def _direction_modes(value: str) -> list[str]:
    modes = [item.strip() for item in parse_csv_strings(value)]
    invalid = sorted(set(modes) - set(DIRECTION_MODES))
    if invalid:
        raise ValueError(f"Unknown direction modes: {invalid}")
    return modes


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


def _load_faithfulness_sample(path: str) -> dict[str, np.ndarray]:
    if not path:
        return {}
    with np.load(path, allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}


def _write_sanity_outputs(
    output_dir: Path,
    *,
    mask: np.ndarray,
    sample_ids: np.ndarray,
    faithfulness: dict[str, np.ndarray],
    ratios: list[float],
) -> dict:
    nested_rows, nested_summary = nested_topr_audit(mask, ratios)
    write_csv(output_dir / "sanity_topr_nesting.csv", nested_rows)
    cosine = channel_cosine_similarity(mask)
    l1_gap = channel_l1_gap(mask)
    channel_rows = [
        {
            "sample_index": int(index),
            "sample_id": str(sample_ids[index]),
            "out_in_cosine": float(cosine[index]),
            "out_in_l1_gap": float(l1_gap[index]),
        }
        for index in range(mask.shape[0])
    ]
    write_csv(output_dir / "sanity_channel_similarity.csv", channel_rows)
    monotonic_rows, monotonic_summary = monotonic_violation_rows(faithfulness, np.asarray(faithfulness.get("ratios", ratios)), sample_ids)
    write_csv(output_dir / "sanity_monotonic_violations.csv", monotonic_rows)
    if faithfulness:
        curve_rows = []
        ratio_values = np.asarray(faithfulness["ratios"], dtype=np.float32)
        for ratio_index, ratio in enumerate(ratio_values):
            for sample in range(mask.shape[0]):
                curve_rows.append(
                    {
                        "sample_index": int(sample),
                        "sample_id": str(sample_ids[sample]),
                        "ratio": float(ratio),
                        "necessity_js_div": float(faithfulness["necessity_js_div"][ratio_index, sample]),
                        "necessity_top1_drop": float(faithfulness["necessity_top1_drop"][ratio_index, sample]),
                        "necessity_correct": float(faithfulness["necessity_correct"][ratio_index, sample]),
                        "sufficiency_js_div": float(faithfulness["sufficiency_js_div"][ratio_index, sample]),
                        "sufficiency_top1_preservation": float(faithfulness["sufficiency_top1_preservation"][ratio_index, sample]),
                    }
                )
        write_csv(output_dir / "sanity_sample_deletion_curves.csv", curve_rows)
    summary = {
        "topr_nesting": nested_summary,
        "channel_cosine_mean": float(np.mean(cosine)) if cosine.size else 0.0,
        "channel_cosine_median": float(np.median(cosine)) if cosine.size else 0.0,
        "channel_cosine_q10": float(np.quantile(cosine, 0.10)) if cosine.size else 0.0,
        "channel_cosine_q90": float(np.quantile(cosine, 0.90)) if cosine.size else 0.0,
        "channel_l1_gap_mean": float(np.mean(l1_gap)) if l1_gap.size else 0.0,
        "monotonic": monotonic_summary,
        "mask_parameter_source_audit": {
            "archive_mask_shape": [int(item) for item in mask.shape],
            "expected_native_shape": "N x 2 x W",
            "code_path": "dmmp/stage_a/dyn_mask.py",
            "implementation_note": "mask_logits is initialized with torch.full_like(x), so optimized parameters have the same independent [B, 2, W] shape as the TAM input.",
        },
        "keep_only_top1_preservation_definition": "p_keep(original_top1_class) / p_original(original_top1_class); this is not merely top1-class identity preservation.",
    }
    write_json(output_dir / "sanity_summary.json", summary)
    return summary


def _dummy_count(tam_row: np.ndarray, budget_ratio: float) -> int:
    original = max(1, int(round(float(np.asarray(tam_row, dtype=np.float32).sum()))))
    return max(1, int(round(original * float(budget_ratio))))


def _flat_counts_from_positions(width: int, positions: np.ndarray, total: int) -> np.ndarray:
    counts = np.zeros((2, int(width)), dtype=np.int32)
    flat_positions = np.asarray(positions, dtype=np.int64).reshape(-1)
    if int(total) <= 0 or flat_positions.size == 0:
        return counts
    chosen = np.resize(flat_positions, int(total))
    for flat in chosen:
        direction = int(flat) // int(width)
        slot = int(flat) % int(width)
        counts[direction, slot] += 1
    return counts


def _random_counts(width: int, total: int, rng: np.random.Generator) -> np.ndarray:
    positions = rng.integers(0, 2 * int(width), size=max(1, int(total)))
    return _flat_counts_from_positions(int(width), positions, int(total))


def _random_window_counts(width: int, total: int, window_length: int, rng: np.random.Generator) -> np.ndarray:
    length = max(1, min(int(width), int(window_length)))
    start = int(rng.integers(0, max(1, int(width) - length + 1)))
    slots = np.arange(start, start + length, dtype=np.int64)
    directions = rng.integers(0, 2, size=max(1, int(total)))
    repeated_slots = np.resize(slots, max(1, int(total)))
    return _flat_counts_from_positions(int(width), directions * int(width) + repeated_slots, int(total))


def _early_counts(width: int, total: int) -> np.ndarray:
    positions = np.asarray([direction * int(width) + slot for slot in range(int(width)) for direction in range(2)], dtype=np.int64)
    return _flat_counts_from_positions(int(width), positions, int(total))


def _magnitude_counts(tam: np.ndarray, total: int) -> np.ndarray:
    width = int(tam.shape[-1])
    order = np.argsort(-np.abs(np.asarray(tam, dtype=np.float32)).reshape(-1), kind="mergesort")
    return _flat_counts_from_positions(width, order, int(total))


def _dynamask_budget_counts(
    tam: np.ndarray,
    windows: list[CandidateWindow],
    action_infos: list[tuple[dict, ActionSpec]],
    total: int,
    *,
    same_position_only: bool,
) -> np.ndarray:
    width = int(tam.shape[-1])
    counts = np.zeros((2, width), dtype=np.int32)
    selected: list[tuple[dict, ActionSpec]] = []
    for window in windows:
        matches = [
            item
            for item in action_infos
            if int(item[0]["window_id"]) == int(window.window_id)
            and (not same_position_only or int(item[0]["offset"]) == 0)
        ]
        if not matches:
            continue
        selected.append(max(matches, key=lambda item: float(item[0]["efficiency_top1_drop"])))
    if not selected or int(total) <= 0:
        return counts
    weights = np.asarray(
        [max(float(row["efficiency_top1_drop"]), 1e-6) * max(float(row["mask_mass"]), 1e-6) for row, _spec in selected],
        dtype=np.float64,
    )
    allocations = allocate_integer(int(total), weights)
    for allocation, (row, _spec) in zip(allocations, selected):
        if int(allocation) <= 0:
            continue
        window = next(item for item in windows if int(item.window_id) == int(row["window_id"]))
        spec = counts_for_action(
            tam,
            window,
            offset=int(row["offset"]),
            dose=int(allocation),
            direction_mode=str(row["direction_mode"]),
        )
        counts += spec.counts
    return counts.astype(np.int32)


def _counts_for_budget_method(
    method: str,
    tam: np.ndarray,
    windows: list[CandidateWindow],
    action_infos: list[tuple[dict, ActionSpec]],
    budget_count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    width = int(tam.shape[-1])
    if method == "random":
        return _random_counts(width, int(budget_count), rng)
    if method == "random_window":
        length = int(np.median([window.length for window in windows])) if windows else max(1, width // 50)
        return _random_window_counts(width, int(budget_count), length, rng)
    if method == "early":
        return _early_counts(width, int(budget_count))
    if method == "magnitude":
        return _magnitude_counts(tam, int(budget_count))
    if method == "dynamask_same":
        return _dynamask_budget_counts(tam, windows, action_infos, int(budget_count), same_position_only=True)
    if method == "dynamask_causal":
        return _dynamask_budget_counts(tam, windows, action_infos, int(budget_count), same_position_only=False)
    raise ValueError(f"Unsupported budget method={method!r}")


def _rows_from_metrics(metrics: dict[str, np.ndarray], index: int) -> dict:
    return {
        "accuracy": float(metrics["accuracy"][index]),
        "flip": float(metrics["flip"][index]),
        "js_div": float(metrics["js_div"][index]),
        "top1_drop": float(metrics["top1_drop"][index]),
        "margin_drop": float(metrics["margin_drop"][index]),
        "entropy_gain": float(metrics["entropy_gain"][index]),
        "evaluated_pred": int(metrics["evaluated_pred"][index]),
    }


def _action_row_base(spec: ActionSpec, sample_id: str, true_label: int, original_packets: int) -> dict:
    dummy = int(spec.counts.sum())
    overhead = float(dummy / max(int(original_packets), 1))
    return {
        "sample_index": int(spec.sample_index),
        "sample_id": str(sample_id),
        "true_label": int(true_label),
        "window_id": int(spec.window_id),
        "affected_direction": "out" if int(spec.affected_direction) == 0 else "in",
        "affected_start": int(spec.affected_start),
        "affected_end": int(spec.affected_end),
        "affected_center": int(spec.affected_center),
        "insert_start": int(spec.insert_start),
        "insert_end": int(spec.insert_end),
        "insert_center": int(spec.insert_center),
        "offset": int(spec.offset),
        "dose": int(spec.dose),
        "direction_mode": str(spec.direction_mode),
        "mask_mass": float(spec.mask_mass),
        "dummy_packets": dummy,
        "bandwidth_overhead": overhead,
        "latency_overhead": 0.0,
        "local_rate_peak": int(spec.local_rate_peak),
        "causal_violation": int(bool(spec.causal_violation)),
        "allowed_violation_count": int(spec.allowed_violation_count),
        "requires_incoming_capability": int(bool(spec.requires_incoming_capability)),
        "renderer_mode": "tam_space_screening",
    }


def _add_efficiency(row: dict) -> None:
    overhead = max(float(row.get("bandwidth_overhead", 0.0)), 1e-8)
    row["efficiency_top1_drop"] = float(row.get("top1_drop", 0.0)) / overhead
    row["efficiency_js_div"] = float(row.get("js_div", 0.0)) / overhead
    row["efficiency_margin_drop"] = float(row.get("margin_drop", 0.0)) / overhead


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


def _run_renderer_check(
    *,
    args: argparse.Namespace,
    attacker,
    renderer_items: list[tuple[int, ActionSpec, dict]],
    source_indices: np.ndarray,
    original_prob: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    output_dir: Path,
    device,
) -> dict:
    if int(args.renderer_top_actions) <= 0 or not renderer_items:
        return {"enabled": False, "rows": 0}
    try:
        raw_rows = _load_raw_rows(str(args.data_root), source_indices, args)
        clean_rows, templates, meta_rows, seeds = [], [], [], []
        for ordinal, (sample_index, spec, row) in enumerate(renderer_items):
            clean_rows.append(raw_rows[int(sample_index)])
            templates.append(
                PaddingTemplate(
                    counts=spec.counts,
                    target_n_pad=int(spec.counts.sum()),
                    actual_n_pad=int(spec.counts.sum()),
                    target_bandwidth=float(row["bandwidth_overhead"]),
                    metadata={"method": "stage_a_additive_top_action_renderer_check"},
                )
            )
            meta_rows.append(row)
            seeds.append(int(args.seed) + int(sample_index) * 1009 + ordinal)
        defended_traces, _origins, stats = render_batch_variable(
            np.asarray(clean_rows, dtype=np.float32),
            templates,
            seeds=seeds,
            strategy=str(args.renderer_strategy),
            coordinate=str(args.renderer_coordinate),
            coordinate_length=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
        )
        padded = crop_or_pad_2d(np.asarray([crop_or_pad_2d(trace, int(args.max_trace_length))[0] for trace in defended_traces]), int(args.max_trace_length))
        tam = build_rf_tam_input(
            padded,
            max_len=int(args.max_trace_length),
            max_load_time=float(args.max_load_time),
            num_slots=int(args.rf_num_slots),
        )
        probs = predict_probabilities(attacker, tam, device=device, batch_size=int(args.batch_size))
        sample_idx = np.asarray([int(item[0]) for item in renderer_items], dtype=np.int64)
        metrics = probability_metrics(original_prob[sample_idx], probs, labels[sample_idx])
        rows = []
        for index, (sample_index, _spec, base_row) in enumerate(renderer_items):
            row = {
                "sample_index": int(sample_index),
                "sample_id": str(sample_ids[int(sample_index)]),
                "window_id": int(base_row["window_id"]),
                "offset": int(base_row["offset"]),
                "dose": int(base_row["dose"]),
                "direction_mode": str(base_row["direction_mode"]),
                "tam_screen_top1_drop": float(base_row["top1_drop"]),
                "tam_screen_js_div": float(base_row["js_div"]),
                "renderer_bandwidth_overhead": float(stats["raw_bandwidth"][index]),
                "renderer_original_retention": float(stats["raw_real_packet_retention"][index]),
                "renderer_raw_length": int(stats["raw_lengths"][index]),
                **_rows_from_metrics(metrics, index),
            }
            _add_efficiency(row)
            rows.append(row)
        write_csv(output_dir / "renderer_top_action_results.csv", rows)
        return {
            "enabled": True,
            "rows": int(len(rows)),
            "mean_top1_drop": float(np.mean([row["top1_drop"] for row in rows])) if rows else 0.0,
            "mean_js_div": float(np.mean([row["js_div"] for row in rows])) if rows else 0.0,
            "mean_bandwidth": float(np.mean([row["renderer_bandwidth_overhead"] for row in rows])) if rows else 0.0,
        }
    except Exception as exc:
        write_json(output_dir / "renderer_top_action_error.json", {"error": repr(exc)})
        return {"enabled": True, "rows": 0, "error": repr(exc)}


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    ratios = parse_csv_floats(args.sanity_ratios)
    offsets = parse_csv_ints(args.offsets)
    doses = parse_csv_ints(args.doses)
    direction_modes = _direction_modes(args.direction_modes)
    budget_ratios = parse_csv_floats(args.budget_ratios)
    budget_methods = _method_values(args.budget_methods)
    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    original_prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    faithfulness = _load_faithfulness_sample(args.faithfulness_sample_npz)
    if faithfulness and int(args.max_samples) > 0:
        for key, value in list(faithfulness.items()):
            arr = np.asarray(value)
            if key == "ratios":
                continue
            if arr.ndim >= 2 and arr.shape[1] >= tam.shape[0]:
                faithfulness[key] = arr[:, : tam.shape[0]]
            elif arr.ndim == 1 and arr.shape[0] >= tam.shape[0]:
                faithfulness[key] = arr[: tam.shape[0]]
    sanity_summary = _write_sanity_outputs(
        output_dir,
        mask=mask,
        sample_ids=sample_ids,
        faithfulness=faithfulness,
        ratios=ratios,
    )
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
    rng = np.random.default_rng(int(args.seed))
    candidate_rows: list[dict] = []
    action_rows: list[dict] = []
    best_action_rows: list[dict] = []
    budget_rows: list[dict] = []
    heatmaps = empty_heatmaps(tam.shape[0], tam.shape[-1])
    best_offset_score = np.full((tam.shape[0], 2, tam.shape[-1]), np.nan, dtype=np.float32)
    renderer_items: list[tuple[int, ActionSpec, dict]] = []
    decision_criticality = mask.astype(np.float32)
    for sample_index in range(tam.shape[0]):
        if args.progress and (sample_index == 0 or (sample_index + 1) % 10 == 0 or sample_index + 1 == tam.shape[0]):
            print(f"[additive] sample {sample_index + 1}/{tam.shape[0]}", flush=True)
        windows = candidate_windows_for_sample(
            mask[sample_index],
            ratio=float(args.ratio),
            closing_kernel=int(args.closing_kernel),
            merge_gap=int(args.merge_gap),
            min_length=int(args.min_window_length),
            min_mass=float(args.min_window_mass),
            max_windows=int(args.max_windows),
            sample_index=int(sample_index),
        )
        for window in windows:
            candidate_rows.append(
                {
                    "sample_index": int(sample_index),
                    "sample_id": str(sample_ids[sample_index]),
                    "window_id": int(window.window_id),
                    "direction": window.direction_name,
                    "start": int(window.start),
                    "end": int(window.end),
                    "center": int(window.center),
                    "length": int(window.length),
                    "mask_mass": float(window.mask_mass),
                }
            )
        specs: list[ActionSpec] = []
        for window in windows:
            for offset in offsets:
                for dose in doses:
                    for mode in direction_modes:
                        specs.append(counts_for_action(tam[sample_index], window, offset=int(offset), dose=int(dose), direction_mode=str(mode)))
        action_infos: list[tuple[dict, ActionSpec]] = []
        if specs:
            variants = np.stack([apply_counts(tam[sample_index], spec.counts) for spec in specs], axis=0)
            probs = predict_probabilities(attacker, variants, device=device, batch_size=int(args.batch_size))
            metrics = probability_metrics(
                np.repeat(original_prob[sample_index : sample_index + 1], len(specs), axis=0),
                probs,
                np.full(len(specs), int(labels[sample_index]), dtype=np.int64),
            )
            original_packets = max(1, int(round(float(tam[sample_index].sum()))))
            for spec_index, spec in enumerate(specs):
                row = _action_row_base(spec, sample_ids[sample_index], int(labels[sample_index]), original_packets)
                row.update(_rows_from_metrics(metrics, spec_index))
                _add_efficiency(row)
                action_rows.append(row)
                action_infos.append((row, spec))
                for direction in range(2):
                    if int(spec.counts[direction].sum()) > 0:
                        update_sparse_best(
                            heatmaps["additive_efficiency"],
                            sample_index,
                            direction,
                            int(spec.insert_center),
                            float(row["efficiency_top1_drop"]),
                            higher_is_better=True,
                        )
                        if float(row["top1_drop"]) >= float(args.min_effective_top1_drop):
                            update_sparse_best(
                                heatmaps["minimum_effective_budget"],
                                sample_index,
                                direction,
                                int(spec.insert_center),
                                float(row["bandwidth_overhead"]),
                                higher_is_better=False,
                            )
            by_window: dict[int, tuple[dict, ActionSpec]] = {}
            for row, spec in action_infos:
                key = int(row["window_id"])
                if key not in by_window or float(row["efficiency_top1_drop"]) > float(by_window[key][0]["efficiency_top1_drop"]):
                    by_window[key] = (row, spec)
            for row, spec in by_window.values():
                best_action_rows.append(row)
                score_position = int(row["affected_center"])
                direction = 0 if row["affected_direction"] == "out" else 1
                old = best_offset_score[sample_index, direction, score_position]
                if np.isnan(old) or float(row["efficiency_top1_drop"]) > float(old):
                    best_offset_score[sample_index, direction, score_position] = float(row["efficiency_top1_drop"])
                    heatmaps["best_causal_offset"][sample_index, direction, score_position] = float(row["offset"])
                    heatmaps["best_insert_position"][sample_index, direction, score_position] = float(row["insert_center"])
            ranked_for_renderer = sorted(action_infos, key=lambda item: float(item[0]["efficiency_top1_drop"]), reverse=True)
            for row, spec in ranked_for_renderer[: max(0, int(args.renderer_top_actions))]:
                renderer_items.append((sample_index, spec, row))
        budget_variants = []
        budget_meta = []
        for budget_ratio in budget_ratios:
            target = _dummy_count(tam[sample_index], float(budget_ratio))
            for method in budget_methods:
                counts = _counts_for_budget_method(str(method), tam[sample_index], windows, action_infos, int(target), rng)
                budget_variants.append(apply_counts(tam[sample_index], counts))
                budget_meta.append((float(budget_ratio), str(method), int(target), counts))
        if budget_variants:
            probs = predict_probabilities(attacker, np.stack(budget_variants, axis=0), device=device, batch_size=int(args.batch_size))
            metrics = probability_metrics(
                np.repeat(original_prob[sample_index : sample_index + 1], len(budget_variants), axis=0),
                probs,
                np.full(len(budget_variants), int(labels[sample_index]), dtype=np.int64),
            )
            original_packets = max(1, int(round(float(tam[sample_index].sum()))))
            for index, (budget_ratio, method, target, counts) in enumerate(budget_meta):
                actual_dummy = int(counts.sum())
                row = {
                    "sample_index": int(sample_index),
                    "sample_id": str(sample_ids[sample_index]),
                    "true_label": int(labels[sample_index]),
                    "method": str(method),
                    "target_budget_ratio": float(budget_ratio),
                    "target_dummy_packets": int(target),
                    "dummy_packets": int(actual_dummy),
                    "bandwidth_overhead": float(actual_dummy / max(original_packets, 1)),
                    "latency_overhead": 0.0,
                    "local_rate_peak": int(counts.sum(axis=0).max()) if counts.size else 0,
                    "requires_incoming_capability": int(bool(counts[1].sum() > 0)),
                    "renderer_mode": "tam_space_screening",
                    **_rows_from_metrics(metrics, index),
                }
                _add_efficiency(row)
                budget_rows.append(row)
    write_csv(output_dir / "candidate_windows.csv", candidate_rows)
    write_csv(output_dir / "action_results.csv", action_rows)
    write_csv(output_dir / "sample_best_actions.csv", best_action_rows)
    write_csv(output_dir / "budget_results.csv", budget_rows)
    np.savez_compressed(
        output_dir / "additive_heatmaps.npz",
        sample_ids=sample_ids,
        labels=labels,
        decision_criticality=decision_criticality,
        additive_efficiency=heatmaps["additive_efficiency"],
        minimum_effective_budget=heatmaps["minimum_effective_budget"],
        best_causal_offset=heatmaps["best_causal_offset"],
        best_insert_position=heatmaps["best_insert_position"],
    )
    renderer_summary = _run_renderer_check(
        args=args,
        attacker=attacker,
        renderer_items=renderer_items,
        source_indices=source_indices,
        original_prob=original_prob,
        labels=labels,
        sample_ids=sample_ids,
        output_dir=output_dir,
        device=device,
    )
    budget_summary = []
    for method in budget_methods:
        for budget_ratio in budget_ratios:
            matched = [row for row in budget_rows if row["method"] == method and abs(float(row["target_budget_ratio"]) - float(budget_ratio)) < 1e-9]
            if not matched:
                continue
            budget_summary.append(
                {
                    "method": str(method),
                    "budget_ratio": float(budget_ratio),
                    "accuracy": float(np.mean([row["accuracy"] for row in matched])),
                    "flip": float(np.mean([row["flip"] for row in matched])),
                    "js_div": float(np.mean([row["js_div"] for row in matched])),
                    "top1_drop": float(np.mean([row["top1_drop"] for row in matched])),
                    "margin_drop": float(np.mean([row["margin_drop"] for row in matched])),
                    "entropy_gain": float(np.mean([row["entropy_gain"] for row in matched])),
                    "efficiency_top1_drop": float(np.mean([row["efficiency_top1_drop"] for row in matched])),
                    "actual_bandwidth": float(np.mean([row["bandwidth_overhead"] for row in matched])),
                }
            )
    write_csv(output_dir / "budget_summary.csv", budget_summary)
    write_json(
        output_dir / "additive_probe_summary.json",
        {
            "archive": str(args.archive),
            "faithfulness_sample_npz": str(args.faithfulness_sample_npz),
            "checkpoint": str(checkpoint),
            "attacker": str(args.attacker),
            "samples": int(tam.shape[0]),
            "width": int(tam.shape[-1]),
            "ratio": float(args.ratio),
            "offsets": [int(item) for item in offsets],
            "doses": [int(item) for item in doses],
            "direction_modes": direction_modes,
            "budget_ratios": [float(item) for item in budget_ratios],
            "budget_methods": budget_methods,
            "candidate_windows": int(len(candidate_rows)),
            "action_rows": int(len(action_rows)),
            "budget_rows": int(len(budget_rows)),
            "sanity": sanity_summary,
            "budget_summary": budget_summary,
            "renderer_top_action_check": renderer_summary,
            "notes": {
                "tam_space_screening": "All action_results and budget_results directly add dummy packet counts in 2x1800 TAM space.",
                "renderer_check": "renderer_top_action_results, when present, applies the top TAM-space action per sample through PaddingTemplate + render_batch_variable, rebuilds 2x1800 TAM, and re-evaluates RF.",
                "incoming_capability": "Rows with requires_incoming_capability=1 require a server/relay-side deployment assumption and should not be mixed with client-only actions.",
            },
        },
    )
    print(f"Stage A additive probing complete: {output_dir}")


if __name__ == "__main__":
    main()
