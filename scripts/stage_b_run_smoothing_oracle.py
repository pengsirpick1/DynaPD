"""Run Stage B2-S keypoint-guided causal smoothing oracle."""

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
from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.objectives import probability_metrics
from dmmp.stage_b.smoothing import (
    SmoothingConfig,
    add_only_dummy_counts,
    causal_delay_trace,
    keypoint_windows,
    local_reductions,
    noncausal_symmetric_tam,
    trace_to_tam,
)
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR, parse_csv_floats, parse_csv_ints, parse_csv_strings


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"
METHODS = (
    "clean",
    "noncausal_symmetric_oracle",
    "causal_delay_smoothing",
    "direct_same_position_dummy",
    "add_only_future_flattening",
    "hybrid_delay_dummy",
)


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
    parser.add_argument("--methods", default="clean,noncausal_symmetric_oracle,causal_delay_smoothing,direct_same_position_dummy,add_only_future_flattening,hybrid_delay_dummy")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--closing_kernel", type=int, default=5)
    parser.add_argument("--merge_gap", type=int, default=8)
    parser.add_argument("--lengths", default="4,8,16,32,64")
    parser.add_argument("--rhos", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--dummy_budgets", default="0.02,0.05,0.08,0.10")
    parser.add_argument("--max_delays", default="0,4,8,16,32")
    parser.add_argument("--quick_grid", action="store_true")
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
    name = args.run_name or f"stage_b2s_smoothing_oracle_{args.attacker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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


def _configs(args: argparse.Namespace) -> list[SmoothingConfig]:
    methods = parse_csv_strings(args.methods)
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    lengths = parse_csv_ints("8,16,32" if args.quick_grid else args.lengths)
    rhos = parse_csv_floats("0.5,0.75,1.0" if args.quick_grid else args.rhos)
    budgets = parse_csv_floats("0.02,0.05,0.10" if args.quick_grid else args.dummy_budgets)
    delays = parse_csv_ints("4,8,16,32" if args.quick_grid else args.max_delays)
    configs: list[SmoothingConfig] = []
    if "clean" in methods:
        configs.append(SmoothingConfig("clean", 0, 0.0, 0.0, 0))
    if "noncausal_symmetric_oracle" in methods:
        for length in lengths:
            for rho in rhos:
                configs.append(SmoothingConfig("noncausal_symmetric_oracle", int(length), float(rho), 0.0, 0))
    if "causal_delay_smoothing" in methods:
        for length in lengths:
            for rho in rhos:
                for delay in delays:
                    if int(delay) <= 0:
                        continue
                    configs.append(SmoothingConfig("causal_delay_smoothing", int(length), float(rho), 0.0, int(delay)))
    if "direct_same_position_dummy" in methods:
        for budget in budgets:
            configs.append(SmoothingConfig("direct_same_position_dummy", 0, 1.0, float(budget), 0))
    if "add_only_future_flattening" in methods:
        for length in lengths:
            for rho in rhos:
                for budget in budgets:
                    configs.append(SmoothingConfig("add_only_future_flattening", int(length), float(rho), float(budget), 0))
    if "hybrid_delay_dummy" in methods:
        for length in lengths:
            for rho in rhos:
                for budget in budgets:
                    for delay in delays:
                        if int(delay) <= 0:
                            continue
                        configs.append(SmoothingConfig("hybrid_delay_dummy", int(length), float(rho), float(budget), int(delay)))
    return configs


def _config_id(config: SmoothingConfig) -> str:
    return f"{config.method}|L={config.window_length}|rho={config.rho:g}|B={config.dummy_budget:g}|D={config.max_delay}"


def _render_dummy_batch(
    *,
    raw_rows: list[np.ndarray],
    dummy_counts: list[np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    if not raw_rows:
        return np.zeros((0, 2, int(args.rf_num_slots)), dtype=np.float32), np.zeros((0, int(args.max_trace_length)), dtype=np.float32), []
    padded_clean = np.vstack([crop_or_pad_2d(row, int(args.max_trace_length))[0] for row in raw_rows]).astype(np.float32)
    templates = [
        PaddingTemplate(
            counts=np.asarray(counts, dtype=np.int32),
            target_n_pad=int(np.asarray(counts).sum()),
            actual_n_pad=int(np.asarray(counts).sum()),
            target_bandwidth=0.0,
            metadata={"method": "stage_b2s_smoothing"},
        )
        for counts in dummy_counts
    ]
    traces, _origins, stats = render_batch_variable(
        padded_clean,
        templates,
        seed=int(args.seed),
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
    stats_rows = [
        {
            "raw_bandwidth": float(stats["raw_bandwidth"][idx]),
            "raw_real_packet_retention": float(stats["raw_real_packet_retention"][idx]),
            "raw_length": int(stats["raw_lengths"][idx]),
        }
        for idx in range(len(raw_rows))
    ]
    return tam.astype(np.float32), padded.astype(np.float32), stats_rows


def _evaluate_config(
    *,
    config: SmoothingConfig,
    tam: np.ndarray,
    mask: np.ndarray,
    raw_rows: np.ndarray,
    labels: np.ndarray,
    sample_ids: np.ndarray,
    original_prob: np.ndarray,
    attacker,
    args: argparse.Namespace,
    device,
) -> tuple[list[dict], np.ndarray]:
    n = int(tam.shape[0])
    width = int(tam.shape[2])
    defended_tams: list[np.ndarray] = []
    transformed_rows: list[np.ndarray] = []
    dummy_counts_rows: list[np.ndarray] = []
    stats_rows: list[dict] = []
    windows_by_sample = [
        keypoint_windows(
            mask[index],
            ratio=float(args.ratio),
            max_windows=int(args.max_windows),
            closing_kernel=int(args.closing_kernel),
            merge_gap=int(args.merge_gap),
            sample_index=int(index),
        )
        for index in range(n)
    ]
    delayed_for_hybrid: list[np.ndarray] = []
    hybrid_dummy: list[np.ndarray] = []
    hybrid_meta: list[dict] = []
    for index in range(n):
        windows = windows_by_sample[index]
        clean_total = max(float(tam[index].sum()), 1.0)
        if config.method == "clean":
            defended = np.asarray(tam[index], dtype=np.float32).copy()
            defended_tams.append(defended)
            stats_rows.append({"raw_bandwidth": 0.0, "raw_real_packet_retention": 1.0, "raw_length": int(np.count_nonzero(raw_rows[index]))})
        elif config.method == "noncausal_symmetric_oracle":
            defended = noncausal_symmetric_tam(tam[index], windows, length=int(config.window_length), rho=float(config.rho))
            defended_tams.append(defended)
            stats_rows.append({"raw_bandwidth": 0.0, "raw_real_packet_retention": 1.0, "raw_length": int(np.count_nonzero(raw_rows[index]))})
        elif config.method == "causal_delay_smoothing":
            trace, avg_delay, max_delay = causal_delay_trace(
                raw_rows[index],
                windows,
                width=width,
                length=int(config.window_length),
                rho=float(config.rho),
                max_delay=int(config.max_delay),
                max_load_time=float(args.max_load_time),
            )
            defended = trace_to_tam(trace, width=width, max_load_time=float(args.max_load_time))
            defended_tams.append(defended)
            stats_rows.append(
                {
                    "raw_bandwidth": 0.0,
                    "raw_real_packet_retention": 1.0,
                    "raw_length": int(trace.size),
                    "average_delay_bins": float(avg_delay),
                    "maximum_delay_bins": int(max_delay),
                }
            )
        elif config.method in {"direct_same_position_dummy", "add_only_future_flattening"}:
            direct = config.method == "direct_same_position_dummy"
            counts = add_only_dummy_counts(
                tam[index],
                windows,
                length=max(1, int(config.window_length)),
                rho=float(config.rho),
                budget=float(config.dummy_budget),
                clean_total=clean_total,
                direct_same_position=bool(direct),
            )
            transformed_rows.append(raw_rows[index])
            dummy_counts_rows.append(counts)
        elif config.method == "hybrid_delay_dummy":
            trace, avg_delay, max_delay = causal_delay_trace(
                raw_rows[index],
                windows,
                width=width,
                length=int(config.window_length),
                rho=float(config.rho),
                max_delay=int(config.max_delay),
                max_load_time=float(args.max_load_time),
            )
            delayed_tam = trace_to_tam(trace, width=width, max_load_time=float(args.max_load_time))
            counts = add_only_dummy_counts(
                delayed_tam,
                windows,
                length=max(1, int(config.window_length)),
                rho=float(config.rho),
                budget=float(config.dummy_budget),
                clean_total=clean_total,
                direct_same_position=False,
            )
            delayed_for_hybrid.append(trace)
            hybrid_dummy.append(counts)
            hybrid_meta.append({"average_delay_bins": float(avg_delay), "maximum_delay_bins": int(max_delay)})
        else:
            raise ValueError(f"Unsupported method={config.method!r}")
    if transformed_rows:
        rendered_tam, _rendered_rows, render_stats = _render_dummy_batch(raw_rows=transformed_rows, dummy_counts=dummy_counts_rows, args=args)
        defended_tams.extend([rendered_tam[idx] for idx in range(rendered_tam.shape[0])])
        stats_rows.extend(render_stats)
    if delayed_for_hybrid:
        rendered_tam, _rendered_rows, render_stats = _render_dummy_batch(raw_rows=delayed_for_hybrid, dummy_counts=hybrid_dummy, args=args)
        defended_tams.extend([rendered_tam[idx] for idx in range(rendered_tam.shape[0])])
        for render_stat, delay_stat in zip(render_stats, hybrid_meta):
            merged = dict(render_stat)
            merged.update(delay_stat)
            stats_rows.append(merged)
    defended = np.stack(defended_tams, axis=0).astype(np.float32)
    probs = predict_probabilities(attacker, defended, device=device, batch_size=int(args.batch_size))
    metrics = probability_metrics(original_prob, probs, labels)
    rows: list[dict] = []
    for index in range(n):
        windows = windows_by_sample[index]
        var_reduction, grad_reduction = local_reductions(
            tam[index],
            defended[index],
            windows,
            length=max(1, int(config.window_length)),
            causal=config.method != "noncausal_symmetric_oracle",
        )
        stats = stats_rows[index]
        row = {
            "sample_index": int(index),
            "sample_id": str(sample_ids[index]),
            "method": str(config.method),
            "config_id": _config_id(config),
            "window_length": int(config.window_length),
            "rho": float(config.rho),
            "dummy_budget": float(config.dummy_budget),
            "max_delay_target": int(config.max_delay),
            "keypoint_window_count": int(len(windows)),
            "accuracy": float(metrics["accuracy"][index]),
            "flip": float(metrics["flip"][index]),
            "original_top1_drop": float(metrics["original_top1_drop"][index]),
            "original_class_probability": float(metrics["original_class_probability"][index]),
            "original_class_margin": float(metrics["original_class_margin"][index]),
            "original_class_margin_drop": float(metrics["original_class_margin_drop"][index]),
            "current_top1_margin": float(metrics["current_top1_margin"][index]),
            "entropy_gain": float(metrics["entropy_gain"][index]),
            "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][index]),
            "js_div": float(metrics["js_div"][index]),
            "original_class_utility": float(metrics["original_class_utility"][index]),
            "actual_bandwidth": float(stats.get("raw_bandwidth", 0.0)),
            "average_delay_bins": float(stats.get("average_delay_bins", 0.0)),
            "maximum_delay_bins": int(stats.get("maximum_delay_bins", 0)),
            "raw_real_packet_retention": float(stats.get("raw_real_packet_retention", 1.0)),
            "raw_length": int(stats.get("raw_length", 0)),
            "local_variance_reduction": float(var_reduction),
            "local_gradient_reduction": float(grad_reduction),
            "renderer_mode": (
                "tam_noncausal" if config.method == "noncausal_symmetric_oracle" else
                "delay_reschedule" if config.method == "causal_delay_smoothing" else
                "clean" if config.method == "clean" else
                "padding_template"
            ),
            "renderer_consistency": float(stats.get("raw_real_packet_retention", 1.0)),
        }
        rows.append(row)
    return rows, defended


def _summarize(sample_rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for row in sample_rows:
        groups.setdefault(str(row["config_id"]), []).append(row)
    metric_keys = [
        "accuracy",
        "flip",
        "original_top1_drop",
        "original_class_probability",
        "original_class_margin",
        "original_class_margin_drop",
        "current_top1_margin",
        "entropy_gain",
        "normalized_entropy_gain",
        "js_div",
        "original_class_utility",
        "actual_bandwidth",
        "average_delay_bins",
        "maximum_delay_bins",
        "raw_real_packet_retention",
        "local_variance_reduction",
        "local_gradient_reduction",
        "renderer_consistency",
    ]
    rows = []
    for config_id, matched in sorted(groups.items()):
        first = matched[0]
        row = {
            "method": first["method"],
            "config_id": config_id,
            "window_length": int(first["window_length"]),
            "rho": float(first["rho"]),
            "dummy_budget": float(first["dummy_budget"]),
            "max_delay_target": int(first["max_delay_target"]),
            "samples": int(len(matched)),
        }
        for key in metric_keys:
            row[key] = float(np.mean([float(item[key]) for item in matched]))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    archive = _load_archive(args.archive, int(args.max_samples))
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    original_prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, args)
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
    configs = _configs(args)
    sample_rows: list[dict] = []
    best_tam = None
    best_config_id = ""
    best_accuracy = 1e9
    best_utility = -1e9
    for index, config in enumerate(configs):
        if args.progress and (index == 0 or (index + 1) % 10 == 0 or index + 1 == len(configs)):
            print(f"[stage_b2s] config {index + 1}/{len(configs)} {_config_id(config)}", flush=True)
        rows, defended = _evaluate_config(
            config=config,
            tam=tam,
            mask=mask,
            raw_rows=raw_rows,
            labels=labels,
            sample_ids=sample_ids,
            original_prob=original_prob,
            attacker=attacker,
            args=args,
            device=device,
        )
        sample_rows.extend(rows)
        accuracy = float(np.mean([row["accuracy"] for row in rows]))
        utility = float(np.mean([row["original_class_utility"] for row in rows]))
        if accuracy < best_accuracy - 1e-12 or (abs(accuracy - best_accuracy) <= 1e-12 and utility > best_utility):
            best_accuracy = accuracy
            best_utility = utility
            best_tam = defended
            best_config_id = _config_id(config)
    summary_rows = _summarize(sample_rows)
    write_csv(output_dir / "smoothing_sample_results.csv", sample_rows)
    write_csv(output_dir / "smoothing_summary.csv", summary_rows)
    if best_tam is not None:
        np.savez_compressed(output_dir / "best_smoothing_tam.npz", tam=np.asarray(best_tam, dtype=np.float32), config_id=np.asarray([best_config_id]))
    write_json(
        output_dir / "smoothing_oracle_summary.json",
        {
            "archive": str(args.archive),
            "checkpoint": str(checkpoint),
            "attacker": str(args.attacker),
            "samples": int(tam.shape[0]),
            "configs": int(len(configs)),
            "quick_grid": bool(args.quick_grid),
            "best_config_id": str(best_config_id),
            "best_accuracy": float(best_accuracy),
            "best_original_class_utility": float(best_utility),
            "objective": "fixed original RF prediction class; labels are post-hoc only",
            "outputs": {
                "sample_results": str(output_dir / "smoothing_sample_results.csv"),
                "summary": str(output_dir / "smoothing_summary.csv"),
                "best_tam": str(output_dir / "best_smoothing_tam.npz"),
            },
        },
    )
    print(f"Stage B2-S smoothing oracle complete: {output_dir}")


if __name__ == "__main__":
    main()
