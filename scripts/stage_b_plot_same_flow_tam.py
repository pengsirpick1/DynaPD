# -*- coding: utf-8 -*-
"""Plot same-flow TAM slot-count comparisons for Stage B2-D defenses."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.evaluation.attack_models import crop_or_pad_2d
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.objectives import probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT

from scripts.stage_b_run_dual_actuator import (
    EvalState,
    _default_checkpoint,
    _initial_state,
    _load_archive,
    _load_raw_rows,
    _predict_one,
    _run_method,
)
from scripts.stage_b_run_target_min_cost import _margin, _run_target_controller


DEFAULT_ARCHIVE = (
    "results/stage_a_rf_native_w1800_n96_s60_seed0/"
    "stage_a_masks_rf/all_masks.npz"
)
DEFAULT_PREVIOUS_CSV = (
    "results/stage_b2d_dual_best_n96_bidir_d32_b10_v1/"
    "dual_sample_results.csv"
)


@dataclass
class Variant:
    key: str
    display: str
    state: EvalState
    protocol: str
    method: str
    budget_bound: float
    delay_bound: int
    runtime_sec: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--previous-result-csv", default=DEFAULT_PREVIOUS_CSV)
    parser.add_argument("--sample-index", type=int, default=-1, help="Default: first flipped sample in previous result CSV.")
    parser.add_argument("--sample-rank", type=int, default=0, help="Which flipped sample to use when --sample-index is omitted.")
    parser.add_argument("--output-dir", default="results/stage_b2d_same_flow_tam_visualization")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trace-length", type=int, default=5000)
    parser.add_argument("--max-load-time", type=float, default=80.0)
    parser.add_argument("--rf-num-slots", type=int, default=1800)
    parser.add_argument("--df-architecture", default="project")
    parser.add_argument("--df-tam-adapter", default="signed_balance")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--delay-budget", type=int, default=32)
    parser.add_argument("--target-budget", type=float, default=0.50)
    parser.add_argument("--target-delay-budget", type=int, default=64)
    parser.add_argument("--target-margin", type=float, default=0.0)
    parser.add_argument("--skip-target", action="store_true", help="Skip the slower target-min-cost variants.")
    parser.add_argument("--zoom-slots", type=int, default=320, help="Also save a zoomed overview for the first N TAM slots; 0 disables it.")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def setup_font() -> font_manager.FontProperties | None:
    for path in [
        Path("C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ]:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    return None


def _auto_sample_index(csv_path: Path, sample_rank: int) -> int:
    if not csv_path.is_file():
        return 0
    hits: list[int] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                if row.get("method") == "alternating3_fast_refresh" and float(row.get("flip", 0.0)) > 0.5:
                    hits.append(int(row["sample_index"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not hits:
        return 0
    rank = min(max(int(sample_rank), 0), len(hits) - 1)
    return int(hits[rank])


def _dual_args(args: argparse.Namespace, *, delay_length: int, max_candidates: int, max_generated_actions: int, max_pair_actions: int, max_dummy_steps: int, max_action_budget: float, max_local_rate_peak: int) -> argparse.Namespace:
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
        delay_length=int(delay_length),
        delay_rho=1.0,
        ratio=0.10,
        max_windows=8,
        max_candidates=int(max_candidates),
        max_generated_actions=int(max_generated_actions),
        max_pair_actions=int(max_pair_actions),
        max_action_budget=float(max_action_budget),
        max_local_rate_peak=int(max_local_rate_peak),
        max_dummy_steps=int(max_dummy_steps),
        confidence_weight=0.40,
        margin_weight=0.40,
        entropy_weight=0.20,
        refresh_stride=32,
        renderer_batch_size=48,
        batch_size=128,
        renderer_coordinate="rf_tam",
        renderer_strategy="uniform_in_patch",
        progress=False,
        rounds=3,
        min_margin_gain=1e-5,
    )


def _clean_relative_resource(state: EvalState, clean_total: float) -> dict[str, Any]:
    clean = max(float(clean_total), 1.0)
    counts = np.asarray(state.dummy_counts, dtype=np.int32)
    total_dummy = int(counts.sum())
    defended = clean + float(total_dummy)
    return {
        "clean_packet_count": float(clean),
        "dummy_packet_count": int(total_dummy),
        "defended_packet_count": float(defended),
        "dummy_bandwidth": float(total_dummy / clean),
        "total_overhead": float((defended - clean) / clean),
        "outgoing_dummy_count": int(counts[0].sum()),
        "incoming_dummy_count": int(counts[1].sum()),
        "average_delay_bins": float(state.avg_delay),
        "p95_delay_bins": float(state.p95_delay),
        "maximum_delay_bins": int(state.max_delay),
        "delay_packet_count": int(len(state.delay_values)),
        "outgoing_delay_packet_count": int(len(state.outgoing_delay_values)),
        "incoming_delay_packet_count": int(len(state.incoming_delay_values)),
    }


def _variant_metrics(
    *,
    variant: Variant,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    sample_index: int,
    sample_id: str,
) -> dict[str, Any]:
    metrics = probability_metrics(
        original_prob.reshape(1, -1),
        variant.state.prob.reshape(1, -1),
        np.asarray([int(label)], dtype=np.int64),
    )
    y0 = int(np.argmax(original_prob))
    tam = np.asarray(variant.state.tam, dtype=np.float32)
    resource = _clean_relative_resource(variant.state, clean_total)
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "variant": variant.key,
        "display": variant.display,
        "protocol": variant.protocol,
        "method": variant.method,
        "budget_bound": float(variant.budget_bound),
        "delay_bound": int(variant.delay_bound),
        "true_label": int(label),
        "original_pred": int(y0),
        "final_pred": int(np.argmax(variant.state.prob)),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "outgoing_tam_packet_count": float(tam[0].sum()),
        "incoming_tam_packet_count": float(tam[1].sum()),
        "total_tam_packet_count": float(tam.sum()),
        "renderer_dummy_bandwidth": float(variant.state.dummy_bandwidth),
        "action_count": int(len(variant.state.selected_actions)),
        "runtime_sec": float(variant.runtime_sec),
        **resource,
    }


def _plot_variant_on_axis(
    ax: plt.Axes,
    *,
    tam: np.ndarray,
    title: str,
    ymax: float,
    x_end: int | None = None,
    show_ylabel: bool = True,
) -> None:
    x_axis = np.arange(int(tam.shape[1]))
    outgoing = np.asarray(tam[0], dtype=np.float32)
    incoming = -np.asarray(tam[1], dtype=np.float32)
    ax.bar(x_axis, outgoing, width=1.0, color="#1358d8", edgecolor="#1358d8", linewidth=0.0)
    ax.bar(x_axis, incoming, width=1.0, color="#d82424", edgecolor="#d82424", linewidth=0.0)
    ax.axhline(0, color="#c17b99", linewidth=0.9, alpha=0.9)
    ax.grid(True, axis="both", alpha=0.23, linewidth=0.6)
    ax.set_ylim(-float(ymax), float(ymax))
    ax.set_xlim(0, int(x_end) if x_end is not None else int(tam.shape[1]))
    if show_ylabel:
        ax.set_ylabel("Pkt. Number", fontsize=9)
    ax.set_title(title, fontsize=9.2, loc="left")


def _title_from_row(row: dict[str, Any]) -> str:
    return (
        f"{row['display']} | pred {row['original_pred']}->{row['final_pred']} "
        f"y={row['true_label']} flip={row['flip']:.0f} "
        f"margin={row['original_class_margin']:.3f} "
        f"dummy={row['dummy_bandwidth'] * 100:.2f}% "
        f"({row['dummy_packet_count']:.0f}/{row['clean_packet_count']:.0f}) "
        f"delay avg/p95={row['average_delay_bins']:.2f}/{row['p95_delay_bins']:.1f} "
        f"actions={row['action_count']}"
    )


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


def _plot_overview(
    output_dir: Path,
    rows: list[dict[str, Any]],
    variants: list[Variant],
    *,
    dpi: int,
    stem: str = "same_flow_tam_slot_counts",
    x_end: int | None = None,
) -> Path:
    width = int(variants[0].state.tam.shape[1])
    shown_width = min(width, int(x_end)) if x_end is not None else width
    ymax = max(float(np.max(np.asarray(v.state.tam))) for v in variants)
    ymax = max(4.0, ymax + max(2.0, ymax * 0.08))
    fig, axes = plt.subplots(len(variants), 1, figsize=(15.6, max(4.0, 2.15 * len(variants))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row, variant in zip(axes, rows, variants):
        _plot_variant_on_axis(ax, tam=variant.state.tam, title=_title_from_row(row), ymax=ymax, x_end=shown_width)
    axes[-1].set_xlabel(f"TAM time slots (0-{shown_width - 1})", fontsize=10)
    axes[0].legend(
        handles=[
            Patch(color="#1358d8", label="Outgoing packets"),
            Patch(color="#d82424", label="Incoming packets"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )
    fig.suptitle(
        f"Same-flow TAM slot-count comparison | sample {rows[0]['sample_index']} / {rows[0]['sample_id']} | slots 0-{shown_width - 1}",
        fontsize=13,
        y=0.996,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _plot_heatmap(output_dir: Path, rows: list[dict[str, Any]], variants: list[Variant], *, dpi: int) -> Path:
    vmax = max(float(np.max(np.asarray(v.state.tam))) for v in variants)
    vmax = max(1.0, vmax)
    fig, axes = plt.subplots(len(variants), 1, figsize=(15.6, max(5.0, 1.75 * len(variants))), sharex=True)
    axes = np.atleast_1d(axes)
    last = None
    for ax, row, variant in zip(axes, rows, variants):
        last = ax.imshow(np.asarray(variant.state.tam, dtype=np.float32), aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["out", "in"], fontsize=8)
        ax.set_title(
            f"{row['display']} | margin={row['original_class_margin']:.3f}, "
            f"dummy={row['dummy_bandwidth'] * 100:.2f}%, delay={row['average_delay_bins']:.2f}",
            fontsize=8.8,
            loc="left",
            pad=6,
        )
    axes[-1].set_xlabel("TAM time slots", fontsize=10)
    if last is not None:
        fig.subplots_adjust(hspace=0.72, right=0.925, top=0.98, bottom=0.055)
        cax = fig.add_axes([0.94, 0.12, 0.012, 0.76])
        fig.colorbar(last, cax=cax, label="slot count")
    path = output_dir / "same_flow_tam_heatmaps.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _plot_individual(output_dir: Path, row: dict[str, Any], variant: Variant, *, ymax: float, dpi: int) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(15.6, 3.4))
    _plot_variant_on_axis(ax, tam=variant.state.tam, title=_title_from_row(row), ymax=ymax)
    ax.set_xlabel(f"TAM time slots (0-{int(variant.state.tam.shape[1]) - 1})", fontsize=10)
    ax.legend(
        handles=[
            Patch(color="#1358d8", label="Outgoing packets"),
            Patch(color="#d82424", label="Incoming packets"),
        ],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )
    fig.tight_layout()
    path = output_dir / f"tam_{variant.key}.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    setup_font()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).resolve()
    archive_path = Path(args.archive).resolve()
    previous_csv = Path(args.previous_result_csv).resolve()
    sample_index = int(args.sample_index)
    if sample_index < 0:
        sample_index = _auto_sample_index(previous_csv, int(args.sample_rank))
    output_dir = output_dir / f"sample_{sample_index:04d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    run_args = _dual_args(
        args,
        delay_length=int(args.delay_budget),
        max_candidates=6,
        max_generated_actions=32,
        max_pair_actions=0,
        max_dummy_steps=3,
        max_action_budget=0.035,
        max_local_rate_peak=16,
    )

    archive = _load_archive(str(archive_path), 0)
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    if not (0 <= sample_index < int(tam.shape[0])):
        raise IndexError(f"sample_index={sample_index} outside archive with {tam.shape[0]} samples")

    raw_rows = _load_raw_rows(str(args.data_root), source_indices, run_args)
    raw_trace = np.asarray(raw_rows[sample_index], dtype=np.float32)
    original_tam = np.asarray(tam[sample_index], dtype=np.float32)
    original_mask = np.asarray(mask[sample_index], dtype=np.float32)
    original_prob = np.asarray(prob[sample_index], dtype=np.float32)
    label = int(labels[sample_index])
    sample_id = str(sample_ids[sample_index])
    clean_total = max(float(original_tam.sum()), 1.0)

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

    variants: list[Variant] = []
    clean_state = _initial_state(raw_trace, original_tam, original_prob)
    clean_state.prob = _predict_one(attacker, original_tam, device=device, args=run_args)
    variants.append(Variant("clean_original", "Clean original", clean_state, "none", "clean", 0.0, 0, 0.0))

    method_specs = [
        ("bidir_delay_only_d32", "Bidir delay only D32", "bidirectional_cooperative", "delay_only"),
        ("bidir_static_hybrid_b10_d32", "Bidir static hybrid B10 D32", "bidirectional_cooperative", "static_hybrid_no_refresh"),
        ("bidir_delay_then_dummy_b10_d32", "Bidir delay->dummy B10 D32", "bidirectional_cooperative", "delay_then_dummy_fast_refresh"),
        ("bidir_alternating3_b10_d32", "Bidir alternating3 B10 D32", "bidirectional_cooperative", "alternating3_fast_refresh"),
        ("client_alternating3_b10_d32", "Client alternating3 B10 D32", "client_only", "alternating3_fast_refresh"),
    ]
    for key, display, protocol, method in method_specs:
        start = time.perf_counter()
        state = _run_method(
            method=method,
            protocol=protocol,
            budget=float(args.budget),
            delay_budget=int(args.delay_budget),
            raw_trace=raw_trace,
            original_tam=original_tam,
            original_mask=original_mask,
            original_prob=original_prob,
            label=label,
            attacker=attacker,
            device=device,
            args=run_args,
        )
        runtime = time.perf_counter() - start
        variants.append(Variant(key, display, state, protocol, method, float(args.budget), int(args.delay_budget), runtime))

    if not bool(args.skip_target):
        target_args = _dual_args(
            args,
            delay_length=int(args.target_delay_budget),
            max_candidates=8,
            max_generated_actions=64,
            max_pair_actions=16,
            max_dummy_steps=12,
            max_action_budget=0.10,
            max_local_rate_peak=64,
        )
        target_specs = [
            ("target_bidir_m0_b50_d64", "Target bidir margin0 B50 D64", "bidirectional_cooperative"),
            ("target_client_m0_b50_d64", "Target client margin0 B50 D64", "client_only"),
        ]
        for key, display, protocol in target_specs:
            start = time.perf_counter()
            state = _run_target_controller(
                protocol=protocol,
                budget=float(args.target_budget),
                delay_budget=int(args.target_delay_budget),
                margin_target=float(args.target_margin),
                raw_trace=raw_trace,
                original_tam=original_tam,
                original_mask=original_mask,
                original_prob=original_prob,
                attacker=attacker,
                device=device,
                args=target_args,
            )
            runtime = time.perf_counter() - start
            variants.append(Variant(key, display, state, protocol, "target_min_cost_margin", float(args.target_budget), int(args.target_delay_budget), runtime))

    rows = [
        _variant_metrics(
            variant=variant,
            original_prob=original_prob,
            label=label,
            clean_total=clean_total,
            sample_index=sample_index,
            sample_id=sample_id,
        )
        for variant in variants
    ]
    _write_csv(output_dir / "same_flow_variant_metrics.csv", rows)
    (output_dir / "same_flow_variant_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    overview = _plot_overview(output_dir, rows, variants, dpi=int(args.dpi))
    zoom_overview: Path | None = None
    if int(args.zoom_slots) > 0:
        zoom_overview = _plot_overview(
            output_dir,
            rows,
            variants,
            dpi=int(args.dpi),
            stem=f"same_flow_tam_slot_counts_zoom_0_{int(args.zoom_slots)}",
            x_end=int(args.zoom_slots),
        )
    heatmap = _plot_heatmap(output_dir, rows, variants, dpi=int(args.dpi))
    ymax = max(float(np.max(np.asarray(v.state.tam))) for v in variants)
    ymax = max(4.0, ymax + max(2.0, ymax * 0.08))
    individual_paths = []
    for row, variant in zip(rows, variants):
        individual_paths.append(str(_plot_individual(individual_dir, row, variant, ymax=ymax, dpi=int(args.dpi))))

    padded_traces = np.vstack([crop_or_pad_2d(v.state.trace, int(args.max_trace_length))[0] for v in variants]).astype(np.float32)
    np.savez_compressed(
        output_dir / "same_flow_tams_and_traces.npz",
        variant_names=np.asarray([v.key for v in variants]),
        tams=np.stack([np.asarray(v.state.tam, dtype=np.float32) for v in variants], axis=0),
        dummy_counts=np.stack([np.asarray(v.state.dummy_counts, dtype=np.int32) for v in variants], axis=0),
        traces=padded_traces,
        rows_json=np.asarray([json.dumps(row, sort_keys=True) for row in rows]),
    )

    manifest = {
        "sample_index": int(sample_index),
        "sample_id": sample_id,
        "true_label": int(label),
        "archive": str(archive_path),
        "checkpoint": str(Path(checkpoint).resolve()),
        "overview_png": str(overview),
        "overview_zoom_png": str(zoom_overview) if zoom_overview is not None else "",
        "heatmap_png": str(heatmap),
        "individual_pngs": individual_paths,
        "metrics_csv": str(output_dir / "same_flow_variant_metrics.csv"),
        "metrics_json": str(output_dir / "same_flow_variant_metrics.json"),
        "arrays_npz": str(output_dir / "same_flow_tams_and_traces.npz"),
        "clean_packet_count": float(clean_total),
        "clean_original_margin": float(_margin(original_prob, original_prob)),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
