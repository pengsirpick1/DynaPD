# -*- coding: utf-8 -*-
"""Plot clean/V1/V2 TAM slot-count figures for one Stage B2-E sample."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
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
)
from scripts.stage_b_run_target_min_cost import _run_target_controller
from scripts.stage_b_run_b2e_diverse_search import _method_config, _run_controller


DEFAULT_ARCHIVE = (
    "results/stage_a_rf_native_w1800_n96_s60_seed0/"
    "stage_a_masks_rf/all_masks.npz"
)


@dataclass
class PlotVariant:
    key: str
    display: str
    state: EvalState
    runtime_sec: float
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--sample-index", type=int, default=14)
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--output-dir", default="results/stage_b2e_same_flow_tam_visualization")
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-trace-length", type=int, default=5000)
    parser.add_argument("--max-load-time", type=float, default=80.0)
    parser.add_argument("--rf-num-slots", type=int, default=1800)
    parser.add_argument("--df-architecture", default="project")
    parser.add_argument("--df-tam-adapter", default="signed_balance")
    parser.add_argument("--max-delay", type=int, default=64)
    parser.add_argument("--margin-target", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay-length", type=int, default=64)
    parser.add_argument("--delay-rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max-windows", type=int, default=8)
    parser.add_argument("--max-dummy-steps", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-generated-actions", type=int, default=64)
    parser.add_argument("--max-pair-actions", type=int, default=16)
    parser.add_argument("--max-action-budget", type=float, default=0.10)
    parser.add_argument("--max-local-rate-peak", type=int, default=64)
    parser.add_argument("--min-margin-gain", type=float, default=1e-5)
    parser.add_argument("--stratified-bucket-k", type=int, default=8)
    parser.add_argument("--stratified-global-k", type=int, default=16)
    parser.add_argument("--random-explore-k", type=int, default=8)
    parser.add_argument("--true-recall-pool-size", type=int, default=0)
    parser.add_argument("--confidence-weight", type=float, default=0.40)
    parser.add_argument("--margin-weight", type=float, default=0.40)
    parser.add_argument("--entropy-weight", type=float, default=0.20)
    parser.add_argument("--renderer-batch-size", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--renderer-coordinate", default="rf_tam")
    parser.add_argument("--renderer-strategy", default="uniform_in_patch")
    parser.add_argument("--zoom-slots", type=int, default=320)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--omit-v1", action="store_true")
    return parser.parse_args()


def _resource(state: EvalState, clean_total: float) -> dict:
    clean = max(float(clean_total), 1.0)
    counts = np.asarray(state.dummy_counts, dtype=np.int32)
    dummy = int(counts.sum())
    actions = list(state.selected_actions)
    dose_counter = Counter(str(int(np.asarray(action.counts).sum())) for action in actions)
    type_counter = Counter(str(action.action_type) for action in actions)
    return {
        "clean_packet_count": float(clean),
        "dummy_packet_count": int(dummy),
        "actual_dummy_bandwidth": float(dummy / clean),
        "outgoing_dummy_count": int(counts[0].sum()),
        "incoming_dummy_count": int(counts[1].sum()),
        "average_delay_bins": float(state.avg_delay),
        "p95_delay_bins": float(state.p95_delay),
        "maximum_delay_bins": int(state.max_delay),
        "accepted_action_count": int(len(actions)),
        "accepted_action_dose_distribution": json.dumps(dict(sorted(dose_counter.items())), sort_keys=True),
        "accepted_action_type_distribution": json.dumps(dict(sorted(type_counter.items())), sort_keys=True),
        "multi_bin_action_rate": float(np.mean([int(np.count_nonzero(np.asarray(action.counts))) > 1 for action in actions])) if actions else 0.0,
    }


def _variant_row(
    *,
    variant: PlotVariant,
    original_prob: np.ndarray,
    label: int,
    clean_total: float,
    sample_index: int,
    sample_id: str,
) -> dict:
    metrics = probability_metrics(
        original_prob.reshape(1, -1),
        variant.state.prob.reshape(1, -1),
        np.asarray([int(label)], dtype=np.int64),
    )
    return {
        "sample_index": int(sample_index),
        "sample_id": str(sample_id),
        "variant": variant.key,
        "display": variant.display,
        "true_label": int(label),
        "original_pred": int(metrics["original_pred"][0]),
        "final_pred": int(metrics["evaluated_pred"][0]),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "runtime_sec": float(variant.runtime_sec),
        "note": str(variant.note),
        **_resource(variant.state, clean_total),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _title(row: dict) -> str:
    return (
        f"{row['display']} | pred {row['original_pred']}->{row['final_pred']} "
        f"y={row['true_label']} flip={row['flip']:.0f} "
        f"margin={row['original_class_margin']:.3f} "
        f"dummy={row['actual_dummy_bandwidth'] * 100:.2f}% "
        f"({row['dummy_packet_count']:.0f}/{row['clean_packet_count']:.0f}) "
        f"delay avg/p95={row['average_delay_bins']:.2f}/{row['p95_delay_bins']:.1f} "
        f"actions={row['accepted_action_count']}"
    )


def _plot_axis(ax: plt.Axes, tam: np.ndarray, title: str, ymax: float, x_end: int | None = None) -> None:
    width = int(tam.shape[1])
    end = min(width, int(x_end)) if x_end is not None else width
    x_axis = np.arange(width)
    outgoing = np.asarray(tam[0], dtype=np.float32)
    incoming = -np.asarray(tam[1], dtype=np.float32)
    ax.bar(x_axis, outgoing, width=1.0, color="#1358d8", edgecolor="#1358d8", linewidth=0.0)
    ax.bar(x_axis, incoming, width=1.0, color="#d82424", edgecolor="#d82424", linewidth=0.0)
    ax.axhline(0, color="#c17b99", linewidth=0.9, alpha=0.9)
    ax.grid(True, axis="both", alpha=0.23, linewidth=0.6)
    ax.set_xlim(0, end)
    ax.set_ylim(-float(ymax), float(ymax))
    ax.set_ylabel("Pkt. Number", fontsize=9)
    ax.set_title(title, fontsize=9.4, loc="left")


def _plot_counts(output_dir: Path, rows: list[dict], variants: list[PlotVariant], *, stem: str, x_end: int | None, dpi: int) -> Path:
    width = int(variants[0].state.tam.shape[1])
    shown_width = min(width, int(x_end)) if x_end is not None else width
    ymax = max(float(np.max(np.asarray(v.state.tam))) for v in variants)
    ymax = max(4.0, ymax + max(2.0, 0.08 * ymax))
    fig, axes = plt.subplots(len(variants), 1, figsize=(15.6, max(4.0, 2.5 * len(variants))), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, row, variant in zip(axes, rows, variants):
        _plot_axis(ax, np.asarray(variant.state.tam, dtype=np.float32), _title(row), ymax, x_end=shown_width)
    axes[0].legend(
        handles=[Patch(color="#1358d8", label="Outgoing packets"), Patch(color="#d82424", label="Incoming packets")],
        loc="upper right",
        frameon=True,
        fontsize=8,
    )
    axes[-1].set_xlabel(f"TAM time slots (0-{shown_width - 1})", fontsize=10)
    fig.suptitle(
        f"Stage B2-E same-flow TAM | sample {rows[0]['sample_index']} / {rows[0]['sample_id']} | slots 0-{shown_width - 1}",
        fontsize=13,
        y=0.996,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    path = output_dir / f"{stem}.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def _plot_heatmap(output_dir: Path, rows: list[dict], variants: list[PlotVariant], *, dpi: int) -> Path:
    vmax = max(float(np.max(np.asarray(v.state.tam))) for v in variants)
    vmax = max(1.0, vmax)
    fig, axes = plt.subplots(len(variants), 1, figsize=(15.6, max(4.2, 1.95 * len(variants))), sharex=True)
    axes = np.atleast_1d(axes)
    last = None
    for ax, row, variant in zip(axes, rows, variants):
        last = ax.imshow(np.asarray(variant.state.tam, dtype=np.float32), aspect="auto", interpolation="nearest", cmap="magma", vmin=0, vmax=vmax)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["out", "in"], fontsize=8)
        ax.set_title(
            f"{row['display']} | pred {row['original_pred']}->{row['final_pred']} | "
            f"dummy={row['actual_dummy_bandwidth'] * 100:.2f}% | margin={row['original_class_margin']:.3f}",
            fontsize=9,
            loc="left",
            pad=6,
        )
    axes[-1].set_xlabel("TAM time slots", fontsize=10)
    if last is not None:
        fig.subplots_adjust(hspace=0.70, right=0.925, top=0.98, bottom=0.08)
        cax = fig.add_axes([0.94, 0.15, 0.012, 0.70])
        fig.colorbar(last, cax=cax, label="slot count")
    path = output_dir / "b2e_same_flow_tam_heatmap.png"
    fig.savefig(path, dpi=int(dpi))
    plt.close(fig)
    return path


def main() -> None:
    import time

    args = parse_args()
    set_seed(int(args.seed))
    output_dir = Path(args.output_dir).resolve() / f"sample_{int(args.sample_index):04d}_{args.method}"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    archive = _load_archive(str(Path(args.archive).resolve()), 0)
    tam = np.asarray(archive["tam"], dtype=np.float32)
    mask = np.asarray(archive["mask"], dtype=np.float32)
    prob = np.asarray(archive["pred_prob"], dtype=np.float32)
    labels = np.asarray(archive["labels"], dtype=np.int64)
    sample_ids = np.asarray(archive.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
    source_indices = np.asarray(archive.get("source_indices", np.arange(tam.shape[0])), dtype=np.int64)
    idx = int(args.sample_index)
    if not 0 <= idx < int(tam.shape[0]):
        raise IndexError(f"sample_index={idx} outside archive with {tam.shape[0]} samples")
    raw_rows = _load_raw_rows(str(args.data_root), source_indices, args)
    raw_trace = np.asarray(raw_rows[idx], dtype=np.float32)
    original_tam = np.asarray(tam[idx], dtype=np.float32)
    original_mask = np.asarray(mask[idx], dtype=np.float32)
    original_prob = np.asarray(prob[idx], dtype=np.float32)
    label = int(labels[idx])
    sample_id = str(sample_ids[idx])
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

    variants: list[PlotVariant] = []
    clean = _initial_state(raw_trace, original_tam, original_prob)
    clean.prob = _predict_one(attacker, clean.tam, device=device, args=args)
    variants.append(PlotVariant("clean_original", "Clean original", clean, 0.0, "clean"))

    if not bool(args.omit_v1):
        start = time.perf_counter()
        v1 = _run_target_controller(
            protocol="bidirectional_cooperative",
            budget=float(args.budget),
            delay_budget=int(args.max_delay),
            margin_target=float(args.margin_target),
            raw_trace=raw_trace,
            original_tam=original_tam,
            original_mask=original_mask,
            original_prob=original_prob,
            attacker=attacker,
            device=device,
            args=args,
        )
        variants.append(PlotVariant("v1_target_b10_d64", "V1 target B10 D64", v1, time.perf_counter() - start, "V1 baseline"))

    config = _method_config(str(args.method))
    start = time.perf_counter()
    v2, aggregate, funnel_rows = _run_controller(
        config=config,
        protocol="bidirectional_cooperative",
        budget=float(args.budget),
        raw_trace=raw_trace,
        original_tam=original_tam,
        original_mask=original_mask,
        original_prob=original_prob,
        label=label,
        sample_index=idx,
        sample_id=sample_id,
        attacker=attacker,
        device=device,
        args=args,
    )
    note = f"V2 {args.method}; stop={aggregate.get('stop_reason', '')}"
    variants.append(PlotVariant(f"v2_{args.method}_b10_d64", f"V2 {args.method} B10 D64", v2, time.perf_counter() - start, note))

    rows = [
        _variant_row(
            variant=variant,
            original_prob=original_prob,
            label=label,
            clean_total=clean_total,
            sample_index=idx,
            sample_id=sample_id,
        )
        for variant in variants
    ]
    _write_csv(output_dir / "b2e_same_flow_variant_metrics.csv", rows)
    (output_dir / "b2e_same_flow_variant_metrics.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(output_dir / "b2e_same_flow_candidate_funnel.csv", funnel_rows)

    full = _plot_counts(output_dir, rows, variants, stem="b2e_same_flow_tam_slot_counts", x_end=None, dpi=int(args.dpi))
    zoom = _plot_counts(output_dir, rows, variants, stem=f"b2e_same_flow_tam_zoom_0_{int(args.zoom_slots)}", x_end=int(args.zoom_slots), dpi=int(args.dpi))
    heatmap = _plot_heatmap(output_dir, rows, variants, dpi=int(args.dpi))
    np.savez_compressed(
        output_dir / "b2e_same_flow_tams_and_traces.npz",
        variant_names=np.asarray([v.key for v in variants]),
        tams=np.stack([np.asarray(v.state.tam, dtype=np.float32) for v in variants], axis=0),
        dummy_counts=np.stack([np.asarray(v.state.dummy_counts, dtype=np.int32) for v in variants], axis=0),
        traces=np.vstack([crop_or_pad_2d(v.state.trace, int(args.max_trace_length))[0] for v in variants]).astype(np.float32),
    )
    manifest = {
        "sample_index": int(idx),
        "sample_id": sample_id,
        "true_label": int(label),
        "method": str(args.method),
        "budget": float(args.budget),
        "overview_png": str(full),
        "zoom_png": str(zoom),
        "heatmap_png": str(heatmap),
        "metrics_csv": str(output_dir / "b2e_same_flow_variant_metrics.csv"),
        "candidate_funnel_csv": str(output_dir / "b2e_same_flow_candidate_funnel.csv"),
        "arrays_npz": str(output_dir / "b2e_same_flow_tams_and_traces.npz"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
