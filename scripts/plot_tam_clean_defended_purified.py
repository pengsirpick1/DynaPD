# -*- coding: utf-8 -*-
"""Plot TAM-style slot-count figures for clean/defended/purified traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz
from dmmp.purifier.config import PurifierConfig

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot TAM slot-count subplots for a matched trace triple.")
    parser.add_argument("--purifier-run-dir", required=True)
    parser.add_argument("--purified-manifest", default="", help="Default: <run-dir>/manifests/purified_dataset_manifest.csv")
    parser.add_argument("--truncated-manifest", default="", help="Optional manifest for a direction-count-truncated purified trace.")
    parser.add_argument("--truncated-row-index", type=int, default=-1, help="Default: use --row-index")
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--num-slots", type=int, default=1000)
    parser.add_argument("--slot-ms", type=float, default=80.0)
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/visualizations/tam")
    parser.add_argument("--language", choices=["paper", "zh"], default="paper")
    parser.add_argument("--y-scale", choices=["shared", "independent"], default="shared")
    parser.add_argument("--omit-defended", action="store_true", help="Plot clean and generated trace only.")
    parser.add_argument("--purified-title-en", default="Purified Trace")
    parser.add_argument("--purified-title-zh", default="净化后流量")
    parser.add_argument("--output-tag", default="", help="Optional suffix tag before language/y-scale parts.")
    return parser.parse_args()


def setup_font() -> font_manager.FontProperties | None:
    for path in [
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
    ]:
        if path.is_file():
            font_manager.fontManager.addfont(str(path))
            prop = font_manager.FontProperties(fname=str(path))
            plt.rcParams["font.family"] = prop.get_name()
            plt.rcParams["font.sans-serif"] = [prop.get_name()]
            plt.rcParams["axes.unicode_minus"] = False
            return prop
    return None


def read_manifest_row(path: Path, row_index: int) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if index == int(row_index):
                return row
    raise IndexError(f"Manifest row index out of range: {row_index}")


def load_clean(clean_path: Path, clean_index: int, seq_length: int) -> np.ndarray:
    x_map = stored_npy_from_npz(clean_path, "X")
    if x_map is not None:
        values = np.asarray(x_map[int(clean_index)], dtype=np.float32).reshape(-1)
    else:
        with np.load(clean_path, allow_pickle=False) as payload:
            values = np.asarray(payload["X"][int(clean_index)], dtype=np.float32).reshape(-1)
    out = np.zeros(int(seq_length), dtype=np.float32)
    take = min(out.size, values.size)
    if take:
        out[:take] = values[:take]
    return out


def load_defended(row: dict[str, str], seq_length: int) -> np.ndarray:
    with np.load(row["defended_path"], allow_pickle=False) as payload:
        flat = np.asarray(payload["flat"], dtype=np.float32)
        offsets = np.asarray(payload["offsets"], dtype=np.int64)
        local = int(row["defended_local_index"])
        values = flat[int(offsets[local]) : int(offsets[local + 1])]
    out = np.zeros(int(seq_length), dtype=np.float32)
    take = min(out.size, values.size)
    if take:
        out[:take] = values[:take]
    return out


def load_purified(row: dict[str, str]) -> np.ndarray:
    with np.load(row["purified_path"], allow_pickle=False) as payload:
        return np.asarray(payload["X"][int(row["purified_index"])], dtype=np.float32).reshape(-1)


def assert_same_trace(base: dict[str, str], other: dict[str, str], *, label: str) -> None:
    for key in ["source_id", "clean_index", "variant_id", "split", "class_id"]:
        if str(base.get(key, "")) != str(other.get(key, "")):
            raise ValueError(f"{label} is not matched on {key}: {base.get(key)} != {other.get(key)}")


def tam_counts(values: np.ndarray, *, num_slots: int, slot_ms: float) -> tuple[np.ndarray, np.ndarray]:
    """Return outgoing positive counts and incoming negative counts per TAM slot."""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    x = x[x != 0]
    out_counts = np.zeros(int(num_slots), dtype=np.float32)
    in_counts = np.zeros(int(num_slots), dtype=np.float32)
    if x.size == 0:
        return out_counts, in_counts
    slot_seconds = float(slot_ms) / 1000.0
    slots = np.floor(np.abs(x) / max(slot_seconds, 1.0e-12)).astype(np.int64)
    slots = np.clip(slots, 0, int(num_slots) - 1)
    positive = x > 0
    negative = x < 0
    np.add.at(out_counts, slots[positive], 1.0)
    np.add.at(in_counts, slots[negative], 1.0)
    return out_counts, -in_counts


def trace_stats(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    nonzero = x[x != 0]
    return {
        "total": int(nonzero.size),
        "outgoing": int(np.sum(nonzero > 0)),
        "incoming": int(np.sum(nonzero < 0)),
        "max_time": float(np.max(np.abs(nonzero))) if nonzero.size else 0.0,
    }


def apply_labels(ax, *, title: str, language: str, font_prop: font_manager.FontProperties | None) -> None:
    if language == "zh":
        xlabel = "时间槽"
        ylabel = "包数量"
    else:
        xlabel = "Time Slots"
        ylabel = "Pkt. Number"
    kwargs = {"fontproperties": font_prop} if font_prop is not None else {}
    ax.set_title(title, fontsize=12, **kwargs)
    ax.set_xlabel(xlabel, fontsize=10, **kwargs)
    ax.set_ylabel(ylabel, fontsize=10, **kwargs)
    ax.grid(True, axis="both", alpha=0.28, linewidth=0.7)
    ax.axhline(0, color="#d9a8c4", linewidth=1.0, alpha=0.9)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        if font_prop is not None:
            label.set_fontproperties(font_prop)


def main() -> None:
    args = parse_args()
    font_prop = setup_font()
    run_dir = Path(args.purifier_run_dir).resolve()
    cfg = PurifierConfig.from_mapping(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    manifest = Path(args.purified_manifest).resolve() if args.purified_manifest else run_dir / "manifests" / "purified_dataset_manifest.csv"
    row = read_manifest_row(manifest, int(args.row_index))

    clean = load_clean(Path(cfg.clean_path), int(row["clean_index"]), int(cfg.seq_length))
    defended = load_defended(row, int(cfg.seq_length))
    purified = load_purified(row)
    traces = [("Clean Trace", "干净流量", clean)]
    if not bool(args.omit_defended):
        traces.append(("Defended Trace", "防御后流量", defended))
    traces.append((str(args.purified_title_en), str(args.purified_title_zh), purified))
    truncated_manifest: Path | None = None
    truncated_row: dict[str, str] | None = None
    if args.truncated_manifest:
        truncated_manifest = Path(args.truncated_manifest).resolve()
        truncated_row_index = int(args.truncated_row_index) if int(args.truncated_row_index) >= 0 else int(args.row_index)
        truncated_row = read_manifest_row(truncated_manifest, truncated_row_index)
        assert_same_trace(row, truncated_row, label="truncated manifest row")
        truncated = load_purified(truncated_row)
        traces.append(("Truncated Purified Trace", "截断后净化流量", truncated))

    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "visualizations" / "tam"
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(traces), 1, figsize=(10.8, 2.6 * len(traces)), sharex=True)
    axes = np.atleast_1d(axes)
    blue = "#0000ff"
    red = "#ff0000"
    x_axis = np.arange(int(args.num_slots))
    all_min = 0.0
    all_max = 0.0
    axis_ranges: list[tuple[float, float]] = []
    summaries: dict[str, Any] = {}
    for (title_en, title_zh, values), ax in zip(traces, axes):
        outgoing, incoming = tam_counts(values, num_slots=int(args.num_slots), slot_ms=float(args.slot_ms))
        stats = trace_stats(values)
        title = title_zh if args.language == "zh" else title_en
        title = f"{title}  (out={stats['outgoing']}, in={stats['incoming']}, total={stats['total']})"
        ax.bar(x_axis, outgoing, width=1.0, color=blue, edgecolor=blue, linewidth=0.0)
        ax.bar(x_axis, incoming, width=1.0, color=red, edgecolor=red, linewidth=0.0)
        apply_labels(ax, title=title, language=str(args.language), font_prop=font_prop)
        local_max = float(np.max(outgoing)) if outgoing.size else 0.0
        local_min = float(np.min(incoming)) if incoming.size else 0.0
        all_max = max(all_max, local_max)
        all_min = min(all_min, local_min)
        axis_ranges.append((local_min, local_max))
        summaries[title_en.lower().replace(" ", "_")] = {
            **stats,
            "max_outgoing_slot_count": float(np.max(outgoing)) if outgoing.size else 0.0,
            "max_incoming_slot_count": float(-np.min(incoming)) if incoming.size else 0.0,
        }

    if args.y_scale == "shared":
        y_margin = max(5.0, 0.08 * max(abs(all_min), abs(all_max), 1.0))
        y_ranges = [(all_min - y_margin, all_max + y_margin)] * len(axes)
    else:
        y_ranges = []
        for local_min, local_max in axis_ranges:
            local_margin = max(2.0, 0.10 * max(abs(local_min), abs(local_max), 1.0))
            y_ranges.append((local_min - local_margin, local_max + local_margin))
    for ax, (y_min, y_max) in zip(axes, y_ranges):
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, int(args.num_slots))
    legend_labels = ("Outgoing Packets", "Incoming Packets") if args.language == "paper" else ("上行包", "下行包")
    legend_kwargs = {"prop": font_prop} if font_prop is not None else {}
    axes[0].legend(
        handles=[Patch(color=blue, label=legend_labels[0]), Patch(color=red, label=legend_labels[1])],
        loc="upper right",
        frameon=True,
        **legend_kwargs,
    )
    if args.language == "zh":
        suptitle = (
            f"TAM 时间槽计数对比：source_id={row['source_id']}，variant={row['variant_id']}，class={row['class_id']}\n"
            f"{int(args.num_slots)} 个时间槽，每格 {float(args.slot_ms):g} ms"
        )
    else:
        suptitle = (
            f"TAM Slot-Count Comparison: source_id={row['source_id']}, variant={row['variant_id']}, class={row['class_id']}\n"
            f"{int(args.num_slots)} time slots, {float(args.slot_ms):g} ms per slot"
        )
    title_kwargs = {"fontproperties": font_prop} if font_prop is not None else {}
    fig.suptitle(suptitle, fontsize=13, **title_kwargs)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    suffix = "zh" if args.language == "zh" else "paper"
    trace_tag = "_clean_generated" if bool(args.omit_defended) else ""
    trace_tag += "_with_truncated" if truncated_row is not None else ""
    safe_tag = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(args.output_tag).strip())
    tag_part = f"_{safe_tag}" if safe_tag else ""
    png_path = output_dir / f"tam_clean_defended_purified_row{int(args.row_index):05d}{tag_part}_{suffix}_{args.y_scale}{trace_tag}.png"
    summary_path = output_dir / f"tam_clean_defended_purified_row{int(args.row_index):05d}{tag_part}_{suffix}_{args.y_scale}{trace_tag}_summary.json"
    fig.savefig(png_path, dpi=220)
    plt.close(fig)

    summary = {
        "method": "TAM slot-count visualization",
        "paper_style_reference": "Figure 1 on page 5 of 2026-f1760-paper.pdf",
        "manifest": str(manifest),
        "truncated_manifest": str(truncated_manifest) if truncated_manifest is not None else "",
        "row_index": int(args.row_index),
        "truncated_row_index": int(args.truncated_row_index) if int(args.truncated_row_index) >= 0 else int(args.row_index),
        "source_id": int(row["source_id"]),
        "variant_id": int(row["variant_id"]),
        "class_id": int(row["class_id"]),
        "clean_index": int(row["clean_index"]),
        "defended_path": row["defended_path"],
        "purified_path": row["purified_path"],
        "truncated_path": str(truncated_row["purified_path"]) if truncated_row is not None else "",
        "num_slots": int(args.num_slots),
        "slot_ms": float(args.slot_ms),
        "y_scale": str(args.y_scale),
        "trace_count": len(traces),
        "output_tag": safe_tag,
        "summaries": summaries,
        "image": str(png_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"image": str(png_path), "summary": str(summary_path)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
