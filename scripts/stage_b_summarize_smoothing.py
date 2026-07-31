"""Summarize Stage B2-S keypoint-guided smoothing oracle results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dmmp.utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--max_visual_samples", type=int, default=4)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _plot_metric(summary: list[dict], figure_dir: Path, metric: str, ylabel: str, *, x_key: str = "actual_bandwidth") -> None:
    groups: dict[str, list[dict]] = {}
    for row in summary:
        if row["method"] == "clean":
            continue
        groups.setdefault(row["method"], []).append(row)
    plt.figure(figsize=(10, 6))
    for method, rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: _float(row, x_key))
        xs = [_float(row, x_key) * (100.0 if "bandwidth" in x_key else 1.0) for row in rows]
        ys = [_float(row, metric) for row in rows]
        plt.scatter(xs, ys, s=28, alpha=0.75, label=method)
    plt.xlabel("actual bandwidth (%)" if x_key == "actual_bandwidth" else x_key)
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs {x_key}")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / f"{metric}_vs_{x_key}.png", dpi=180)
    plt.close()


def _plot_pareto(summary: list[dict], figure_dir: Path) -> None:
    rows = [row for row in summary if row["method"] != "clean"]
    if not rows:
        return
    xs = np.asarray([_float(row, "actual_bandwidth") * 100.0 for row in rows], dtype=np.float32)
    ys = np.asarray([_float(row, "average_delay_bins") for row in rows], dtype=np.float32)
    colors = np.asarray([_float(row, "accuracy") for row in rows], dtype=np.float32)
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(xs, ys, c=colors, cmap="viridis_r", s=42, alpha=0.85)
    plt.colorbar(sc, label="RF accuracy")
    plt.xlabel("actual bandwidth (%)")
    plt.ylabel("average delay (bins)")
    plt.title("Bandwidth-Delay-Accuracy Pareto Map")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / "bandwidth_delay_accuracy_pareto.png", dpi=180)
    plt.close()


def _best_configs(summary: list[dict]) -> list[dict]:
    methods = sorted({row["method"] for row in summary})
    rows = []
    for method in methods:
        matched = [row for row in summary if row["method"] == method]
        best = sorted(matched, key=lambda row: (_float(row, "accuracy"), -_float(row, "original_class_utility")))[0]
        rows.append(
            {
                "method": method,
                "config_id": best["config_id"],
                "accuracy": _float(best, "accuracy"),
                "flip": _float(best, "flip"),
                "original_top1_drop": _float(best, "original_top1_drop"),
                "original_class_margin": _float(best, "original_class_margin"),
                "original_class_utility": _float(best, "original_class_utility"),
                "actual_bandwidth": _float(best, "actual_bandwidth"),
                "average_delay_bins": _float(best, "average_delay_bins"),
                "maximum_delay_bins": _float(best, "maximum_delay_bins"),
                "local_variance_reduction": _float(best, "local_variance_reduction"),
                "local_gradient_reduction": _float(best, "local_gradient_reduction"),
            }
        )
    return rows


def _visualize_best(result_dir: Path, figure_dir: Path, max_samples: int) -> list[str]:
    run_path = result_dir / "smoothing_oracle_summary.json"
    best_path = result_dir / "best_smoothing_tam.npz"
    if not run_path.exists() or not best_path.exists():
        return []
    run = json.loads(run_path.read_text(encoding="utf-8"))
    archive_path = Path(run["archive"])
    if not archive_path.is_absolute():
        archive_path = ROOT / archive_path
    if not archive_path.exists():
        return []
    with np.load(archive_path, allow_pickle=False) as archive:
        original = np.asarray(archive["tam"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=np.float32)
    with np.load(best_path, allow_pickle=False) as payload:
        defended = np.asarray(payload["tam"], dtype=np.float32)
        config_id = str(np.asarray(payload["config_id"])[0])
    count = min(int(max_samples), original.shape[0], defended.shape[0])
    paths = []
    for index in range(count):
        fig, axes = plt.subplots(3, 1, figsize=(11, 5), sharex=True)
        axes[0].imshow(original[index], aspect="auto", cmap="magma")
        axes[0].set_ylabel("clean TAM")
        axes[1].imshow(mask[index], aspect="auto", cmap="viridis")
        axes[1].set_ylabel("DynaMask")
        axes[2].imshow(defended[index], aspect="auto", cmap="magma")
        axes[2].set_ylabel("defended TAM")
        axes[2].set_xlabel("RF TAM bin")
        fig.suptitle(f"Best smoothing config: {config_id} | sample {index}")
        plt.tight_layout()
        figure_dir.mkdir(parents=True, exist_ok=True)
        path = figure_dir / f"tam_mask_defended_sample_{index:03d}.png"
        plt.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(str(path))
    return paths


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    figure_dir = output_dir / "figures"
    summary = _read_csv(result_dir / "smoothing_summary.csv")
    sample_rows = _read_csv(result_dir / "smoothing_sample_results.csv") if (result_dir / "smoothing_sample_results.csv").exists() else []
    for metric, ylabel in (
        ("accuracy", "RF accuracy"),
        ("flip", "prediction flip"),
        ("original_top1_drop", "original top-1 drop"),
        ("original_class_margin", "original-class margin"),
        ("original_class_utility", "original-class utility"),
        ("local_variance_reduction", "local variance reduction"),
        ("local_gradient_reduction", "local gradient reduction"),
    ):
        _plot_metric(summary, figure_dir, metric, ylabel)
    _plot_metric(summary, figure_dir, "accuracy", "RF accuracy", x_key="average_delay_bins")
    _plot_pareto(summary, figure_dir)
    best = _best_configs(summary)
    write_csv(output_dir / "smoothing_best_configs.csv", best)
    visual_paths = _visualize_best(result_dir, figure_dir, int(args.max_visual_samples))
    report = {
        "result_dir": str(result_dir),
        "figures": str(figure_dir),
        "summary_rows": int(len(summary)),
        "sample_rows": int(len(sample_rows)),
        "best_configs": best,
        "visualizations": visual_paths,
    }
    run_path = result_dir / "smoothing_oracle_summary.json"
    if run_path.exists():
        report["run"] = json.loads(run_path.read_text(encoding="utf-8"))
    write_json(output_dir / "smoothing_summary_report.json", report)
    print(f"Stage B2-S smoothing summary complete: {result_dir}")


if __name__ == "__main__":
    main()
