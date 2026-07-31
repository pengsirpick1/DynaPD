"""Summarize and plot Stage B0 sequential oracle results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.utils import write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", default="")
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _figure_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) if args.output_dir else Path(args.result_dir) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _plot_metric(summary_rows: list[dict], figure_dir: Path, metric: str, title: str, ylabel: str) -> None:
    if not summary_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    protocols = list(dict.fromkeys(row["protocol"] for row in summary_rows))
    methods = list(dict.fromkeys(row["method"] for row in summary_rows))
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for protocol in protocols:
        for method in methods:
            matched = [row for row in summary_rows if row["protocol"] == protocol and row["method"] == method]
            if not matched:
                continue
            matched = sorted(matched, key=lambda row: float(row["budget"]))
            label = f"{protocol}:{method}"
            ax.plot([float(row["actual_bandwidth"]) * 100.0 for row in matched], [float(row[metric]) for row in matched], marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel("actual bandwidth (%)")
    ax.set_ylabel(ylabel)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / f"{metric}_vs_bandwidth.png", dpi=180)
    plt.close(fig)


def _plot_marginal(step_rows: list[dict], figure_dir: Path) -> None:
    if not step_rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keys = sorted({(row["protocol"], row["method"]) for row in step_rows})
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    for protocol, method in keys:
        matched = [row for row in step_rows if row["protocol"] == protocol and row["method"] == method]
        if not matched:
            continue
        max_step = max(int(row["step"]) for row in matched)
        xs, ys = [], []
        for step in range(1, max_step + 1):
            step_rows_local = [row for row in matched if int(row["step"]) == step]
            if not step_rows_local:
                continue
            xs.append(step)
            ys.append(float(np.mean([float(row["marginal_gain"]) for row in step_rows_local])))
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{protocol}:{method}")
    ax.set_title("Sequential Marginal Gain")
    ax.set_xlabel("step")
    ax.set_ylabel("mean marginal utility gain")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "sequential_marginal_gain_curve.png", dpi=180)
    plt.close(fig)


def _best_rows(summary_rows: list[dict]) -> list[dict]:
    best = []
    for protocol in sorted({row["protocol"] for row in summary_rows}):
        for budget in sorted({float(row["budget"]) for row in summary_rows}):
            matched = [row for row in summary_rows if row["protocol"] == protocol and abs(float(row["budget"]) - budget) < 1e-9]
            if not matched:
                continue
            best_acc = min(matched, key=lambda row: float(row["accuracy"]))
            best_top1 = max(matched, key=lambda row: float(row["top1_drop"]))
            best.append(
                {
                    "protocol": protocol,
                    "budget": budget,
                    "best_accuracy_method": best_acc["method"],
                    "best_accuracy": float(best_acc["accuracy"]),
                    "best_top1_method": best_top1["method"],
                    "best_top1_drop": float(best_top1["top1_drop"]),
                    "best_margin_drop": float(best_top1["margin_drop"]),
                    "best_actual_bandwidth": float(best_top1["actual_bandwidth"]),
                }
            )
    return best


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = _figure_dir(args)
    summary_rows = _read_csv(result_dir / "oracle_summary.csv")
    step_rows = _read_csv(result_dir / "oracle_step_results.csv")
    _plot_metric(summary_rows, figure_dir, "accuracy", "RF Accuracy vs Bandwidth", "accuracy")
    _plot_metric(summary_rows, figure_dir, "flip", "Prediction Flip Rate vs Bandwidth", "flip rate")
    _plot_metric(summary_rows, figure_dir, "top1_drop", "Top1 Drop vs Bandwidth", "top1 drop")
    _plot_metric(summary_rows, figure_dir, "margin_drop", "Margin Drop vs Bandwidth", "margin drop")
    _plot_metric(summary_rows, figure_dir, "js_div", "JS Divergence vs Bandwidth", "JS divergence")
    _plot_metric(summary_rows, figure_dir, "selected_action_count", "Selected Action Count vs Bandwidth", "selected actions")
    _plot_marginal(step_rows, figure_dir)
    best = _best_rows(summary_rows)
    write_csv(result_dir / "oracle_best_methods.csv", best)
    payload = {
        "result_dir": str(result_dir),
        "figures": str(figure_dir),
        "summary_rows": len(summary_rows),
        "step_rows": len(step_rows),
        "best_methods": best,
    }
    source_summary = result_dir / "sequential_oracle_summary.json"
    if source_summary.is_file():
        with source_summary.open("r", encoding="utf-8") as handle:
            payload["run"] = json.load(handle)
    write_json(result_dir / "oracle_summary_report.json", payload)
    print(f"Stage B0 oracle summary complete: {result_dir}")


if __name__ == "__main__":
    main()
