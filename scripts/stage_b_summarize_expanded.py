"""Summarize and plot Stage B1 expanded oracle results."""

from __future__ import annotations

import argparse
import csv
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
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _plot_metric(summary_rows: list[dict], figure_dir: Path, metric: str, title: str, ylabel: str, *, x_key: str = "actual_bandwidth") -> None:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in summary_rows:
        groups.setdefault((row["protocol"], row["method"]), []).append(row)
    plt.figure(figsize=(11, 6))
    for (protocol, method), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda row: _float(row, x_key))
        xs = [_float(row, x_key) * (100.0 if "bandwidth" in x_key else 1.0) for row in rows]
        ys = [_float(row, metric) for row in rows]
        plt.plot(xs, ys, marker="o", linewidth=1.8, label=f"{protocol}:{method}")
    xlabel = "actual bandwidth (%)" if x_key == "actual_bandwidth" else x_key
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / f"{metric}_vs_{x_key}.png", dpi=180)
    plt.close()


def _best_methods(summary_rows: list[dict]) -> list[dict]:
    keys = sorted({(row["protocol"], float(row["budget"])) for row in summary_rows})
    best = []
    for protocol, budget in keys:
        rows = [row for row in summary_rows if row["protocol"] == protocol and abs(float(row["budget"]) - budget) < 1e-9]
        by_acc = min(rows, key=lambda row: _float(row, "accuracy"))
        by_util = max(rows, key=lambda row: _float(row, "original_class_utility"))
        by_budget = max(rows, key=lambda row: _float(row, "budget_utilization"))
        best.append(
            {
                "protocol": protocol,
                "budget": budget,
                "best_accuracy_method": by_acc["method"],
                "best_accuracy": _float(by_acc, "accuracy"),
                "best_utility_method": by_util["method"],
                "best_original_class_utility": _float(by_util, "original_class_utility"),
                "best_original_class_margin": _float(by_util, "original_class_margin"),
                "best_budget_utilization_method": by_budget["method"],
                "best_budget_utilization": _float(by_budget, "budget_utilization"),
                "best_actual_bandwidth": _float(by_budget, "actual_bandwidth"),
            }
        )
    return best


def _step_curve(step_rows: list[dict], figure_dir: Path) -> None:
    if not step_rows:
        return
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in step_rows:
        groups.setdefault((row["protocol"], row["method"]), []).append(row)
    plt.figure(figsize=(11, 6))
    for (protocol, method), rows in sorted(groups.items()):
        steps: dict[int, list[float]] = {}
        for row in rows:
            steps.setdefault(int(float(row["step"])), []).append(_float(row, "marginal_gain"))
        xs = sorted(steps)
        ys = [float(np.mean(steps[x])) for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=1.6, label=f"{protocol}:{method}")
    plt.xlabel("selected action step")
    plt.ylabel("mean marginal original-class utility gain")
    plt.title("Stage B1 Marginal Gain Per Step")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figure_dir / "marginal_gain_per_step.png", dpi=180)
    plt.close()


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    summary_rows = _read_csv(result_dir / "expanded_summary.csv")
    step_rows = _read_csv(result_dir / "expanded_step_results.csv") if (result_dir / "expanded_step_results.csv").exists() else []
    figure_dir = output_dir / "figures"
    for metric, title, ylabel in (
        ("accuracy", "RF Accuracy vs Bandwidth", "accuracy"),
        ("flip", "Prediction Flip vs Bandwidth", "flip rate"),
        ("original_top1_drop", "Original Top-1 Drop vs Bandwidth", "original top-1 drop"),
        ("original_class_margin", "Original-Class Margin vs Bandwidth", "original-class margin"),
        ("original_class_utility", "Original-Class Utility vs Bandwidth", "utility"),
        ("budget_utilization", "Budget Utilization vs Bandwidth", "utilization"),
        ("selected_action_count", "Selected Actions vs Bandwidth", "selected actions"),
        ("secondary_action_ratio", "Secondary Action Ratio vs Bandwidth", "secondary ratio"),
    ):
        _plot_metric(summary_rows, figure_dir, metric, title, ylabel)
    _step_curve(step_rows, figure_dir)
    best = _best_methods(summary_rows)
    write_csv(output_dir / "expanded_best_methods.csv", best)
    report = {
        "result_dir": str(result_dir),
        "figures": str(figure_dir),
        "summary_rows": int(len(summary_rows)),
        "step_rows": int(len(step_rows)),
        "best_methods": best,
    }
    run_json = result_dir / "expanded_oracle_summary.json"
    if run_json.exists():
        import json

        report["run"] = json.loads(run_json.read_text(encoding="utf-8"))
    write_json(output_dir / "expanded_summary_report.json", report)
    print(f"Stage B1 expanded summary complete: {result_dir}")


if __name__ == "__main__":
    main()
