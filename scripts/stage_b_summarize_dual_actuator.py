"""Summarize and plot Stage B2-D dual-actuator controller results."""

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


B1_SEQUENTIAL_CAUSAL_ACC = 0.6042
B2S_STATIC_HYBRID_ACC = 0.6458


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--b1_accuracy", type=float, default=B1_SEQUENTIAL_CAUSAL_ACC)
    parser.add_argument("--b2s_static_hybrid_accuracy", type=float, default=B2S_STATIC_HYBRID_ACC)
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(row: dict, key: str) -> float:
    value = row.get(key, 0.0)
    return float(value) if value not in {"", None} else 0.0


def _label(row: dict) -> str:
    return f"{row['protocol']}:{row['method']}"


def _dynamic_rows(summary_rows: list[dict]) -> list[dict]:
    return [row for row in summary_rows if row["method"] not in {"delay_only", "dummy_only", "static_hybrid_no_refresh"}]


def _plot_accuracy_vs_bandwidth(summary_rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    delays = sorted({int(float(row["max_delay_budget"])) for row in summary_rows})
    for delay in delays:
        rows = [row for row in summary_rows if int(float(row["max_delay_budget"])) == delay]
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            groups.setdefault((row["protocol"], row["method"]), []).append(row)
        plt.figure(figsize=(11, 6))
        for (protocol, method), matched in sorted(groups.items()):
            matched = sorted(matched, key=lambda row: _float(row, "dummy_bandwidth"))
            xs = [_float(row, "dummy_bandwidth") * 100.0 for row in matched]
            ys = [_float(row, "accuracy") * 100.0 for row in matched]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"{protocol}:{method}")
        plt.axhline(B1_SEQUENTIAL_CAUSAL_ACC * 100.0, color="black", linestyle="--", linewidth=1.0, label="B1 sequential causal")
        plt.axhline(B2S_STATIC_HYBRID_ACC * 100.0, color="gray", linestyle=":", linewidth=1.0, label="B2-S static hybrid")
        plt.xlabel("actual dummy bandwidth (%)")
        plt.ylabel("RF accuracy (%)")
        plt.title(f"Accuracy vs Bandwidth at Delay Budget D={delay}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(figure_dir / f"accuracy_vs_bandwidth_D{delay}.png", dpi=180)
        plt.close()


def _plot_accuracy_vs_delay(summary_rows: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    budgets = sorted({float(row["dummy_budget"]) for row in summary_rows})
    for budget in budgets:
        rows = [row for row in summary_rows if abs(float(row["dummy_budget"]) - budget) < 1e-9]
        groups: dict[tuple[str, str], list[dict]] = {}
        for row in rows:
            groups.setdefault((row["protocol"], row["method"]), []).append(row)
        plt.figure(figsize=(11, 6))
        for (protocol, method), matched in sorted(groups.items()):
            matched = sorted(matched, key=lambda row: _float(row, "max_delay_budget"))
            xs = [_float(row, "max_delay_budget") for row in matched]
            ys = [_float(row, "accuracy") * 100.0 for row in matched]
            plt.plot(xs, ys, marker="o", linewidth=1.5, label=f"{protocol}:{method}")
        plt.axhline(B1_SEQUENTIAL_CAUSAL_ACC * 100.0, color="black", linestyle="--", linewidth=1.0, label="B1 sequential causal")
        plt.axhline(B2S_STATIC_HYBRID_ACC * 100.0, color="gray", linestyle=":", linewidth=1.0, label="B2-S static hybrid")
        plt.xlabel("maximum delay budget (bins)")
        plt.ylabel("RF accuracy (%)")
        plt.title(f"Accuracy vs Delay at Dummy Budget B={budget:.0%}")
        plt.grid(alpha=0.25)
        plt.legend(fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(figure_dir / f"accuracy_vs_delay_B{int(round(budget * 100)):02d}.png", dpi=180)
        plt.close()


def _plot_pareto(summary_rows: list[dict], figure_dir: Path) -> None:
    rows = [row for row in summary_rows if row["method"] != "delay_only"]
    if not rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    xs = np.asarray([_float(row, "dummy_bandwidth") * 100.0 for row in rows], dtype=np.float32)
    ys = np.asarray([_float(row, "average_delay_bins") for row in rows], dtype=np.float32)
    colors = np.asarray([_float(row, "accuracy") * 100.0 for row in rows], dtype=np.float32)
    plt.figure(figsize=(9, 6))
    sc = plt.scatter(xs, ys, c=colors, cmap="viridis_r", s=50, alpha=0.85)
    plt.colorbar(sc, label="RF accuracy (%)")
    plt.xlabel("actual dummy bandwidth (%)")
    plt.ylabel("average delay (bins)")
    plt.title("Bandwidth-Delay-Accuracy Pareto Map")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "bandwidth_delay_accuracy_pareto.png", dpi=180)
    plt.close()


def _plot_static_vs_dynamic(summary_rows: list[dict], figure_dir: Path) -> None:
    rows = []
    for protocol in sorted({row["protocol"] for row in summary_rows}):
        matched = [row for row in summary_rows if row["protocol"] == protocol and row["method"] != "delay_only" and row["method"] != "dummy_only"]
        if not matched:
            continue
        best_by_method: dict[str, dict] = {}
        for method in sorted({row["method"] for row in matched}):
            candidates = [row for row in matched if row["method"] == method]
            best_by_method[method] = min(candidates, key=lambda row: _float(row, "accuracy"))
        rows.extend(best_by_method.values())
    if not rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    labels = [_label(row) for row in rows]
    values = [_float(row, "accuracy") * 100.0 for row in rows]
    plt.figure(figsize=(max(10, len(rows) * 0.55), 6))
    plt.bar(np.arange(len(rows)), values)
    plt.axhline(B1_SEQUENTIAL_CAUSAL_ACC * 100.0, color="black", linestyle="--", linewidth=1.0, label="B1 sequential causal")
    plt.axhline(B2S_STATIC_HYBRID_ACC * 100.0, color="gray", linestyle=":", linewidth=1.0, label="B2-S static hybrid")
    plt.xticks(np.arange(len(rows)), labels, rotation=35, ha="right", fontsize=7)
    plt.ylabel("best RF accuracy (%)")
    plt.title("Static Hybrid vs Dynamic Hybrid Best Points")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "static_vs_dynamic_best_accuracy.png", dpi=180)
    plt.close()


def _plot_distributions(sample_rows: list[dict], figure_dir: Path) -> None:
    if not sample_rows:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    for key, title, filename in (
        ("synergy_utility", "Synergy Utility Distribution", "synergy_distribution.png"),
        ("keypoint_displacement", "Keypoint Displacement Distribution", "keypoint_displacement_distribution.png"),
        ("selected_dummy_position_displacement", "Dummy Position Displacement Distribution", "dummy_position_displacement_distribution.png"),
    ):
        values = [_float(row, key) for row in sample_rows if row.get(key, "") not in {"", None}]
        if not values:
            continue
        plt.figure(figsize=(8, 5))
        plt.hist(values, bins=30, alpha=0.85)
        plt.xlabel(key)
        plt.ylabel("count")
        plt.title(title)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.savefig(figure_dir / filename, dpi=180)
        plt.close()


def _plot_synergy(summary_rows: list[dict], figure_dir: Path) -> None:
    rows = _dynamic_rows(summary_rows)
    values = [_float(row, "synergy_utility") for row in rows]
    if not values:
        return
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=24, alpha=0.85)
    plt.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("synergy utility")
    plt.ylabel("config count")
    plt.title("Dynamic Joint Synergy Distribution")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(figure_dir / "synergy_distribution.png", dpi=180)
    plt.close()


def _best_configs(summary_rows: list[dict], *, top_k: int = 16) -> list[dict]:
    ranked = sorted(summary_rows, key=lambda row: (_float(row, "accuracy"), -_float(row, "original_class_utility"), _float(row, "dummy_bandwidth"), _float(row, "average_delay_bins")))
    out = []
    for rank, row in enumerate(ranked[: int(top_k)], start=1):
        out.append(
            {
                "rank": rank,
                "protocol": row["protocol"],
                "method": row["method"],
                "dummy_budget": _float(row, "dummy_budget"),
                "max_delay_budget": int(float(row["max_delay_budget"])),
                "samples": int(float(row["samples"])),
                "accuracy": _float(row, "accuracy"),
                "flip": _float(row, "flip"),
                "original_class_probability": _float(row, "original_class_probability"),
                "original_class_margin": _float(row, "original_class_margin"),
                "original_class_utility": _float(row, "original_class_utility"),
                "synergy_utility": _float(row, "synergy_utility"),
                "dummy_bandwidth": _float(row, "dummy_bandwidth"),
                "average_delay_bins": _float(row, "average_delay_bins"),
                "p95_delay_bins": _float(row, "p95_delay_bins"),
                "maximum_delay_bins": _float(row, "maximum_delay_bins"),
                "action_count": _float(row, "action_count"),
                "client_only_legal": _float(row, "client_only_legal"),
                "old_new_mask_overlap": _float(row, "old_new_mask_overlap"),
                "keypoint_displacement": _float(row, "keypoint_displacement"),
                "selected_dummy_position_displacement": _float(row, "selected_dummy_position_displacement"),
            }
        )
    return out


def _success_report(summary_rows: list[dict], *, b1_accuracy: float, b2s_static_hybrid_accuracy: float) -> dict:
    dynamic = _dynamic_rows(summary_rows)
    best_dynamic = min(dynamic, key=lambda row: _float(row, "accuracy")) if dynamic else None
    best_static = min(
        [row for row in summary_rows if row["method"] == "static_hybrid_no_refresh"],
        key=lambda row: _float(row, "accuracy"),
        default=None,
    )
    positive_synergy = [row for row in dynamic if _float(row, "synergy_utility") > 0.0]
    return {
        "best_dynamic": dict(best_dynamic) if best_dynamic else None,
        "best_static_hybrid_no_refresh": dict(best_static) if best_static else None,
        "condition_joint_below_b1": bool(best_dynamic and _float(best_dynamic, "accuracy") < float(b1_accuracy)),
        "condition_dynamic_below_b2s_static_hybrid": bool(best_dynamic and _float(best_dynamic, "accuracy") < float(b2s_static_hybrid_accuracy)),
        "condition_dynamic_below_internal_static": bool(best_dynamic and best_static and _float(best_dynamic, "accuracy") < _float(best_static, "accuracy")),
        "positive_synergy_rows": int(len(positive_synergy)),
        "best_positive_synergy": dict(max(positive_synergy, key=lambda row: _float(row, "synergy_utility"))) if positive_synergy else None,
        "b1_accuracy_reference": float(b1_accuracy),
        "b2s_static_hybrid_accuracy_reference": float(b2s_static_hybrid_accuracy),
    }


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir) if args.output_dir else result_dir
    figure_dir = output_dir / "figures"
    summary_rows = _read_csv(result_dir / "dual_summary.csv")
    sample_rows = _read_csv(result_dir / "dual_sample_results.csv") if (result_dir / "dual_sample_results.csv").exists() else []
    _plot_accuracy_vs_bandwidth(summary_rows, figure_dir)
    _plot_accuracy_vs_delay(summary_rows, figure_dir)
    _plot_pareto(summary_rows, figure_dir)
    _plot_static_vs_dynamic(summary_rows, figure_dir)
    _plot_distributions(sample_rows, figure_dir)
    _plot_synergy(summary_rows, figure_dir)
    best = _best_configs(summary_rows)
    success = _success_report(summary_rows, b1_accuracy=float(args.b1_accuracy), b2s_static_hybrid_accuracy=float(args.b2s_static_hybrid_accuracy))
    write_csv(output_dir / "dual_best_configs.csv", best)
    report = {
        "result_dir": str(result_dir),
        "figures": str(figure_dir),
        "summary_rows": int(len(summary_rows)),
        "sample_rows": int(len(sample_rows)),
        "best_configs": best,
        "success_report": success,
    }
    run_json = result_dir / "dual_oracle_summary.json"
    if run_json.exists():
        report["run"] = json.loads(run_json.read_text(encoding="utf-8"))
    write_json(output_dir / "dual_summary_report.json", report)
    print(f"Stage B2-D dual actuator summary complete: {result_dir}")


if __name__ == "__main__":
    main()
