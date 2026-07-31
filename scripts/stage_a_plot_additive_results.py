"""Plot Stage A additive probing outputs."""

from __future__ import annotations

import argparse
import csv
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--top_mapping_rows", type=int, default=300)
    return parser.parse_args()


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _figure_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) if args.output_dir else Path(args.result_dir) / "figures"
    target.mkdir(parents=True, exist_ok=True)
    return target


def _mean_heatmap(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(arr, axis=0)
    return np.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)


def _plot_heatmap(values: np.ndarray, title: str, save_path: Path, cmap: str = "magma") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 3.2))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("TAM bin")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["out", "in"])
    plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def _plot_budget_curves(rows: list[dict], figure_dir: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = [
        ("accuracy", "Accuracy"),
        ("flip", "Prediction Flip Rate"),
        ("js_div", "JS Divergence"),
        ("top1_drop", "Top1 Confidence Drop"),
        ("entropy_gain", "Entropy Gain"),
        ("efficiency_top1_drop", "Top1 Drop / Bandwidth"),
    ]
    methods = list(dict.fromkeys(row["method"] for row in rows))
    budgets = sorted({float(row["budget_ratio"]) for row in rows})
    for metric, title in metrics:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        for method in methods:
            ys = []
            for budget in budgets:
                matched = [row for row in rows if row["method"] == method and abs(float(row["budget_ratio"]) - budget) < 1e-9]
                ys.append(float(matched[0][metric]) if matched else np.nan)
            ax.plot([budget * 100.0 for budget in budgets], ys, marker="o", label=method)
        ax.set_title(f"Additive Budget Curve: {title}")
        ax.set_xlabel("dummy bandwidth budget (%)")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(figure_dir / f"budget_curve_{metric}.png", dpi=180)
        plt.close(fig)


def _plot_dose_response(rows: list[dict], figure_dir: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups: dict[int, list[float]] = defaultdict(list)
    eff_groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        groups[int(row["dose"])].append(float(row["top1_drop"]))
        eff_groups[int(row["dose"])].append(float(row["efficiency_top1_drop"]))
    doses = sorted(groups)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(doses, [float(np.mean(groups[dose])) for dose in doses], marker="o", label="top1_drop")
    ax.plot(doses, [float(np.mean(eff_groups[dose])) for dose in doses], marker="o", label="efficiency")
    ax.set_xscale("log", base=2)
    ax.set_title("Mean Dose Response")
    ax.set_xlabel("dummy dose")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(figure_dir / "dose_response_top1_drop_efficiency.png", dpi=180)
    plt.close(fig)


def _plot_mapping(rows: list[dict], figure_dir: Path, limit: int) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: float(row["efficiency_top1_drop"]), reverse=True)[: max(1, int(limit))]
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    x = [int(row["affected_center"]) for row in ordered]
    y = [int(row["insert_center"]) for row in ordered]
    c = [float(row["efficiency_top1_drop"]) for row in ordered]
    sc = ax.scatter(x, y, c=c, s=18, cmap="viridis", alpha=0.85)
    ax.plot([0, 1800], [0, 1800], color="black", linewidth=0.8, alpha=0.4)
    ax.set_title("Keypoint Position -> Best Insertion Position")
    ax.set_xlabel("affected keypoint TAM bin")
    ax.set_ylabel("insertion TAM bin")
    plt.colorbar(sc, ax=ax, label="top1-drop efficiency")
    plt.tight_layout()
    plt.savefig(figure_dir / "keypoint_to_insertion_mapping.png", dpi=180)
    plt.close(fig)


def _plot_channel_similarity(rows: list[dict], figure_dir: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [float(row["out_in_cosine"]) for row in rows]
    fig = plt.figure(figsize=(7, 4.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, bins=24, color="#2a6fbb", alpha=0.85)
    ax.set_title("Out/In Mask Cosine Similarity")
    ax.set_xlabel("cosine")
    ax.set_ylabel("samples")
    plt.tight_layout()
    plt.savefig(figure_dir / "mask_channel_cosine_hist.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    result_dir = Path(args.result_dir)
    figure_dir = _figure_dir(args)
    heatmap_path = result_dir / "additive_heatmaps.npz"
    if heatmap_path.is_file():
        with np.load(heatmap_path, allow_pickle=False) as arrays:
            payload = {key: arrays[key] for key in arrays.files}
        _plot_heatmap(_mean_heatmap(payload["decision_criticality"]), "Decision-Criticality S(d,t)", figure_dir / "decision_criticality_heatmap.png")
        _plot_heatmap(_mean_heatmap(payload["additive_efficiency"]), "Additive Efficiency E(d,t)", figure_dir / "additive_efficiency_heatmap.png")
        _plot_heatmap(_mean_heatmap(payload["minimum_effective_budget"]), "Minimum Effective Budget B*(d,t)", figure_dir / "minimum_effective_budget_heatmap.png", cmap="viridis")
        _plot_heatmap(_mean_heatmap(payload["best_causal_offset"]), "Best Causal Offset O*(d,t)", figure_dir / "best_causal_offset_heatmap.png", cmap="coolwarm")
    _plot_budget_curves(_rows(result_dir / "budget_summary.csv"), figure_dir)
    _plot_dose_response(_rows(result_dir / "action_results.csv"), figure_dir)
    _plot_mapping(_rows(result_dir / "sample_best_actions.csv"), figure_dir, int(args.top_mapping_rows))
    _plot_channel_similarity(_rows(result_dir / "sanity_channel_similarity.csv"), figure_dir)
    print(f"Stage A additive plots complete: {figure_dir}")


if __name__ == "__main__":
    main()
