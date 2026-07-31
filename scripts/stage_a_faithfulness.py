"""Run Stage A keypoint faithfulness tests against equal-budget baselines."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.faithfulness import (
    METHODS,
    aopc,
    apply_deletion,
    apply_keep_only,
    method_ratio_masks,
    predict_probabilities,
    sample_probability_metrics,
    score_probabilities,
)
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.utils import resolve_device, set_seed, write_csv, write_json
from dmmp.utils.config import DEFAULT_OUTPUT_DIR, parse_csv_floats


DEFAULT_FIXED_DF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt"
DEFAULT_FIXED_RF = r"D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt"


def _default_checkpoint(attacker: str) -> str:
    return DEFAULT_FIXED_DF if str(attacker).lower() == "df" else DEFAULT_FIXED_RF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--ratios", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--methods", default="dynamask,random,random_block,magnitude,early")
    parser.add_argument("--random_repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    return parser.parse_args()


def _method_values(value: str) -> list[str]:
    names = [item.strip().lower() for item in str(value).replace(";", ",").split(",") if item.strip()]
    invalid = sorted(set(names) - set(METHODS))
    if invalid:
        raise ValueError(f"Unknown methods: {invalid}")
    return names


def _run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        name = args.run_name
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"stage_a_faithfulness_{args.attacker}_{stamp}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _plot_curves(rows: list[dict], output_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(row["method"] for row in rows if row["mode"] == "necessity"))
    ratios = sorted({float(row["ratio"]) for row in rows})
    metric_specs = [
        ("accuracy", "Accuracy", False),
        ("flip_rate", "Prediction Flip Rate", True),
        ("js_div", "JS Divergence", True),
        ("top1_drop", "Top1 Confidence Drop", True),
        ("entropy_gain", "Entropy Gain", True),
    ]
    for metric, title, _higher in metric_specs:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        for method in methods:
            values = []
            for ratio in ratios:
                matched = [
                    row
                    for row in rows
                    if row["mode"] == "necessity" and row["method"] == method and abs(float(row["ratio"]) - ratio) < 1e-9
                ]
                values.append(float(np.mean([row[metric] for row in matched])) if matched else np.nan)
            ax.plot([r * 100.0 for r in ratios], values, marker="o", label=method)
        ax.set_title(f"Deletion Curve: {title}")
        ax.set_xlabel("deleted positions (%)")
        ax.set_ylabel(metric)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(figure_dir / f"deletion_curve_{metric}.png", dpi=180)
        plt.close(fig)

    dyna_suff = [row for row in rows if row["method"] == "dynamask"]
    if dyna_suff:
        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        for mode, metric, label in [
            ("necessity", "js_div", "delete JS (higher better)"),
            ("sufficiency", "js_div", "keep-only JS (lower better)"),
            ("sufficiency", "top1_preservation", "keep-only top1 preservation"),
        ]:
            values = []
            for ratio in ratios:
                matched = [row for row in dyna_suff if row["mode"] == mode and abs(float(row["ratio"]) - ratio) < 1e-9]
                values.append(float(np.mean([row[metric] for row in matched])) if matched else np.nan)
            ax.plot([r * 100.0 for r in ratios], values, marker="o", label=label)
        ax.set_title("DynaMask Necessity vs Sufficiency")
        ax.set_xlabel("selected positions (%)")
        ax.legend(loc="best")
        plt.tight_layout()
        plt.savefig(figure_dir / "dynamask_necessity_sufficiency.png", dpi=180)
        plt.close(fig)


def _row_from_metrics(mode: str, method: str, ratio: float, repeat: int, metrics) -> dict:
    return {
        "mode": mode,
        "method": method,
        "ratio": float(ratio),
        "repeat": int(repeat),
        "accuracy": float(metrics.accuracy),
        "flip_rate": float(metrics.flip_rate),
        "js_div": float(metrics.js_div),
        "top1_drop": float(metrics.top1_drop),
        "entropy_gain": float(metrics.entropy_gain),
        "top1_preservation": float(metrics.top1_preservation),
    }


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    output_dir = _run_dir(args)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    ratios = parse_csv_floats(args.ratios)
    methods = _method_values(args.methods)
    with np.load(args.archive, allow_pickle=False) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    tam = np.asarray(payload["tam"], dtype=np.float32)
    soft_mask = np.asarray(payload["mask"], dtype=np.float32)
    baseline = np.asarray(payload["tam_base"], dtype=np.float32)
    original_prob = np.asarray(payload["pred_prob"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
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
    rows: list[dict] = []
    sample_metric_rows: list[dict[str, np.ndarray | float]] = []
    rng_root = np.random.default_rng(int(args.seed))
    for ratio in ratios:
        for method in methods:
            repeats = int(args.random_repeats) if method in {"random", "random_block"} else 1
            for repeat in range(repeats):
                rng = np.random.default_rng(int(rng_root.integers(0, 2**31 - 1)))
                hard = method_ratio_masks(method, soft_mask=soft_mask, tam=tam, ratio=float(ratio), rng=rng)
                deleted = apply_deletion(tam, baseline, hard)
                kept = apply_keep_only(tam, baseline, hard)
                deleted_prob = predict_probabilities(attacker, deleted, device=device, batch_size=int(args.batch_size))
                kept_prob = predict_probabilities(attacker, kept, device=device, batch_size=int(args.batch_size))
                deleted_metrics = score_probabilities(original_prob, deleted_prob, labels)
                kept_metrics = score_probabilities(original_prob, kept_prob, labels)
                rows.append(_row_from_metrics("necessity", method, float(ratio), repeat, deleted_metrics))
                rows.append(_row_from_metrics("sufficiency", method, float(ratio), repeat, kept_metrics))
                if method == "dynamask" and repeat == 0:
                    necessity_sample = sample_probability_metrics(original_prob, deleted_prob, labels)
                    sufficiency_sample = sample_probability_metrics(original_prob, kept_prob, labels)
                    sample_metric_rows.append(
                        {
                            "ratio": float(ratio),
                            "necessity_correct": necessity_sample["correct"],
                            "necessity_flip": necessity_sample["flip"],
                            "necessity_js_div": necessity_sample["js_div"],
                            "necessity_top1_drop": necessity_sample["top1_drop"],
                            "necessity_entropy_gain": necessity_sample["entropy_gain"],
                            "necessity_top1_preservation": necessity_sample["top1_preservation"],
                            "sufficiency_correct": sufficiency_sample["correct"],
                            "sufficiency_flip": sufficiency_sample["flip"],
                            "sufficiency_js_div": sufficiency_sample["js_div"],
                            "sufficiency_top1_drop": sufficiency_sample["top1_drop"],
                            "sufficiency_entropy_gain": sufficiency_sample["entropy_gain"],
                            "sufficiency_top1_preservation": sufficiency_sample["top1_preservation"],
                        }
                    )
                print(
                    f"[faithfulness] ratio={ratio:.2f} method={method} repeat={repeat} "
                    f"delete_acc={rows[-2]['accuracy']:.4f} delete_js={rows[-2]['js_div']:.4f} "
                    f"keep_js={rows[-1]['js_div']:.4f}",
                    flush=True,
                )
    write_csv(output_dir / "faithfulness_metrics.csv", rows)
    sample_metric_path = output_dir / "dynamask_sample_metrics.npz"
    if sample_metric_rows:
        ratios_array = np.asarray([item["ratio"] for item in sample_metric_rows], dtype=np.float32)
        np.savez_compressed(
            sample_metric_path,
            sample_ids=np.asarray(payload.get("sample_ids", np.arange(tam.shape[0]))).astype(str),
            labels=labels.astype(np.int64),
            ratios=ratios_array,
            necessity_correct=np.stack([item["necessity_correct"] for item in sample_metric_rows], axis=0).astype(np.float32),
            necessity_flip=np.stack([item["necessity_flip"] for item in sample_metric_rows], axis=0).astype(np.float32),
            necessity_js_div=np.stack([item["necessity_js_div"] for item in sample_metric_rows], axis=0).astype(np.float32),
            necessity_top1_drop=np.stack([item["necessity_top1_drop"] for item in sample_metric_rows], axis=0).astype(np.float32),
            necessity_entropy_gain=np.stack([item["necessity_entropy_gain"] for item in sample_metric_rows], axis=0).astype(np.float32),
            necessity_top1_preservation=np.stack([item["necessity_top1_preservation"] for item in sample_metric_rows], axis=0).astype(np.float32),
            sufficiency_correct=np.stack([item["sufficiency_correct"] for item in sample_metric_rows], axis=0).astype(np.float32),
            sufficiency_flip=np.stack([item["sufficiency_flip"] for item in sample_metric_rows], axis=0).astype(np.float32),
            sufficiency_js_div=np.stack([item["sufficiency_js_div"] for item in sample_metric_rows], axis=0).astype(np.float32),
            sufficiency_top1_drop=np.stack([item["sufficiency_top1_drop"] for item in sample_metric_rows], axis=0).astype(np.float32),
            sufficiency_entropy_gain=np.stack([item["sufficiency_entropy_gain"] for item in sample_metric_rows], axis=0).astype(np.float32),
            sufficiency_top1_preservation=np.stack([item["sufficiency_top1_preservation"] for item in sample_metric_rows], axis=0).astype(np.float32),
        )
    aopc_rows = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method and row["mode"] == "necessity"]
        aopc_rows.append(
            {
                "method": method,
                "aopc_top1_drop": aopc(method_rows, method, "top1_drop"),
                "mean_js_div": float(np.mean([row["js_div"] for row in method_rows])) if method_rows else 0.0,
                "mean_flip_rate": float(np.mean([row["flip_rate"] for row in method_rows])) if method_rows else 0.0,
                "mean_accuracy": float(np.mean([row["accuracy"] for row in method_rows])) if method_rows else 0.0,
            }
        )
    write_csv(output_dir / "aopc_summary.csv", aopc_rows)
    _plot_curves(rows, output_dir)
    write_json(
        output_dir / "faithfulness_summary.json",
        {
            "archive": str(args.archive),
            "checkpoint": str(checkpoint),
            "attacker": str(args.attacker),
            "samples": int(tam.shape[0]),
            "width": int(tam.shape[-1]),
            "ratios": [float(r) for r in ratios],
            "methods": methods,
            "random_repeats": int(args.random_repeats),
            "dynamask_sample_metrics": str(sample_metric_path) if sample_metric_rows else "",
            "aopc": aopc_rows,
            "figures": str(output_dir / "figures"),
        },
    )
    print(f"Stage A faithfulness complete: {output_dir}")


if __name__ == "__main__":
    main()
