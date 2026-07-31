"""Summarize Stage A mask and clustering artifacts into interpretable metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--cluster_result", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--early_fraction", type=float, default=0.25)
    return parser.parse_args()


def _mask_stats(mask: np.ndarray, early_fraction: float) -> dict[str, float]:
    values = np.asarray(mask, dtype=np.float64)
    width = values.shape[-1]
    slots = np.arange(width, dtype=np.float64)
    total = float(values.sum())
    if total <= 1e-12:
        return {
            "mask_mean": float(values.mean()),
            "mass_total": 0.0,
            "time_center": 0.0,
            "time_center_norm": 0.0,
            "early_mass_ratio": 0.0,
            "outgoing_mass_ratio": 0.0,
            "incoming_mass_ratio": 0.0,
        }
    per_time = values.sum(axis=0)
    early_end = max(1, int(round(width * float(early_fraction))))
    out = float(values[0].sum())
    inc = float(values[1].sum())
    center = float((per_time * slots).sum() / total)
    return {
        "mask_mean": float(values.mean()),
        "mass_total": total,
        "time_center": center,
        "time_center_norm": float(center / max(width - 1, 1)),
        "early_mass_ratio": float(per_time[:early_end].sum() / total),
        "outgoing_mass_ratio": float(out / total),
        "incoming_mass_ratio": float(inc / total),
    }


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    cluster_path = Path(args.cluster_result)
    with np.load(archive_path, allow_pickle=False) as arrays:
        archive = {key: arrays[key] for key in arrays.files}
    with np.load(cluster_path, allow_pickle=False) as arrays:
        cluster = {key: arrays[key] for key in arrays.files}
    masks = archive["mask"]
    cluster_labels = cluster["cluster_labels"].astype(np.int64)
    proto = cluster["proto_masks"]
    rows = []
    for k in range(int(cluster_labels.max()) + 1):
        member = np.flatnonzero(cluster_labels == k)
        sample_metrics = {
            "cluster": int(k),
            "size": int(member.size),
            "original_accuracy": float(np.mean(archive["pred_labels"][member] == archive["labels"][member])) if member.size else 0.0,
            "masked_accuracy": float(np.mean(archive["masked_pred_labels"][member] == archive["labels"][member])) if member.size else 0.0,
            "mean_js_div": float(np.mean(archive["js_div"][member])) if member.size else 0.0,
            "mean_entropy_gain": float(np.mean(archive["entropy_gain"][member])) if member.size else 0.0,
            "mean_top1_drop": float(np.mean(archive["top1_drop"][member])) if member.size else 0.0,
            "changed_pred_rate": float(np.mean(archive["pred_labels"][member] != archive["masked_pred_labels"][member])) if member.size else 0.0,
        }
        rows.append(sample_metrics | _mask_stats(proto[k], float(args.early_fraction)))
    global_summary = {
        "archive": str(archive_path),
        "cluster_result": str(cluster_path),
        "samples": int(masks.shape[0]),
        "width": int(masks.shape[-1]),
        "best_k": int(cluster_labels.max()) + 1,
        "mean_js_div": float(np.mean(archive["js_div"])),
        "mean_entropy_gain": float(np.mean(archive["entropy_gain"])),
        "mean_top1_drop": float(np.mean(archive["top1_drop"])),
        "mean_mask": float(np.mean(masks)),
        "changed_pred_rate": float(np.mean(archive["pred_labels"] != archive["masked_pred_labels"])),
        "original_accuracy": float(np.mean(archive["pred_labels"] == archive["labels"])),
        "masked_accuracy": float(np.mean(archive["masked_pred_labels"] == archive["labels"])),
        "global_mask_stats": _mask_stats(masks.mean(axis=0), float(args.early_fraction)),
        "clusters": rows,
    }
    output = Path(args.output) if args.output else cluster_path.with_name("analysis_summary.json")
    write_json(output, global_summary)
    print(f"Stage A analysis summary written: {output}")


if __name__ == "__main__":
    main()
