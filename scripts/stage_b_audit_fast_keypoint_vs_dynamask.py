# -*- coding: utf-8 -*-
"""Audit fast keypoints against DynaMask on matched source traces."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.faithfulness import predict_probabilities
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.objectives import probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_b2e_diverse_search import DEFAULT_ARCHIVE, _default_checkpoint


DEFAULT_FAST_ARCHIVE = "results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dynamask_archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--fast_archive", default=DEFAULT_FAST_ARCHIVE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="stage_b_fast_vs_dynamask_audit_n96")
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--deletion_ratios", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--overlap_ratios", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--write_aligned_fast_archive", action="store_true")
    return parser.parse_args()


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) / str(args.run_name)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _rows_by_source(fast_source: np.ndarray, requested_source: np.ndarray) -> np.ndarray:
    position = {int(source): row for row, source in enumerate(np.asarray(fast_source, dtype=np.int64).tolist())}
    missing = [int(source) for source in np.asarray(requested_source, dtype=np.int64).tolist() if int(source) not in position]
    if missing:
        raise ValueError(f"Fast archive missing {len(missing)} source indices; first missing={missing[:5]}")
    return np.asarray([position[int(source)] for source in np.asarray(requested_source, dtype=np.int64).tolist()], dtype=np.int64)


def _top_indices(mask: np.ndarray, ratio: float) -> np.ndarray:
    flat = np.asarray(mask, dtype=np.float32).reshape(-1)
    k = max(1, int(round(float(ratio) * flat.size)))
    return np.argsort(-flat, kind="mergesort")[:k].astype(np.int64)


def _overlap_rows(dyn_mask: np.ndarray, fast_mask: np.ndarray, ratios: list[float]) -> list[dict[str, Any]]:
    rows = []
    for sample_index in range(int(dyn_mask.shape[0])):
        d_flat = np.asarray(dyn_mask[sample_index], dtype=np.float32).reshape(-1)
        f_flat = np.asarray(fast_mask[sample_index], dtype=np.float32).reshape(-1)
        d_norm = d_flat / max(float(np.linalg.norm(d_flat)), 1e-8)
        f_norm = f_flat / max(float(np.linalg.norm(f_flat)), 1e-8)
        base = {
            "sample_index": int(sample_index),
            "cosine": float(np.dot(d_norm, f_norm)),
            "pearson": float(np.corrcoef(d_flat, f_flat)[0, 1]) if float(np.std(d_flat)) > 1e-8 and float(np.std(f_flat)) > 1e-8 else 0.0,
        }
        for ratio in ratios:
            d_top = set(_top_indices(d_flat, ratio).tolist())
            f_top = set(_top_indices(f_flat, ratio).tolist())
            inter = len(d_top & f_top)
            union = len(d_top | f_top)
            rows.append(
                {
                    **base,
                    "top_ratio": float(ratio),
                    "intersection": int(inter),
                    "jaccard": float(inter / max(union, 1)),
                    "recall_dyn_in_fast": float(inter / max(len(d_top), 1)),
                    "recall_fast_in_dyn": float(inter / max(len(f_top), 1)),
                }
            )
    return rows


def _delete_top(tam: np.ndarray, mask: np.ndarray, ratio: float) -> np.ndarray:
    out = np.asarray(tam, dtype=np.float32).copy()
    for sample_index in range(int(out.shape[0])):
        flat = out[sample_index].reshape(-1)
        idx = _top_indices(mask[sample_index], float(ratio))
        flat[idx] = 0.0
    return out


def _deletion_rows(
    *,
    tam: np.ndarray,
    masks: dict[str, np.ndarray],
    original_prob: np.ndarray,
    labels: np.ndarray,
    attacker,
    device,
    batch_size: int,
    ratios: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for mask_name, mask in masks.items():
        for ratio in ratios:
            deleted = _delete_top(tam, mask, float(ratio))
            prob = predict_probabilities(attacker, deleted, device=device, batch_size=int(batch_size))
            metrics = probability_metrics(original_prob, prob, labels)
            for sample_index in range(int(tam.shape[0])):
                sample_rows.append(
                    {
                        "sample_index": int(sample_index),
                        "mask_type": str(mask_name),
                        "deletion_ratio": float(ratio),
                        "accuracy": float(metrics["accuracy"][sample_index]),
                        "flip": float(metrics["flip"][sample_index]),
                        "original_class_probability": float(metrics["original_class_probability"][sample_index]),
                        "original_class_margin": float(metrics["original_class_margin"][sample_index]),
                        "original_class_margin_drop": float(metrics["original_class_margin_drop"][sample_index]),
                        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][sample_index]),
                        "js_div": float(metrics["js_div"][sample_index]),
                    }
                )
            summary_rows.append(
                {
                    "mask_type": str(mask_name),
                    "deletion_ratio": float(ratio),
                    "samples": int(tam.shape[0]),
                    "accuracy": float(np.mean(metrics["accuracy"])),
                    "flip": float(np.mean(metrics["flip"])),
                    "mean_original_class_probability": float(np.mean(metrics["original_class_probability"])),
                    "mean_original_class_margin": float(np.mean(metrics["original_class_margin"])),
                    "mean_original_class_margin_drop": float(np.mean(metrics["original_class_margin_drop"])),
                    "mean_entropy_gain": float(np.mean(metrics["normalized_entropy_gain"])),
                    "mean_js_div": float(np.mean(metrics["js_div"])),
                }
            )
    return sample_rows, summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = _run_dir(args)
    device = resolve_device(args.device)
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    deletion_ratios = _parse_floats(args.deletion_ratios)
    overlap_ratios = _parse_floats(args.overlap_ratios)
    with np.load(args.dynamask_archive, allow_pickle=False) as dyn, np.load(args.fast_archive, allow_pickle=False) as fast:
        dyn_source = np.asarray(dyn["source_indices"], dtype=np.int64)
        fast_rows = _rows_by_source(np.asarray(fast["source_indices"], dtype=np.int64), dyn_source)
        tam = np.asarray(dyn["tam"], dtype=np.float32)
        dyn_mask = np.asarray(dyn["mask"], dtype=np.float32)
        fast_mask = np.asarray(fast["mask"][fast_rows], dtype=np.float32)
        original_prob = np.asarray(dyn["pred_prob"], dtype=np.float32)
        labels = np.asarray(dyn["labels"], dtype=np.int64)
        sample_ids = np.asarray(dyn.get("sample_ids", np.arange(tam.shape[0]))).astype(str)
        if bool(args.write_aligned_fast_archive):
            np.savez_compressed(
                output_dir / "aligned_fast_keypoint_archive.npz",
                tam=np.asarray(fast["tam"][fast_rows]),
                mask=np.asarray(fast["mask"][fast_rows]),
                pred_prob=np.asarray(fast["pred_prob"][fast_rows], dtype=np.float32),
                pred_labels=np.asarray(fast["pred_labels"][fast_rows], dtype=np.int64),
                labels=np.asarray(fast["labels"][fast_rows], dtype=np.int64),
                source_indices=np.asarray(fast["source_indices"][fast_rows], dtype=np.int64),
                sample_ids=np.asarray(fast["sample_ids"][fast_rows]).astype(str),
            )
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(original_prob.shape[1]),
        device=device,
        max_trace_length=int(args.max_trace_length),
        rf_num_slots=int(tam.shape[-1]),
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    overlap = _overlap_rows(dyn_mask, fast_mask, overlap_ratios)
    deletion_samples, deletion_summary = _deletion_rows(
        tam=tam,
        masks={"dynamask": dyn_mask, "fast_keypoint": fast_mask},
        original_prob=original_prob,
        labels=labels,
        attacker=attacker,
        device=device,
        batch_size=int(args.batch_size),
        ratios=deletion_ratios,
    )
    overlap_summary = []
    for ratio in overlap_ratios:
        rows = [row for row in overlap if abs(float(row["top_ratio"]) - float(ratio)) < 1e-12]
        overlap_summary.append(
            {
                "top_ratio": float(ratio),
                "samples": int(len(rows)),
                "mean_jaccard": float(np.mean([row["jaccard"] for row in rows])) if rows else 0.0,
                "mean_recall_dyn_in_fast": float(np.mean([row["recall_dyn_in_fast"] for row in rows])) if rows else 0.0,
                "mean_cosine": float(np.mean([row["cosine"] for row in rows])) if rows else 0.0,
                "mean_pearson": float(np.mean([row["pearson"] for row in rows])) if rows else 0.0,
            }
        )
    _write_csv(output_dir / "mask_overlap_samples.csv", overlap)
    _write_csv(output_dir / "mask_overlap_summary.csv", overlap_summary)
    _write_csv(output_dir / "deletion_sample_results.csv", deletion_samples)
    _write_csv(output_dir / "deletion_summary.csv", deletion_summary)
    manifest = {
        "dynamask_archive": str(args.dynamask_archive),
        "fast_archive": str(args.fast_archive),
        "samples": int(tam.shape[0]),
        "classes": int(len(np.unique(labels))),
        "deletion_ratios": deletion_ratios,
        "overlap_ratios": overlap_ratios,
        "sample_ids_first": sample_ids[:10].tolist(),
        "overlap_summary": str(output_dir / "mask_overlap_summary.csv"),
        "deletion_summary": str(output_dir / "deletion_summary.csv"),
        "aligned_fast_archive": str(output_dir / "aligned_fast_keypoint_archive.npz") if bool(args.write_aligned_fast_archive) else "",
    }
    (output_dir / "fast_vs_dynamask_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
