# -*- coding: utf-8 -*-
"""Audit a full Stage B archive by explicit train/val/test split indices."""

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

from dmmp.utils.config import DEFAULT_OUTPUT_DIR


DEFAULT_ARCHIVE = "results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz"
DEFAULT_SPLIT_FILE = "results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="stage_b_full_archive_split_audit")
    return parser.parse_args()


def _run_dir(args: argparse.Namespace) -> Path:
    target = Path(args.output_dir) / str(args.run_name)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _positions_for_source_indices(archive_source: np.ndarray, requested: np.ndarray) -> np.ndarray:
    position = {int(source): row for row, source in enumerate(np.asarray(archive_source, dtype=np.int64).tolist())}
    missing = [int(source) for source in np.asarray(requested, dtype=np.int64).tolist() if int(source) not in position]
    if missing:
        raise ValueError(f"Archive is missing {len(missing)} requested source indices; first missing={missing[:5]}")
    return np.asarray([position[int(source)] for source in np.asarray(requested, dtype=np.int64).tolist()], dtype=np.int64)


def main() -> None:
    args = parse_args()
    output_dir = _run_dir(args)
    rows = []
    with np.load(args.archive, allow_pickle=False) as archive, np.load(args.split_file, allow_pickle=False) as splits:
        source = np.asarray(archive["source_indices"], dtype=np.int64)
        for split_name in ("train", "val", "test"):
            split_indices = np.asarray(splits[f"{split_name}_indices"], dtype=np.int64)
            rows_in_archive = _positions_for_source_indices(source, split_indices)
            labels = np.asarray(archive["labels"][rows_in_archive], dtype=np.int64)
            pred = np.asarray(archive["pred_labels"][rows_in_archive], dtype=np.int64)
            prob = np.asarray(archive["pred_prob"][rows_in_archive], dtype=np.float32)
            conf = prob[np.arange(len(pred)), pred]
            per_class = [int(np.sum(labels == label)) for label in np.unique(labels)]
            rows.append(
                {
                    "split": split_name,
                    "samples": int(len(rows_in_archive)),
                    "classes": int(len(np.unique(labels))),
                    "clean_rf_accuracy": float(np.mean(pred == labels)),
                    "mean_pred_confidence": float(np.mean(conf)),
                    "min_per_class": int(min(per_class)) if per_class else 0,
                    "max_per_class": int(max(per_class)) if per_class else 0,
                    "archive_row_min": int(rows_in_archive.min()) if rows_in_archive.size else -1,
                    "archive_row_max": int(rows_in_archive.max()) if rows_in_archive.size else -1,
                }
            )
    with (output_dir / "split_clean_accuracy.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "archive": str(args.archive),
        "split_file": str(args.split_file),
        "rows": rows,
        "csv": str(output_dir / "split_clean_accuracy.csv"),
    }
    (output_dir / "split_clean_accuracy.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
