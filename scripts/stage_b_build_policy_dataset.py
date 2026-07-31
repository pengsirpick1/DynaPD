# -*- coding: utf-8 -*-
"""Build the balanced policy dataset split for Stage B teacher/student work."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.data import load_cw_data
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR


DEFAULT_ARCHIVE = (
    "results/stage_a_rf_native_w1800_n96_s60_seed0/"
    "stage_a_masks_rf/all_masks.npz"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--classes", type=int, default=95)
    parser.add_argument("--full_cw", action="store_true", help="Use the full CW train/val/test splits instead of fixed per-class policy counts.")
    parser.add_argument("--train_per_class", type=int, default=70)
    parser.add_argument("--val_per_class", type=int, default=10)
    parser.add_argument("--test_per_class", type=int, default=20)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--budget_points", default="0.01,0.02,0.05,0.08,0.10")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _parse_budgets(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b_policy_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _load_labels_and_splits(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], str]:
    cfg = SimpleNamespace(
        data_root=str(args.data_root),
        seed=int(args.seed),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        max_samples=0,
        max_classes=0,
    )
    _raw, labels, trace_ids, splits, source = load_cw_data(cfg)
    return np.asarray(labels, dtype=np.int64), np.asarray(trace_ids).astype(str), splits, str(source)


def _choose_per_class(
    labels: np.ndarray,
    indices: np.ndarray,
    *,
    classes: np.ndarray,
    per_class: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    selected: list[int] = []
    for label in classes.tolist():
        candidates = np.asarray(indices[np.asarray(labels[indices], dtype=np.int64) == int(label)], dtype=np.int64)
        if len(candidates) < int(per_class):
            raise ValueError(f"Class {label} has only {len(candidates)} samples in split, need {per_class}.")
        picked = rng.choice(candidates, size=int(per_class), replace=False)
        selected.extend(int(item) for item in picked.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def _balanced_budgets(n: int, budgets: list[float], seed: int) -> np.ndarray:
    if not budgets:
        return np.zeros(int(n), dtype=np.float32)
    values = np.asarray(budgets, dtype=np.float32)
    tiled = np.resize(values, int(n)).astype(np.float32)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(tiled)
    return tiled


def _counts_rows(name: str, labels: np.ndarray, indices: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    counts = Counter(int(item) for item in labels[np.asarray(indices, dtype=np.int64)].tolist())
    for label in sorted(counts):
        rows.append({"split": name, "label": int(label), "count": int(counts[label])})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _archive_audit(path: str, classes: int) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        return {"archive": str(path), "exists": False}
    with np.load(target, allow_pickle=False) as arrays:
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        sample_ids = np.asarray(arrays.get("sample_ids", np.arange(labels.size))).astype(str)
    counts = Counter(int(item) for item in labels.tolist())
    strict_one_each = bool(len(labels) == int(classes) and len(counts) == int(classes) and all(count == 1 for count in counts.values()))
    return {
        "archive": str(path),
        "exists": True,
        "total": int(labels.size),
        "num_classes": int(len(counts)),
        "strict_one_per_class": strict_one_each,
        "min_per_class": int(min(counts.values())) if counts else 0,
        "max_per_class": int(max(counts.values())) if counts else 0,
        "duplicate_classes": {str(label): int(count) for label, count in sorted(counts.items()) if int(count) > 1},
        "missing_classes": [int(label) for label in range(int(classes)) if int(label) not in counts],
        "first_sample_ids": sample_ids[:10].tolist(),
        "counts": {str(label): int(count) for label, count in sorted(counts.items())},
    }


def main() -> None:
    args = parse_args()
    output_dir = _run_dir(args)
    budgets = _parse_budgets(args.budget_points)
    labels, trace_ids, splits, source = _load_labels_and_splits(args)
    all_classes = np.unique(labels)
    if len(all_classes) != int(args.classes):
        raise ValueError(f"Expected {args.classes} classes, found {len(all_classes)} classes in {source}.")

    if bool(args.full_cw):
        train_idx = np.asarray(splits["train"], dtype=np.int64)
        val_idx = np.asarray(splits["val"], dtype=np.int64)
        test_idx = np.asarray(splits["test"], dtype=np.int64)
    else:
        train_idx = _choose_per_class(labels, splits["train"], classes=all_classes, per_class=int(args.train_per_class), seed=int(args.seed) + 11)
        val_idx = _choose_per_class(labels, splits["val"], classes=all_classes, per_class=int(args.val_per_class), seed=int(args.seed) + 23)
        test_idx = _choose_per_class(labels, splits["test"], classes=all_classes, per_class=int(args.test_per_class), seed=int(args.seed) + 37)
    train_budget = _balanced_budgets(len(train_idx), budgets, int(args.seed) + 101)
    val_budget = _balanced_budgets(len(val_idx), budgets, int(args.seed) + 103)
    test_budget = _balanced_budgets(len(test_idx), budgets, int(args.seed) + 107)

    np.savez_compressed(
        output_dir / "policy_splits.npz",
        train_indices=train_idx,
        val_indices=val_idx,
        test_indices=test_idx,
        train_labels=labels[train_idx],
        val_labels=labels[val_idx],
        test_labels=labels[test_idx],
        train_sample_ids=trace_ids[train_idx],
        val_sample_ids=trace_ids[val_idx],
        test_sample_ids=trace_ids[test_idx],
        train_budget=train_budget,
        val_budget=val_budget,
        test_budget=test_budget,
        budget_points=np.asarray(budgets, dtype=np.float32),
    )

    rows: list[dict[str, Any]] = []
    rows.extend(_counts_rows("policy_train", labels, train_idx))
    rows.extend(_counts_rows("policy_val", labels, val_idx))
    rows.extend(_counts_rows("policy_test", labels, test_idx))
    _write_csv(output_dir / "policy_split_class_counts.csv", rows)

    manifest = {
        "source": source,
        "classes": int(len(all_classes)),
        "mode": "full_cw" if bool(args.full_cw) else "balanced_policy",
        "policy_total": int(len(train_idx) + len(val_idx) + len(test_idx)),
        "policy_train": int(len(train_idx)),
        "policy_val": int(len(val_idx)),
        "policy_test": int(len(test_idx)),
        "per_class": {
            "train": "full_split" if bool(args.full_cw) else int(args.train_per_class),
            "val": "full_split" if bool(args.full_cw) else int(args.val_per_class),
            "test": "full_split" if bool(args.full_cw) else int(args.test_per_class),
            "total": "full_split" if bool(args.full_cw) else int(args.train_per_class + args.val_per_class + args.test_per_class),
        },
        "budget_points": budgets,
        "split_file": str(output_dir / "policy_splits.npz"),
        "class_count_csv": str(output_dir / "policy_split_class_counts.csv"),
        "pilot_n96_archive_audit": _archive_audit(args.archive, int(args.classes)),
    }
    (output_dir / "policy_dataset_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
