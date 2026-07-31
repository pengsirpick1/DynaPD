# -*- coding: utf-8 -*-
"""Generate a random additive defense baseline in purified-manifest format."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz
from dmmp.purifier.config import PurifierConfig
from dmmp.utils import write_json


MANIFEST_COLUMNS = [
    "source_id",
    "clean_index",
    "defended_index",
    "defended_local_index",
    "variant_id",
    "split",
    "class_id",
    "defended_path",
    "defended_length",
    "purified_path",
    "purified_index",
    "purifier_checkpoint",
    "diffusion_steps",
    "sampling_steps",
    "sampling_seed",
    "representation",
    "legalization_version",
    "output_length_policy",
    "output_length",
    "random_policy",
    "random_seed",
    "clean_positive_packets",
    "clean_negative_packets",
    "clean_total_packets",
    "dummy_positive_packets",
    "dummy_negative_packets",
    "dummy_total_packets",
    "output_positive_packets",
    "output_negative_packets",
    "output_total_packets",
    "extra_positive_packets",
    "extra_negative_packets",
    "extra_total_packets",
    "overhead_ratio",
    "aggregate_target_overhead",
    "additive_count_feasible",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random add-only dummy-packet baseline.")
    parser.add_argument("--purifier-run-dir", required=True, help="Run dir used only for config/test manifest provenance.")
    parser.add_argument("--output-manifest", default="", help="Default: <run-dir>/manifests/random_additive_b045_manifest.csv")
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/random_additive_b045")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--overhead", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=76045)
    parser.add_argument("--shard-size", type=int, default=1024)
    parser.add_argument("--dummy-time", choices=["uniform_trace", "uniform_80"], default="uniform_trace")
    parser.add_argument("--dummy-direction", choices=["balanced_random", "uniform_random"], default="uniform_random")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


class CleanStore:
    def __init__(self, path: str | Path):
        clean_path = Path(path)
        x_map = stored_npy_from_npz(clean_path, "X")
        y_map = stored_npy_from_npz(clean_path, "y")
        self._payload = None
        if x_map is not None and y_map is not None:
            self.x = x_map
            self.y = np.asarray(y_map, dtype=np.int64)
        else:
            self._payload = np.load(clean_path, allow_pickle=False)
            self.x = self._payload["X"]
            self.y = np.asarray(self._payload["y"], dtype=np.int64)

    def row(self, index: int, seq_length: int) -> np.ndarray:
        values = np.asarray(self.x[int(index)], dtype=np.float32).reshape(-1)
        out = np.zeros(int(seq_length), dtype=np.float32)
        take = min(out.size, values.size)
        if take:
            out[:take] = values[:take]
        return out

    def close(self) -> None:
        if self._payload is not None:
            self._payload.close()


def _manifest_for_split(cfg: PurifierConfig, split: str) -> str:
    if split == "train":
        return cfg.train_manifest
    if split == "validation":
        return cfg.validation_manifest
    if split == "test":
        return cfg.test_manifest
    raise ValueError(f"Unsupported split {split!r}")


def _read_rows(paths: list[str | Path], wanted_splits: set[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with Path(path).open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["split"] in wanted_splits:
                    rows.append(row)
    return rows


def _counts(values: np.ndarray) -> dict[str, int]:
    x = np.asarray(values)
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    return {"positive": pos, "negative": neg, "total": pos + neg}


def _tail_length(values: np.ndarray) -> int:
    nz = np.flatnonzero(np.asarray(values) != 0)
    return int(nz[-1] + 1) if nz.size else 0


def _row_int(row: dict[str, str], *names: str, default: int = 0) -> int:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value != "":
            return int(value)
    return int(default)


def _make_dummy_events(
    clean_nonzero: np.ndarray,
    *,
    dummy_count: int,
    rng: np.random.Generator,
    seq_length: int,
    value_clip: float,
    dummy_time: str,
    dummy_direction: str,
) -> tuple[np.ndarray, dict[str, int]]:
    if int(dummy_count) <= 0:
        return np.zeros(0, dtype=np.float32), {"positive": 0, "negative": 0, "total": 0}
    if dummy_direction == "balanced_random":
        positive_count = int(dummy_count) // 2
        negative_count = int(dummy_count) - positive_count
        signs = np.concatenate([np.ones(positive_count, dtype=np.float32), -np.ones(negative_count, dtype=np.float32)])
        rng.shuffle(signs)
    else:
        signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=int(dummy_count), replace=True)
    if dummy_time == "uniform_80" or clean_nonzero.size == 0:
        max_time = float(value_clip)
    else:
        max_time = float(np.max(np.abs(clean_nonzero)))
        max_time = min(max(max_time, 1.0e-6), float(value_clip))
    times = rng.uniform(0.0, max_time, size=int(dummy_count)).astype(np.float32)
    # Avoid exactly zero, which would disappear in the signed representation.
    times = np.maximum(times, np.float32(1.0e-6))
    events = signs.astype(np.float32) * times
    stats = _counts(events)
    return events.astype(np.float32, copy=False), stats


def _random_defense_trace(
    clean: np.ndarray,
    *,
    overhead: float,
    rng: np.random.Generator,
    seq_length: int,
    value_clip: float,
    dummy_time: str,
    dummy_direction: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    clean_nonzero = np.asarray(clean, dtype=np.float32).reshape(-1)
    clean_nonzero = clean_nonzero[clean_nonzero != 0]
    clean_counts = _counts(clean_nonzero)
    requested_dummy = int(round(float(overhead) * int(clean_counts["total"])))
    max_dummy = max(0, int(seq_length) - int(clean_counts["total"]))
    dummy_count = min(requested_dummy, max_dummy)
    dummies, dummy_counts = _make_dummy_events(
        clean_nonzero,
        dummy_count=dummy_count,
        rng=rng,
        seq_length=int(seq_length),
        value_clip=float(value_clip),
        dummy_time=str(dummy_time),
        dummy_direction=str(dummy_direction),
    )
    combined = np.concatenate([clean_nonzero, dummies]).astype(np.float32, copy=False)
    order = np.argsort(np.abs(combined), kind="stable")
    combined = combined[order]
    out = np.zeros(int(seq_length), dtype=np.float32)
    take = min(out.size, combined.size)
    if take:
        out[:take] = combined[:take]
    out_counts = _counts(out)
    extra_pos = int(out_counts["positive"]) - int(clean_counts["positive"])
    extra_neg = int(out_counts["negative"]) - int(clean_counts["negative"])
    extra_total = int(out_counts["total"]) - int(clean_counts["total"])
    return out, {
        "clean": clean_counts,
        "dummy": dummy_counts,
        "output": out_counts,
        "requested_dummy_total": int(requested_dummy),
        "capped_dummy_total": int(dummy_count),
        "extra_positive": int(extra_pos),
        "extra_negative": int(extra_neg),
        "extra_total": int(extra_total),
        "overhead_ratio": float(extra_total / max(int(clean_counts["total"]), 1)),
        "tail_length": _tail_length(out),
        "additive_count_feasible": int(extra_pos >= 0 and extra_neg >= 0),
    }


def _flush_shard(
    *,
    output_path: Path,
    rows: list[dict[str, Any]],
    arrays: list[np.ndarray],
    writer: csv.DictWriter,
) -> int:
    if not rows:
        return 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.stack(arrays, axis=0).astype(np.float32, copy=False)
    np.savez_compressed(
        output_path,
        X=x,
        y=np.asarray([row["class_id"] for row in rows], dtype=np.int64),
        source_id=np.asarray([row["source_id"] for row in rows], dtype=np.int64),
        clean_index=np.asarray([row["clean_index"] for row in rows], dtype=np.int64),
        defended_index=np.asarray([row["defended_index"] for row in rows], dtype=np.int64),
        defended_local_index=np.asarray([row["defended_local_index"] for row in rows], dtype=np.int64),
        variant_id=np.asarray([row["variant_id"] for row in rows], dtype=np.int64),
        clean_positive_packets=np.asarray([row["clean_positive_packets"] for row in rows], dtype=np.int64),
        clean_negative_packets=np.asarray([row["clean_negative_packets"] for row in rows], dtype=np.int64),
        dummy_positive_packets=np.asarray([row["dummy_positive_packets"] for row in rows], dtype=np.int64),
        dummy_negative_packets=np.asarray([row["dummy_negative_packets"] for row in rows], dtype=np.int64),
    )
    for local_index, row in enumerate(rows):
        row_out = dict(row)
        row_out["purified_path"] = str(output_path.resolve())
        row_out["purified_index"] = int(local_index)
        writer.writerow(row_out)
    return len(rows)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.purifier_run_dir).resolve()
    cfg = PurifierConfig.from_mapping(json.loads((run_dir / "config.json").read_text(encoding="utf-8")))
    wanted_splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    split_paths = [_manifest_for_split(cfg, split) for split in sorted(wanted_splits)]
    input_rows = _read_rows(split_paths, wanted_splits)
    if not input_rows:
        raise ValueError("No manifest rows selected.")
    output_manifest = Path(args.output_manifest).resolve() if args.output_manifest else run_dir / "manifests" / "random_additive_b045_manifest.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "random_additive_b045"
    if output_manifest.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output_manifest}")
    if output_dir.exists() and any(output_dir.glob("**/*.npz")) and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing shards under: {output_dir}")

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    clean_store = CleanStore(cfg.clean_path)
    rng = np.random.default_rng(int(args.seed))
    shard_rows: list[dict[str, Any]] = []
    shard_arrays: list[np.ndarray] = []
    shard_index_by_split: dict[str, int] = {split: 0 for split in wanted_splits}
    split_counts: Counter[str] = Counter()
    sums = Counter()
    feasible_count = 0
    total_count = 0
    try:
        with output_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            active_split: str | None = None
            for row_index, row in enumerate(input_rows, start=1):
                split = row["split"]
                if active_split is None:
                    active_split = split
                if split != active_split or len(shard_rows) >= int(args.shard_size):
                    output_path = output_dir / str(active_split) / f"random_additive_{active_split}_shard{shard_index_by_split[str(active_split)]:05d}.npz"
                    written = _flush_shard(output_path=output_path, rows=shard_rows, arrays=shard_arrays, writer=writer)
                    split_counts[str(active_split)] += written
                    print(f"[random additive] split={active_split} shard={shard_index_by_split[str(active_split)]} rows={written}", flush=True)
                    shard_rows, shard_arrays = [], []
                    shard_index_by_split[str(active_split)] += 1
                    active_split = split

                clean = clean_store.row(int(row["clean_index"]), int(cfg.seq_length))
                defended, stats = _random_defense_trace(
                    clean,
                    overhead=float(args.overhead),
                    rng=rng,
                    seq_length=int(cfg.seq_length),
                    value_clip=float(cfg.value_clip),
                    dummy_time=str(args.dummy_time),
                    dummy_direction=str(args.dummy_direction),
                )
                feasible_count += int(stats["additive_count_feasible"])
                total_count += 1
                sums["clean_total"] += float(stats["clean"]["total"])
                sums["output_total"] += float(stats["output"]["total"])
                sums["extra_total"] += float(stats["extra_total"])
                sums["extra_positive"] += float(stats["extra_positive"])
                sums["extra_negative"] += float(stats["extra_negative"])
                sums["overhead_ratio"] += float(stats["overhead_ratio"])
                row_out: dict[str, Any] = {
                    "source_id": _row_int(row, "source_id"),
                    "clean_index": _row_int(row, "clean_index"),
                    "defended_index": _row_int(row, "defended_index", "defended_global_index"),
                    "defended_local_index": _row_int(row, "defended_local_index", "shard_local_index"),
                    "variant_id": _row_int(row, "variant_id"),
                    "split": split,
                    "class_id": _row_int(row, "class_id"),
                    "defended_path": row["defended_path"],
                    "defended_length": row.get("defended_length", ""),
                    "purified_path": "",
                    "purified_index": 0,
                    "purifier_checkpoint": "",
                    "diffusion_steps": 0,
                    "sampling_steps": 0,
                    "sampling_seed": int(args.seed),
                    "representation": "fixed_length_signed_time_sequence_v1_scaled",
                    "legalization_version": "random_additive_v1",
                    "output_length_policy": f"random_additive_{float(args.overhead):.3f}",
                    "output_length": int(stats["tail_length"]),
                    "random_policy": f"dummy_time={args.dummy_time};dummy_direction={args.dummy_direction}",
                    "random_seed": int(args.seed),
                    "clean_positive_packets": int(stats["clean"]["positive"]),
                    "clean_negative_packets": int(stats["clean"]["negative"]),
                    "clean_total_packets": int(stats["clean"]["total"]),
                    "dummy_positive_packets": int(stats["dummy"]["positive"]),
                    "dummy_negative_packets": int(stats["dummy"]["negative"]),
                    "dummy_total_packets": int(stats["dummy"]["total"]),
                    "output_positive_packets": int(stats["output"]["positive"]),
                    "output_negative_packets": int(stats["output"]["negative"]),
                    "output_total_packets": int(stats["output"]["total"]),
                    "extra_positive_packets": int(stats["extra_positive"]),
                    "extra_negative_packets": int(stats["extra_negative"]),
                    "extra_total_packets": int(stats["extra_total"]),
                    "overhead_ratio": float(stats["overhead_ratio"]),
                    "aggregate_target_overhead": float(args.overhead),
                    "additive_count_feasible": int(stats["additive_count_feasible"]),
                }
                shard_rows.append(row_out)
                shard_arrays.append(defended.astype(np.float32, copy=False))
                if row_index == 1 or row_index % 5000 == 0 or row_index == len(input_rows):
                    print(f"[random additive] row={row_index}/{len(input_rows)} feasible={feasible_count}/{total_count}", flush=True)
            if shard_rows:
                assert active_split is not None
                output_path = output_dir / str(active_split) / f"random_additive_{active_split}_shard{shard_index_by_split[str(active_split)]:05d}.npz"
                written = _flush_shard(output_path=output_path, rows=shard_rows, arrays=shard_arrays, writer=writer)
                split_counts[str(active_split)] += written
                print(f"[random additive] split={active_split} shard={shard_index_by_split[str(active_split)]} rows={written}", flush=True)
    finally:
        clean_store.close()

    summary = {
        "output_manifest": str(output_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "splits": sorted(wanted_splits),
        "rows": int(total_count),
        "additive_count_feasible_rows": int(feasible_count),
        "additive_count_feasible_rate": float(feasible_count / max(total_count, 1)),
        "target_overhead": float(args.overhead),
        "dummy_time": str(args.dummy_time),
        "dummy_direction": str(args.dummy_direction),
        "seed": int(args.seed),
        "split_counts": dict(split_counts),
        "mean_clean_total_packets": sums["clean_total"] / max(total_count, 1),
        "mean_output_total_packets": sums["output_total"] / max(total_count, 1),
        "mean_extra_total_packets": sums["extra_total"] / max(total_count, 1),
        "mean_extra_positive_packets": sums["extra_positive"] / max(total_count, 1),
        "mean_extra_negative_packets": sums["extra_negative"] / max(total_count, 1),
        "mean_overhead_ratio": sums["overhead_ratio"] / max(total_count, 1),
        "aggregate_overhead_ratio": sums["extra_total"] / max(sums["clean_total"], 1.0),
        "note": "Random add-only baseline in the same fixed_length_signed_time_sequence_v1_scaled format.",
    }
    summary_path = output_manifest.with_name(output_manifest.stem + "_summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
