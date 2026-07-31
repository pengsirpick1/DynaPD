# -*- coding: utf-8 -*-
"""Truncate generated/purified traces by clean directional packet counts.

For each generated trace, scan from the beginning and keep the shortest prefix
whose positive and negative packet counts are both at least the corresponding
clean trace counts. This is a count-level feasibility transform for testing
add-only-by-direction defense constraints.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import stored_npy_from_npz
from dmmp.purifier.config import PurifierConfig
from dmmp.utils import write_json


EXTRA_COLUMNS = [
    "truncation_policy",
    "clean_positive_packets",
    "clean_negative_packets",
    "clean_total_packets",
    "truncated_positive_packets",
    "truncated_negative_packets",
    "truncated_total_packets",
    "extra_positive_packets",
    "extra_negative_packets",
    "extra_total_packets",
    "overhead_ratio",
    "direction_count_feasible",
    "simple_prefix_feasible",
    "missing_direction_repaired",
    "compact_quota_repaired",
    "truncation_stop_position",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a direction-count-truncated generated dataset.")
    parser.add_argument("--purifier-run-dir", required=True)
    parser.add_argument("--input-manifest", default="", help="Default: <run-dir>/manifests/purified_dataset_manifest.csv")
    parser.add_argument("--output-manifest", default="", help="Default: <run-dir>/manifests/diffusion_defense_truncated_manifest.csv")
    parser.add_argument("--output-dir", default="", help="Default: <run-dir>/diffusion_defense_truncated")
    parser.add_argument("--splits", default="test")
    parser.add_argument("--shard-size", type=int, default=1024)
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


class NpzCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max(1, int(max_items))
        self.cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def get(self, path: str | Path) -> dict[str, np.ndarray]:
        text = str(Path(path).resolve())
        if text in self.cache:
            item = self.cache.pop(text)
            self.cache[text] = item
            return item
        payload = np.load(text, allow_pickle=False)
        try:
            arrays = {key: np.asarray(payload[key]) for key in payload.files}
        finally:
            payload.close()
        self.cache[text] = arrays
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        return arrays


def _read_csv(path: str | Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def _counts(values: np.ndarray) -> dict[str, int]:
    x = np.asarray(values)
    pos = int(np.sum(x > 0))
    neg = int(np.sum(x < 0))
    return {"positive": pos, "negative": neg, "total": pos + neg}


def _tail_length(values: np.ndarray) -> int:
    nz = np.flatnonzero(np.asarray(values) != 0)
    return int(nz[-1] + 1) if nz.size else 0


def _first_position_reaching(gen: np.ndarray, *, positive_target: int | None = None, negative_target: int | None = None) -> tuple[int, int, int]:
    pos = 0
    neg = 0
    for index, value in enumerate(gen):
        if value > 0:
            pos += 1
        elif value < 0:
            neg += 1
        pos_ok = positive_target is None or pos >= int(positive_target)
        neg_ok = negative_target is None or neg >= int(negative_target)
        if pos_ok and neg_ok:
            return int(index), int(pos), int(neg)
    return int(len(gen) - 1), int(pos), int(neg)


def _append_missing_direction(
    out: np.ndarray,
    *,
    write_start: int,
    missing_positive: int,
    missing_negative: int,
    value_clip: float,
) -> int:
    write = int(write_start)
    if missing_positive > 0:
        end = min(out.size, write + int(missing_positive))
        out[write:end] = abs(float(value_clip))
        write = end
    if missing_negative > 0:
        end = min(out.size, write + int(missing_negative))
        out[write:end] = -abs(float(value_clip))
        write = end
    return int(write)


def _compact_quota_repair(gen: np.ndarray, clean_counts: dict[str, int], *, value_clip: float) -> tuple[np.ndarray, int, int]:
    out = np.zeros_like(gen, dtype=np.float32)
    pos = 0
    neg = 0
    write = 0
    target_pos = int(clean_counts["positive"])
    target_neg = int(clean_counts["negative"])
    for value in gen:
        if write >= out.size:
            break
        if value > 0 and pos < target_pos:
            out[write] = value
            pos += 1
            write += 1
        elif value < 0 and neg < target_neg:
            out[write] = value
            neg += 1
            write += 1
        if pos >= target_pos and neg >= target_neg:
            break
    missing_positive = max(0, target_pos - pos)
    missing_negative = max(0, target_neg - neg)
    before_append = write
    write = _append_missing_direction(
        out,
        write_start=write,
        missing_positive=missing_positive,
        missing_negative=missing_negative,
        value_clip=float(value_clip),
    )
    return out, int(before_append), int(write)


def _truncate(generated: np.ndarray, clean_counts: dict[str, int], *, value_clip: float) -> tuple[np.ndarray, dict[str, Any]]:
    gen = np.asarray(generated, dtype=np.float32).reshape(-1)
    pos = 0
    neg = 0
    stop = len(gen) - 1
    simple_feasible = False
    repaired = False
    compact_repaired = False
    for index, value in enumerate(gen):
        if value > 0:
            pos += 1
        elif value < 0:
            neg += 1
        if pos >= int(clean_counts["positive"]) and neg >= int(clean_counts["negative"]):
            stop = index
            simple_feasible = True
            break
    out = np.zeros_like(gen, dtype=np.float32)
    if simple_feasible and stop >= 0:
        out[: stop + 1] = gen[: stop + 1]
    else:
        repaired = True
        full_counts = _counts(gen)
        if int(full_counts["positive"]) >= int(clean_counts["positive"]):
            stop, pos, neg = _first_position_reaching(gen, positive_target=int(clean_counts["positive"]))
        elif int(full_counts["negative"]) >= int(clean_counts["negative"]):
            stop, pos, neg = _first_position_reaching(gen, negative_target=int(clean_counts["negative"]))
        else:
            stop = -1
            pos = 0
            neg = 0
        if stop >= 0:
            out[: stop + 1] = gen[: stop + 1]
        missing_positive = max(0, int(clean_counts["positive"]) - int(pos))
        missing_negative = max(0, int(clean_counts["negative"]) - int(neg))
        _append_missing_direction(
            out,
            write_start=int(stop) + 1,
            missing_positive=missing_positive,
            missing_negative=missing_negative,
            value_clip=float(value_clip),
        )
    out_counts = _counts(out)
    feasible = out_counts["positive"] >= clean_counts["positive"] and out_counts["negative"] >= clean_counts["negative"]
    if not feasible:
        compact_repaired = True
        repaired = True
        out, stop, _ = _compact_quota_repair(gen, clean_counts, value_clip=float(value_clip))
        out_counts = _counts(out)
        feasible = out_counts["positive"] >= clean_counts["positive"] and out_counts["negative"] >= clean_counts["negative"]
    return out, {
        "feasible": bool(feasible),
        "simple_feasible": bool(simple_feasible),
        "repaired": bool(repaired),
        "compact_repaired": bool(compact_repaired),
        "stop_position": int(stop),
        "positive": int(out_counts["positive"]),
        "negative": int(out_counts["negative"]),
        "total": int(out_counts["total"]),
        "tail_length": _tail_length(out),
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
        truncated_positive_packets=np.asarray([row["truncated_positive_packets"] for row in rows], dtype=np.int64),
        truncated_negative_packets=np.asarray([row["truncated_negative_packets"] for row in rows], dtype=np.int64),
        truncation_stop_position=np.asarray([row["truncation_stop_position"] for row in rows], dtype=np.int64),
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
    input_manifest = Path(args.input_manifest).resolve() if args.input_manifest else run_dir / "manifests" / "purified_dataset_manifest.csv"
    output_manifest = Path(args.output_manifest).resolve() if args.output_manifest else run_dir / "manifests" / "diffusion_defense_truncated_manifest.csv"
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "diffusion_defense_truncated"
    if output_manifest.exists() and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing manifest: {output_manifest}")
    if output_dir.exists() and any(output_dir.glob("**/*.npz")) and not bool(args.overwrite):
        raise FileExistsError(f"Refusing to overwrite existing shards under: {output_dir}")

    wanted_splits = {item.strip() for item in str(args.splits).split(",") if item.strip()}
    input_rows, input_columns = _read_csv(input_manifest)
    input_rows = [row for row in input_rows if row.get("split", "") in wanted_splits]
    if not input_rows:
        raise ValueError(f"No rows selected from {input_manifest}")
    output_columns = list(input_columns)
    for column in EXTRA_COLUMNS:
        if column not in output_columns:
            output_columns.append(column)

    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    cache = NpzCache(max_items=8)
    clean_store = CleanStore(cfg.clean_path)
    shard_rows: list[dict[str, Any]] = []
    shard_arrays: list[np.ndarray] = []
    shard_index_by_split: dict[str, int] = {split: 0 for split in wanted_splits}
    split_counts: dict[str, int] = {split: 0 for split in wanted_splits}
    feasible_count = 0
    simple_feasible_count = 0
    repaired_count = 0
    compact_repaired_count = 0
    total_count = 0
    sums = {
        "clean_total": 0.0,
        "truncated_total": 0.0,
        "extra_total": 0.0,
        "extra_positive": 0.0,
        "extra_negative": 0.0,
        "overhead_ratio": 0.0,
    }
    try:
        with output_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=output_columns)
            writer.writeheader()
            active_split: str | None = None
            for row_index, row in enumerate(input_rows, start=1):
                split = row["split"]
                if active_split is None:
                    active_split = split
                if split != active_split or len(shard_rows) >= int(args.shard_size):
                    output_path = output_dir / str(active_split) / f"truncated_{active_split}_shard{shard_index_by_split[str(active_split)]:05d}.npz"
                    written = _flush_shard(output_path=output_path, rows=shard_rows, arrays=shard_arrays, writer=writer)
                    split_counts[str(active_split)] += written
                    print(f"[truncate] split={active_split} shard={shard_index_by_split[str(active_split)]} rows={written}", flush=True)
                    shard_rows, shard_arrays = [], []
                    shard_index_by_split[str(active_split)] += 1
                    active_split = split

                clean = clean_store.row(int(row["clean_index"]), int(cfg.seq_length))
                clean_counts = _counts(clean)
                payload = cache.get(row["purified_path"])
                generated = np.asarray(payload["X"][int(row["purified_index"])], dtype=np.float32)
                truncated, info = _truncate(generated, clean_counts, value_clip=float(cfg.value_clip))
                feasible_count += int(bool(info["feasible"]))
                simple_feasible_count += int(bool(info["simple_feasible"]))
                repaired_count += int(bool(info["repaired"]))
                compact_repaired_count += int(bool(info["compact_repaired"]))
                total_count += 1
                extra_pos = int(info["positive"]) - int(clean_counts["positive"])
                extra_neg = int(info["negative"]) - int(clean_counts["negative"])
                extra_total = int(info["total"]) - int(clean_counts["total"])
                overhead_ratio = float(extra_total) / max(float(clean_counts["total"]), 1.0)
                sums["clean_total"] += float(clean_counts["total"])
                sums["truncated_total"] += float(info["total"])
                sums["extra_total"] += float(extra_total)
                sums["extra_positive"] += float(extra_pos)
                sums["extra_negative"] += float(extra_neg)
                sums["overhead_ratio"] += float(overhead_ratio)

                row_out: dict[str, Any] = dict(row)
                row_out.update(
                    {
                        "output_length_policy": "direction_count_truncated",
                        "output_length": int(info["tail_length"]),
                        "truncation_policy": "shortest_prefix_pos_neg_at_least_clean",
                        "clean_positive_packets": int(clean_counts["positive"]),
                        "clean_negative_packets": int(clean_counts["negative"]),
                        "clean_total_packets": int(clean_counts["total"]),
                        "truncated_positive_packets": int(info["positive"]),
                        "truncated_negative_packets": int(info["negative"]),
                        "truncated_total_packets": int(info["total"]),
                        "extra_positive_packets": int(extra_pos),
                        "extra_negative_packets": int(extra_neg),
                        "extra_total_packets": int(extra_total),
                        "overhead_ratio": float(overhead_ratio),
                        "direction_count_feasible": int(bool(info["feasible"])),
                        "simple_prefix_feasible": int(bool(info["simple_feasible"])),
                        "missing_direction_repaired": int(bool(info["repaired"])),
                        "compact_quota_repaired": int(bool(info["compact_repaired"])),
                        "truncation_stop_position": int(info["stop_position"]),
                    }
                )
                shard_rows.append(row_out)
                shard_arrays.append(truncated.astype(np.float32, copy=False))
                if row_index == 1 or row_index % 5000 == 0 or row_index == len(input_rows):
                    print(f"[truncate] row={row_index}/{len(input_rows)} feasible={feasible_count}/{total_count}", flush=True)

            if shard_rows:
                assert active_split is not None
                output_path = output_dir / str(active_split) / f"truncated_{active_split}_shard{shard_index_by_split[str(active_split)]:05d}.npz"
                written = _flush_shard(output_path=output_path, rows=shard_rows, arrays=shard_arrays, writer=writer)
                split_counts[str(active_split)] += written
                print(f"[truncate] split={active_split} shard={shard_index_by_split[str(active_split)]} rows={written}", flush=True)
    finally:
        clean_store.close()

    summary = {
        "input_manifest": str(input_manifest.resolve()),
        "output_manifest": str(output_manifest.resolve()),
        "output_dir": str(output_dir.resolve()),
        "splits": sorted(wanted_splits),
        "rows": int(total_count),
        "direction_count_feasible_rows": int(feasible_count),
        "direction_count_feasible_rate": float(feasible_count / max(total_count, 1)),
        "simple_prefix_feasible_rows": int(simple_feasible_count),
        "simple_prefix_feasible_rate": float(simple_feasible_count / max(total_count, 1)),
        "missing_direction_repaired_rows": int(repaired_count),
        "missing_direction_repaired_rate": float(repaired_count / max(total_count, 1)),
        "compact_quota_repaired_rows": int(compact_repaired_count),
        "compact_quota_repaired_rate": float(compact_repaired_count / max(total_count, 1)),
        "split_counts": split_counts,
        "mean_clean_total_packets": sums["clean_total"] / max(total_count, 1),
        "mean_truncated_total_packets": sums["truncated_total"] / max(total_count, 1),
        "mean_extra_total_packets": sums["extra_total"] / max(total_count, 1),
        "mean_extra_positive_packets": sums["extra_positive"] / max(total_count, 1),
        "mean_extra_negative_packets": sums["extra_negative"] / max(total_count, 1),
        "mean_overhead_ratio": sums["overhead_ratio"] / max(total_count, 1),
        "aggregate_overhead_ratio": sums["extra_total"] / max(sums["clean_total"], 1.0),
        "policy": "shortest_prefix_pos_neg_at_least_clean_else_append_missing_direction_else_compact_quota",
        "note": "This guarantees directional packet counts >= clean counts, but does not prove exact original packet preservation.",
    }
    summary_path = output_manifest.with_name(output_manifest.stem + "_summary.json")
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
