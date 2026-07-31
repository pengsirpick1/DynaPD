"""Audit offline purified datasets against the authoritative pair manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.purifier.config import PurifierConfig
from dmmp.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit generated purified datasets.")
    parser.add_argument("--purifier-run-dir", required=True)
    parser.add_argument("--purified-manifest", default="", help="Default: <purifier-run-dir>/manifests/purified_dataset_manifest.csv")
    parser.add_argument("--report-json", default="", help="Default: <purifier-run-dir>/manifests/purified_dataset_audit_report.json")
    parser.add_argument("--report-md", default="", help="Default: <purifier-run-dir>/manifests/purified_dataset_audit_report.md")
    parser.add_argument("--seq-length", type=int, default=0)
    parser.add_argument("--value-clip", type=float, default=0.0)
    return parser.parse_args()


def _pair_key(row: dict[str, str]) -> tuple[str, str, int, int, int, int]:
    return (
        str(row["split"]),
        str(Path(row["defended_path"]).resolve()),
        int(row["defended_local_index"]),
        int(row["source_id"]),
        int(row["clean_index"]),
        int(row["variant_id"]),
    )


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_checkpoint_config(purified_rows: list[dict[str, str]], run_dir: Path) -> tuple[PurifierConfig, dict[str, Any]]:
    checkpoint = Path(purified_rows[0]["purifier_checkpoint"]).resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = PurifierConfig.from_mapping(payload["config"])
    selection_path = run_dir / "checkpoint_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8")) if selection_path.is_file() else {}
    return cfg, selection


def _original_rows(cfg: PurifierConfig) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in [cfg.train_manifest, cfg.validation_manifest, cfg.test_manifest]:
        rows.extend(_load_rows(Path(path)))
    return rows


def _row_digest(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Purified Dataset Audit",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- purified manifest: `{report['purified_manifest']}`",
        f"- total purified pairs: `{report['counts']['total_pairs']}`",
        f"- missing pairs: `{report['counts']['missing_pairs']}`",
        f"- extra pairs: `{report['counts']['extra_pairs']}`",
        f"- NaN rows: `{report['quality']['nan_rows']}`",
        f"- Inf rows: `{report['quality']['inf_rows']}`",
        f"- all-zero rows: `{report['quality']['all_zero_rows']}`",
        f"- constant rows: `{report['quality']['constant_rows']}`",
        f"- range violations: `{report['quality']['range_violation_rows']}`",
    ]
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {item}" for item in report["errors"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.purifier_run_dir).resolve()
    purified_manifest = Path(args.purified_manifest).resolve() if args.purified_manifest else run_dir / "manifests" / "purified_dataset_manifest.csv"
    report_json = Path(args.report_json).resolve() if args.report_json else run_dir / "manifests" / "purified_dataset_audit_report.json"
    report_md = Path(args.report_md).resolve() if args.report_md else run_dir / "manifests" / "purified_dataset_audit_report.md"
    purified_rows = _load_rows(purified_manifest)
    if not purified_rows:
        raise SystemExit(f"Empty purified manifest: {purified_manifest}")
    cfg, selection = _load_checkpoint_config(purified_rows, run_dir)
    seq_length = int(args.seq_length or cfg.seq_length)
    value_clip = float(args.value_clip or cfg.value_clip)
    original = _original_rows(cfg)
    original_counter = Counter(_pair_key(row) for row in original)
    purified_counter = Counter(_pair_key(row) for row in purified_rows)
    missing = list((original_counter - purified_counter).elements())
    extra = list((purified_counter - original_counter).elements())
    duplicate_purified = sum(count - 1 for count in purified_counter.values() if count > 1)

    rows_by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in purified_rows:
        rows_by_path[str(Path(row["purified_path"]).resolve())].append(row)

    errors: list[str] = []
    split_pairs: Counter[str] = Counter()
    split_sources: dict[str, set[int]] = defaultdict(set)
    source_split: dict[int, str] = {}
    variants_by_source: dict[tuple[str, int], set[int]] = defaultdict(set)
    fingerprints: set[str] = set()
    duplicate_fingerprints = 0
    quality = Counter()
    abnormal_lengths = 0
    checked_rows = 0

    for path_text, rows in sorted(rows_by_path.items()):
        path = Path(path_text)
        if not path.is_file():
            errors.append(f"missing purified shard: {path}")
            continue
        payload = np.load(path, allow_pickle=False)
        try:
            if "X" not in payload.files:
                errors.append(f"{path} missing X")
                continue
            x = np.asarray(payload["X"], dtype=np.float32)
            if x.ndim != 2 or int(x.shape[1]) != seq_length:
                errors.append(f"{path} invalid X shape {x.shape}, expected (*,{seq_length})")
                continue
            n = int(x.shape[0])
            for row in rows:
                local = int(row["purified_index"])
                if local < 0 or local >= n:
                    errors.append(f"{path} purified_index {local} out of range")
                    continue
                values = x[local]
                checked_rows += 1
                split = str(row["split"])
                source_id = int(row["source_id"])
                variant_id = int(row["variant_id"])
                split_pairs[split] += 1
                split_sources[split].add(source_id)
                variants_by_source[(split, source_id)].add(variant_id)
                previous_split = source_split.setdefault(source_id, split)
                if previous_split != split:
                    errors.append(f"source_id {source_id} crosses split {previous_split}/{split}")
                finite = np.isfinite(values)
                if np.isnan(values).any():
                    quality["nan_rows"] += 1
                if np.isinf(values).any():
                    quality["inf_rows"] += 1
                if not finite.all():
                    continue
                if np.any(np.abs(values) > value_clip + 1.0e-4):
                    quality["range_violation_rows"] += 1
                nonzero = np.flatnonzero(values != 0)
                if len(nonzero) == 0:
                    quality["all_zero_rows"] += 1
                else:
                    length = int(nonzero[-1] + 1)
                    if length > seq_length:
                        abnormal_lengths += 1
                    nz_values = values[nonzero]
                    if float(np.std(nz_values)) < 1.0e-8:
                        quality["constant_rows"] += 1
                digest = _row_digest(values)
                if digest in fingerprints:
                    duplicate_fingerprints += 1
                fingerprints.add(digest)
        finally:
            payload.close()

    variant_bad = {
        f"{split}:{source}": sorted(variants)
        for (split, source), variants in variants_by_source.items()
        if len(variants) != 3
    }
    if missing:
        errors.append(f"missing purified pairs: {len(missing)}")
    if extra:
        errors.append(f"extra purified pairs: {len(extra)}")
    if duplicate_purified:
        errors.append(f"duplicate purified pair rows: {duplicate_purified}")
    if quality["nan_rows"] or quality["inf_rows"]:
        errors.append("purified data contains NaN or Inf")
    if quality["range_violation_rows"]:
        errors.append(f"range violations: {quality['range_violation_rows']}")
    if variant_bad:
        errors.append(f"sources with variant count != 3: {len(variant_bad)}")
    test_unused = bool(selection.get("selection_scope") == "validation_only" and selection.get("test_metric_used") is False and selection.get("test_loader_constructed") is False)
    if not test_unused:
        errors.append("checkpoint_selection.json does not prove validation-only selection")

    report = {
        "verdict": "PASS" if not errors else "FAIL",
        "purified_manifest": str(purified_manifest.resolve()),
        "source_manifests": {
            "train": cfg.train_manifest,
            "validation": cfg.validation_manifest,
            "test": cfg.test_manifest,
        },
        "counts": {
            "total_pairs": int(len(purified_rows)),
            "checked_rows": int(checked_rows),
            "expected_pairs": int(len(original)),
            "missing_pairs": int(len(missing)),
            "extra_pairs": int(len(extra)),
            "duplicate_purified_pair_rows": int(duplicate_purified),
            "split_pairs": {key: int(value) for key, value in sorted(split_pairs.items())},
            "split_unique_sources": {key: int(len(value)) for key, value in sorted(split_sources.items())},
        },
        "quality": {
            "nan_rows": int(quality["nan_rows"]),
            "inf_rows": int(quality["inf_rows"]),
            "all_zero_rows": int(quality["all_zero_rows"]),
            "constant_rows": int(quality["constant_rows"]),
            "range_violation_rows": int(quality["range_violation_rows"]),
            "abnormal_length_rows": int(abnormal_lengths),
            "duplicate_fingerprint_rows": int(duplicate_fingerprints),
        },
        "assertions": {
            "each_defended_pair_has_one_purified_result": len(missing) == 0 and len(extra) == 0 and duplicate_purified == 0,
            "source_split_inherited": not any("crosses split" in item for item in errors),
            "no_nan_or_inf": quality["nan_rows"] == 0 and quality["inf_rows"] == 0,
            "shape_legal": not any("invalid X shape" in item for item in errors),
            "range_legal": quality["range_violation_rows"] == 0,
            "all_sources_have_three_variants": len(variant_bad) == 0,
            "test_not_used_for_checkpoint_selection": test_unused,
        },
        "variant_bad_examples": dict(list(variant_bad.items())[:20]),
        "errors": errors,
    }
    write_json(report_json, report)
    _write_markdown(report_md, report)
    print(json.dumps({"verdict": report["verdict"], "report": str(report_json.resolve()), "counts": report["counts"], "quality": report["quality"]}, indent=2, ensure_ascii=False))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
