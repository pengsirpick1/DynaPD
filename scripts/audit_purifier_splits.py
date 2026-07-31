"""Audit purifier clean/defended split manifests across train/validation/test.

This is the strict Step 2/3 gate before conditional purifier training. It
accepts sharded train/validation defended datasets and the already audited
fresh test defended dataset, then writes split-specific and unified manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import resolve_cw_path, stored_npy_from_npz  # noqa: E402


DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "dmmpv3_rf_tam_shape_v2_fullcw_seed0_20260719_201036"
MANIFEST_COLUMNS = [
    "source_id",
    "clean_index",
    "split",
    "class_id",
    "variant_id",
    "defense_seed",
    "defense_round",
    "budget",
    "policy_id",
    "clean_path",
    "clean_local_index",
    "clean_fingerprint",
    "defended_path",
    "defended_local_index",
    "defended_global_index",
    "defended_fingerprint",
    "defense_profile_id",
    "shard_id",
    "shard_index",
    "shard_local_index",
    "pairing_status",
    "pairing_evidence",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _open_clean_arrays(clean_path: Path) -> tuple[np.ndarray, np.ndarray, Any | None]:
    x_map = stored_npy_from_npz(clean_path, "X")
    y_map = stored_npy_from_npz(clean_path, "y")
    if x_map is not None and y_map is not None:
        return x_map, np.asarray(y_map, dtype=np.int64), None
    payload = np.load(clean_path, allow_pickle=False)
    return payload["X"], np.asarray(payload["y"], dtype=np.int64), payload


def _nonzero_trace(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    return values[values != 0]


def _sha256_array(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    digest = hashlib.sha256()
    digest.update(str(arr.dtype).encode("ascii"))
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _sha256_defended(trace: np.ndarray, origin: np.ndarray) -> str:
    trace_arr = np.asarray(trace, dtype=np.float32).reshape(-1)
    origin_arr = np.asarray(origin, dtype=np.uint8).reshape(-1)
    digest = hashlib.sha256()
    digest.update(b"trace:")
    digest.update(str(trace_arr.shape).encode("ascii"))
    digest.update(trace_arr.tobytes(order="C"))
    digest.update(b"|origin:")
    digest.update(str(origin_arr.shape).encode("ascii"))
    digest.update(origin_arr.tobytes(order="C"))
    return digest.hexdigest()


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    payload = _read_json(run_dir / "split_indices.json")
    return {
        "train": np.asarray(payload["train"], dtype=np.int64),
        "validation": np.asarray(payload["val"], dtype=np.int64),
        "test": np.asarray(payload["test"], dtype=np.int64),
    }


def _split_maps(splits: dict[str, np.ndarray]) -> tuple[dict[int, str], dict[int, int]]:
    split_by_source: dict[int, str] = {}
    local_by_source: dict[int, int] = {}
    for split, values in splits.items():
        for local, source_id in enumerate(values.tolist()):
            source = int(source_id)
            if source in split_by_source:
                raise RuntimeError(f"source_id {source} appears in both {split_by_source[source]} and {split}")
            split_by_source[source] = split
            local_by_source[source] = int(local)
    return split_by_source, local_by_source


def _load_ragged(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    payload = np.load(path, allow_pickle=False)
    flat = np.asarray(payload["flat"], dtype=np.float32)
    offsets = np.asarray(payload["offsets"], dtype=np.int64)
    origin_flat = np.asarray(payload["origin_flat"], dtype=np.uint8)
    metadata = {key: np.asarray(payload[key]) for key in payload.files if key not in {"flat", "offsets", "origin_flat"}}
    return flat, offsets, origin_flat, metadata


def _validate_lengths(path: Path, flat: np.ndarray, offsets: np.ndarray, origin_flat: np.ndarray, metadata: dict[str, np.ndarray]) -> int:
    if len(offsets) == 0 or int(offsets[0]) != 0 or int(offsets[-1]) != len(flat):
        raise RuntimeError(f"{path} has invalid ragged offsets")
    if np.any(offsets[1:] < offsets[:-1]):
        raise RuntimeError(f"{path} offsets are not monotonic")
    if len(origin_flat) != len(flat):
        raise RuntimeError(f"{path} origin_flat length differs from flat length")
    n = len(offsets) - 1
    for required in ("y", "clean_index", "budget", "keep_ratio"):
        if required not in metadata:
            raise RuntimeError(f"{path} missing required field {required!r}")
    for key, value in metadata.items():
        if value.ndim > 0 and value.shape[0] != n:
            raise RuntimeError(f"{path}:{key} first dimension {value.shape[0]} != sample count {n}")
    return n


def _find_profile_id(path: Path) -> str:
    parts = list(path.parts)
    for marker in ("purifier", "profiles"):
        if marker in parts:
            idx = parts.index(marker)
            if marker == "purifier" and idx + 2 < len(parts):
                return parts[idx + 2]
            if marker == "profiles" and idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def _infer_shard(path: Path) -> tuple[str, int]:
    match = re.search(r"(shard(\d{4})-of-\d{4})", path.stem)
    if match:
        return match.group(1), int(match.group(2))
    return path.stem, 0


def _split_from_purifier_path(path: Path) -> str | None:
    parts = list(path.parts)
    if "purifier" not in parts:
        return None
    idx = parts.index("purifier")
    if idx + 1 >= len(parts):
        return None
    split = parts[idx + 1]
    return "validation" if split == "val" else split


def _defended_files_for_split(run_dir: Path, split: str) -> list[Path]:
    purifier_root = run_dir / "defended_datasets" / "purifier" / split
    files = []
    if purifier_root.is_dir():
        files.extend(path for path in purifier_root.glob("**/*.npz") if not path.name.endswith("_policy_statistics.npz"))
    if split == "test":
        profile_root = run_dir / "defended_datasets" / "profiles"
        if profile_root.is_dir():
            files.extend(
                path
                for path in profile_root.glob("test_*/fresh_deployment_test*.npz")
                if not path.name.endswith("_policy_statistics.npz")
            )
    return sorted(set(files))


def _clean_fingerprints(clean_x: np.ndarray, *, progress: bool) -> list[str]:
    result: list[str] = []
    total = int(clean_x.shape[0])
    for index in range(total):
        if progress and index and index % 10000 == 0:
            print(f"[purifier audit] clean fingerprints: {index}/{total}", flush=True)
        result.append(_sha256_array(_nonzero_trace(clean_x[index])))
    return result


def _cross_split_clean_duplicates(clean_fps: list[str], splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    by_fp: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split, values in splits.items():
        for source_id in values.tolist():
            by_fp[clean_fps[int(source_id)]].append((split, int(source_id)))
    duplicates = []
    for fp, rows in by_fp.items():
        row_splits = {split for split, _ in rows}
        if len(row_splits) > 1:
            duplicates.append({"fingerprint": fp, "splits": sorted(row_splits), "source_ids": [source for _, source in rows[:20]], "count": len(rows)})
    return duplicates


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in MANIFEST_COLUMNS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit purifier defended splits and write authoritative manifests.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--manifest-dir", default="", help="Default: <run-dir>/manifests")
    parser.add_argument("--expected-repeats", type=int, default=0, help="Default: run_config deployment_repeats")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run_config = _read_json(run_dir / "run_config.json")
    expected_repeats = int(args.expected_repeats or run_config.get("deployment_repeats", 3))
    requested_splits = ["validation" if item.strip() == "val" else item.strip() for item in str(args.splits).split(",") if item.strip()]
    splits = _load_splits(run_dir)
    split_by_source, local_by_source = _split_maps(splits)
    clean_path = resolve_cw_path(run_config.get("data_source") or run_config.get("data_root") or PROJECT_ROOT.parents[1] / "datasets" / "CW")
    clean_x, clean_y, clean_payload = _open_clean_arrays(clean_path)
    try:
        if len(clean_x) != len(clean_y):
            raise RuntimeError("clean X/y length mismatch")
        if set(split_by_source) != set(range(len(clean_y))):
            raise RuntimeError("split_indices do not cover all clean samples exactly")
        print(f"[purifier audit] computing clean fingerprints for {len(clean_y)} samples", flush=True)
        clean_fps = _clean_fingerprints(clean_x, progress=not bool(args.quiet))
        errors: list[str] = []
        manifests: dict[str, list[dict[str, Any]]] = {split: [] for split in requested_splits}
        split_source_counts: dict[str, Counter[int]] = {split: Counter() for split in requested_splits}
        split_file_reports: dict[str, list[dict[str, Any]]] = {split: [] for split in requested_splits}
        defended_fps_by_split: dict[str, list[tuple[str, str, int]]] = defaultdict(list)
        global_index_by_split: Counter[str] = Counter()
        label_mismatches = 0
        structure_failures = 0
        unresolved = 0

        for split in requested_splits:
            files = _defended_files_for_split(run_dir, split)
            if not files:
                errors.append(f"missing defended files for split {split}")
                continue
            for path in files:
                print(f"[purifier audit] split={split} file={path}", flush=True)
                flat, offsets, origin_flat, metadata = _load_ragged(path)
                n = _validate_lengths(path, flat, offsets, origin_flat, metadata)
                y = np.asarray(metadata["y"], dtype=np.int64)
                clean_index = np.asarray(metadata["clean_index"], dtype=np.int64)
                budgets = np.asarray(metadata["budget"], dtype=np.float32)
                combination_index = np.asarray(metadata.get("combination_index", np.full(n, -1)), dtype=np.int64)
                shard_id, shard_index = _infer_shard(path)
                profile_id = _find_profile_id(path)
                source_counts = Counter(int(value) for value in clean_index.tolist())
                file_splits = {split_by_source.get(int(value), "out_of_range") for value in source_counts}
                if file_splits != {split}:
                    errors.append(f"{path} has source ids outside declared split {split}: {sorted(file_splits)}")
                local_duplicate_fps = 0
                local_seen_fps: set[str] = set()
                per_file_unresolved = 0
                for row in range(n):
                    source_id = int(clean_index[row])
                    status = "paired"
                    evidence = ["explicit clean_index metadata", "source_id = clean_index"]
                    if source_id < 0 or source_id >= len(clean_y):
                        status = "unresolved"
                        unresolved += 1
                        per_file_unresolved += 1
                        evidence.append("source id out of range")
                    elif int(y[row]) != int(clean_y[source_id]):
                        status = "unresolved"
                        unresolved += 1
                        per_file_unresolved += 1
                        label_mismatches += 1
                        evidence.append(f"label mismatch clean={int(clean_y[source_id])} defended={int(y[row])}")
                    start, end = int(offsets[row]), int(offsets[row + 1])
                    trace = flat[start:end]
                    origin = origin_flat[start:end]
                    defended_fp = _sha256_defended(trace, origin)
                    if defended_fp in local_seen_fps:
                        local_duplicate_fps += 1
                    local_seen_fps.add(defended_fp)
                    defended_fps_by_split[defended_fp].append((split, str(path), int(row)))
                    if status == "paired":
                        clean_trace = _nonzero_trace(clean_x[source_id])
                        recovered = trace[origin.astype(bool)]
                        if len(recovered) != len(clean_trace) or not np.array_equal(recovered.astype(np.float32, copy=False), clean_trace):
                            status = "unresolved"
                            unresolved += 1
                            per_file_unresolved += 1
                            structure_failures += 1
                            evidence.append("origin-marked subsequence does not equal clean nonzero trace")
                        else:
                            evidence.append("origin-marked subsequence exactly equals clean nonzero trace")
                    source_round = split_source_counts[split][source_id]
                    split_source_counts[split][source_id] += 1
                    global_row = int(global_index_by_split[split])
                    global_index_by_split[split] += 1
                    manifests[split].append(
                        {
                            "source_id": source_id,
                            "clean_index": source_id,
                            "split": split,
                            "class_id": int(y[row]) if row < len(y) else "",
                            "variant_id": int(source_round),
                            "defense_seed": int(run_config.get("seed", 0)),
                            "defense_round": int(source_round),
                            "budget": float(budgets[row]) if row < len(budgets) else "",
                            "policy_id": int(combination_index[row]) if row < len(combination_index) else "",
                            "clean_path": str(clean_path.resolve()),
                            "clean_local_index": local_by_source.get(source_id, ""),
                            "clean_fingerprint": clean_fps[source_id] if 0 <= source_id < len(clean_fps) else "",
                            "defended_path": str(path.resolve()),
                            "defended_local_index": int(row),
                            "defended_global_index": global_row,
                            "defended_fingerprint": defended_fp,
                            "defense_profile_id": profile_id,
                            "shard_id": shard_id,
                            "shard_index": int(shard_index),
                            "shard_local_index": int(row),
                            "pairing_status": status,
                            "pairing_evidence": "; ".join(evidence),
                        }
                    )
                split_file_reports[split].append(
                    {
                        "path": str(path.resolve()),
                        "samples": int(n),
                        "unique_sources": int(len(source_counts)),
                        "source_repeat_min": int(min(source_counts.values())) if source_counts else 0,
                        "source_repeat_max": int(max(source_counts.values())) if source_counts else 0,
                        "unresolved": int(per_file_unresolved),
                        "duplicate_defended_fingerprints_within_file": int(local_duplicate_fps),
                        "profile_id": profile_id,
                        "shard_id": shard_id,
                        "shard_index": int(shard_index),
                    }
                )

        clean_cross_split_duplicates = _cross_split_clean_duplicates(clean_fps, splits)
        if clean_cross_split_duplicates:
            errors.append(f"clean fingerprint repeats across splits: {len(clean_cross_split_duplicates)} groups")

        defended_cross_split_duplicates = []
        for fp, rows in defended_fps_by_split.items():
            row_splits = {split for split, _, _ in rows}
            if len(row_splits) > 1:
                defended_cross_split_duplicates.append(
                    {
                        "fingerprint": fp,
                        "splits": sorted(row_splits),
                        "rows": [{"split": split, "path": path, "defended_local_index": index} for split, path, index in rows[:20]],
                        "count": len(rows),
                    }
                )
        if defended_cross_split_duplicates:
            errors.append(f"defended fingerprint repeats across splits: {len(defended_cross_split_duplicates)} groups")

        split_reports: dict[str, dict[str, Any]] = {}
        for split in requested_splits:
            expected_sources = set(int(value) for value in splits[split].tolist())
            actual_sources = set(split_source_counts[split])
            missing = sorted(expected_sources - actual_sources)
            unexpected = sorted(actual_sources - expected_sources)
            repeat_bad = {source: count for source, count in split_source_counts[split].items() if count != expected_repeats}
            if missing:
                errors.append(f"{split} missing {len(missing)} clean sources")
            if unexpected:
                errors.append(f"{split} has {len(unexpected)} unexpected clean sources")
            if repeat_bad:
                errors.append(f"{split} has {len(repeat_bad)} sources whose defended variant count != {expected_repeats}")
            split_reports[split] = {
                "expected_source_count": int(len(expected_sources)),
                "actual_source_count": int(len(actual_sources)),
                "pair_count": int(len(manifests[split])),
                "expected_pair_count": int(len(expected_sources) * expected_repeats),
                "missing_source_count": int(len(missing)),
                "unexpected_source_count": int(len(unexpected)),
                "bad_repeat_source_count": int(len(repeat_bad)),
                "repeat_min": int(min(split_source_counts[split].values())) if split_source_counts[split] else 0,
                "repeat_max": int(max(split_source_counts[split].values())) if split_source_counts[split] else 0,
                "files": split_file_reports[split],
            }

        if label_mismatches:
            errors.append(f"label mismatches: {label_mismatches}")
        if structure_failures:
            errors.append(f"structure invariant failures: {structure_failures}")
        if unresolved:
            errors.append(f"unresolved manifest rows: {unresolved}")

        manifest_dir = Path(args.manifest_dir).resolve() if str(args.manifest_dir).strip() else run_dir / "manifests"
        unified_rows = []
        for split in requested_splits:
            split_path = manifest_dir / f"purifier_{split}_pairs.csv"
            _write_csv(split_path, manifests[split])
            unified_rows.extend(manifests[split])
        unified_path = manifest_dir / "purifier_clean_defended_pairs.csv"
        _write_csv(unified_path, unified_rows)
        report = {
            "verdict": "PASS" if not errors else "FAIL",
            "run_dir": str(run_dir.resolve()),
            "clean_path": str(clean_path.resolve()),
            "manifest_dir": str(manifest_dir.resolve()),
            "unified_manifest": str(unified_path.resolve()),
            "expected_repeats": int(expected_repeats),
            "split_reports": split_reports,
            "source_intersections": {
                "train_validation": int(len(set(splits["train"]) & set(splits["validation"]))),
                "train_test": int(len(set(splits["train"]) & set(splits["test"]))),
                "validation_test": int(len(set(splits["validation"]) & set(splits["test"]))),
            },
            "duplicates": {
                "clean_cross_split_total_groups": int(len(clean_cross_split_duplicates)),
                "clean_cross_split": clean_cross_split_duplicates[:100],
                "defended_cross_split_total_groups": int(len(defended_cross_split_duplicates)),
                "defended_cross_split": defended_cross_split_duplicates[:100],
            },
            "counts": {
                "label_mismatches": int(label_mismatches),
                "structure_failures": int(structure_failures),
                "unresolved": int(unresolved),
                "total_pairs": int(len(unified_rows)),
            },
            "assertions": {
                "train_validation_disjoint": int(len(set(splits["train"]) & set(splits["validation"]))) == 0,
                "train_test_disjoint": int(len(set(splits["train"]) & set(splits["test"]))) == 0,
                "validation_test_disjoint": int(len(set(splits["validation"]) & set(splits["test"]))) == 0,
                "no_clean_cross_split_fingerprint_duplicates": len(clean_cross_split_duplicates) == 0,
                "no_defended_cross_split_fingerprint_duplicates": len(defended_cross_split_duplicates) == 0,
                "no_unresolved": unresolved == 0,
                "labels_match": label_mismatches == 0,
                "structure_invariant_holds": structure_failures == 0,
                "all_sources_have_expected_repeats": all(row["bad_repeat_source_count"] == 0 for row in split_reports.values()),
                "all_expected_sources_present": all(row["missing_source_count"] == 0 for row in split_reports.values()),
                "no_unexpected_sources": all(row["unexpected_source_count"] == 0 for row in split_reports.values()),
            },
            "errors": errors,
        }
        report_path = manifest_dir / "purifier_split_audit_report.json"
        _write_json(report_path, report)
        print(json.dumps({"verdict": report["verdict"], "report": str(report_path.resolve()), "unified_manifest": report["unified_manifest"], "counts": report["counts"]}, indent=2, ensure_ascii=False))
        if errors:
            raise SystemExit(1)
    finally:
        if clean_payload is not None:
            clean_payload.close()


if __name__ == "__main__":
    main()
