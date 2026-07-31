"""Audit clean-to-defended instance pairing for a DMMPv3 run.

This script is intentionally read-only for existing datasets. It writes a
manifest and audit report under the selected run directory, and exits non-zero
when provenance or pairing assertions fail.
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
    "class_id",
    "split",
    "clean_path",
    "clean_index",
    "clean_local_index",
    "clean_fingerprint",
    "defended_path",
    "defended_index",
    "defended_fingerprint",
    "defense_method",
    "defense_seed",
    "defense_round",
    "defense_profile_id",
    "defense_budget",
    "defense_keep_ratio",
    "policy_id",
    "pairing_status",
    "pairing_evidence",
]


class AuditFailure(RuntimeError):
    """Raised when one or more strict audit assertions fail."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _nonzero_trace(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float32).reshape(-1)
    return values[values != 0]


def _open_clean_arrays(clean_path: Path) -> tuple[np.ndarray, np.ndarray, Any | None]:
    x_map = stored_npy_from_npz(clean_path, "X")
    y_map = stored_npy_from_npz(clean_path, "y")
    if x_map is not None and y_map is not None:
        return x_map, np.asarray(y_map, dtype=np.int64), None
    payload = np.load(clean_path, allow_pickle=False)
    return payload["X"], np.asarray(payload["y"], dtype=np.int64), payload


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    payload = _read_json(run_dir / "split_indices.json")
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _split_maps(splits: dict[str, np.ndarray]) -> tuple[dict[int, str], dict[int, int]]:
    split_by_source: dict[int, str] = {}
    local_by_source: dict[int, int] = {}
    for split, values in splits.items():
        for local_index, source_id in enumerate(np.asarray(values, dtype=np.int64).tolist()):
            source = int(source_id)
            if source in split_by_source:
                raise AuditFailure(f"source_id {source} appears in both {split_by_source[source]} and {split}")
            split_by_source[source] = split
            local_by_source[source] = int(local_index)
    return split_by_source, local_by_source


def _defended_files(run_dir: Path) -> list[Path]:
    root = run_dir / "defended_datasets"
    if not root.is_dir():
        return []
    files = []
    for path in root.glob("profiles/**/*.npz"):
        if path.name.endswith("_policy_statistics.npz"):
            continue
        files.append(path)
    return sorted(files)


def _infer_role(path: Path) -> str:
    name = path.stem
    match = re.match(r"(.+?)_b\d", name)
    return match.group(1) if match else name


def _infer_profile_id(path: Path) -> str:
    try:
        parts = path.parts
        idx = parts.index("profiles")
        return parts[idx + 1]
    except (ValueError, IndexError):
        return ""


def _metrics_path(path: Path) -> Path:
    return path.with_name(path.stem + "_metrics.json")


def _load_ragged(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    payload = np.load(path, allow_pickle=False)
    flat = np.asarray(payload["flat"], dtype=np.float32)
    offsets = np.asarray(payload["offsets"], dtype=np.int64)
    origin_flat = np.asarray(payload["origin_flat"], dtype=np.uint8)
    metadata = {key: np.asarray(payload[key]) for key in payload.files if key not in {"flat", "offsets", "origin_flat"}}
    return flat, offsets, origin_flat, metadata


def _validate_ragged_lengths(path: Path, flat: np.ndarray, offsets: np.ndarray, origin_flat: np.ndarray, metadata: dict[str, np.ndarray]) -> int:
    if offsets.ndim != 1 or len(offsets) == 0:
        raise AuditFailure(f"{path} has invalid offsets")
    if int(offsets[0]) != 0 or int(offsets[-1]) != len(flat):
        raise AuditFailure(f"{path} offsets do not span flat array")
    if np.any(offsets[1:] < offsets[:-1]):
        raise AuditFailure(f"{path} offsets are not monotonic")
    if len(origin_flat) != len(flat):
        raise AuditFailure(f"{path} origin_flat length differs from flat length")
    n = len(offsets) - 1
    for key, value in metadata.items():
        if key in {"primitive_weights", "selected_primitive_mask", "profile_mask"}:
            if value.shape[0] != n:
                raise AuditFailure(f"{path}:{key} first dimension {value.shape[0]} != defended sample count {n}")
        elif value.ndim >= 1 and value.shape[0] != n:
            raise AuditFailure(f"{path}:{key} length {value.shape[0]} != defended sample count {n}")
    for required in ("y", "clean_index", "budget", "keep_ratio"):
        if required not in metadata:
            raise AuditFailure(f"{path} is missing required metadata field {required!r}")
    return n


def _clean_fingerprints(clean_x: np.ndarray, *, progress: bool) -> list[str]:
    fingerprints: list[str] = []
    total = int(clean_x.shape[0])
    for index in range(total):
        if progress and index and index % 10000 == 0:
            print(f"[audit] clean fingerprints: {index}/{total}", flush=True)
        fingerprints.append(_sha256_array(_nonzero_trace(clean_x[index])))
    return fingerprints


def _cross_split_duplicates(fingerprints: list[str], splits: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    by_fp: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for split, values in splits.items():
        for source_id in np.asarray(values, dtype=np.int64).tolist():
            by_fp[fingerprints[int(source_id)]].append((split, int(source_id)))
    duplicates = []
    for fp, rows in by_fp.items():
        split_names = {split for split, _ in rows}
        if len(split_names) > 1:
            duplicates.append(
                {
                    "fingerprint": fp,
                    "splits": sorted(split_names),
                    "source_ids": [source_id for _, source_id in rows[:20]],
                    "count": len(rows),
                }
            )
    return duplicates


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    counter = Counter(int(labels[int(index)]) for index in np.asarray(indices, dtype=np.int64).tolist())
    return {str(key): int(counter[key]) for key in sorted(counter)}


def _distance_probe(
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    clean_indices: np.ndarray,
    flat: np.ndarray,
    offsets: np.ndarray,
    origin_flat: np.ndarray,
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    if len(clean_indices) == 0:
        return {"samples": 0}
    rng = np.random.default_rng(int(seed) + 991)
    rows = np.arange(len(clean_indices), dtype=np.int64)
    if len(rows) > int(sample_count):
        rows = np.sort(rng.choice(rows, size=int(sample_count), replace=False).astype(np.int64))
    by_label: dict[int, list[int]] = defaultdict(list)
    unique_sources = sorted(set(int(value) for value in clean_indices.tolist()))
    for source_id in unique_sources:
        by_label[int(clean_y[source_id])].append(source_id)
    paired_distances = []
    random_distances = []
    skipped = 0
    for row in rows.tolist():
        source_id = int(clean_indices[row])
        clean = _nonzero_trace(clean_x[source_id])
        start, end = int(offsets[row]), int(offsets[row + 1])
        origin = origin_flat[start:end].astype(bool)
        recovered = flat[start:end][origin].astype(np.float32, copy=False)
        if len(recovered) != len(clean):
            skipped += 1
            continue
        paired_distances.append(float(np.mean(np.abs(recovered - clean))) if len(clean) else 0.0)
        candidates = [item for item in by_label[int(clean_y[source_id])] if item != source_id]
        if not candidates:
            skipped += 1
            continue
        other_source = int(candidates[int(rng.integers(0, len(candidates)))])
        other = _nonzero_trace(clean_x[other_source])
        m = min(len(recovered), len(other))
        if m == 0:
            random_distances.append(0.0)
        else:
            random_distances.append(float(np.mean(np.abs(recovered[:m] - other[:m]))))
    return {
        "samples": int(len(rows)),
        "skipped": int(skipped),
        "paired_origin_l1_mean": float(np.mean(paired_distances)) if paired_distances else None,
        "paired_origin_l1_max": float(np.max(paired_distances)) if paired_distances else None,
        "same_class_mismatch_l1_mean": float(np.mean(random_distances)) if random_distances else None,
        "same_class_mismatch_l1_min": float(np.min(random_distances)) if random_distances else None,
    }


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in MANIFEST_COLUMNS})


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Clean/Defended Pairing Audit",
        "",
        f"- verdict: `{report['verdict']}`",
        f"- run_dir: `{report['run_dir']}`",
        f"- manifest: `{report['manifest_path']}`",
        f"- clean samples: `{report['counts']['clean_total']}`",
        f"- defended samples: `{report['counts']['defended_total']}`",
        f"- unresolved samples: `{report['counts']['unresolved']}`",
        f"- structure failures: `{report['counts']['structure_failures']}`",
        f"- label mismatches: `{report['counts']['label_mismatches']}`",
        f"- cross-split clean fingerprint duplicates: `{len(report['duplicates']['clean_cross_split'])}`",
        "",
        "## Defended Files",
        "",
    ]
    for item in report["defended_files"]:
        lines.append(
            "- `{path}`: samples=`{samples}`, unique_sources=`{unique_sources}`, split=`{split}`, "
            "budget=`{budget}`, keep_ratio=`{keep_ratio}`, order_matches_expected=`{order_matches_expected}`".format(**item)
        )
    if report["errors"]:
        lines.extend(["", "## Errors", ""])
        lines.extend(f"- {error}" for error in report["errors"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    run_config = _read_json(run_dir / "run_config.json")
    splits = _load_splits(run_dir)
    split_by_source, local_by_source = _split_maps(splits)
    clean_path = resolve_cw_path(run_config.get("data_source") or run_config.get("data_root") or (PROJECT_ROOT.parents[1] / "datasets" / "CW"))
    clean_x, clean_y, clean_payload = _open_clean_arrays(clean_path)
    try:
        if int(clean_y.shape[0]) != int(clean_x.shape[0]):
            raise AuditFailure("clean X and y lengths differ")
        if set(split_by_source) != set(range(int(clean_y.shape[0]))):
            raise AuditFailure("split_indices do not cover exactly all clean sample ids")

        defended_paths = _defended_files(run_dir)
        if not defended_paths:
            raise AuditFailure(f"No final defended NPZ files found under {run_dir / 'defended_datasets'}")

        print(f"[audit] computing clean fingerprints for {len(clean_y)} samples", flush=True)
        clean_fps = _clean_fingerprints(clean_x, progress=not args.quiet)
        clean_cross_split_duplicates = _cross_split_duplicates(clean_fps, splits)

        errors: list[str] = []
        manifest_rows: list[dict[str, Any]] = []
        defended_file_reports: list[dict[str, Any]] = []
        defended_split_fps: dict[str, list[tuple[str, int]]] = defaultdict(list)
        defended_counts_by_config: Counter[tuple[str, str, str, str]] = Counter()
        label_mismatches = 0
        structure_failures = 0
        unresolved = 0
        missing_sources_total = 0
        duplicate_sources_total = 0
        distance_probes: list[dict[str, Any]] = []
        expected_repeats = int(run_config.get("deployment_repeats", 1) or 1)
        defense_method = str(run_config.get("implementation") or run_config.get("method") or run_dir.name)
        defense_seed = int(run_config.get("seed", 0))

        for defended_path in defended_paths:
            print(f"[audit] auditing defended file {defended_path}", flush=True)
            flat, offsets, origin_flat, metadata = _load_ragged(defended_path)
            n = _validate_ragged_lengths(defended_path, flat, offsets, origin_flat, metadata)
            y = np.asarray(metadata["y"], dtype=np.int64)
            clean_indices = np.asarray(metadata["clean_index"], dtype=np.int64)
            budgets = np.asarray(metadata["budget"], dtype=np.float32)
            keep_ratios = np.asarray(metadata["keep_ratio"], dtype=np.float32)
            combination_index = np.asarray(metadata.get("combination_index", np.full(n, -1)), dtype=np.int64)
            role = _infer_role(defended_path)
            profile_id = _infer_profile_id(defended_path)
            metrics = _read_json(_metrics_path(defended_path)) if _metrics_path(defended_path).is_file() else {}
            if int(metrics.get("generated_traces", n)) != n:
                errors.append(f"{defended_path} generated_traces={metrics.get('generated_traces')} but NPZ has {n}")
            unique_sources = sorted(set(int(value) for value in clean_indices.tolist()))
            source_counts = Counter(int(value) for value in clean_indices.tolist())
            duplicate_sources_total += sum(1 for count in source_counts.values() if count > 1)
            source_splits = sorted({split_by_source.get(source_id, "out_of_range") for source_id in unique_sources})
            file_split = source_splits[0] if len(source_splits) == 1 else "mixed_or_invalid"
            expected_order = None
            if file_split in splits and len(unique_sources) == len(splits[file_split]) and n == len(splits[file_split]) * expected_repeats:
                expected_order = np.repeat(np.asarray(splits[file_split], dtype=np.int64), expected_repeats)
            order_matches_expected = bool(expected_order is not None and np.array_equal(clean_indices, expected_order))
            expected_source_set = set(np.asarray(splits.get(file_split, []), dtype=np.int64).tolist()) if file_split in splits else set()
            missing_sources = sorted(expected_source_set - set(unique_sources))
            missing_sources_total += len(missing_sources)
            if expected_order is not None and not order_matches_expected:
                errors.append(f"{defended_path} clean_index order does not match split order repeated {expected_repeats} times")
            if file_split == "mixed_or_invalid":
                errors.append(f"{defended_path} contains source ids from multiple or invalid splits: {source_splits}")
            if metrics and "deployment_repeats" in metrics and int(metrics["deployment_repeats"]) != expected_repeats:
                errors.append(f"{defended_path} deployment_repeats metric differs from run_config")
            if n != len(clean_indices) or n != len(y):
                errors.append(f"{defended_path} metadata lengths are inconsistent")
            if missing_sources:
                errors.append(f"{defended_path} is missing {len(missing_sources)} sources from split {file_split}")
            if any(count != expected_repeats for count in source_counts.values()) and n == len(unique_sources) * expected_repeats:
                errors.append(f"{defended_path} source repeat counts are not uniformly {expected_repeats}")

            occurrence_counter: Counter[int] = Counter()
            defended_duplicate_local = 0
            seen_defended_fps: set[str] = set()
            for defended_index in range(n):
                source_id = int(clean_indices[defended_index])
                status = "paired"
                evidence_parts = ["explicit clean_index metadata"]
                if source_id < 0 or source_id >= len(clean_y):
                    status = "unresolved"
                    unresolved += 1
                    evidence_parts.append("source_id out of range")
                source_split = split_by_source.get(source_id, "out_of_range")
                class_id = int(y[defended_index]) if defended_index < len(y) else -1
                if status == "paired" and class_id != int(clean_y[source_id]):
                    status = "unresolved"
                    unresolved += 1
                    label_mismatches += 1
                    evidence_parts.append(f"label mismatch clean={int(clean_y[source_id])} defended={class_id}")
                start, end = int(offsets[defended_index]), int(offsets[defended_index + 1])
                trace = flat[start:end]
                origin = origin_flat[start:end]
                defended_fp = _sha256_defended(trace, origin)
                if defended_fp in seen_defended_fps:
                    defended_duplicate_local += 1
                seen_defended_fps.add(defended_fp)
                defended_split_fps[defended_fp].append((source_split, str(defended_path), defended_index))
                if status == "paired":
                    clean_trace = _nonzero_trace(clean_x[source_id])
                    origin_bool = origin.astype(bool)
                    recovered = trace[origin_bool]
                    if len(recovered) != len(clean_trace) or not np.array_equal(recovered.astype(np.float32, copy=False), clean_trace):
                        status = "unresolved"
                        unresolved += 1
                        structure_failures += 1
                        evidence_parts.append("origin-marked subsequence does not equal clean nonzero trace")
                    else:
                        evidence_parts.append("origin-marked subsequence exactly equals clean nonzero trace")
                round_index = occurrence_counter[source_id]
                occurrence_counter[source_id] += 1
                manifest_rows.append(
                    {
                        "source_id": source_id,
                        "class_id": class_id,
                        "split": source_split,
                        "clean_path": str(clean_path.resolve()),
                        "clean_index": source_id,
                        "clean_local_index": local_by_source.get(source_id, ""),
                        "clean_fingerprint": clean_fps[source_id] if 0 <= source_id < len(clean_fps) else "",
                        "defended_path": str(defended_path.resolve()),
                        "defended_index": defended_index,
                        "defended_fingerprint": defended_fp,
                        "defense_method": defense_method,
                        "defense_seed": defense_seed,
                        "defense_round": round_index,
                        "defense_profile_id": profile_id,
                        "defense_budget": float(budgets[defended_index]) if defended_index < len(budgets) else "",
                        "defense_keep_ratio": float(keep_ratios[defended_index]) if defended_index < len(keep_ratios) else "",
                        "policy_id": int(combination_index[defended_index]) if defended_index < len(combination_index) else "",
                        "pairing_status": status,
                        "pairing_evidence": "; ".join(evidence_parts),
                    }
                )
                defended_counts_by_config[(defense_method, str(defense_seed), f"{float(budgets[defended_index]):.2f}", f"{float(keep_ratios[defended_index]):.2f}")] += 1

            distance_probes.append(
                {
                    "path": str(defended_path.resolve()),
                    **_distance_probe(
                        clean_x,
                        clean_y,
                        clean_indices,
                        flat,
                        offsets,
                        origin_flat,
                        sample_count=int(args.distance_samples),
                        seed=defense_seed,
                    ),
                }
            )
            defended_file_reports.append(
                {
                    "path": str(defended_path.resolve()),
                    "role": role,
                    "profile_id": profile_id,
                    "samples": int(n),
                    "unique_sources": int(len(unique_sources)),
                    "split": file_split,
                    "source_id_min": int(min(unique_sources)) if unique_sources else None,
                    "source_id_max": int(max(unique_sources)) if unique_sources else None,
                    "missing_sources": int(len(missing_sources)),
                    "source_repeat_min": int(min(source_counts.values())) if source_counts else 0,
                    "source_repeat_max": int(max(source_counts.values())) if source_counts else 0,
                    "expected_repeats": int(expected_repeats),
                    "duplicate_sources": int(sum(1 for count in source_counts.values() if count > 1)),
                    "duplicate_defended_fingerprints_within_file": int(defended_duplicate_local),
                    "budget": float(np.unique(budgets)[0]) if len(np.unique(budgets)) == 1 else "mixed",
                    "keep_ratio": float(np.unique(keep_ratios)[0]) if len(np.unique(keep_ratios)) == 1 else "mixed",
                    "order_matches_expected": bool(order_matches_expected),
                    "metrics_generated_traces": int(metrics.get("generated_traces", n)),
                }
            )

        defended_cross_split_duplicates = []
        for fp, rows in defended_split_fps.items():
            splits_for_fp = {split for split, _, _ in rows}
            if len(splits_for_fp) > 1:
                defended_cross_split_duplicates.append(
                    {
                        "fingerprint": fp,
                        "splits": sorted(splits_for_fp),
                        "rows": [{"path": path, "defended_index": int(index)} for _, path, index in rows[:20]],
                        "count": len(rows),
                    }
                )

        manifest_path = Path(args.manifest_path) if args.manifest_path else run_dir / "manifests" / "clean_defended_pair_manifest.csv"
        report_json_path = Path(args.report_json) if args.report_json else run_dir / "manifests" / "clean_defended_pair_audit_report.json"
        report_md_path = Path(args.report_md) if args.report_md else run_dir / "manifests" / "clean_defended_pair_audit_report.md"
        _write_manifest(manifest_path, manifest_rows)

        if clean_cross_split_duplicates:
            errors.append(f"clean fingerprints repeat across splits: {len(clean_cross_split_duplicates)} duplicate groups")
        if defended_cross_split_duplicates:
            errors.append(f"defended fingerprints repeat across splits: {len(defended_cross_split_duplicates)} duplicate groups")
        if unresolved:
            errors.append(f"manifest contains {unresolved} unresolved samples")
        if label_mismatches:
            errors.append(f"clean/defended label mismatches: {label_mismatches}")
        if structure_failures:
            errors.append(f"structure invariant failures: {structure_failures}")
        if len(manifest_rows) != sum(item["samples"] for item in defended_file_reports):
            errors.append("manifest row count does not equal defended sample count")

        report = {
            "verdict": "PASS" if not errors else "FAIL",
            "run_dir": str(run_dir),
            "clean_path": str(clean_path.resolve()),
            "manifest_path": str(manifest_path.resolve()),
            "report_json_path": str(report_json_path.resolve()),
            "report_md_path": str(report_md_path.resolve()),
            "counts": {
                "clean_total": int(len(clean_y)),
                "defended_total": int(len(manifest_rows)),
                "unresolved": int(unresolved),
                "label_mismatches": int(label_mismatches),
                "structure_failures": int(structure_failures),
                "missing_sources": int(missing_sources_total),
                "duplicate_source_ids_within_files": int(duplicate_sources_total),
            },
            "split_counts": {key: int(len(value)) for key, value in splits.items()},
            "clean_class_counts_by_split": {key: _class_counts(clean_y, value) for key, value in splits.items()},
            "defended_counts_by_config": {
                "|".join(key): int(value) for key, value in sorted(defended_counts_by_config.items())
            },
            "defended_files": defended_file_reports,
            "distance_probes": distance_probes,
            "duplicates": {
                "clean_cross_split": clean_cross_split_duplicates[:100],
                "clean_cross_split_total_groups": len(clean_cross_split_duplicates),
                "defended_cross_split": defended_cross_split_duplicates[:100],
                "defended_cross_split_total_groups": len(defended_cross_split_duplicates),
            },
            "assertions": {
                "each_defended_has_exactly_one_clean_source": unresolved == 0,
                "source_ids_legal_and_in_split": all(item["split"] != "mixed_or_invalid" for item in defended_file_reports),
                "labels_match": label_mismatches == 0,
                "source_id_not_cross_split": True,
                "no_clean_cross_split_fingerprint_duplicates": len(clean_cross_split_duplicates) == 0,
                "no_defended_cross_split_fingerprint_duplicates": len(defended_cross_split_duplicates) == 0,
                "manifest_has_no_unresolved": unresolved == 0,
                "field_lengths_consistent": True,
                "actual_counts_match_report": len(manifest_rows) == sum(item["samples"] for item in defended_file_reports),
            },
            "errors": errors,
        }
        _write_json(report_json_path, report)
        _write_markdown(report_md_path, report)
        return report
    finally:
        if clean_payload is not None:
            clean_payload.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit DMMPv3 clean/defended pairing and write an authoritative manifest.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="DMMPv3 run directory to audit.")
    parser.add_argument("--manifest-path", default="", help="Output CSV manifest path. Defaults to <run-dir>/manifests.")
    parser.add_argument("--report-json", default="", help="Output JSON report path. Defaults to <run-dir>/manifests.")
    parser.add_argument("--report-md", default="", help="Output Markdown report path. Defaults to <run-dir>/manifests.")
    parser.add_argument("--distance-samples", type=int, default=256, help="Number of samples for auxiliary pairing-vs-mismatch distance probe.")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging.")
    return parser.parse_args()


def main() -> None:
    try:
        report = audit(parse_args())
    except Exception as exc:
        print(f"[audit] FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(json.dumps({"verdict": report["verdict"], "counts": report["counts"], "manifest": report["manifest_path"]}, ensure_ascii=False, indent=2))
    if report["verdict"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
