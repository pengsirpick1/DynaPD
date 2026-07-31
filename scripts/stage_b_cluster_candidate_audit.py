# -*- coding: utf-8 -*-
"""Audit whether traffic clustering can support shared Stage B candidates.

The script intentionally does not change the Teacher controller. It answers the
precondition question for cluster-aware candidate retrieval:

1. Do same-cluster traces have more similar fast-keypoint maps?
2. Do same-cluster traces choose more similar first Teacher actions?
3. Can a cluster-level candidate template library cover each sample's
   per-sample Teacher-optimal first action with low regret?
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_b.policy_data import ACTION_FEATURE_NAMES
from dmmp.utils.config import DEFAULT_OUTPUT_DIR


DEFAULT_ARCHIVE = "results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz"
DEFAULT_SPLIT_FILE = "results/stage_b_policy_dataset_950_seed0_8_1_1/policy_splits.npz"
DEFAULT_TEACHER_RECORDS = ",".join(
    [
        "results/stage_b2e_teacher_950_train_tensorized_seed0_a16req/teacher_records.csv",
        "results/stage_b2e_teacher_950_val_tensorized_seed0_a16req/teacher_records.csv",
        "results/stage_b2e_teacher_950_test_tensorized_seed0_a16req/teacher_records.csv",
    ]
)


@dataclass
class SampleInfo:
    local_index: int
    archive_index: int
    source_index: int
    sample_id: str
    split: str
    label: int


@dataclass
class FirstActionRecord:
    sample_index: int
    split: str
    record_path: str
    selected_signature: tuple[int, ...]
    oracle_signature: tuple[int, ...]
    selected_gain: float
    oracle_gain: float
    candidate_signatures: list[tuple[int, ...]]
    candidate_gains: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default=DEFAULT_SPLIT_FILE)
    parser.add_argument("--teacher_records_csvs", default=DEFAULT_TEACHER_RECORDS)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--num_clusters", type=int, default=32)
    parser.add_argument("--segments", type=int, default=32)
    parser.add_argument("--signature_time_bins", type=int, default=64)
    parser.add_argument("--top_ratios", default="0.01,0.02,0.05,0.10")
    parser.add_argument("--template_topks", default="16,32,64,128")
    parser.add_argument("--near_eps", default="0.005,0.01,0.02")
    parser.add_argument("--max_pairs_per_cluster", type=int, default=2000)
    parser.add_argument("--max_random_pairs", type=int, default=20000)
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--frequency_weight", type=float, default=0.01)
    parser.add_argument("--selected_weight", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    name = args.run_name or f"stage_b_cluster_candidate_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    target = Path(args.output_dir) / name
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _archive_rows_for_splits(archive_path: str | Path, split_file: str | Path) -> list[SampleInfo]:
    with np.load(archive_path, allow_pickle=False) as archive:
        n = int(archive["tam"].shape[0])
        source_indices = np.asarray(archive.get("source_indices", np.arange(n)), dtype=np.int64)
        sample_ids = np.asarray(archive.get("sample_ids", np.arange(n))).astype(str)
        labels = np.asarray(archive["labels"], dtype=np.int64)
    position = {int(source): row for row, source in enumerate(source_indices.tolist())}
    infos: list[SampleInfo] = []
    local_index = 0
    with np.load(split_file, allow_pickle=False) as splits:
        for split_name in ("train", "val", "test"):
            requested = np.asarray(splits[f"{split_name}_indices"], dtype=np.int64)
            missing = [int(source) for source in requested.tolist() if int(source) not in position]
            if missing:
                raise ValueError(f"Archive is missing {len(missing)} {split_name} sources; first={missing[:5]}")
            for source in requested.tolist():
                row = int(position[int(source)])
                infos.append(
                    SampleInfo(
                        local_index=int(local_index),
                        archive_index=int(row),
                        source_index=int(source_indices[row]),
                        sample_id=str(sample_ids[row]),
                        split=str(split_name),
                        label=int(labels[row]),
                    )
                )
                local_index += 1
    return infos


def _load_archive_arrays(path: str | Path, infos: list[SampleInfo]) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray([item.archive_index for item in infos], dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        tam = np.asarray(archive["tam"][rows], dtype=np.float32)
        mask = np.asarray(archive["mask"][rows], dtype=np.float32)
    return tam, mask


def _segment_sums(values: np.ndarray, segments: int) -> np.ndarray:
    width = int(values.shape[-1])
    cuts = np.linspace(0, width, int(segments) + 1).astype(np.int64)
    parts = [values[..., cuts[i] : cuts[i + 1]].sum(axis=-1) for i in range(int(segments))]
    return np.stack(parts, axis=-1).astype(np.float32)


def _distribution_features(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float32)
    n, width = int(counts.shape[0]), int(counts.shape[-1])
    bins = np.arange(width, dtype=np.float32)
    total = np.maximum(counts.sum(axis=-1), 1.0)
    mean = (counts * bins.reshape(1, -1)).sum(axis=-1) / total
    var = (counts * (bins.reshape(1, -1) - mean.reshape(-1, 1)) ** 2).sum(axis=-1) / total
    active = np.count_nonzero(counts > 0, axis=-1)
    peak = np.max(counts, axis=-1)
    p90 = np.percentile(counts, 90, axis=-1)
    prob = counts / total.reshape(-1, 1)
    entropy = -np.sum(prob * np.log(np.maximum(prob, 1e-8)), axis=-1) / math.log(max(width, 2))
    return np.stack(
        [
            mean / max(width - 1, 1),
            np.sqrt(np.maximum(var, 0.0)) / max(width - 1, 1),
            np.log1p(active),
            np.log1p(peak),
            np.log1p(p90),
            entropy,
        ],
        axis=1,
    ).astype(np.float32)


def _structure_features(tam: np.ndarray, mask: np.ndarray, segments: int) -> np.ndarray:
    tam = np.asarray(tam, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    total_by_dir = tam.sum(axis=-1)
    total = np.maximum(total_by_dir.sum(axis=1, keepdims=True), 1.0)
    dir_props = total_by_dir / total
    log_totals = np.log1p(total_by_dir)
    tam_segments = _segment_sums(tam, int(segments))
    tam_seg_total = np.maximum(tam_segments.sum(axis=(1, 2), keepdims=True), 1.0)
    tam_seg_props = (tam_segments / tam_seg_total).reshape(tam.shape[0], -1)
    signed_segments = (tam_segments[:, 0, :] - tam_segments[:, 1, :]) / np.maximum(tam_segments.sum(axis=1), 1.0)
    mask_segments = _segment_sums(mask, int(segments))
    mask_seg_total = np.maximum(mask_segments.sum(axis=(1, 2), keepdims=True), 1.0)
    mask_seg_props = (mask_segments / mask_seg_total).reshape(mask.shape[0], -1)
    all_counts = tam.sum(axis=1)
    all_mask = mask.sum(axis=1)
    dist = np.concatenate(
        [
            _distribution_features(all_counts),
            _distribution_features(tam[:, 0, :]),
            _distribution_features(tam[:, 1, :]),
            _distribution_features(all_mask),
        ],
        axis=1,
    )
    return np.concatenate([log_totals, dir_props, tam_seg_props, signed_segments, mask_seg_props, dist], axis=1).astype(np.float32)


def _standardize(train_x: np.ndarray, all_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(train_x, axis=0)
    std = np.std(train_x, axis=0)
    keep = std > 1e-8
    if not np.any(keep):
        raise ValueError("All clustering features have near-zero variance.")
    mean = mean[keep]
    std = std[keep]
    return ((train_x[:, keep] - mean) / std).astype(np.float32), ((all_x[:, keep] - mean) / std).astype(np.float32), keep


def _sqdist(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    return np.sum((x[:, None, :] - centers[None, :, :]) ** 2, axis=2)


def _kmeans_plus_plus(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = int(x.shape[0])
    centers = np.empty((int(k), int(x.shape[1])), dtype=np.float32)
    first = int(rng.integers(0, n))
    centers[0] = x[first]
    closest = np.sum((x - centers[0]) ** 2, axis=1)
    for idx in range(1, int(k)):
        total = float(np.sum(closest))
        if total <= 1e-12:
            centers[idx] = x[int(rng.integers(0, n))]
        else:
            choice = int(rng.choice(np.arange(n), p=closest / total))
            centers[idx] = x[choice]
        closest = np.minimum(closest, np.sum((x - centers[idx]) ** 2, axis=1))
    return centers


def _fit_kmeans(train_x: np.ndarray, all_x: np.ndarray, k: int, max_iter: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    k = min(int(k), int(train_x.shape[0]))
    centers = _kmeans_plus_plus(train_x, k, rng)
    train_labels = np.zeros(int(train_x.shape[0]), dtype=np.int64)
    for _iter in range(int(max_iter)):
        dist = _sqdist(train_x, centers)
        new_labels = np.argmin(dist, axis=1).astype(np.int64)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = train_x[new_labels == cluster_id]
            if len(members):
                new_centers[cluster_id] = np.mean(members, axis=0)
            else:
                new_centers[cluster_id] = train_x[int(rng.integers(0, train_x.shape[0]))]
        shift = float(np.max(np.sqrt(np.sum((new_centers - centers) ** 2, axis=1))))
        centers = new_centers.astype(np.float32)
        train_labels = new_labels
        if shift < 1e-4:
            break
    all_labels = np.argmin(_sqdist(all_x, centers), axis=1).astype(np.int64)
    return centers, all_labels


def _top_indices(mask_flat: np.ndarray, ratio: float) -> np.ndarray:
    flat = np.asarray(mask_flat, dtype=np.float32).reshape(-1)
    k = max(1, int(round(float(ratio) * flat.size)))
    if k >= flat.size:
        return np.arange(flat.size, dtype=np.int64)
    idx = np.argpartition(-flat, k - 1)[:k]
    return idx[np.argsort(-flat[idx], kind="mergesort")].astype(np.int64)


def _sample_within_pairs(assignments: np.ndarray, rng: np.random.Generator, max_pairs_per_cluster: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for cluster_id in sorted(set(int(x) for x in assignments.tolist())):
        members = np.flatnonzero(assignments == int(cluster_id)).astype(np.int64)
        if len(members) < 2:
            continue
        total = len(members) * (len(members) - 1) // 2
        if total <= int(max_pairs_per_cluster):
            for a_pos in range(len(members)):
                for b_pos in range(a_pos + 1, len(members)):
                    pairs.append((int(members[a_pos]), int(members[b_pos])))
        else:
            seen: set[tuple[int, int]] = set()
            while len(seen) < int(max_pairs_per_cluster):
                a, b = rng.choice(members, size=2, replace=False).tolist()
                pair = (int(min(a, b)), int(max(a, b)))
                seen.add(pair)
            pairs.extend(sorted(seen))
    return pairs


def _sample_different_pairs(assignments: np.ndarray, rng: np.random.Generator, max_pairs: int) -> list[tuple[int, int]]:
    n = int(assignments.size)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    target = int(max_pairs)
    while len(pairs) < target and attempts < target * 50:
        attempts += 1
        a, b = rng.choice(np.arange(n), size=2, replace=False).tolist()
        if int(assignments[a]) == int(assignments[b]):
            continue
        pairs.add((int(min(a, b)), int(max(a, b))))
    return sorted(pairs)


def _normalized_rows(values: np.ndarray, centered: bool) -> np.ndarray:
    flat = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    if centered:
        flat = flat - np.mean(flat, axis=1, keepdims=True)
    norm = np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-8)
    return (flat / norm).astype(np.float32)


def _pair_similarity_rows(
    *,
    masks: np.ndarray,
    assignments: np.ndarray,
    pairs: list[tuple[int, int]],
    pair_group: str,
    ratios: list[float],
) -> list[dict[str, Any]]:
    flat_masks = np.asarray(masks, dtype=np.float32).reshape(masks.shape[0], -1)
    raw_norm = _normalized_rows(masks, centered=False)
    centered_norm = _normalized_rows(masks, centered=True)
    top_sets = {
        float(ratio): [set(_top_indices(flat_masks[i], float(ratio)).tolist()) for i in range(int(flat_masks.shape[0]))]
        for ratio in ratios
    }
    rows: list[dict[str, Any]] = []
    for ratio in ratios:
        values = []
        for a, b in pairs:
            a_top = top_sets[float(ratio)][a]
            b_top = top_sets[float(ratio)][b]
            inter = len(a_top & b_top)
            union = len(a_top | b_top)
            values.append(inter / max(union, 1))
        rows.append(
            {
                "pair_group": str(pair_group),
                "metric": "keypoint_topk_jaccard",
                "top_ratio": float(ratio),
                "pairs": int(len(pairs)),
                "mean": float(np.mean(values)) if values else 0.0,
                "median": float(np.median(values)) if values else 0.0,
                "p90": float(np.percentile(values, 90)) if values else 0.0,
            }
        )
    cosine = [float(np.dot(raw_norm[a], raw_norm[b])) for a, b in pairs]
    pearson = [float(np.dot(centered_norm[a], centered_norm[b])) for a, b in pairs]
    for metric, values in (("importance_cosine", cosine), ("importance_pearson", pearson)):
        rows.append(
            {
                "pair_group": str(pair_group),
                "metric": metric,
                "top_ratio": "",
                "pairs": int(len(pairs)),
                "mean": float(np.mean(values)) if values else 0.0,
                "median": float(np.median(values)) if values else 0.0,
                "p90": float(np.percentile(values, 90)) if values else 0.0,
            }
        )
    return rows


def _feature_index(name: str) -> int:
    return int(ACTION_FEATURE_NAMES.index(str(name)))


def _signature_from_feature_row(features: np.ndarray, *, time_bins: int) -> tuple[int, ...]:
    row = np.asarray(features, dtype=np.float32).reshape(-1)
    affected_center = int(round(float(row[_feature_index("affected_center_norm")]) * int(time_bins)))
    insert_center = int(round(float(row[_feature_index("insert_center_norm")]) * int(time_bins)))
    insert_width = int(round(float(row[_feature_index("insert_width_norm")]) * int(time_bins)))
    relative_offset = int(round(float(row[_feature_index("relative_offset_norm")]) * int(time_bins)))
    dose = int(round(float(np.expm1(max(float(row[_feature_index("dose_log")]), 0.0)))))
    dummy = int(round(float(np.expm1(max(float(row[_feature_index("dummy_count_log")]), 0.0)))))
    nonzero = int(round(float(np.expm1(max(float(row[_feature_index("nonzero_bins_log")]), 0.0)))))
    return (
        int(round(float(row[_feature_index("action_type_id")]))),
        int(round(float(row[_feature_index("direction_mode_id")]))),
        int(round(float(row[_feature_index("direction_bucket_id")]))),
        int(round(float(row[_feature_index("affected_direction_id")]))),
        int(affected_center),
        int(insert_center),
        int(insert_width),
        int(relative_offset),
        int(dose),
        int(dummy),
        int(nonzero),
    )


def _action_type_signature(signature: tuple[int, ...]) -> tuple[int, int, int, int]:
    return tuple(int(x) for x in signature[:4])


def _signatures(action_features: np.ndarray, *, time_bins: int) -> list[tuple[int, ...]]:
    return [_signature_from_feature_row(row, time_bins=int(time_bins)) for row in np.asarray(action_features, dtype=np.float32)]


def _csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_first_action_records(
    csv_paths: list[str],
    *,
    sample_split: dict[int, str],
    time_bins: int,
) -> dict[int, FirstActionRecord]:
    first_rows: dict[int, dict[str, str]] = {}
    for csv_path in csv_paths:
        for row in _csv_rows(csv_path):
            if str(row.get("event_type", "")).strip() != "action":
                continue
            sample_index = int(float(row["sample_index"]))
            round_index = int(float(row.get("round_index", 0)))
            step_index = int(float(row.get("step_index", 0)))
            if round_index != 0 or step_index != 0:
                continue
            if sample_index not in first_rows:
                first_rows[sample_index] = row
    records: dict[int, FirstActionRecord] = {}
    for sample_index, row in first_rows.items():
        record_path = str(row["record_path"])
        with np.load(record_path, allow_pickle=False) as arrays:
            action_features = np.asarray(arrays["action_features"], dtype=np.float32)
            gains = np.asarray(arrays["candidate_gains"], dtype=np.float32)
            selected_index = int(np.asarray(arrays["selected_index"]).item())
        if action_features.shape[0] == 0 or gains.size == 0 or selected_index < 0:
            continue
        signatures = _signatures(action_features, time_bins=int(time_bins))
        oracle_index = int(np.argmax(gains))
        records[int(sample_index)] = FirstActionRecord(
            sample_index=int(sample_index),
            split=str(sample_split.get(int(sample_index), "unknown")),
            record_path=str(record_path),
            selected_signature=signatures[int(selected_index)],
            oracle_signature=signatures[int(oracle_index)],
            selected_gain=float(gains[int(selected_index)]),
            oracle_gain=float(gains[int(oracle_index)]),
            candidate_signatures=signatures,
            candidate_gains=gains.astype(np.float32),
        )
    return records


def _action_agreement_rows(
    *,
    records_by_local: dict[int, FirstActionRecord],
    pairs: list[tuple[int, int]],
    pair_group: str,
) -> list[dict[str, Any]]:
    exact_values = []
    type_values = []
    for a, b in pairs:
        ra = records_by_local.get(int(a))
        rb = records_by_local.get(int(b))
        if ra is None or rb is None:
            continue
        exact_values.append(float(ra.selected_signature == rb.selected_signature))
        type_values.append(float(_action_type_signature(ra.selected_signature) == _action_type_signature(rb.selected_signature)))
    return [
        {
            "pair_group": str(pair_group),
            "metric": "teacher_first_action_signature_agreement",
            "pairs": int(len(exact_values)),
            "mean": float(np.mean(exact_values)) if exact_values else 0.0,
        },
        {
            "pair_group": str(pair_group),
            "metric": "teacher_first_action_type_agreement",
            "pairs": int(len(type_values)),
            "mean": float(np.mean(type_values)) if type_values else 0.0,
        },
    ]


def _record_unique_candidate_contrib(record: FirstActionRecord) -> dict[tuple[int, ...], float]:
    best: dict[tuple[int, ...], float] = {}
    for sig, gain in zip(record.candidate_signatures, record.candidate_gains.tolist()):
        value = float(gain)
        if sig not in best or value > best[sig]:
            best[sig] = value
    return best


def _build_template_libraries(
    *,
    records_by_local: dict[int, FirstActionRecord],
    assignments: np.ndarray,
    infos: list[SampleInfo],
    frequency_weight: float,
    selected_weight: float,
) -> dict[str, dict[int, list[tuple[tuple[int, ...], float, int, float]]]]:
    stats: dict[str, dict[int, dict[tuple[int, ...], dict[str, float]]]] = {
        "cluster": defaultdict(lambda: defaultdict(lambda: {"gain_sum": 0.0, "count": 0.0, "selected": 0.0})),
        "global": defaultdict(lambda: defaultdict(lambda: {"gain_sum": 0.0, "count": 0.0, "selected": 0.0})),
    }
    for local_index, info in enumerate(infos):
        if info.split != "train":
            continue
        record = records_by_local.get(int(local_index))
        if record is None:
            continue
        cluster_id = int(assignments[local_index])
        for scope, key in (("cluster", cluster_id), ("global", -1)):
            for sig, gain in _record_unique_candidate_contrib(record).items():
                item = stats[scope][key][sig]
                item["gain_sum"] += float(gain)
                item["count"] += 1.0
            stats[scope][key][record.selected_signature]["selected"] += 1.0
    libraries: dict[str, dict[int, list[tuple[tuple[int, ...], float, int, float]]]] = {"cluster": {}, "global": {}}
    for scope, by_key in stats.items():
        for key, by_sig in by_key.items():
            ranked = []
            for sig, item in by_sig.items():
                count = max(float(item["count"]), 1.0)
                mean_gain = float(item["gain_sum"] / count)
                score = mean_gain + float(frequency_weight) * math.log1p(count) + float(selected_weight) * math.log1p(float(item["selected"]))
                ranked.append((sig, float(score), int(count), float(mean_gain)))
            ranked.sort(key=lambda item: (-item[1], item[0]))
            libraries[scope][int(key)] = ranked
    return libraries


def _unique_top_signatures(signatures: list[tuple[int, ...]], gains: np.ndarray, k: int) -> set[tuple[int, ...]]:
    order = np.argsort(-np.asarray(gains, dtype=np.float32), kind="mergesort")
    out: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for idx in order.tolist():
        sig = signatures[int(idx)]
        if sig in seen:
            continue
        seen.add(sig)
        out.append(sig)
        if len(out) >= int(k):
            break
    return set(out)


def _template_coverage_rows(
    *,
    records_by_local: dict[int, FirstActionRecord],
    assignments: np.ndarray,
    infos: list[SampleInfo],
    libraries: dict[str, dict[int, list[tuple[tuple[int, ...], float, int, float]]]],
    topks: list[int],
    eps_values: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sample_rows: list[dict[str, Any]] = []
    for local_index, info in enumerate(infos):
        record = records_by_local.get(int(local_index))
        if record is None:
            continue
        cluster_id = int(assignments[local_index])
        for scope in ("cluster", "global"):
            library_key = cluster_id if scope == "cluster" else -1
            ranked = libraries.get(scope, {}).get(int(library_key), [])
            for topk in topks:
                template = set(sig for sig, _score, _count, _mean_gain in ranked[: int(topk)])
                oracle_gain = float(record.oracle_gain)
                selected_gain = float(record.selected_gain)
                shared_gains = [
                    float(gain)
                    for sig, gain in zip(record.candidate_signatures, record.candidate_gains.tolist())
                    if sig in template
                ]
                best_shared_gain = float(max(shared_gains)) if shared_gains else 0.0
                regret = float(max(oracle_gain - best_shared_gain, 0.0))
                sample_top = _unique_top_signatures(record.candidate_signatures, record.candidate_gains, int(topk))
                overlap = len(sample_top & template) / max(len(sample_top), 1)
                row: dict[str, Any] = {
                    "sample_index": int(info.archive_index),
                    "source_index": int(info.source_index),
                    "sample_id": str(info.sample_id),
                    "split": str(info.split),
                    "cluster": int(cluster_id),
                    "template_scope": str(scope),
                    "topk": int(topk),
                    "template_size": int(len(template)),
                    "candidate_count": int(len(record.candidate_signatures)),
                    "shared_candidate_count": int(len(shared_gains)),
                    "oracle_gain": float(oracle_gain),
                    "selected_gain": float(selected_gain),
                    "best_shared_gain": float(best_shared_gain),
                    "regret": float(regret),
                    "oracle_signature_covered": int(record.oracle_signature in template),
                    "selected_signature_covered": int(record.selected_signature in template),
                    "candidate_topk_overlap": float(overlap),
                }
                for eps in eps_values:
                    row[f"near_optimal_eps_{str(eps).replace('.', 'p')}"] = int(regret <= float(eps))
                sample_rows.append(row)
    summary_rows: list[dict[str, Any]] = []
    group_keys = sorted({(row["split"], row["template_scope"], int(row["topk"])) for row in sample_rows})
    for split, scope, topk in group_keys:
        rows = [row for row in sample_rows if row["split"] == split and row["template_scope"] == scope and int(row["topk"]) == int(topk)]
        summary: dict[str, Any] = {
            "split": str(split),
            "template_scope": str(scope),
            "topk": int(topk),
            "samples": int(len(rows)),
            "mean_template_size": float(np.mean([row["template_size"] for row in rows])) if rows else 0.0,
            "mean_shared_candidate_count": float(np.mean([row["shared_candidate_count"] for row in rows])) if rows else 0.0,
            "oracle_signature_coverage": float(np.mean([row["oracle_signature_covered"] for row in rows])) if rows else 0.0,
            "selected_signature_coverage": float(np.mean([row["selected_signature_covered"] for row in rows])) if rows else 0.0,
            "mean_candidate_topk_overlap": float(np.mean([row["candidate_topk_overlap"] for row in rows])) if rows else 0.0,
            "mean_regret": float(np.mean([row["regret"] for row in rows])) if rows else 0.0,
            "median_regret": float(np.median([row["regret"] for row in rows])) if rows else 0.0,
        }
        for eps in eps_values:
            key = f"near_optimal_eps_{str(eps).replace('.', 'p')}"
            summary[key] = float(np.mean([row[key] for row in rows])) if rows else 0.0
        summary_rows.append(summary)
    return sample_rows, summary_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _cluster_rows(infos: list[SampleInfo], assignments: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    assignment_rows = []
    for info, cluster_id in zip(infos, assignments.tolist()):
        assignment_rows.append(
            {
                "local_index": int(info.local_index),
                "archive_index": int(info.archive_index),
                "source_index": int(info.source_index),
                "sample_id": str(info.sample_id),
                "split": str(info.split),
                "label_eval_only": int(info.label),
                "cluster": int(cluster_id),
            }
        )
    summary_rows = []
    for cluster_id in sorted(set(int(x) for x in assignments.tolist())):
        rows = [row for row in assignment_rows if int(row["cluster"]) == int(cluster_id)]
        split_counts = Counter(row["split"] for row in rows)
        label_counts = Counter(int(row["label_eval_only"]) for row in rows)
        probs = np.asarray(list(label_counts.values()), dtype=np.float64)
        probs = probs / max(float(probs.sum()), 1.0)
        entropy = -float(np.sum(probs * np.log(np.maximum(probs, 1e-12)))) / math.log(max(len(label_counts), 2))
        summary_rows.append(
            {
                "cluster": int(cluster_id),
                "samples": int(len(rows)),
                "train": int(split_counts.get("train", 0)),
                "val": int(split_counts.get("val", 0)),
                "test": int(split_counts.get("test", 0)),
                "eval_only_label_count": int(len(label_counts)),
                "eval_only_label_entropy": float(entropy),
                "eval_only_top_label": int(label_counts.most_common(1)[0][0]) if label_counts else -1,
                "eval_only_top_label_count": int(label_counts.most_common(1)[0][1]) if label_counts else 0,
            }
        )
    return assignment_rows, summary_rows


def main() -> None:
    args = parse_args()
    output_dir = _run_dir(args)
    rng = np.random.default_rng(int(args.seed))
    top_ratios = _parse_floats(args.top_ratios)
    topks = _parse_ints(args.template_topks)
    eps_values = _parse_floats(args.near_eps)
    infos = _archive_rows_for_splits(args.archive, args.split_file)
    tam, mask = _load_archive_arrays(args.archive, infos)
    features = _structure_features(tam, mask, int(args.segments))
    train_local = np.asarray([info.local_index for info in infos if info.split == "train"], dtype=np.int64)
    train_x, all_x, keep = _standardize(features[train_local], features)
    centers, assignments = _fit_kmeans(train_x, all_x, int(args.num_clusters), int(args.kmeans_max_iter), int(args.seed))

    sample_split = {int(info.archive_index): str(info.split) for info in infos}
    local_by_archive = {int(info.archive_index): int(info.local_index) for info in infos}
    csv_paths = [item.strip() for item in str(args.teacher_records_csvs).split(",") if item.strip()]
    first_by_archive = _load_first_action_records(csv_paths, sample_split=sample_split, time_bins=int(args.signature_time_bins))
    records_by_local = {
        int(local_by_archive[archive_index]): record
        for archive_index, record in first_by_archive.items()
        if int(archive_index) in local_by_archive
    }

    assignment_rows, cluster_summary_rows = _cluster_rows(infos, assignments)
    within_pairs = _sample_within_pairs(assignments, rng, int(args.max_pairs_per_cluster))
    different_pairs = _sample_different_pairs(assignments, rng, int(args.max_random_pairs))
    similarity_rows = []
    similarity_rows.extend(
        _pair_similarity_rows(masks=mask, assignments=assignments, pairs=within_pairs, pair_group="within_cluster", ratios=top_ratios)
    )
    similarity_rows.extend(
        _pair_similarity_rows(masks=mask, assignments=assignments, pairs=different_pairs, pair_group="different_cluster", ratios=top_ratios)
    )
    action_rows = []
    action_rows.extend(_action_agreement_rows(records_by_local=records_by_local, pairs=within_pairs, pair_group="within_cluster"))
    action_rows.extend(_action_agreement_rows(records_by_local=records_by_local, pairs=different_pairs, pair_group="different_cluster"))

    libraries = _build_template_libraries(
        records_by_local=records_by_local,
        assignments=assignments,
        infos=infos,
        frequency_weight=float(args.frequency_weight),
        selected_weight=float(args.selected_weight),
    )
    coverage_samples, coverage_summary = _template_coverage_rows(
        records_by_local=records_by_local,
        assignments=assignments,
        infos=infos,
        libraries=libraries,
        topks=topks,
        eps_values=eps_values,
    )

    _write_csv(output_dir / "cluster_assignments.csv", assignment_rows)
    _write_csv(output_dir / "cluster_summary.csv", cluster_summary_rows)
    _write_csv(output_dir / "keypoint_pair_similarity_summary.csv", similarity_rows)
    _write_csv(output_dir / "teacher_first_action_agreement_summary.csv", action_rows)
    _write_csv(output_dir / "candidate_template_coverage_samples.csv", coverage_samples)
    _write_csv(output_dir / "candidate_template_coverage_summary.csv", coverage_summary)
    np.savez_compressed(
        output_dir / "cluster_model_arrays.npz",
        centers=centers.astype(np.float32),
        assignments=assignments.astype(np.int64),
        feature_keep_mask=keep.astype(np.bool_),
        features=features.astype(np.float32),
    )
    manifest = {
        "archive": str(args.archive),
        "split_file": str(args.split_file),
        "teacher_records_csvs": csv_paths,
        "samples": int(len(infos)),
        "train_samples": int(sum(1 for item in infos if item.split == "train")),
        "val_samples": int(sum(1 for item in infos if item.split == "val")),
        "test_samples": int(sum(1 for item in infos if item.split == "test")),
        "num_clusters": int(args.num_clusters),
        "segments": int(args.segments),
        "signature_time_bins": int(args.signature_time_bins),
        "top_ratios": top_ratios,
        "template_topks": topks,
        "near_eps": eps_values,
        "first_action_records": int(len(records_by_local)),
        "within_cluster_pairs": int(len(within_pairs)),
        "different_cluster_pairs": int(len(different_pairs)),
        "outputs": {
            "cluster_assignments": str(output_dir / "cluster_assignments.csv"),
            "cluster_summary": str(output_dir / "cluster_summary.csv"),
            "keypoint_pair_similarity": str(output_dir / "keypoint_pair_similarity_summary.csv"),
            "teacher_first_action_agreement": str(output_dir / "teacher_first_action_agreement_summary.csv"),
            "candidate_template_coverage_summary": str(output_dir / "candidate_template_coverage_summary.csv"),
            "candidate_template_coverage_samples": str(output_dir / "candidate_template_coverage_samples.csv"),
            "cluster_model_arrays": str(output_dir / "cluster_model_arrays.npz"),
        },
    }
    (output_dir / "cluster_candidate_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
