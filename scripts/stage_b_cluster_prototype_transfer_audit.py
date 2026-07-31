# -*- coding: utf-8 -*-
"""Audit whether cluster prototype Teacher policies transfer to members.

This is an end-to-end check for cluster-aware Stage B acceleration.  Clustering
is fit on the explicit train split only.  Each cluster is represented by a real
train medoid trace, the formal Teacher is run only on those prototypes, and the
prototype action trajectory is replayed or lightly verified on cluster members.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.data.cw import resolve_cw_path, stored_npy_from_npz
from dmmp.stage_a.modeling import load_stage_a_attacker
from dmmp.stage_b.expanded_generator import ExpandedAction, action_identity
from dmmp.stage_b.objectives import probability_metrics
from dmmp.utils import resolve_device, set_seed
from dmmp.utils.config import DEFAULT_DATA_ROOT, DEFAULT_OUTPUT_DIR

from scripts.stage_b_run_b2e_diverse_search import (
    MethodConfig,
    _action_dummy_count,
    _default_checkpoint,
    _evaluate_actions,
    _margin,
    _method_config,
    _resource_fields,
    _run_controller,
    _runtime_args,
    _sample_row as _teacher_sample_row,
)
from scripts.stage_b_run_dual_actuator import (
    EvalState,
    _apply_delay,
    _fast_refresh_mask,
    _initial_state,
    _load_raw_rows,
    _predict_one,
    _render_dummy,
)


DEFAULT_ARCHIVE = "results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz"
DEFAULT_FULL_SPLIT = "results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz"
DEFAULT_OUT = "results/stage_b_cluster_prototype_transfer_audit_20260730"
DEFAULT_TEACHER_SUMMARIES = ",".join(
    [
        "results/stage_b2e_teacher_950_train_tensorized_seed0_a16req/teacher_sample_summary.csv",
        "results/stage_b2e_teacher_950_val_tensorized_seed0_a16req/teacher_sample_summary.csv",
        "results/stage_b2e_teacher_950_test_tensorized_seed0_a16req/teacher_sample_summary.csv",
    ]
)


@dataclass(frozen=True)
class SampleInfo:
    local_index: int
    archive_index: int
    source_index: int
    sample_id: str
    split: str
    label: int


@dataclass(frozen=True)
class ClusterSpec:
    representation: str
    num_clusters: int
    cluster_id: int
    cluster_size: int
    prototype_local_index: int
    prototype_archive_index: int
    prototype_source_index: int
    prototype_sample_id: str
    prototype_distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE)
    parser.add_argument("--split_file", default=DEFAULT_FULL_SPLIT)
    parser.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", default=DEFAULT_OUT)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--representations", default="normalized_flat_tam,pca_normalized_tam")
    parser.add_argument("--num_clusters", default="100,300,600")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--kmeans_batch_size", type=int, default=4096)
    parser.add_argument("--kmeans_max_iter", type=int, default=100)
    parser.add_argument("--top_relative_keypoints", type=int, default=64)
    parser.add_argument("--transfer_modes", default="absolute_replay,relative_replay,relative_top4_verify")
    parser.add_argument("--transfer_splits", default="val,test")
    parser.add_argument("--transfer_max_samples", type=int, default=0)
    parser.add_argument("--prototype_limit", type=int, default=0)
    parser.add_argument("--prototype_only", action="store_true")
    parser.add_argument("--prototype_policy_dir", default="")
    parser.add_argument("--teacher_sample_summary_csvs", default=DEFAULT_TEACHER_SUMMARIES)
    parser.add_argument("--attacker", choices=["df", "rf"], default="rf")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--protocol", default="bidirectional_cooperative")
    parser.add_argument("--method", default="stratified_top128")
    parser.add_argument("--budget", type=float, default=0.10)
    parser.add_argument("--margin_target", type=float, default=0.0)
    parser.add_argument("--max_delay", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--delay_length", type=int, default=64)
    parser.add_argument("--delay_rho", type=float, default=1.0)
    parser.add_argument("--ratio", type=float, default=0.10)
    parser.add_argument("--max_windows", type=int, default=8)
    parser.add_argument("--max_dummy_steps", type=int, default=8)
    parser.add_argument("--max_action_budget", type=float, default=0.10)
    parser.add_argument("--max_local_rate_peak", type=int, default=64)
    parser.add_argument("--stratified_bucket_k", type=int, default=8)
    parser.add_argument("--stratified_global_k", type=int, default=16)
    parser.add_argument("--random_explore_k", type=int, default=8)
    parser.add_argument("--true_recall_pool_size", type=int, default=0)
    parser.add_argument("--confidence_weight", type=float, default=0.40)
    parser.add_argument("--margin_weight", type=float, default=0.40)
    parser.add_argument("--entropy_weight", type=float, default=0.20)
    parser.add_argument("--renderer_batch_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_num_slots", type=int, default=1800)
    parser.add_argument("--df_architecture", default="project")
    parser.add_argument("--df_tam_adapter", default="signed_balance")
    parser.add_argument("--renderer_coordinate", default="rf_tam")
    parser.add_argument("--renderer_strategy", default="uniform_in_patch")
    parser.add_argument("--compact_candidate_generation", action="store_true", default=True)
    parser.add_argument("--deferred_materialize_oversample", type=int, default=1)
    parser.add_argument("--candidate_batch_size", type=int, default=8192)
    parser.add_argument("--materialization_batch_size", type=int, default=128)
    parser.add_argument("--candidate_device", choices=["auto", "cpu", "cuda"], default="cuda")
    parser.add_argument("--candidate_eval_mode", choices=["renderer", "gpu_tam"], default="gpu_tam")
    parser.add_argument("--include_existing_teacher_upper_bound", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--progress", action="store_true")
    return parser.parse_args()


def _parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _run_dir(args: argparse.Namespace) -> Path:
    base = Path(args.output_dir)
    target = base / args.run_name if str(args.run_name).strip() else base
    if target.exists() and any(target.iterdir()) and not bool(args.force):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {target}; pass --force.")
    target.mkdir(parents=True, exist_ok=True)
    (target / "prototype_policies").mkdir(exist_ok=True)
    (target / "sample_manifests").mkdir(exist_ok=True)
    return target


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(str(key))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _archive_infos(archive_path: str | Path, split_file: str | Path) -> list[SampleInfo]:
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


def _load_archive_arrays(path: str | Path, infos: list[SampleInfo]) -> dict[str, np.ndarray]:
    rows = np.asarray([info.archive_index for info in infos], dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        payload: dict[str, np.ndarray] = {}
        n = int(archive["tam"].shape[0])
        for key in archive.files:
            arr = archive[key]
            if arr.shape[:1] == (n,):
                payload[key] = np.asarray(arr[rows])
            else:
                payload[key] = np.asarray(arr)
    return payload


def _load_archive_keys(path: str | Path, archive_rows: list[int] | np.ndarray, keys: list[str]) -> dict[str, np.ndarray]:
    rows = np.asarray(archive_rows, dtype=np.int64)
    with np.load(path, allow_pickle=False) as archive:
        payload: dict[str, np.ndarray] = {}
        for key in keys:
            payload[key] = np.asarray(archive[key][rows])
    return payload


def _load_raw_selected(data_root: str | Path, source_indices: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    selected = np.asarray(source_indices, dtype=np.int64)
    if selected.size == 0:
        return np.zeros((0, 2, int(args.max_trace_length)), dtype=np.float32)
    path = resolve_cw_path(data_root)
    if path.name.lower() == "train.npz" and (path.parent / "valid.npz").is_file() and (path.parent / "test.npz").is_file():
        return _load_raw_rows(str(data_root), selected, args)
    x_map = stored_npy_from_npz(path, "X")
    if x_map is not None:
        raw = np.empty((int(selected.size), *x_map.shape[1:]), dtype=np.float32)
        for start in range(0, int(selected.size), 512):
            end = min(start + 512, int(selected.size))
            raw[start:end] = np.asarray(x_map[selected[start:end]], dtype=np.float32)
        return raw
    with np.load(path, allow_pickle=False) as arrays:
        return np.asarray(arrays["X"][selected], dtype=np.float32)


def _normalized_tam_flat(tam: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tam = np.asarray(tam, dtype=np.float32)
    totals = tam.sum(axis=-1).astype(np.float32)
    per_dir = tam / np.maximum(totals[:, :, None], 1.0)
    global_total = np.maximum(totals.sum(axis=1, keepdims=True), 1.0)
    extras = np.concatenate([np.log1p(totals), totals / global_total], axis=1).astype(np.float32)
    return per_dir.reshape(tam.shape[0], -1).astype(np.float32), extras


def _features_for_representation(
    *,
    tam: np.ndarray,
    train_mask: np.ndarray,
    representation: str,
    pca_dim: int,
    batch_size: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    flat, extras = _normalized_tam_flat(tam)
    representation = str(representation)
    meta: dict[str, Any] = {"representation": representation}
    if representation == "normalized_flat_tam":
        raw = np.concatenate([flat, extras], axis=1).astype(np.float32)
        scaler = StandardScaler(copy=False)
        train_x = scaler.fit_transform(raw[train_mask]).astype(np.float32)
        all_x = scaler.transform(raw).astype(np.float32)
        meta.update({"feature_dim": int(all_x.shape[1]), "pca_dim": 0})
        return all_x, meta
    if representation == "pca_normalized_tam":
        dim = min(int(pca_dim), int(flat.shape[1]), int(np.count_nonzero(train_mask)) - 1)
        if dim <= 0:
            raise ValueError("PCA representation needs at least two train samples.")
        scaler0 = StandardScaler(copy=True)
        train_flat = scaler0.fit_transform(flat[train_mask]).astype(np.float32)
        ipca = IncrementalPCA(n_components=dim, batch_size=max(int(batch_size), dim * 4))
        for start in range(0, train_flat.shape[0], max(int(batch_size), dim * 4)):
            ipca.partial_fit(train_flat[start : start + max(int(batch_size), dim * 4)])
        transformed = np.empty((flat.shape[0], dim), dtype=np.float32)
        for start in range(0, flat.shape[0], max(int(batch_size), dim * 4)):
            chunk = scaler0.transform(flat[start : start + max(int(batch_size), dim * 4)]).astype(np.float32)
            transformed[start : start + chunk.shape[0]] = ipca.transform(chunk).astype(np.float32)
        raw = np.concatenate([transformed, extras], axis=1).astype(np.float32)
        scaler1 = StandardScaler(copy=False)
        scaler1.fit(raw[train_mask])
        all_x = scaler1.transform(raw).astype(np.float32)
        meta.update(
            {
                "feature_dim": int(all_x.shape[1]),
                "pca_dim": int(dim),
                "pca_explained_variance_sum": float(np.sum(ipca.explained_variance_ratio_)),
            }
        )
        return all_x, meta
    raise ValueError(f"Unknown representation: {representation}")


def _fit_assignments(
    *,
    features: np.ndarray,
    train_mask: np.ndarray,
    k: int,
    batch_size: int,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    train_x = np.asarray(features[train_mask], dtype=np.float32)
    actual_k = min(int(k), int(train_x.shape[0]))
    model = MiniBatchKMeans(
        n_clusters=actual_k,
        random_state=int(seed),
        batch_size=max(int(batch_size), actual_k),
        max_iter=int(max_iter),
        n_init=3,
        reassignment_ratio=0.01,
    )
    model.fit(train_x)
    labels = model.predict(np.asarray(features, dtype=np.float32)).astype(np.int64)
    return labels, np.asarray(model.cluster_centers_, dtype=np.float32)


def _cluster_specs(
    *,
    representation: str,
    k: int,
    features: np.ndarray,
    assignments: np.ndarray,
    centers: np.ndarray,
    infos: list[SampleInfo],
) -> list[ClusterSpec]:
    specs: list[ClusterSpec] = []
    for cluster_id in range(int(centers.shape[0])):
        train_members = [
            idx
            for idx, info in enumerate(infos)
            if info.split == "train" and int(assignments[idx]) == int(cluster_id)
        ]
        if not train_members:
            continue
        member_x = features[np.asarray(train_members, dtype=np.int64)]
        distances = np.sum((member_x - centers[int(cluster_id)].reshape(1, -1)) ** 2, axis=1)
        best_pos = int(np.argmin(distances))
        proto_idx = int(train_members[best_pos])
        proto = infos[proto_idx]
        cluster_size = int(np.count_nonzero(assignments == int(cluster_id)))
        specs.append(
            ClusterSpec(
                representation=str(representation),
                num_clusters=int(k),
                cluster_id=int(cluster_id),
                cluster_size=int(cluster_size),
                prototype_local_index=int(proto_idx),
                prototype_archive_index=int(proto.archive_index),
                prototype_source_index=int(proto.source_index),
                prototype_sample_id=str(proto.sample_id),
                prototype_distance=float(distances[best_pos]),
            )
        )
    return specs


def _keypoint_anchors(mask: np.ndarray, top_k: int) -> list[tuple[int, int, float]]:
    flat = np.asarray(mask, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        return []
    k = min(max(int(top_k), 1), int(flat.size))
    idx = np.argpartition(-flat, k - 1)[:k]
    idx = idx[np.argsort(-flat[idx], kind="mergesort")]
    width = int(np.asarray(mask).shape[-1])
    return [(int(item // width), int(item % width), float(flat[item])) for item in idx.tolist()]


def _counts_sparse(counts: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(counts, dtype=np.int32)
    nz = np.argwhere(arr != 0)
    return {
        "shape": [int(x) for x in arr.shape],
        "direction": [int(x[0]) for x in nz.tolist()],
        "bin": [int(x[1]) for x in nz.tolist()],
        "count": [int(arr[int(x[0]), int(x[1])]) for x in nz.tolist()],
    }


def _counts_from_sparse(payload: dict[str, Any]) -> np.ndarray:
    shape = tuple(int(x) for x in payload.get("shape", [2, 1800]))
    arr = np.zeros(shape, dtype=np.int32)
    for direction, bin_index, count in zip(payload.get("direction", []), payload.get("bin", []), payload.get("count", [])):
        arr[int(direction), int(bin_index)] += int(count)
    return arr


def _action_center(action: ExpandedAction, counts: np.ndarray) -> int:
    center = int(getattr(action, "insert_center", -1))
    if center >= 0:
        return center
    nz = np.argwhere(np.asarray(counts, dtype=np.int32) > 0)
    if nz.size == 0:
        return 0
    weights = np.asarray(counts, dtype=np.float32)[nz[:, 0], nz[:, 1]]
    return int(round(float(np.average(nz[:, 1], weights=np.maximum(weights, 1e-6)))))


def _action_to_dict(
    action: ExpandedAction,
    *,
    mask: np.ndarray,
    gain: float,
    score: float,
    rank: int,
    selected: bool,
    top_relative_keypoints: int,
) -> dict[str, Any]:
    counts = np.asarray(action.counts, dtype=np.int32)
    anchors = _keypoint_anchors(mask, top_relative_keypoints)
    center = _action_center(action, counts)
    if anchors:
        nearest_rank, (anchor_direction, anchor_bin, anchor_value) = min(
            enumerate(anchors), key=lambda item: abs(int(item[1][1]) - int(center))
        )
    else:
        nearest_rank, anchor_direction, anchor_bin, anchor_value = 0, 0, center, 0.0
    fields = asdict(action)
    fields.pop("counts", None)
    return {
        "fields": _jsonable(fields),
        "counts": _counts_sparse(counts),
        "gain": float(gain),
        "score": float(score),
        "candidate_rank": int(rank),
        "selected": int(bool(selected)),
        "relative_anchor_rank": int(nearest_rank),
        "relative_anchor_direction": int(anchor_direction),
        "relative_anchor_bin": int(anchor_bin),
        "relative_anchor_value": float(anchor_value),
        "relative_offset_bins": int(center - int(anchor_bin)),
        "original_insert_center": int(center),
    }


def _action_from_dict(
    payload: dict[str, Any],
    *,
    counts: np.ndarray | None = None,
    sample_index: int,
    sample_id: str,
    label: int,
) -> ExpandedAction:
    fields = dict(payload.get("fields", {}))
    counts_arr = np.asarray(counts if counts is not None else _counts_from_sparse(payload["counts"]), dtype=np.int32)
    width = int(counts_arr.shape[-1])
    nz = np.argwhere(counts_arr > 0)
    if nz.size:
        start = int(nz[:, 1].min())
        end = int(nz[:, 1].max() + 1)
        center = int(round(float(np.average(nz[:, 1], weights=counts_arr[nz[:, 0], nz[:, 1]]))))
    else:
        start = end = center = 0
    fields.update(
        {
            "sample_index": int(sample_index),
            "sample_id": str(sample_id),
            "true_label": int(label),
            "counts": counts_arr,
            "dummy_count": int(counts_arr.sum()),
            "outgoing_dummy_count": int(counts_arr[0].sum()) if counts_arr.shape[0] > 0 else 0,
            "incoming_dummy_count": int(counts_arr[1].sum()) if counts_arr.shape[0] > 1 else 0,
            "nonzero_bin_count": int(np.count_nonzero(counts_arr)),
            "active_bin_count": int(len(np.unique(nz[:, 1]))) if nz.size else 0,
            "count_signature": tuple(int(x) for x in counts_arr[counts_arr > 0].tolist()),
            "insert_start": max(0, min(width, int(start))),
            "insert_end": max(0, min(width, int(end))),
            "insert_center": max(0, min(width - 1, int(center))),
            "affected_start": max(0, min(width, int(fields.get("affected_start", start)))),
            "affected_end": max(0, min(width, int(fields.get("affected_end", end)))),
            "affected_center": max(0, min(width - 1, int(fields.get("affected_center", center)))),
        }
    )
    return ExpandedAction(**fields)


def _shift_counts_strict(counts: np.ndarray, shift: int) -> np.ndarray | None:
    arr = np.asarray(counts, dtype=np.int32)
    out = np.zeros_like(arr, dtype=np.int32)
    nz = np.argwhere(arr > 0)
    for direction, bin_index in nz.tolist():
        target = int(bin_index) + int(shift)
        if target < 0 or target >= arr.shape[-1]:
            return None
        out[int(direction), target] += int(arr[int(direction), int(bin_index)])
    return out


def _scale_counts_to_total(counts: np.ndarray, target_total: int) -> np.ndarray:
    arr = np.asarray(counts, dtype=np.int32)
    target = max(0, int(target_total))
    out = np.zeros_like(arr, dtype=np.int32)
    original_total = int(arr.sum())
    if target <= 0 or original_total <= 0:
        return out
    nz = np.argwhere(arr > 0)
    weights = arr[nz[:, 0], nz[:, 1]].astype(np.float64) / float(original_total)
    exact = weights * float(target)
    base = np.floor(exact).astype(np.int32)
    remainder = int(target - int(base.sum()))
    if remainder > 0:
        order = np.argsort(-(exact - base), kind="mergesort")
        base[order[:remainder]] += 1
    for (direction, bin_index), value in zip(nz.tolist(), base.tolist()):
        if int(value) > 0:
            out[int(direction), int(bin_index)] = int(value)
    return out


def _mapped_action(
    payload: dict[str, Any],
    *,
    mode: str,
    member_mask: np.ndarray,
    sample_index: int,
    sample_id: str,
    label: int,
    top_relative_keypoints: int,
) -> ExpandedAction | None:
    base_counts = _counts_from_sparse(payload["counts"])
    if str(mode) == "absolute_replay":
        counts = base_counts
    else:
        anchors = _keypoint_anchors(member_mask, top_relative_keypoints)
        rank = int(payload.get("relative_anchor_rank", 0))
        if not anchors:
            return None
        anchor_bin = int(anchors[min(rank, len(anchors) - 1)][1])
        target_center = int(anchor_bin + int(payload.get("relative_offset_bins", 0)))
        shift = int(target_center - int(payload.get("original_insert_center", target_center)))
        counts = _shift_counts_strict(base_counts, shift)
        if counts is None:
            return None
    if int(np.asarray(counts, dtype=np.int32).sum()) <= 0:
        return None
    return _action_from_dict(payload, counts=counts, sample_index=sample_index, sample_id=sample_id, label=label)


def _budget_normalized_absolute_action(
    payload: dict[str, Any],
    *,
    target_dummy: int,
    sample_index: int,
    sample_id: str,
    label: int,
) -> ExpandedAction | None:
    counts = _scale_counts_to_total(_counts_from_sparse(payload["counts"]), int(target_dummy))
    if int(counts.sum()) <= 0:
        return None
    return _action_from_dict(payload, counts=counts, sample_index=sample_index, sample_id=sample_id, label=label)


def _is_legal_action(
    action: ExpandedAction,
    *,
    state: EvalState,
    clean_total: float,
    budget: float,
    max_local_rate_peak: int,
    used: set[tuple],
) -> tuple[bool, str]:
    counts = np.asarray(action.counts, dtype=np.int32)
    if counts.shape != np.asarray(state.dummy_counts, dtype=np.int32).shape:
        return False, "shape_mismatch"
    if np.any(counts < 0):
        return False, "negative_count"
    dummy = int(counts.sum())
    if dummy <= 0:
        return False, "empty_action"
    max_dummy = int(np.floor(float(clean_total) * float(budget) + 1e-9))
    current = np.asarray(state.dummy_counts, dtype=np.int32)
    if int(current.sum()) + dummy > max_dummy:
        return False, "bandwidth_exhausted"
    if int(np.max(current + counts)) > int(max_local_rate_peak):
        return False, "local_rate_exceeded"
    key = tuple(action_identity(action))
    if key in used:
        return False, "duplicate_action"
    return True, ""


def _accept_action(
    *,
    state: EvalState,
    action: ExpandedAction,
    attacker,
    device,
    args: argparse.Namespace,
) -> EvalState:
    selected_counts = np.asarray(state.dummy_counts, dtype=np.int32) + np.asarray(action.counts, dtype=np.int32)
    trace, tam, stats = _render_dummy(base_trace=state.trace, counts=selected_counts, args=args)
    prob = _predict_one(attacker, tam, device=device, args=args)
    return EvalState(
        trace=trace,
        tam=tam,
        prob=prob,
        dummy_counts=np.asarray(selected_counts, dtype=np.int32),
        dummy_bandwidth=float(stats["raw_bandwidth"]),
        avg_delay=float(state.avg_delay),
        p95_delay=float(state.p95_delay),
        max_delay=int(state.max_delay),
        delay_values=tuple(state.delay_values),
        outgoing_delay_values=tuple(state.outgoing_delay_values),
        incoming_delay_values=tuple(state.incoming_delay_values),
        selected_actions=list(state.selected_actions) + [action],
    )


def _safe_percentile(values: list[float], q: float) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, q)) if arr.size else 0.0


def _load_teacher_summaries(paths: str) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for item in _parse_csv_strings(paths):
        path = Path(item)
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows[int(float(row.get("sample_index", -1)))] = row
                except Exception:
                    continue
    return rows


def _resource_probe() -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import psutil  # type: ignore

        proc = psutil.Process()
        result["process_rss_mb"] = float(proc.memory_info().rss / 1024 / 1024)
    except Exception:
        result["process_rss_mb"] = ""
    if torch.cuda.is_available():
        result["cuda_max_allocated_mb"] = float(torch.cuda.max_memory_allocated() / 1024 / 1024)
        result["cuda_max_reserved_mb"] = float(torch.cuda.max_memory_reserved() / 1024 / 1024)
    else:
        result["cuda_max_allocated_mb"] = 0.0
        result["cuda_max_reserved_mb"] = 0.0
    return result


def _prototype_policy_path(output_dir: Path, archive_index: int, args: argparse.Namespace) -> Path:
    name = f"prototype_archive_{int(archive_index):06d}.json"
    local = output_dir / "prototype_policies" / name
    if local.exists() or not str(getattr(args, "prototype_policy_dir", "")).strip():
        return local
    source = Path(str(args.prototype_policy_dir)) / name
    return source if source.exists() else local


class PrototypePolicyRecorder:
    def __init__(self, *, topk: int) -> None:
        self.topk = int(topk)
        self.events: list[dict[str, Any]] = []
        self.action_records = 0
        self.stop_records = 0
        self.candidate_total = 0
        self.candidate_positive = 0
        self.selected_gains: list[float] = []
        self.best_gains: list[float] = []

    def __call__(self, payload: dict[str, Any]) -> None:
        actions = list(payload.get("candidate_actions", []))
        gains = np.asarray(payload.get("candidate_gains", np.zeros(len(actions), dtype=np.float32)), dtype=np.float32)
        scores = np.asarray(payload.get("candidate_scores", np.zeros(len(actions), dtype=np.float32)), dtype=np.float32)
        selected_index = int(payload.get("selected_index", -1))
        mask = np.asarray(payload.get("mask"), dtype=np.float32)
        ranking = np.argsort(-gains, kind="mergesort")[: min(self.topk, len(actions))]
        top_candidates = [
            _action_to_dict(
                actions[int(idx)],
                mask=mask,
                gain=float(gains[int(idx)]) if int(idx) < len(gains) else 0.0,
                score=float(scores[int(idx)]) if int(idx) < len(scores) else 0.0,
                rank=int(pos),
                selected=bool(int(idx) == selected_index),
                top_relative_keypoints=max(1, self.topk * 16),
            )
            for pos, idx in enumerate(ranking.tolist())
        ]
        selected_action = None
        if 0 <= selected_index < len(actions):
            selected_action = _action_to_dict(
                actions[selected_index],
                mask=mask,
                gain=float(payload.get("selected_gain", gains[selected_index] if selected_index < len(gains) else 0.0)),
                score=float(scores[selected_index]) if selected_index < len(scores) else 0.0,
                rank=int(selected_index),
                selected=True,
                top_relative_keypoints=max(1, self.topk * 16),
            )
        event_type = str(payload.get("event_type", ""))
        if event_type == "action":
            self.action_records += 1
        if event_type == "stop":
            self.stop_records += 1
        self.candidate_total += int(len(actions))
        self.candidate_positive += int(np.count_nonzero(gains > 0.0))
        if gains.size:
            self.best_gains.append(float(np.max(gains)))
        if selected_action is not None:
            self.selected_gains.append(float(selected_action["gain"]))
        self.events.append(
            {
                "event_type": event_type,
                "stop_reason": str(payload.get("stop_reason", "")),
                "round_index": int(payload.get("round_index", -1)),
                "step_index": int(payload.get("step_index", -1)),
                "selected_kind": str(payload.get("selected_kind", "")),
                "selected_gain": float(payload.get("selected_gain", 0.0)),
                "clean_total": float(payload.get("clean_total", 1.0)),
                "remaining_dummy": int(payload.get("remaining_dummy", 0)),
                "selected_action": selected_action,
                "top_candidates": top_candidates,
            }
        )


def _run_prototype_teacher(
    *,
    info: SampleInfo,
    tam: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    raw_trace: np.ndarray,
    config: MethodConfig,
    attacker,
    device,
    run_args: argparse.Namespace,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    policy_path = _prototype_policy_path(output_dir, int(info.archive_index), args)
    if policy_path.exists():
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
        return policy_path, dict(payload.get("summary", {}))
    recorder = PrototypePolicyRecorder(topk=4)
    timing: dict[str, float] = {}
    setattr(run_args, "timing_accumulator", timing)
    start = time.perf_counter()
    state, aggregate, _funnel = _run_controller(
        config=config,
        protocol=str(args.protocol),
        budget=float(args.budget),
        raw_trace=np.asarray(raw_trace, dtype=np.float32),
        original_tam=np.asarray(tam, dtype=np.float32),
        original_mask=np.asarray(mask, dtype=np.float32),
        original_prob=np.asarray(prob, dtype=np.float32),
        label=int(info.label),
        sample_index=int(info.archive_index),
        sample_id=str(info.sample_id),
        attacker=attacker,
        device=device,
        args=run_args,
        teacher_observer=recorder,
    )
    runtime_sec = float(time.perf_counter() - start)
    clean_total = max(float(np.asarray(tam, dtype=np.float32).sum()), 1.0)
    sample_row = _teacher_sample_row(
        sample_index=int(info.archive_index),
        sample_id=str(info.sample_id),
        protocol=str(args.protocol),
        config=config,
        budget=float(args.budget),
        margin_target=float(args.margin_target),
        original_prob=np.asarray(prob, dtype=np.float32),
        state=state,
        label=int(info.label),
        clean_total=clean_total,
        runtime=runtime_sec,
        aggregate=aggregate,
    )
    summary = {
        "prototype_archive_index": int(info.archive_index),
        "prototype_source_index": int(info.source_index),
        "prototype_sample_id": str(info.sample_id),
        "true_label": int(info.label),
        "runtime_sec": float(runtime_sec),
        "action_records": int(recorder.action_records),
        "stop_records": int(recorder.stop_records),
        "candidate_total_count": int(recorder.candidate_total),
        "candidate_positive_count": int(recorder.candidate_positive),
        "mean_selected_gain": float(np.mean(recorder.selected_gains)) if recorder.selected_gains else 0.0,
        "mean_best_gain": float(np.mean(recorder.best_gains)) if recorder.best_gains else 0.0,
        **sample_row,
        **{f"timing_{key}": float(value) for key, value in timing.items()},
    }
    policy = {
        "summary": _jsonable(summary),
        "events": _jsonable(recorder.events),
        "config": {
            "method": str(config.name),
            "protocol": str(args.protocol),
            "budget": float(args.budget),
            "rounds": int(args.rounds),
            "max_delay": int(args.max_delay),
            "max_dummy_steps": int(args.max_dummy_steps),
        },
    }
    policy_path.write_text(json.dumps(policy, indent=2), encoding="utf-8")
    return policy_path, summary


def _events_by_round(events: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if str(event.get("event_type", "")) == "action":
            by_round[int(event.get("round_index", 0))].append(event)
    for items in by_round.values():
        items.sort(key=lambda item: int(item.get("step_index", 0)))
    return by_round


def _try_replay_candidate_queue(
    *,
    mode: str,
    event: dict[str, Any],
    current: EvalState,
    current_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    info: SampleInfo,
    clean_total: float,
    used: set[tuple],
    attacker,
    device,
    run_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[EvalState, dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    selected = event.get("selected_action")
    if isinstance(selected, dict):
        candidates.append(selected)
    for item in event.get("top_candidates", []):
        if not isinstance(item, dict):
            continue
        if selected is not None and item.get("counts") == selected.get("counts"):
            continue
        candidates.append(item)
    attempted = 0
    invalid = Counter()
    for payload in candidates:
        attempted += 1
        action = _mapped_action(
            payload,
            mode=mode,
            member_mask=current_mask,
            sample_index=int(info.archive_index),
            sample_id=str(info.sample_id),
            label=int(label),
            top_relative_keypoints=int(args.top_relative_keypoints),
        )
        if action is None:
            invalid["mapping_failed"] += 1
            continue
        legal, reason = _is_legal_action(
            action,
            state=current,
            clean_total=clean_total,
            budget=float(args.budget),
            max_local_rate_peak=int(args.max_local_rate_peak),
            used=used,
        )
        if not legal:
            invalid[reason] += 1
            continue
        next_state = _accept_action(state=current, action=action, attacker=attacker, device=device, args=run_args)
        used.add(tuple(action_identity(action)))
        margin_before = _margin(current.prob, original_prob)
        margin_after = _margin(next_state.prob, original_prob)
        return next_state, {
            "executed": 1,
            "attempted": int(attempted),
            "invalid": int(sum(invalid.values())),
            "invalid_reasons": dict(invalid),
            "candidate_rf_eval_count": 0,
            "state_rf_eval_count": 1,
            "gain": float(margin_before - margin_after),
            "prototype_gain": float(payload.get("gain", 0.0)),
            "action_type": str(action.action_type),
        }
    return current, {
        "executed": 0,
        "attempted": int(attempted),
        "invalid": int(sum(invalid.values())),
        "invalid_reasons": dict(invalid),
        "candidate_rf_eval_count": 0,
        "state_rf_eval_count": 0,
        "gain": 0.0,
        "prototype_gain": 0.0,
        "action_type": "",
    }


def _try_top4_verify(
    *,
    event: dict[str, Any],
    current: EvalState,
    current_mask: np.ndarray,
    original_prob: np.ndarray,
    label: int,
    info: SampleInfo,
    clean_total: float,
    used: set[tuple],
    attacker,
    device,
    run_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[EvalState, dict[str, Any]]:
    invalid = Counter()
    actions: list[ExpandedAction] = []
    payloads = [item for item in event.get("top_candidates", []) if isinstance(item, dict)]
    for payload in payloads[:4]:
        action = _mapped_action(
            payload,
            mode="relative_replay",
            member_mask=current_mask,
            sample_index=int(info.archive_index),
            sample_id=str(info.sample_id),
            label=int(label),
            top_relative_keypoints=int(args.top_relative_keypoints),
        )
        if action is None:
            invalid["mapping_failed"] += 1
            continue
        legal, reason = _is_legal_action(
            action,
            state=current,
            clean_total=clean_total,
            budget=float(args.budget),
            max_local_rate_peak=int(args.max_local_rate_peak),
            used=used,
        )
        if not legal:
            invalid[reason] += 1
            continue
        actions.append(action)
    if not actions:
        return current, {
            "executed": 0,
            "attempted": int(len(payloads[:4])),
            "invalid": int(sum(invalid.values())),
            "invalid_reasons": dict(invalid),
            "candidate_rf_eval_count": 0,
            "state_rf_eval_count": 0,
            "gain": 0.0,
            "prototype_gain": 0.0,
            "action_type": "",
            "no_positive": 1,
        }
    gains, _probs, _metrics = _evaluate_actions(
        state=current,
        actions=actions,
        original_prob=np.asarray(original_prob, dtype=np.float32),
        label=int(label),
        attacker=attacker,
        device=device,
        args=run_args,
    )
    best_idx = int(np.argmax(gains)) if len(gains) else -1
    best_gain = float(gains[best_idx]) if best_idx >= 0 else 0.0
    if best_idx < 0 or best_gain <= 0.0:
        return current, {
            "executed": 0,
            "attempted": int(len(payloads[:4])),
            "invalid": int(sum(invalid.values())),
            "invalid_reasons": dict(invalid),
            "candidate_rf_eval_count": int(len(actions)),
            "state_rf_eval_count": 0,
            "gain": float(best_gain),
            "prototype_gain": 0.0,
            "action_type": "",
            "no_positive": 1,
        }
    action = actions[best_idx]
    next_state = _accept_action(state=current, action=action, attacker=attacker, device=device, args=run_args)
    used.add(tuple(action_identity(action)))
    return next_state, {
        "executed": 1,
        "attempted": int(len(payloads[:4])),
        "invalid": int(sum(invalid.values())),
        "invalid_reasons": dict(invalid),
        "candidate_rf_eval_count": int(len(actions)),
        "state_rf_eval_count": 1,
        "gain": float(best_gain),
        "prototype_gain": float(event.get("selected_gain", 0.0)),
        "action_type": str(action.action_type),
        "no_positive": 0,
    }


def _try_budget_normalized_absolute(
    *,
    event: dict[str, Any],
    current: EvalState,
    original_prob: np.ndarray,
    label: int,
    info: SampleInfo,
    clean_total: float,
    used: set[tuple],
    prototype_cumulative_dummy: int,
    member_previous_target_dummy: int,
    attacker,
    device,
    run_args: argparse.Namespace,
    args: argparse.Namespace,
) -> tuple[EvalState, dict[str, Any]]:
    selected = event.get("selected_action")
    if not isinstance(selected, dict):
        return current, {
            "executed": 0,
            "attempted": 0,
            "invalid": 0,
            "invalid_reasons": {},
            "candidate_rf_eval_count": 0,
            "state_rf_eval_count": 0,
            "gain": 0.0,
            "prototype_gain": 0.0,
            "action_type": "",
            "zero_dose_skip": 1,
            "budget_clipped_action": 0,
            "fully_scaled_action": 0,
            "prototype_action_dummy": 0,
            "scaled_action_dummy": 0,
            "member_next_target_dummy": int(member_previous_target_dummy),
            "prototype_next_cumulative_dummy": int(prototype_cumulative_dummy),
        }
    prototype_action_dummy = int(np.asarray(_counts_from_sparse(selected["counts"]), dtype=np.int32).sum())
    prototype_next_cumulative = int(prototype_cumulative_dummy + prototype_action_dummy)
    prototype_clean_total = max(float(event.get("clean_total", clean_total)), 1.0)
    prototype_budget = max(1, int(np.floor(float(args.budget) * prototype_clean_total + 1e-9)))
    member_budget = max(0, int(np.floor(float(args.budget) * float(clean_total) + 1e-9)))
    ratio = min(float(prototype_next_cumulative) / float(prototype_budget), 1.0)
    member_next_target = int(round(float(ratio) * float(member_budget)))
    desired_dummy = max(0, int(member_next_target - int(member_previous_target_dummy)))
    remaining_dummy = max(0, int(member_budget - int(np.asarray(current.dummy_counts, dtype=np.int32).sum())))
    scaled_dummy = min(int(desired_dummy), int(remaining_dummy))
    budget_clipped = int(scaled_dummy < desired_dummy)
    if scaled_dummy <= 0:
        return current, {
            "executed": 0,
            "attempted": 1,
            "invalid": 0,
            "invalid_reasons": {},
            "candidate_rf_eval_count": 0,
            "state_rf_eval_count": 0,
            "gain": 0.0,
            "prototype_gain": float(selected.get("gain", 0.0)),
            "action_type": "",
            "zero_dose_skip": 1,
            "budget_clipped_action": int(budget_clipped),
            "fully_scaled_action": 0,
            "prototype_action_dummy": int(prototype_action_dummy),
            "scaled_action_dummy": int(scaled_dummy),
            "member_next_target_dummy": int(member_next_target),
            "prototype_next_cumulative_dummy": int(prototype_next_cumulative),
        }
    action = _budget_normalized_absolute_action(
        selected,
        target_dummy=int(scaled_dummy),
        sample_index=int(info.archive_index),
        sample_id=str(info.sample_id),
        label=int(label),
    )
    if action is None:
        return current, {
            "executed": 0,
            "attempted": 1,
            "invalid": 0,
            "invalid_reasons": {},
            "candidate_rf_eval_count": 0,
            "state_rf_eval_count": 0,
            "gain": 0.0,
            "prototype_gain": float(selected.get("gain", 0.0)),
            "action_type": "",
            "zero_dose_skip": 1,
            "budget_clipped_action": int(budget_clipped),
            "fully_scaled_action": 0,
            "prototype_action_dummy": int(prototype_action_dummy),
            "scaled_action_dummy": int(scaled_dummy),
            "member_next_target_dummy": int(member_next_target),
            "prototype_next_cumulative_dummy": int(prototype_next_cumulative),
        }
    legal, reason = _is_legal_action(
        action,
        state=current,
        clean_total=clean_total,
        budget=float(args.budget),
        max_local_rate_peak=int(args.max_local_rate_peak),
        used=used,
    )
    if not legal:
        return current, {
            "executed": 0,
            "attempted": 1,
            "invalid": 1,
            "invalid_reasons": {str(reason): 1},
            "candidate_rf_eval_count": 0,
            "state_rf_eval_count": 0,
            "gain": 0.0,
            "prototype_gain": float(selected.get("gain", 0.0)),
            "action_type": "",
            "zero_dose_skip": 0,
            "budget_clipped_action": int(budget_clipped),
            "fully_scaled_action": 0,
            "prototype_action_dummy": int(prototype_action_dummy),
            "scaled_action_dummy": int(scaled_dummy),
            "member_next_target_dummy": int(member_next_target),
            "prototype_next_cumulative_dummy": int(prototype_next_cumulative),
        }
    next_state = _accept_action(state=current, action=action, attacker=attacker, device=device, args=run_args)
    used.add(tuple(action_identity(action)))
    margin_before = _margin(current.prob, original_prob)
    margin_after = _margin(next_state.prob, original_prob)
    return next_state, {
        "executed": 1,
        "attempted": 1,
        "invalid": 0,
        "invalid_reasons": {},
        "candidate_rf_eval_count": 0,
        "state_rf_eval_count": 1,
        "gain": float(margin_before - margin_after),
        "prototype_gain": float(selected.get("gain", 0.0)),
        "action_type": str(action.action_type),
        "zero_dose_skip": 0,
        "budget_clipped_action": int(budget_clipped),
        "fully_scaled_action": int(not budget_clipped),
        "prototype_action_dummy": int(prototype_action_dummy),
        "scaled_action_dummy": int(scaled_dummy),
        "member_next_target_dummy": int(member_next_target),
        "prototype_next_cumulative_dummy": int(prototype_next_cumulative),
    }


def _transfer_policy(
    *,
    mode: str,
    policy_path: Path,
    info: SampleInfo,
    cluster: ClusterSpec,
    member_distance: float,
    raw_trace: np.ndarray,
    tam: np.ndarray,
    mask: np.ndarray,
    prob: np.ndarray,
    attacker,
    device,
    run_args: argparse.Namespace,
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy = json.loads(Path(policy_path).read_text(encoding="utf-8"))
    events = list(policy.get("events", []))
    by_round = _events_by_round(events)
    clean_total = max(float(np.asarray(tam, dtype=np.float32).sum()), 1.0)
    current = _initial_state(np.asarray(raw_trace, dtype=np.float32), np.asarray(tam, dtype=np.float32), np.asarray(prob, dtype=np.float32))
    used: set[tuple] = set()
    runtime_start = time.perf_counter()
    attempted = 0
    invalid = 0
    executed = 0
    candidate_rf = 0
    state_rf = 0
    no_positive = 0
    zero_dose_skip = 0
    budget_clipped_action = 0
    fully_scaled_action = 0
    prototype_action_dummy_sum = 0
    scaled_action_dummy_sum = 0
    gain_values: list[float] = []
    prototype_gain_values: list[float] = []
    accepted_types: list[str] = []
    invalid_reasons: Counter[str] = Counter()
    stop_reason = "prototype_sequence_exhausted"
    prototype_cumulative_dummy = 0
    member_previous_target_dummy = 0
    for round_index in range(max(1, int(args.rounds))):
        if int(args.max_delay) > 0:
            mask0 = _fast_refresh_mask(attacker, current.tam, np.asarray(prob, dtype=np.float32), device=device)
            current = _apply_delay(
                state=current,
                mask=mask0,
                protocol=str(args.protocol),
                delay_budget=max(1, int(round(int(args.max_delay) / max(1, int(args.rounds))))),
                args=run_args,
            )
            current.prob = _predict_one(attacker, current.tam, device=device, args=run_args)
            state_rf += 1
        if _margin(current.prob, prob) <= float(args.margin_target):
            stop_reason = "target_reached"
            break
        current_mask = _fast_refresh_mask(attacker, current.tam, np.asarray(prob, dtype=np.float32), device=device)
        round_events = by_round.get(int(round_index), [])
        if not round_events:
            continue
        for event in round_events:
            if int(np.asarray(current.dummy_counts, dtype=np.int32).sum()) >= int(math.floor(clean_total * float(args.budget) + 1e-9)):
                stop_reason = "bandwidth_10pct_reached"
                break
            if str(mode) == "absolute_budget_normalized_replay":
                next_state, stats = _try_budget_normalized_absolute(
                    event=event,
                    current=current,
                    original_prob=np.asarray(prob, dtype=np.float32),
                    label=int(info.label),
                    info=info,
                    clean_total=clean_total,
                    used=used,
                    prototype_cumulative_dummy=int(prototype_cumulative_dummy),
                    member_previous_target_dummy=int(member_previous_target_dummy),
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                )
                prototype_cumulative_dummy = int(stats.get("prototype_next_cumulative_dummy", prototype_cumulative_dummy))
                member_previous_target_dummy = int(stats.get("member_next_target_dummy", member_previous_target_dummy))
            elif str(mode) == "relative_top4_verify":
                next_state, stats = _try_top4_verify(
                    event=event,
                    current=current,
                    current_mask=current_mask,
                    original_prob=np.asarray(prob, dtype=np.float32),
                    label=int(info.label),
                    info=info,
                    clean_total=clean_total,
                    used=used,
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                )
            else:
                next_state, stats = _try_replay_candidate_queue(
                    mode=str(mode),
                    event=event,
                    current=current,
                    current_mask=current_mask,
                    original_prob=np.asarray(prob, dtype=np.float32),
                    label=int(info.label),
                    info=info,
                    clean_total=clean_total,
                    used=used,
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                )
            attempted += int(stats["attempted"])
            invalid += int(stats["invalid"])
            invalid_reasons.update({str(k): int(v) for k, v in dict(stats.get("invalid_reasons", {})).items()})
            candidate_rf += int(stats["candidate_rf_eval_count"])
            state_rf += int(stats["state_rf_eval_count"])
            zero_dose_skip += int(stats.get("zero_dose_skip", 0))
            budget_clipped_action += int(stats.get("budget_clipped_action", 0))
            fully_scaled_action += int(stats.get("fully_scaled_action", 0))
            prototype_action_dummy_sum += int(stats.get("prototype_action_dummy", 0))
            scaled_action_dummy_sum += int(stats.get("scaled_action_dummy", 0))
            if int(stats["executed"]) <= 0:
                no_positive += int(stats.get("no_positive", 0))
                if str(mode) == "absolute_budget_normalized_replay" and int(stats.get("zero_dose_skip", 0)) > 0 and int(stats.get("invalid", 0)) <= 0:
                    stop_reason = "prototype_sequence_exhausted"
                    continue
                stop_reason = "no_positive_transfer" if str(mode) == "relative_top4_verify" else "invalid_transfer_action"
                break
            current = next_state
            executed += 1
            gain_values.append(float(stats.get("gain", 0.0)))
            prototype_gain_values.append(float(stats.get("prototype_gain", 0.0)))
            if str(stats.get("action_type", "")):
                accepted_types.append(str(stats["action_type"]))
            if _margin(current.prob, prob) <= float(args.margin_target):
                stop_reason = "target_reached"
                break
            current_mask = _fast_refresh_mask(attacker, current.tam, np.asarray(prob, dtype=np.float32), device=device)
        if stop_reason in {"target_reached", "bandwidth_10pct_reached", "invalid_transfer_action", "no_positive_transfer"}:
            break
    runtime = float(time.perf_counter() - runtime_start)
    aggregate = {
        "stop_reason": str(stop_reason),
        "accepted_single_count": int(executed),
        "accepted_pair_count": 0,
        "candidate_step_count": int(max(executed, attempted)),
        "rf_eval_count": int(candidate_rf),
        "best_single_gain_seen": float(max(gain_values) if gain_values else 0.0),
        "best_pair_gain_seen": 0.0,
        "proxy_best_gain_recall_values": [],
        "true_best_gain_recall_values": [],
    }
    metrics = probability_metrics(
        np.asarray(prob, dtype=np.float32).reshape(1, -1),
        np.asarray(current.prob, dtype=np.float32).reshape(1, -1),
        np.asarray([int(info.label)], dtype=np.int64),
    )
    resource = _resource_fields(current, clean_total)
    return {
        "representation": str(cluster.representation),
        "num_clusters": int(cluster.num_clusters),
        "cluster_id": int(cluster.cluster_id),
        "cluster_size": int(cluster.cluster_size),
        "mode": str(mode),
        "split": str(info.split),
        "sample_index": int(info.archive_index),
        "sample_id": str(info.sample_id),
        "source_index": int(info.source_index),
        "true_label": int(info.label),
        "prototype_sample_index": int(cluster.prototype_archive_index),
        "prototype_source_index": int(cluster.prototype_source_index),
        "prototype_sample_id": str(cluster.prototype_sample_id),
        "member_distance_to_center": float(member_distance),
        "prototype_distance_to_center": float(cluster.prototype_distance),
        "distance_ratio_to_prototype": float(member_distance / max(float(cluster.prototype_distance), 1e-8)),
        "clean_accuracy": float(int(int(np.argmax(prob)) == int(info.label))),
        "accuracy": float(metrics["accuracy"][0]),
        "flip": float(metrics["flip"][0]),
        "target_margin_success": int(float(metrics["original_class_margin"][0]) <= float(args.margin_target)),
        "original_pred": int(metrics["original_pred"][0]),
        "final_pred": int(metrics["evaluated_pred"][0]),
        "original_class_probability": float(metrics["original_class_probability"][0]),
        "original_class_margin": float(metrics["original_class_margin"][0]),
        "original_class_margin_drop": float(metrics["original_class_margin_drop"][0]),
        "normalized_entropy_gain": float(metrics["normalized_entropy_gain"][0]),
        "js_div": float(metrics["js_div"][0]),
        "attempted_transfer_actions": int(attempted),
        "invalid_transfer_actions": int(invalid),
        "invalid_transfer_action_rate": float(invalid / max(attempted, 1)),
        "invalid_transfer_reasons": json.dumps(dict(sorted(invalid_reasons.items())), sort_keys=True),
        "executed_actions": int(executed),
        "no_positive_verify_steps": int(no_positive),
        "candidate_rf_eval_count": int(candidate_rf),
        "state_rf_eval_count": int(state_rf),
        "total_rf_eval_count": int(candidate_rf + state_rf),
        "mean_exact_gain": float(np.mean(gain_values)) if gain_values else 0.0,
        "mean_prototype_gain": float(np.mean(prototype_gain_values)) if prototype_gain_values else 0.0,
        "accepted_action_type_distribution": json.dumps(dict(sorted(Counter(accepted_types).items())), sort_keys=True),
        "zero_dose_skip": int(zero_dose_skip),
        "budget_clipped_action": int(budget_clipped_action),
        "fully_scaled_action": int(fully_scaled_action),
        "prototype_action_dummy_sum": int(prototype_action_dummy_sum),
        "scaled_action_dummy_sum": int(scaled_action_dummy_sum),
        "stop_reason": str(stop_reason),
        "runtime_sec": float(runtime),
        **resource,
        **aggregate,
    }


def _summarize_transfer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["representation"]), int(row["num_clusters"]), str(row["mode"]), str(row["split"]))].append(row)
    summary: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        rep, k, mode, split = key
        n = len(items)
        runtime = [float(row.get("runtime_sec", 0.0)) for row in items]
        summary.append(
            {
                "representation": rep,
                "num_clusters": int(k),
                "mode": mode,
                "split": split,
                "samples": int(n),
                "clean_rf_accuracy": float(np.mean([float(row["clean_accuracy"]) for row in items])) if items else 0.0,
                "defended_rf_accuracy": float(np.mean([float(row["accuracy"]) for row in items])) if items else 0.0,
                "flip_rate": float(np.mean([float(row["flip"]) for row in items])) if items else 0.0,
                "target_reached_rate": float(np.mean([float(row["target_margin_success"]) for row in items])) if items else 0.0,
                "mean_actual_bw": float(np.mean([float(row["actual_dummy_bandwidth"]) for row in items])) if items else 0.0,
                "p95_actual_bw": _safe_percentile([float(row["actual_dummy_bandwidth"]) for row in items], 95),
                "max_actual_bw": float(max([float(row["actual_dummy_bandwidth"]) for row in items], default=0.0)),
                "mean_average_delay_bins": float(np.mean([float(row["average_delay_bins"]) for row in items])) if items else 0.0,
                "p95_delay_bins": _safe_percentile([float(row["p95_delay_bins"]) for row in items], 95),
                "mean_executed_actions": float(np.mean([float(row["executed_actions"]) for row in items])) if items else 0.0,
                "mean_candidate_rf_eval_count": float(np.mean([float(row["candidate_rf_eval_count"]) for row in items])) if items else 0.0,
                "mean_total_rf_eval_count": float(np.mean([float(row["total_rf_eval_count"]) for row in items])) if items else 0.0,
                "invalid_transfer_action_rate": float(np.mean([float(row["invalid_transfer_action_rate"]) for row in items])) if items else 0.0,
                "invalid_sample_rate": float(np.mean([str(row.get("stop_reason", "")) == "invalid_transfer_action" for row in items])) if items else 0.0,
                "mean_zero_dose_skip": float(np.mean([float(row.get("zero_dose_skip", 0.0) or 0.0) for row in items])) if items else 0.0,
                "mean_budget_clipped_action": float(np.mean([float(row.get("budget_clipped_action", 0.0) or 0.0) for row in items])) if items else 0.0,
                "mean_fully_scaled_action": float(np.mean([float(row.get("fully_scaled_action", 0.0) or 0.0) for row in items])) if items else 0.0,
                "mean_scaled_action_dummy_sum": float(np.mean([float(row.get("scaled_action_dummy_sum", 0.0) or 0.0) for row in items])) if items else 0.0,
                "total_wall_time_sec": float(sum(runtime)),
                "mean_seconds_per_trace": float(np.mean(runtime)) if runtime else 0.0,
                "median_seconds_per_trace": float(np.median(runtime)) if runtime else 0.0,
                "p95_seconds_per_trace": _safe_percentile(runtime, 95),
                "traces_per_hour": float(3600.0 * n / max(sum(runtime), 1e-8)),
                "stop_reason_distribution": json.dumps(dict(sorted(Counter(str(row["stop_reason"]) for row in items).items())), sort_keys=True),
            }
        )
    return summary


def _invalid_reason_total(items: list[dict[str, Any]], reason: str) -> int:
    total = 0
    for row in items:
        value = row.get("invalid_transfer_reasons", "")
        if not value:
            continue
        try:
            payload = json.loads(str(value))
        except Exception:
            continue
        total += int(payload.get(reason, 0))
    return int(total)


def _length_bucket(clean_packet_count: float) -> str:
    value = float(clean_packet_count)
    if value < 500:
        return "[0,500)"
    if value < 1000:
        return "[500,1000)"
    if value < 2000:
        return "[1000,2000)"
    return "[2000,+inf)"


def _summarize_by_length_bucket(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        bucket = _length_bucket(float(row.get("clean_packet_count", 0.0) or 0.0))
        groups[(str(row["representation"]), int(row["num_clusters"]), str(row["mode"]), str(row["split"]), bucket)].append(row)
    out: list[dict[str, Any]] = []
    order = {"[0,500)": 0, "[500,1000)": 1, "[1000,2000)": 2, "[2000,+inf)": 3}
    for key, items in sorted(groups.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], item[0][3], order.get(item[0][4], 99))):
        rep, k, mode, split, bucket = key
        out.append(
            {
                "representation": rep,
                "num_clusters": int(k),
                "mode": mode,
                "split": split,
                "length_bucket": bucket,
                "samples": int(len(items)),
                "clean_rf_accuracy": float(np.mean([float(row["clean_accuracy"]) for row in items])) if items else 0.0,
                "defended_rf_accuracy": float(np.mean([float(row["accuracy"]) for row in items])) if items else 0.0,
                "flip_rate": float(np.mean([float(row["flip"]) for row in items])) if items else 0.0,
                "target_reached_rate": float(np.mean([float(row["target_margin_success"]) for row in items])) if items else 0.0,
                "invalid_sample_rate": float(np.mean([str(row.get("stop_reason", "")) == "invalid_transfer_action" for row in items])) if items else 0.0,
                "bandwidth_exhausted_attempts": _invalid_reason_total(items, "bandwidth_exhausted"),
                "duplicate_action_attempts": _invalid_reason_total(items, "duplicate_action"),
                "local_rate_exceeded_attempts": _invalid_reason_total(items, "local_rate_exceeded"),
                "mean_executed_actions": float(np.mean([float(row["executed_actions"]) for row in items])) if items else 0.0,
                "mean_actual_bw": float(np.mean([float(row["actual_dummy_bandwidth"]) for row in items])) if items else 0.0,
                "p95_actual_bw": _safe_percentile([float(row["actual_dummy_bandwidth"]) for row in items], 95),
                "mean_zero_dose_skip": float(np.mean([float(row.get("zero_dose_skip", 0.0) or 0.0) for row in items])) if items else 0.0,
                "mean_budget_clipped_action": float(np.mean([float(row.get("budget_clipped_action", 0.0) or 0.0) for row in items])) if items else 0.0,
                "mean_fully_scaled_action": float(np.mean([float(row.get("fully_scaled_action", 0.0) or 0.0) for row in items])) if items else 0.0,
                "stop_reason_distribution": json.dumps(dict(sorted(Counter(str(row["stop_reason"]) for row in items).items())), sort_keys=True),
            }
        )
    return out


def _summarize_prototypes(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["representation"]), int(row["num_clusters"]))].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        rep, k = key
        weights = np.asarray([float(row.get("cluster_size", 1.0) or 1.0) for row in items], dtype=np.float64)
        weights = np.maximum(weights, 1.0)

        def vals(field: str) -> np.ndarray:
            result = []
            for row in items:
                try:
                    result.append(float(row.get(field, 0.0) or 0.0))
                except Exception:
                    result.append(0.0)
            return np.asarray(result, dtype=np.float64)

        clean = np.asarray(
            [
                1.0
                if int(float(row.get("teacher_original_pred", -1) or -1)) == int(float(row.get("teacher_true_label", -2) or -2))
                else 0.0
                for row in items
            ],
            dtype=np.float64,
        )
        runtime = vals("teacher_runtime_sec")
        cluster_sizes = vals("cluster_size")
        stop_counter = Counter(str(row.get("teacher_stop_reason", "")) for row in items)
        weighted_stop: dict[str, float] = defaultdict(float)
        for row, weight in zip(items, weights):
            weighted_stop[str(row.get("teacher_stop_reason", ""))] += float(weight)
        out.append(
            {
                "representation": rep,
                "num_clusters": int(k),
                "prototype_count": int(len(items)),
                "represented_samples": int(np.sum(weights)),
                "cluster_size_min": int(np.min(cluster_sizes)) if cluster_sizes.size else 0,
                "cluster_size_median": float(np.median(cluster_sizes)) if cluster_sizes.size else 0.0,
                "cluster_size_p95": _safe_percentile(cluster_sizes.tolist(), 95),
                "cluster_size_max": int(np.max(cluster_sizes)) if cluster_sizes.size else 0,
                "prototype_clean_accuracy": float(np.mean(clean)) if clean.size else 0.0,
                "weighted_clean_accuracy": float(np.average(clean, weights=weights)) if clean.size else 0.0,
                "prototype_defended_accuracy": float(np.mean(vals("teacher_accuracy"))) if items else 0.0,
                "weighted_defended_accuracy": float(np.average(vals("teacher_accuracy"), weights=weights)) if items else 0.0,
                "prototype_flip_rate": float(np.mean(vals("teacher_flip"))) if items else 0.0,
                "weighted_flip_rate": float(np.average(vals("teacher_flip"), weights=weights)) if items else 0.0,
                "prototype_target_reached_rate": float(np.mean(vals("teacher_target_margin_success"))) if items else 0.0,
                "weighted_target_reached_rate": float(np.average(vals("teacher_target_margin_success"), weights=weights)) if items else 0.0,
                "prototype_mean_actual_bw": float(np.mean(vals("teacher_actual_dummy_bandwidth"))) if items else 0.0,
                "weighted_mean_actual_bw": float(np.average(vals("teacher_actual_dummy_bandwidth"), weights=weights)) if items else 0.0,
                "prototype_p95_actual_bw": _safe_percentile(vals("teacher_actual_dummy_bandwidth").tolist(), 95),
                "prototype_max_actual_bw": float(np.max(vals("teacher_actual_dummy_bandwidth"))) if items else 0.0,
                "prototype_mean_average_delay_bins": float(np.mean(vals("teacher_average_delay_bins"))) if items else 0.0,
                "weighted_mean_average_delay_bins": float(np.average(vals("teacher_average_delay_bins"), weights=weights)) if items else 0.0,
                "prototype_p95_delay_bins": _safe_percentile(vals("teacher_p95_delay_bins").tolist(), 95),
                "prototype_mean_actions": float(np.mean(vals("teacher_accepted_action_count"))) if items else 0.0,
                "weighted_mean_actions": float(np.average(vals("teacher_accepted_action_count"), weights=weights)) if items else 0.0,
                "prototype_mean_rf_eval_count": float(np.mean(vals("teacher_rf_eval_count"))) if items else 0.0,
                "weighted_mean_rf_eval_count": float(np.average(vals("teacher_rf_eval_count"), weights=weights)) if items else 0.0,
                "prototype_teacher_runtime_sec_sum": float(np.sum(runtime)),
                "prototype_teacher_runtime_sec_mean": float(np.mean(runtime)) if runtime.size else 0.0,
                "prototype_teacher_runtime_sec_p95": _safe_percentile(runtime.tolist(), 95),
                "prototype_traces_per_hour": float(3600.0 * len(items) / max(np.sum(runtime), 1e-8)),
                "stop_reason_distribution": json.dumps(dict(sorted(stop_counter.items())), sort_keys=True),
                "weighted_stop_reason_distribution": json.dumps(dict(sorted(weighted_stop.items())), sort_keys=True),
            }
        )
    return out


def _summarize_by_cluster(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["representation"]), int(row["num_clusters"]), str(row["mode"]), int(row["cluster_id"]))].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        rep, k, mode, cluster_id = key
        out.append(
            {
                "representation": rep,
                "num_clusters": int(k),
                "mode": mode,
                "cluster_id": int(cluster_id),
                "samples": int(len(items)),
                "cluster_size": int(items[0]["cluster_size"]) if items else 0,
                "mean_member_distance_to_center": float(np.mean([float(row["member_distance_to_center"]) for row in items])) if items else 0.0,
                "defended_rf_accuracy": float(np.mean([float(row["accuracy"]) for row in items])) if items else 0.0,
                "flip_rate": float(np.mean([float(row["flip"]) for row in items])) if items else 0.0,
                "mean_actual_bw": float(np.mean([float(row["actual_dummy_bandwidth"]) for row in items])) if items else 0.0,
                "invalid_transfer_action_rate": float(np.mean([float(row["invalid_transfer_action_rate"]) for row in items])) if items else 0.0,
                "mean_runtime_sec": float(np.mean([float(row["runtime_sec"]) for row in items])) if items else 0.0,
            }
        )
    return out


def _correlation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["representation"]), int(row["num_clusters"]), str(row["mode"]))].append(row)
    out: list[dict[str, Any]] = []
    for key, items in sorted(groups.items()):
        rep, k, mode = key
        for x_key, y_key in (
            ("member_distance_to_center", "accuracy"),
            ("member_distance_to_center", "flip"),
            ("distance_ratio_to_prototype", "accuracy"),
            ("cluster_size", "accuracy"),
        ):
            x = np.asarray([float(row[x_key]) for row in items], dtype=np.float64)
            y = np.asarray([float(row[y_key]) for row in items], dtype=np.float64)
            corr = 0.0
            if x.size >= 3 and np.std(x) > 1e-12 and np.std(y) > 1e-12:
                corr = float(np.corrcoef(x, y)[0, 1])
            out.append(
                {
                    "representation": rep,
                    "num_clusters": int(k),
                    "mode": mode,
                    "x": x_key,
                    "y": y_key,
                    "samples": int(len(items)),
                    "pearson": corr,
                }
            )
    return out


def _existing_teacher_rows(
    *,
    transfer_infos: list[SampleInfo],
    assignments: np.ndarray,
    specs_by_cluster: dict[int, ClusterSpec],
    features: np.ndarray,
    centers: np.ndarray,
    representation: str,
    k: int,
    teacher_summaries: dict[int, dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for info in transfer_infos:
        source = teacher_summaries.get(int(info.archive_index))
        if source is None:
            continue
        cluster_id = int(assignments[int(info.local_index)])
        cluster = specs_by_cluster.get(cluster_id)
        if cluster is None:
            continue
        member_distance = float(np.sum((features[int(info.local_index)] - centers[cluster_id]) ** 2))
        rows.append(
            {
                "representation": str(representation),
                "num_clusters": int(k),
                "cluster_id": int(cluster_id),
                "cluster_size": int(cluster.cluster_size),
                "mode": "per_sample_teacher",
                "split": str(info.split),
                "sample_index": int(info.archive_index),
                "sample_id": str(info.sample_id),
                "source_index": int(info.source_index),
                "true_label": int(info.label),
                "prototype_sample_index": int(cluster.prototype_archive_index),
                "prototype_source_index": int(cluster.prototype_source_index),
                "prototype_sample_id": str(cluster.prototype_sample_id),
                "member_distance_to_center": float(member_distance),
                "prototype_distance_to_center": float(cluster.prototype_distance),
                "distance_ratio_to_prototype": float(member_distance / max(float(cluster.prototype_distance), 1e-8)),
                "clean_accuracy": float(1.0 if int(float(source.get("original_pred", -1))) == int(info.label) else 0.0),
                "accuracy": float(source.get("accuracy", 0.0) or 0.0),
                "flip": float(source.get("flip", 0.0) or 0.0),
                "target_margin_success": int(float(source.get("target_margin_success", 0.0) or 0.0)),
                "original_pred": int(float(source.get("original_pred", -1) or -1)),
                "final_pred": int(float(source.get("final_pred", -1) or -1)),
                "original_class_probability": float(source.get("original_class_probability", 0.0) or 0.0),
                "original_class_margin": float(source.get("original_class_margin", 0.0) or 0.0),
                "original_class_margin_drop": float(source.get("original_class_margin_drop", 0.0) or 0.0),
                "attempted_transfer_actions": "",
                "invalid_transfer_actions": "",
                "invalid_transfer_action_rate": "",
                "invalid_transfer_reasons": "",
                "executed_actions": int(float(source.get("accepted_action_count", 0) or 0)),
                "no_positive_verify_steps": "",
                "candidate_rf_eval_count": int(float(source.get("rf_eval_count", 0) or 0)),
                "state_rf_eval_count": "",
                "total_rf_eval_count": int(float(source.get("rf_eval_count", 0) or 0)),
                "mean_exact_gain": "",
                "mean_prototype_gain": "",
                "accepted_action_type_distribution": source.get("accepted_action_type_distribution", ""),
                "stop_reason": source.get("stop_reason", ""),
                "runtime_sec": float(source.get("runtime_sec", 0.0) or 0.0),
                "clean_packet_count": float(source.get("clean_packet_count", 0.0) or 0.0),
                "dummy_packet_count": int(float(source.get("dummy_packet_count", 0) or 0)),
                "defended_packet_count": float(source.get("defended_packet_count", 0.0) or 0.0),
                "actual_dummy_bandwidth": float(source.get("actual_dummy_bandwidth", 0.0) or 0.0),
                "dummy_overhead": float(source.get("dummy_overhead", 0.0) or 0.0),
                "total_overhead": float(source.get("total_overhead", 0.0) or 0.0),
                "outgoing_dummy_packet_count": int(float(source.get("outgoing_dummy_packet_count", 0) or 0)),
                "incoming_dummy_packet_count": int(float(source.get("incoming_dummy_packet_count", 0) or 0)),
                "average_delay_bins": float(source.get("average_delay_bins", 0.0) or 0.0),
                "p95_delay_bins": float(source.get("p95_delay_bins", 0.0) or 0.0),
                "maximum_delay_bins": int(float(source.get("maximum_delay_bins", 0) or 0)),
                "delay_packet_count": int(float(source.get("delay_packet_count", 0) or 0)),
                "accepted_action_count": int(float(source.get("accepted_action_count", 0) or 0)),
            }
        )
    return rows


def _write_doc(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    prototype_summary_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> None:
    doc_path = ROOT / "docs" / "stage_b_cluster_prototype_transfer_audit_20260730.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage B Cluster Prototype Transfer Audit",
        "",
        "This audit tests whether a real train-split cluster medoid can run the formal Teacher once and transfer its action trajectory to other traces.",
        "",
        "## Run",
        "",
        f"- Output dir: `{output_dir}`",
        f"- Archive: `{audit.get('archive')}`",
        f"- Split file: `{audit.get('split_file')}`",
        f"- Transfer modes: `{', '.join(audit.get('transfer_modes', []))}`",
        "",
        "## Prototype-Only Teacher Results",
        "",
    ]
    if prototype_summary_rows:
        lines.append("| representation | K | prototypes | represented | weighted clean acc | weighted defended acc | weighted BW | weighted delay | weighted actions | runtime |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in prototype_summary_rows:
            lines.append(
                "| {representation} | {num_clusters} | {prototype_count} | {represented_samples} | {clean:.2f}% | {acc:.2f}% | {bw:.2f}% | {delay:.2f} | {actions:.2f} | {runtime:.1f}s |".format(
                    representation=row["representation"],
                    num_clusters=int(row["num_clusters"]),
                    prototype_count=int(row["prototype_count"]),
                    represented_samples=int(row["represented_samples"]),
                    clean=100.0 * float(row["weighted_clean_accuracy"]),
                    acc=100.0 * float(row["weighted_defended_accuracy"]),
                    bw=100.0 * float(row["weighted_mean_actual_bw"]),
                    delay=float(row["weighted_mean_average_delay_bins"]),
                    actions=float(row["weighted_mean_actions"]),
                    runtime=float(row["prototype_teacher_runtime_sec_sum"]),
                )
            )
    else:
        lines.append("No prototype Teacher rows were generated.")
    lines.extend(
        [
            "",
        "## Aggregate Results",
        "",
        ]
    )
    if summary_rows:
        lines.append("| representation | K | mode | split | n | clean acc | defended acc | mean BW | p95 BW | mean delay | eval/sample | traces/hour |")
        lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in summary_rows:
            lines.append(
                "| {representation} | {num_clusters} | {mode} | {split} | {samples} | {clean:.2f}% | {acc:.2f}% | {bw:.2f}% | {p95:.2f}% | {delay:.2f} | {evals:.2f} | {tph:.1f} |".format(
                    representation=row["representation"],
                    num_clusters=int(row["num_clusters"]),
                    mode=row["mode"],
                    split=row["split"],
                    samples=int(row["samples"]),
                    clean=100.0 * float(row["clean_rf_accuracy"]),
                    acc=100.0 * float(row["defended_rf_accuracy"]),
                    bw=100.0 * float(row["mean_actual_bw"]),
                    p95=100.0 * float(row["p95_actual_bw"]),
                    delay=float(row["mean_average_delay_bins"]),
                    evals=float(row["mean_total_rf_eval_count"]),
                    tph=float(row["traces_per_hour"]),
                )
            )
    else:
        lines.append("No transfer rows were generated.")
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- `absolute_replay` checks whether absolute prototype bins are reusable. A weak result means absolute timing is too instance-specific.",
            "- `relative_replay` checks whether keypoint-relative mapping is enough without per-member RF verification.",
            "- `relative_top4_verify` checks whether a tiny exact verification set can approach the per-sample Teacher while cutting candidate evaluations.",
            "- The distance/effect correlations in `audit.json` indicate whether cluster compactness predicts transfer quality.",
        ]
    )
    doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = _run_dir(args)
    run_args = _runtime_args(args)
    setattr(run_args, "max_delay", int(args.max_delay))
    setattr(run_args, "rounds", int(args.rounds))
    setattr(run_args, "max_dummy_steps", int(args.max_dummy_steps))
    setattr(run_args, "stratified_bucket_k", int(args.stratified_bucket_k))
    setattr(run_args, "stratified_global_k", int(args.stratified_global_k))
    setattr(run_args, "random_explore_k", int(args.random_explore_k))
    setattr(run_args, "true_recall_pool_size", int(args.true_recall_pool_size))
    setattr(run_args, "margin_target", float(args.margin_target))
    device = resolve_device(args.device)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    checkpoint = args.checkpoint or _default_checkpoint(args.attacker)
    infos = _archive_infos(args.archive, args.split_file)
    tam = np.asarray(_load_archive_keys(args.archive, [info.archive_index for info in infos], ["tam"])["tam"], dtype=np.float32)
    with np.load(args.archive, allow_pickle=False) as archive_probe:
        num_classes = int(archive_probe["pred_prob"].shape[1])
    attacker = load_stage_a_attacker(
        checkpoint,
        attacker=str(args.attacker),
        num_classes=int(num_classes),
        device=device,
        df_architecture=str(args.df_architecture),
        df_tam_adapter=str(args.df_tam_adapter),
    )
    config = _method_config(args.method)
    representations = _parse_csv_strings(args.representations)
    ks = _parse_csv_ints(args.num_clusters)
    modes = _parse_csv_strings(args.transfer_modes)
    transfer_splits = set(_parse_csv_strings(args.transfer_splits))
    train_mask = np.asarray([info.split == "train" for info in infos], dtype=bool)
    transfer_infos = [] if bool(args.prototype_only) else [info for info in infos if info.split in transfer_splits]
    if int(args.transfer_max_samples) > 0:
        transfer_infos = transfer_infos[: int(args.transfer_max_samples)]
    teacher_summaries = _load_teacher_summaries(args.teacher_sample_summary_csvs)

    cluster_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    cluster_result_index: dict[tuple[str, int], dict[str, Any]] = {}
    global_start = time.perf_counter()

    for representation in representations:
        feature_start = time.perf_counter()
        features, feature_meta = _features_for_representation(
            tam=tam,
            train_mask=train_mask,
            representation=representation,
            pca_dim=int(args.pca_dim),
            batch_size=int(args.kmeans_batch_size),
            seed=int(args.seed),
        )
        if args.progress:
            print(f"[features] {representation} dim={features.shape[1]} sec={time.perf_counter() - feature_start:.2f}", flush=True)
        for k in ks:
            cluster_start = time.perf_counter()
            assignments, centers = _fit_assignments(
                features=features,
                train_mask=train_mask,
                k=int(k),
                batch_size=int(args.kmeans_batch_size),
                max_iter=int(args.kmeans_max_iter),
                seed=int(args.seed) + int(k),
            )
            specs = _cluster_specs(
                representation=representation,
                k=int(k),
                features=features,
                assignments=assignments,
                centers=centers,
                infos=infos,
            )
            specs_by_cluster = {int(spec.cluster_id): spec for spec in specs}
            cluster_runtime = float(time.perf_counter() - cluster_start)
            cluster_result_index[(str(representation), int(k))] = {
                "assignments": assignments,
                "centers": centers,
                "features": features,
                "specs_by_cluster": specs_by_cluster,
                "feature_meta": feature_meta,
            }
            for spec in specs:
                cluster_rows.append(asdict(spec))
            for info in infos:
                cid = int(assignments[int(info.local_index)])
                center = centers[cid]
                distance = float(np.sum((features[int(info.local_index)] - center) ** 2))
                assignment_rows.append(
                    {
                        "representation": str(representation),
                        "num_clusters": int(k),
                        "local_index": int(info.local_index),
                        "archive_index": int(info.archive_index),
                        "source_index": int(info.source_index),
                        "sample_id": str(info.sample_id),
                        "split": str(info.split),
                        "label": int(info.label),
                        "cluster_id": int(cid),
                        "distance_to_center": float(distance),
                        "is_prototype": int(specs_by_cluster.get(cid) is not None and specs_by_cluster[cid].prototype_local_index == info.local_index),
                    }
                )
            runtime_rows.append(
                {
                    "representation": str(representation),
                    "num_clusters": int(k),
                    "phase": "cluster_fit_assign",
                    "samples": int(len(infos)),
                    "runtime_sec": float(cluster_runtime),
                    "traces_per_hour": float(3600.0 * len(infos) / max(cluster_runtime, 1e-8)),
                    **feature_meta,
                }
            )
            specs_for_teacher = list(specs)
            if int(args.prototype_limit) > 0:
                specs_for_teacher = specs_for_teacher[: int(args.prototype_limit)]
            needed_local_indices = sorted(
                set(int(spec.prototype_local_index) for spec in specs_for_teacher)
                | set(int(info.local_index) for info in transfer_infos)
            )
            needed_archive_rows = [int(infos[idx].archive_index) for idx in needed_local_indices]
            needed_source_indices = np.asarray([int(infos[idx].source_index) for idx in needed_local_indices], dtype=np.int64)
            selected_archive = _load_archive_keys(args.archive, needed_archive_rows, ["mask", "pred_prob"])
            selected_mask = np.asarray(selected_archive["mask"], dtype=np.float32)
            selected_prob = np.asarray(selected_archive["pred_prob"], dtype=np.float32)
            selected_raw = _load_raw_selected(args.data_root, needed_source_indices, run_args)
            selected_pos = {int(local_index): pos for pos, local_index in enumerate(needed_local_indices)}
            for pos, spec in enumerate(specs_for_teacher):
                proto_info = infos[int(spec.prototype_local_index)]
                proto_pos = int(selected_pos[int(proto_info.local_index)])
                if args.progress:
                    print(
                        f"[prototype] {representation} K={k} {pos + 1}/{len(specs_for_teacher)} "
                        f"cluster={spec.cluster_id} archive={proto_info.archive_index}",
                        flush=True,
                    )
                policy_path, proto_summary = _run_prototype_teacher(
                    info=proto_info,
                    tam=tam[int(proto_info.local_index)],
                    mask=selected_mask[proto_pos],
                    prob=selected_prob[proto_pos],
                    raw_trace=selected_raw[proto_pos],
                    config=config,
                    attacker=attacker,
                    device=device,
                    run_args=run_args,
                    args=args,
                    output_dir=output_dir,
                )
                prototype_rows.append(
                    {
                        **asdict(spec),
                        "policy_path": str(policy_path),
                        **{f"teacher_{key}": value for key, value in proto_summary.items() if key not in {"sample_index", "sample_id"}},
                    }
                )
            if int(args.prototype_limit) > 0:
                allowed_clusters = {int(spec.cluster_id) for spec in specs_for_teacher}
            else:
                allowed_clusters = set(int(spec.cluster_id) for spec in specs)
            for mode in modes:
                mode_start = time.perf_counter()
                mode_rows_before = len(transfer_rows)
                for idx, info in enumerate(transfer_infos):
                    cluster_id = int(assignments[int(info.local_index)])
                    if cluster_id not in allowed_clusters:
                        continue
                    spec = specs_by_cluster.get(cluster_id)
                    if spec is None:
                        continue
                    policy_path = _prototype_policy_path(output_dir, int(spec.prototype_archive_index), args)
                    if not policy_path.exists():
                        continue
                    if args.progress and idx % max(1, min(50, len(transfer_infos))) == 0:
                        print(f"[transfer] {representation} K={k} mode={mode} {idx + 1}/{len(transfer_infos)}", flush=True)
                    member_distance = float(np.sum((features[int(info.local_index)] - centers[cluster_id]) ** 2))
                    member_pos = int(selected_pos[int(info.local_index)])
                    row = _transfer_policy(
                        mode=mode,
                        policy_path=policy_path,
                        info=info,
                        cluster=spec,
                        member_distance=member_distance,
                        raw_trace=selected_raw[member_pos],
                        tam=tam[int(info.local_index)],
                        mask=selected_mask[member_pos],
                        prob=selected_prob[member_pos],
                        attacker=attacker,
                        device=device,
                        run_args=run_args,
                        args=args,
                    )
                    transfer_rows.append(row)
                mode_runtime = float(time.perf_counter() - mode_start)
                produced = len(transfer_rows) - mode_rows_before
                runtime_rows.append(
                    {
                        "representation": str(representation),
                        "num_clusters": int(k),
                        "phase": f"transfer_{mode}",
                        "samples": int(produced),
                        "runtime_sec": float(mode_runtime),
                        "traces_per_hour": float(3600.0 * produced / max(mode_runtime, 1e-8)),
                    }
                )
            if bool(args.include_existing_teacher_upper_bound):
                transfer_rows.extend(
                    _existing_teacher_rows(
                        transfer_infos=transfer_infos,
                        assignments=assignments,
                        specs_by_cluster=specs_by_cluster,
                        features=features,
                        centers=centers,
                        representation=representation,
                        k=int(k),
                        teacher_summaries=teacher_summaries,
                    )
                )

    summary_rows = _summarize_transfer(transfer_rows)
    by_cluster_rows = _summarize_by_cluster(transfer_rows)
    by_length_rows = _summarize_by_length_bucket(transfer_rows)
    corr_rows = _correlation_rows([row for row in transfer_rows if str(row.get("mode")) != "per_sample_teacher"])
    _write_csv(output_dir / "cluster_prototypes.csv", cluster_rows)
    _write_csv(output_dir / "cluster_assignments.csv", assignment_rows)
    prototype_summary_rows = _summarize_prototypes(prototype_rows)
    _write_csv(output_dir / "prototype_teacher_summary.csv", prototype_rows)
    _write_csv(output_dir / "prototype_weighted_summary.csv", prototype_summary_rows)
    _write_csv(output_dir / "transfer_results.csv", transfer_rows)
    _write_csv(output_dir / "transfer_results_by_cluster.csv", by_cluster_rows)
    _write_csv(output_dir / "transfer_results_by_length_bucket.csv", by_length_rows)
    _write_csv(output_dir / "runtime_comparison.csv", runtime_rows + summary_rows)
    _write_csv(output_dir / "distance_effect_correlations.csv", corr_rows)
    (output_dir / "sample_manifests" / "selected_transfer_samples.json").write_text(
        json.dumps(
            {
                "transfer_splits": sorted(transfer_splits),
                "transfer_max_samples": int(args.transfer_max_samples),
                "sample_count": int(len(transfer_infos)),
                "samples": [_jsonable(asdict(info)) for info in transfer_infos],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    audit = {
        "archive": str(args.archive),
        "split_file": str(args.split_file),
        "output_dir": str(output_dir),
        "representations": representations,
        "num_clusters": ks,
        "transfer_modes": modes,
        "transfer_splits": sorted(transfer_splits),
        "transfer_sample_count": int(len(transfer_infos)),
        "prototype_limit": int(args.prototype_limit),
        "budget": float(args.budget),
        "max_delay": int(args.max_delay),
        "method": str(args.method),
        "protocol": str(args.protocol),
        "runtime_sec": float(time.perf_counter() - global_start),
        "summary": summary_rows,
        "length_bucket_summary": by_length_rows,
        "prototype_weighted_summary": prototype_summary_rows,
        "distance_effect_correlations": corr_rows,
        "resource_probe": _resource_probe(),
    }
    (output_dir / "audit.json").write_text(json.dumps(_jsonable(audit), indent=2), encoding="utf-8")
    _write_doc(output_dir, summary_rows, prototype_summary_rows, audit)
    print(json.dumps(_jsonable(audit), indent=2), flush=True)


if __name__ == "__main__":
    main()
