"""Stage 1 prefix leakage profiling for DMMPv3."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..encoders.prefix import nonzero_trace, normalize01, prefix_patch_counts
from ..utils import write_json


def _resample(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(int(size), dtype=np.float32)
    if values.size == 1:
        return np.full(int(size), float(values[0]), dtype=np.float32)
    source = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    target = np.linspace(0.0, 1.0, int(size), dtype=np.float32)
    return np.interp(target, source, values).astype(np.float32)


def _burst_lengths(directions: np.ndarray) -> np.ndarray:
    directions = np.asarray(directions, dtype=np.float32).reshape(-1)
    if directions.size == 0:
        return np.asarray([], dtype=np.float32)
    changes = np.flatnonzero(directions[1:] != directions[:-1]) + 1
    starts = np.concatenate([np.asarray([0]), changes])
    ends = np.concatenate([changes, np.asarray([directions.size])])
    return (ends - starts).astype(np.float32)


def build_view_features(
    raw: np.ndarray,
    prefix_n: int,
    patch_num: int,
    *,
    max_trace_length: int = 5000,
    max_load_time: float = 80.0,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[np.ndarray]] = {
        "V_raw": [],
        "V_count": [],
        "V_interval": [],
        "V_burst": [],
        "V_rate": [],
        "V_cumul": [],
        "V_patch": [],
    }
    for trace in raw:
        prefix = nonzero_trace(trace)[: int(prefix_n)]
        directions = np.sign(prefix).astype(np.float32)
        times = np.abs(prefix).astype(np.float32)
        gaps = np.diff(times).astype(np.float32) if times.size > 1 else np.asarray([], dtype=np.float32)
        positive_gaps = gaps[gaps > 0]
        counts = prefix_patch_counts(
            prefix,
            int(prefix_n),
            int(patch_num),
            max_trace_length=int(max_trace_length),
            max_load_time=float(max_load_time),
        )
        patch_total = counts.sum(axis=0)
        out_count = float(np.sum(directions > 0))
        in_count = float(np.sum(directions < 0))
        prefix_len = float(prefix.size)
        duration = float(max(times[-1] - times[0], 1e-6)) if times.size > 1 else 1e-6
        bursts = _burst_lengths(directions)
        cumul = np.cumsum(directions) if directions.size else np.asarray([], dtype=np.float32)
        local_rate = patch_total / max(float(prefix_len), 1.0)

        rows["V_raw"].append(
            np.concatenate([_resample(directions, 32), _resample(normalize01(times), 32), _resample(normalize01(gaps), 16)])
        )
        rows["V_count"].append(
            np.asarray(
                [
                    prefix_len,
                    out_count,
                    in_count,
                    out_count - in_count,
                    out_count / max(prefix_len, 1.0),
                    in_count / max(prefix_len, 1.0),
                ],
                dtype=np.float32,
            )
        )
        rows["V_interval"].append(
            np.asarray(
                [
                    float(positive_gaps.mean()) if positive_gaps.size else 0.0,
                    float(np.median(positive_gaps)) if positive_gaps.size else 0.0,
                    float(positive_gaps.max()) if positive_gaps.size else 0.0,
                    float(positive_gaps.std()) if positive_gaps.size else 0.0,
                    float(np.percentile(positive_gaps, 75)) if positive_gaps.size else 0.0,
                    float(np.percentile(positive_gaps, 90)) if positive_gaps.size else 0.0,
                    float(np.sum(positive_gaps > np.percentile(positive_gaps, 90)) / max(prefix_len, 1.0))
                    if positive_gaps.size
                    else 0.0,
                ],
                dtype=np.float32,
            )
        )
        rows["V_burst"].append(
            np.asarray(
                [
                    float(bursts.size),
                    float(bursts.mean()) if bursts.size else 0.0,
                    float(bursts.max()) if bursts.size else 0.0,
                    float(bursts.std()) if bursts.size else 0.0,
                    float(np.sum(directions[1:] != directions[:-1]) / max(prefix_len - 1.0, 1.0))
                    if directions.size > 1
                    else 0.0,
                    float(directions[0]) if directions.size else 0.0,
                    float(directions[-1]) if directions.size else 0.0,
                ],
                dtype=np.float32,
            )
        )
        rows["V_rate"].append(
            np.concatenate(
                [
                    np.asarray([prefix_len / duration, float(local_rate.max()) if local_rate.size else 0.0], dtype=np.float32),
                    _resample(local_rate, 32),
                ]
            )
        )
        rows["V_cumul"].append(_resample(cumul / max(prefix_len, 1.0), 64))
        rows["V_patch"].append(counts.reshape(-1) / max(float(prefix_n), 1.0))
    return {key: np.stack(value, axis=0).astype(np.float32) for key, value in rows.items()}


def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0 or float(np.nanmax(arr) - np.nanmin(arr)) <= 1e-12:
        return np.zeros(arr.shape, dtype=np.int64)
    qs = np.linspace(0.0, 100.0, int(n_bins) + 1)[1:-1]
    edges = np.unique(np.nanpercentile(arr, qs))
    if edges.size == 0:
        return np.zeros(arr.shape, dtype=np.int64)
    return np.digitize(arr, edges, right=False).astype(np.int64)


def mutual_information_discrete(feature_codes: np.ndarray, labels: np.ndarray) -> float:
    x = np.asarray(feature_codes, dtype=np.int64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if x.size != y.size or x.size == 0:
        return 0.0
    _, x_inv = np.unique(x, return_inverse=True)
    _, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((int(x_inv.max()) + 1, int(y_inv.max()) + 1), dtype=np.float64)
    np.add.at(joint, (x_inv, y_inv), 1.0)
    joint /= float(x.size)
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    expected = px @ py
    valid = joint > 0
    return float((joint[valid] * np.log((joint[valid] + 1e-12) / (expected[valid] + 1e-12))).sum())


def score_feature_matrix(matrix: np.ndarray, labels: np.ndarray, n_bins: int = 8, top_features: int = 32) -> tuple[float, np.ndarray]:
    features = np.asarray(matrix, dtype=np.float32)
    if features.ndim == 1:
        features = features.reshape(-1, 1)
    scores = np.zeros(features.shape[1], dtype=np.float32)
    for column in range(features.shape[1]):
        scores[column] = mutual_information_discrete(_quantile_bins(features[:, column], int(n_bins)), labels)
    if scores.size == 0:
        return 0.0, scores
    take = min(int(top_features), scores.size)
    return float(np.mean(np.sort(scores)[-take:])), scores


def _centroid_logits(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = features[:, None, :] - centroids[None, :, :]
    logits = -np.sum(diff * diff, axis=2)
    return (logits - logits.max(axis=1, keepdims=True)).astype(np.float32)


def _cross_entropy_from_logits(logits: np.ndarray, labels: np.ndarray, class_to_pos: dict[int, int]) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)
    positions = np.asarray([class_to_pos[int(label)] for label in labels], dtype=np.int64)
    return -np.log(np.maximum(probs[np.arange(labels.size), positions], 1e-12)).astype(np.float32)


def masking_scores(features: np.ndarray, labels: np.ndarray, *, max_samples: int = 2048, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.shape[0] > int(max_samples) > 0:
        idx = rng.choice(features.shape[0], size=int(max_samples), replace=False)
        features = features[idx]
        labels = labels[idx]
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / np.maximum(std, 1e-6)
    fill_value = features.mean(axis=0)
    classes = np.unique(labels)
    class_to_pos = {int(label): pos for pos, label in enumerate(classes.tolist())}
    centroids = np.stack([features[labels == label].mean(axis=0) for label in classes], axis=0).astype(np.float32)
    base_logits = _centroid_logits(features, centroids)
    base_ce = _cross_entropy_from_logits(base_logits, labels, class_to_pos)
    base_dist = -base_logits
    scores = np.zeros(features.shape[1], dtype=np.float32)
    for column in range(features.shape[1]):
        old = features[:, column : column + 1]
        new = np.full_like(old, float(fill_value[column]))
        delta = (new - centroids[:, column].reshape(1, -1)) ** 2 - (old - centroids[:, column].reshape(1, -1)) ** 2
        masked_logits = -(base_dist + delta)
        masked_ce = _cross_entropy_from_logits(masked_logits.astype(np.float32), labels, class_to_pos)
        scores[column] = float(np.mean(masked_ce - base_ce))
    return np.maximum(scores, 0.0).astype(np.float32)


def profile_prefix_leakage(
    raw: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    *,
    prefix_n: int = 500,
    patch_num: int = 200,
    max_trace_length: int = 5000,
    max_load_time: float = 80.0,
    seed: int = 0,
    topk_cells: int = 80,
    mi_bins: int = 8,
    masking_max_samples: int = 2048,
    command: str = "",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = np.asarray(raw)
    labels = np.asarray(labels, dtype=np.int64)
    view_features = build_view_features(
        raw,
        int(prefix_n),
        int(patch_num),
        max_trace_length=int(max_trace_length),
        max_load_time=float(max_load_time),
    )
    view_scores: dict[str, float] = {}
    for name, matrix in view_features.items():
        score, _ = score_feature_matrix(matrix, labels, n_bins=int(mi_bins))
        view_scores[name] = float(score)

    patch_matrix = view_features["V_patch"]
    _, cell_mi_flat = score_feature_matrix(patch_matrix, labels, n_bins=int(mi_bins), top_features=patch_matrix.shape[1])
    cell_mask_flat = masking_scores(patch_matrix, labels, max_samples=int(masking_max_samples), seed=int(seed))
    cell_mi = cell_mi_flat.reshape(2, int(patch_num))
    cell_mask = cell_mask_flat.reshape(2, int(patch_num))
    cell_leakage = normalize01(cell_mi) + normalize01(cell_mask)
    if float(cell_leakage.max()) > 1e-8:
        cell_leakage = cell_leakage / float(cell_leakage.max())
    patch_leakage = cell_leakage.sum(axis=0).astype(np.float32)
    flat_order = np.argsort(-cell_leakage.reshape(-1), kind="mergesort")
    topk = max(1, min(int(topk_cells), flat_order.size))
    topk_mask = np.zeros_like(cell_leakage, dtype=np.float32).reshape(-1)
    topk_mask[flat_order[:topk]] = 1.0
    topk_mask = topk_mask.reshape(2, int(patch_num)).astype(np.float32)

    top_views = [{"view": key, "score": float(value)} for key, value in sorted(view_scores.items(), key=lambda item: item[1], reverse=True)]
    top_cells = [
        {
            "direction": "outgoing" if int(index // int(patch_num)) == 0 else "incoming",
            "patch": int(index % int(patch_num)),
            "score": float(cell_leakage.reshape(-1)[index]),
        }
        for index in flat_order[: min(topk, 20)]
    ]
    top_patches = [{"patch": int(idx), "score": float(patch_leakage[idx])} for idx in np.argsort(-patch_leakage, kind="mergesort")[:20]]

    write_json(output_dir / "view_leakage.json", {"view_scores": view_scores, "ranked_views": top_views, "mi_bins": int(mi_bins)})
    write_json(output_dir / "topk_views.json", {"topk_cells": int(topk), "ranked_views": top_views, "top_cells": top_cells, "top_patches": top_patches})
    np.save(output_dir / "patch_leakage.npy", patch_leakage.astype(np.float32))
    np.save(output_dir / "cell_leakage.npy", cell_leakage.astype(np.float32))
    np.save(output_dir / "cell_mi_leakage.npy", cell_mi.astype(np.float32))
    np.save(output_dir / "cell_masking_leakage.npy", cell_mask.astype(np.float32))
    np.save(output_dir / "topk_candidate_mask.npy", topk_mask.astype(np.float32))

    early_top = sum(1 for item in top_cells if item["patch"] < max(1, int(0.25 * int(patch_num))))
    lines = [
        "# Stage 1: Prefix leakage profiling",
        "",
        f"- samples: {len(labels)}",
        f"- classes: {len(np.unique(labels))}",
        f"- prefix_n: {int(prefix_n)}",
        f"- patch_num: {int(patch_num)}",
        f"- topk_cells: {int(topk)}",
        "",
        "## View Ranking",
        "",
        *[f"- {row['view']}: {row['score']:.6f}" for row in top_views],
        "",
        "## Top Cells",
        "",
        f"- early top cells in first 25% patches: {early_top}",
        f"- top patches: {', '.join(str(item['patch']) for item in top_patches[:5])}",
        f"- command: {command or 'N/A'}",
        "",
    ]
    (output_dir / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "view_scores": view_scores,
        "patch_leakage": patch_leakage.astype(np.float32),
        "cell_leakage": cell_leakage.astype(np.float32),
        "topk_candidate_mask": topk_mask.astype(np.float32),
        "top_views": top_views,
        "output_dir": str(output_dir),
    }

