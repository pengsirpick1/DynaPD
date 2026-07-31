"""PCA + clustering analysis for Stage A masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


@dataclass
class ClusterScanResult:
    rows: list[dict]
    best_k: int
    best_labels: np.ndarray
    pca_features: np.ndarray
    pca_2d: np.ndarray
    proto_masks: np.ndarray
    cluster_sizes: np.ndarray
    distances: np.ndarray
    representative_indices: dict[int, list[int]]


def flatten_masks(masks: np.ndarray) -> np.ndarray:
    values = np.asarray(masks, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected masks [N, 2, W], got {values.shape}")
    return values.reshape(values.shape[0], -1)


def _fit_pca(flat: np.ndarray, n_components: int) -> tuple[np.ndarray, np.ndarray, PCA]:
    n = flat.shape[0]
    max_components = max(1, min(int(n_components), flat.shape[1], n))
    pca = PCA(n_components=max_components, random_state=0)
    features = pca.fit_transform(flat)
    pca2_components = max(1, min(2, flat.shape[1], n))
    pca2 = PCA(n_components=pca2_components, random_state=0).fit_transform(flat)
    if pca2.shape[1] == 1:
        pca2 = np.concatenate([pca2, np.zeros_like(pca2)], axis=1)
    return features.astype(np.float32), pca2.astype(np.float32), pca


def _proto_masks(masks: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    proto = np.zeros((int(k), masks.shape[1], masks.shape[2]), dtype=np.float32)
    sizes = np.zeros((int(k),), dtype=np.int64)
    for cluster in range(int(k)):
        member = labels == cluster
        sizes[cluster] = int(member.sum())
        if sizes[cluster] > 0:
            proto[cluster] = masks[member].mean(axis=0)
    return proto, sizes


def scan_kmeans(
    masks: np.ndarray,
    *,
    k_values: list[int],
    pca_components: int = 16,
    seed: int = 0,
    representatives_per_cluster: int = 4,
) -> ClusterScanResult:
    values = np.asarray(masks, dtype=np.float32)
    flat = flatten_masks(values)
    features, pca2, _ = _fit_pca(flat, int(pca_components))
    rows: list[dict] = []
    fitted: dict[int, tuple[np.ndarray, KMeans]] = {}
    for raw_k in k_values:
        k = int(raw_k)
        if k < 2 or k >= len(values):
            continue
        model = KMeans(n_clusters=k, n_init=20, random_state=int(seed))
        labels = model.fit_predict(features)
        unique = np.unique(labels)
        if unique.size < 2:
            continue
        silhouette = float(silhouette_score(features, labels))
        dbi = float(davies_bouldin_score(features, labels))
        ch = float(calinski_harabasz_score(features, labels))
        sizes = np.bincount(labels, minlength=k)
        row = {
            "k": int(k),
            "silhouette": silhouette,
            "davies_bouldin": dbi,
            "calinski_harabasz": ch,
            "min_cluster_size": int(sizes.min()),
            "max_cluster_size": int(sizes.max()),
        }
        rows.append(row)
        fitted[k] = (labels.astype(np.int64), model)
    if not rows:
        raise ValueError("No valid K values for clustering; need at least 3 mask samples")
    best_row = sorted(rows, key=lambda row: (-float(row["silhouette"]), float(row["davies_bouldin"]), int(row["k"])))[0]
    best_k = int(best_row["k"])
    best_labels, best_model = fitted[best_k]
    proto, sizes = _proto_masks(values, best_labels, best_k)
    distances = best_model.transform(features).astype(np.float32)
    reps: dict[int, list[int]] = {}
    for cluster in range(best_k):
        members = np.flatnonzero(best_labels == cluster)
        if members.size == 0:
            reps[cluster] = []
            continue
        order = members[np.argsort(distances[members, cluster], kind="mergesort")]
        reps[cluster] = [int(item) for item in order[: int(representatives_per_cluster)]]
    return ClusterScanResult(
        rows=rows,
        best_k=best_k,
        best_labels=best_labels.astype(np.int64),
        pca_features=features,
        pca_2d=pca2,
        proto_masks=proto,
        cluster_sizes=sizes,
        distances=distances,
        representative_indices=reps,
    )


def class_cluster_matrix(labels: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    c = np.asarray(cluster_labels, dtype=np.int64)
    classes = np.unique(y)
    k_count = int(c.max()) + 1 if c.size else 0
    matrix = np.zeros((int(classes.max()) + 1 if classes.size else 0, k_count), dtype=np.int64)
    for label, cluster in zip(y, c, strict=False):
        matrix[int(label), int(cluster)] += 1
    return matrix


def load_mask_archive(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as arrays:
        return {key: arrays[key] for key in arrays.files}
