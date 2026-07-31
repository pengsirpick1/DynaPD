"""Plotting helpers for Stage A artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_sample_mask(
    tam: np.ndarray,
    mask: np.ndarray,
    tam_masked: np.ndarray,
    tam_keypoint_only: np.ndarray,
    pred_prob: np.ndarray,
    masked_prob: np.ndarray,
    save_path: str | Path,
) -> None:
    plt = _plt()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    pred_label = int(np.argmax(pred_prob))
    masked_label = int(np.argmax(masked_prob))
    js_title = f"before={pred_label} ({pred_prob[pred_label]:.3f}) after={masked_label} ({masked_prob[masked_label]:.3f})"
    fig = plt.figure(figsize=(12, 10))
    rows = [
        ("Original TAM", tam, "viridis"),
        ("Deletion Keypoint Mask", mask, "magma"),
        ("Deleted-Keypoint TAM", tam_masked, "viridis"),
        ("Keypoint-Only TAM", tam_keypoint_only, "viridis"),
    ]
    for index, (title, values, cmap) in enumerate(rows, start=1):
        ax = fig.add_subplot(5, 1, index)
        im = ax.imshow(values, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_ylabel("dir")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["out", "in"])
        if index == len(rows):
            ax.set_xlabel("TAM slot")
        plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    ax = fig.add_subplot(5, 1, 5)
    top_indices = np.argsort(-pred_prob)[: min(8, pred_prob.size)]
    x = np.arange(len(top_indices))
    ax.bar(x - 0.18, pred_prob[top_indices], width=0.36, label="original")
    ax.bar(x + 0.18, masked_prob[top_indices], width=0.36, label="deleted")
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(idx)) for idx in top_indices])
    ax.set_ylim(0.0, max(1e-6, float(max(pred_prob[top_indices].max(), masked_prob[top_indices].max()))) * 1.15)
    ax.set_title(f"Prediction Change | {js_title}")
    ax.set_ylabel("prob")
    ax.set_xlabel("class")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_cluster_prototypes(proto_masks: np.ndarray, save_path: str | Path) -> None:
    plt = _plt()
    proto = np.asarray(proto_masks, dtype=np.float32)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    k_count = int(proto.shape[0])
    fig = plt.figure(figsize=(12, max(2.0, 1.6 * k_count)))
    for k in range(k_count):
        ax = fig.add_subplot(k_count, 1, k + 1)
        im = ax.imshow(proto[k], aspect="auto", cmap="magma", vmin=0.0, vmax=max(1e-6, float(proto.max())))
        ax.set_title(f"Cluster {k} Prototype Mask")
        ax.set_ylabel("dir")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["out", "in"])
        if k == k_count - 1:
            ax.set_xlabel("TAM slot")
        plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_class_cluster_matrix(matrix: np.ndarray, save_path: str | Path) -> None:
    plt = _plt()
    values = np.asarray(matrix)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, max(5, min(18, 0.17 * values.shape[0] + 2))))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title("Website Class vs Keypoint Cluster")
    ax.set_xlabel("cluster")
    ax.set_ylabel("class")
    plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_cluster_selection(rows: list[dict], save_path: str | Path) -> None:
    plt = _plt()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    k_list = [int(row["k"]) for row in rows]
    silhouette = [float(row.get("silhouette", 0.0)) for row in rows]
    dbi = [float(row.get("davies_bouldin", 0.0)) for row in rows]
    ch = [float(row.get("calinski_harabasz", 0.0)) for row in rows]
    fig = plt.figure(figsize=(10, 8))
    for index, (title, values) in enumerate(
        [
            ("Silhouette Score", silhouette),
            ("Davies-Bouldin Index", dbi),
            ("Calinski-Harabasz Score", ch),
        ],
        start=1,
    ):
        ax = fig.add_subplot(3, 1, index)
        ax.plot(k_list, values, marker="o")
        ax.set_title(title)
        ax.set_xlabel("K")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_cluster_scatter(features_2d: np.ndarray, labels: np.ndarray, save_path: str | Path) -> None:
    plt = _plt()
    points = np.asarray(features_2d, dtype=np.float32)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1)
    scatter = ax.scatter(points[:, 0], points[:, 1], c=np.asarray(labels), s=32, cmap="tab10")
    ax.set_title("Mask PCA Scatter")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, fraction=0.035, pad=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def plot_cluster_sizes(cluster_sizes: np.ndarray, save_path: str | Path) -> None:
    plt = _plt()
    sizes = np.asarray(cluster_sizes, dtype=np.int64)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.bar(np.arange(sizes.size), sizes)
    ax.set_title("Cluster Sizes")
    ax.set_xlabel("cluster")
    ax.set_ylabel("samples")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)
