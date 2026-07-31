"""Shape-normalized pooled clustering and stability checks for Stage A masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.clustering import class_cluster_matrix
from dmmp.stage_a.viz import plot_class_cluster_matrix, plot_cluster_scatter, plot_cluster_sizes
from dmmp.utils import write_csv, write_json
from dmmp.utils.config import parse_csv_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pooled_width", type=int, default=200)
    parser.add_argument("--k_values", default="3,4,5,6,7,8,9,10,11,12")
    parser.add_argument("--macro_k", type=int, default=4)
    parser.add_argument("--pca_components", type=int, default=16)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--bootstrap_fraction", type=float, default=0.80)
    parser.add_argument("--min_cluster_size", type=int, default=5)
    parser.add_argument("--representatives_per_cluster", type=int, default=6)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--faithfulness_sample_npz", default="")
    parser.add_argument("--faithfulness_ratio", type=float, default=0.10)
    return parser.parse_args()


def sum_pool_mask(mask: np.ndarray, pooled_width: int) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] != 2:
        raise ValueError(f"Expected mask [N, 2, W], got {values.shape}")
    bins = np.array_split(np.arange(values.shape[-1]), int(pooled_width))
    pooled = np.zeros((values.shape[0], 2, int(pooled_width)), dtype=np.float32)
    for target, source in enumerate(bins):
        if source.size:
            pooled[:, :, target] = values[:, :, source].sum(axis=2)
    return pooled


def l1_normalize(mask: np.ndarray) -> np.ndarray:
    values = np.asarray(mask, dtype=np.float32)
    denom = values.reshape(values.shape[0], -1).sum(axis=1).reshape(-1, 1, 1)
    return (values / np.maximum(denom, 1e-8)).astype(np.float32)


def flatten(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.float32).reshape(mask.shape[0], -1)


def _fit_features(flat: np.ndarray, pca_components: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_components = max(1, min(int(pca_components), flat.shape[0], flat.shape[1]))
    pca = PCA(n_components=n_components, random_state=int(seed))
    features = pca.fit_transform(flat).astype(np.float32)
    pca2_components = max(1, min(2, flat.shape[0], flat.shape[1]))
    pca2 = PCA(n_components=pca2_components, random_state=int(seed)).fit_transform(flat).astype(np.float32)
    if pca2.shape[1] == 1:
        pca2 = np.concatenate([pca2, np.zeros_like(pca2)], axis=1)
    return features, pca2


def _prototype(mask: np.ndarray, labels: np.ndarray, k: int, reducer: str = "mean") -> np.ndarray:
    proto = np.zeros((int(k), mask.shape[1], mask.shape[2]), dtype=np.float32)
    for cluster in range(int(k)):
        member = mask[labels == cluster]
        if member.size == 0:
            continue
        if reducer == "median":
            proto[cluster] = np.median(member, axis=0).astype(np.float32)
        elif reducer == "q25":
            proto[cluster] = np.quantile(member, 0.25, axis=0).astype(np.float32)
        elif reducer == "q75":
            proto[cluster] = np.quantile(member, 0.75, axis=0).astype(np.float32)
        else:
            proto[cluster] = member.mean(axis=0).astype(np.float32)
    return proto


def _prototype_similarity(mask: np.ndarray, labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    ka = int(labels_a.max()) + 1
    kb = int(labels_b.max()) + 1
    pa = _prototype(mask, labels_a, ka).reshape(ka, -1)
    pb = _prototype(mask, labels_b, kb).reshape(kb, -1)
    pa = pa / np.maximum(np.linalg.norm(pa, axis=1, keepdims=True), 1e-8)
    pb = pb / np.maximum(np.linalg.norm(pb, axis=1, keepdims=True), 1e-8)
    sim = pa @ pb.T
    return float(np.mean(np.max(sim, axis=1)))


def _cluster_once(features: np.ndarray, k: int, seed: int, indices: np.ndarray | None) -> np.ndarray:
    train = features if indices is None else features[indices]
    model = KMeans(n_clusters=int(k), n_init=20, random_state=int(seed))
    model.fit(train)
    return model.predict(features).astype(np.int64)


def _metrics_for_labels(features: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1)
    result = {
        "silhouette": float(silhouette_score(features, labels)) if np.unique(labels).size > 1 else 0.0,
        "davies_bouldin": float(davies_bouldin_score(features, labels)) if np.unique(labels).size > 1 else 0.0,
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)) if np.unique(labels).size > 1 else 0.0,
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "singleton_clusters": int(np.sum(sizes <= 1)),
    }
    return result


def _cluster_stability(features: np.ndarray, shape_masks: np.ndarray, k: int, args: argparse.Namespace) -> tuple[list[np.ndarray], list[dict]]:
    rng = np.random.default_rng(int(args.base_seed) + int(k) * 1009)
    labels_list = []
    rows = []
    n = features.shape[0]
    subset_size = max(int(k), min(n, int(round(n * float(args.bootstrap_fraction)))))
    for run in range(int(args.seeds)):
        seed = int(args.base_seed) + int(k) * 1000 + run
        subset = rng.choice(n, size=subset_size, replace=True)
        labels = _cluster_once(features, int(k), seed, subset)
        labels_list.append(labels)
        rows.append({"run": run, **_metrics_for_labels(features, labels)})
    return labels_list, rows


def _pairwise(values: list[np.ndarray], fn) -> tuple[float, float]:
    scores = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            scores.append(float(fn(values[i], values[j])))
    if not scores:
        return 0.0, 0.0
    return float(np.mean(scores)), float(np.std(scores))


def choose_auto_k(rows: list[dict], min_cluster_size: int) -> int:
    valid = [row for row in rows if int(row["min_cluster_size_mean"]) >= int(min_cluster_size) and float(row["singleton_run_rate"]) <= 0.0]
    candidates = valid if valid else rows
    if not candidates:
        raise ValueError("No stability rows available")
    best = max(candidates, key=lambda row: float(row["silhouette_mean"]) + 0.25 * float(row["ari_mean"]))
    best_score = float(best["silhouette_mean"]) + 0.25 * float(best["ari_mean"])
    near = [
        row
        for row in candidates
        if float(row["silhouette_mean"]) + 0.25 * float(row["ari_mean"]) >= 0.95 * best_score
    ]
    return int(min(near, key=lambda row: int(row["k"]))["k"])


def _plot_audit_card(
    mean_proto: np.ndarray,
    median_proto: np.ndarray,
    q25: np.ndarray,
    q75: np.ndarray,
    labels: np.ndarray,
    cluster: int,
    stats: dict,
    save_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_path.parent.mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 100.0, mean_proto.shape[-1])
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(4, 1, 1)
    im = ax.imshow(mean_proto, aspect="auto", cmap="magma")
    ax.set_title(
        f"Cluster {cluster} mean | n={int(stats['size'])} "
        f"necessity_js={float(stats.get('necessity_js', 0.0)):.3f} "
        f"suff_js={float(stats.get('sufficiency_js', 0.0)):.3f}"
    )
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["out", "in"])
    plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    ax = fig.add_subplot(4, 1, 2)
    im = ax.imshow(median_proto, aspect="auto", cmap="magma")
    ax.set_title("Median prototype")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["out", "in"])
    plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012)
    ax = fig.add_subplot(4, 1, 3)
    for direction, name in enumerate(["out", "in"]):
        mean_curve = mean_proto[direction]
        ax.plot(x, mean_curve, label=f"{name} mean")
        ax.fill_between(x, q25[direction], q75[direction], alpha=0.20)
    ax.set_title("Directional marginal curves with 25%-75% envelope")
    ax.set_xlabel("normalized time (%)")
    ax.legend(loc="best")
    ax = fig.add_subplot(4, 1, 4)
    ax.axis("off")
    text = "\n".join(
        [
            f"size: {int(stats['size'])}",
            f"mask_mass: {float(stats.get('mask_mass', 0.0)):.6f}",
            f"mean_top1_drop: {float(stats.get('mean_top1_drop', 0.0)):.6f}",
            f"top1_preservation: {float(stats.get('top1_preservation', 0.0)):.6f}",
            f"necessity_js: {float(stats.get('necessity_js', 0.0)):.6f}",
            f"sufficiency_js: {float(stats.get('sufficiency_js', 0.0)):.6f}",
            f"incoming_mass_ratio: {float(stats.get('incoming_mass_ratio', 0.0)):.6f}",
            f"time_center_norm: {float(stats.get('time_center_norm', 0.0)):.6f}",
        ]
    )
    ax.text(0.01, 0.95, text, va="top", family="monospace")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close(fig)


def _mask_stats(mask: np.ndarray) -> dict[str, float]:
    values = np.asarray(mask, dtype=np.float64)
    total = float(values.sum())
    width = values.shape[-1]
    if total <= 1e-12:
        return {"mask_mass": 0.0, "incoming_mass_ratio": 0.0, "time_center_norm": 0.0}
    time_mass = values.sum(axis=0)
    center = float((time_mass * np.arange(width)).sum() / total)
    return {
        "mask_mass": total,
        "incoming_mass_ratio": float(values[1].sum() / total),
        "time_center_norm": float(center / max(width - 1, 1)),
    }


def _save_cluster_result(
    output_dir: Path,
    name: str,
    shape_masks: np.ndarray,
    features: np.ndarray,
    pca2: np.ndarray,
    labels: np.ndarray,
    sample_labels: np.ndarray,
    sample_ids: np.ndarray,
    faithfulness: dict[str, np.ndarray],
    representatives_per_cluster: int,
) -> dict:
    target = output_dir / name
    figure_dir = target / "figures"
    target.mkdir(parents=True, exist_ok=True)
    k = int(labels.max()) + 1
    mean_proto = _prototype(shape_masks, labels, k, "mean")
    median_proto = _prototype(shape_masks, labels, k, "median")
    q25 = _prototype(shape_masks, labels, k, "q25")
    q75 = _prototype(shape_masks, labels, k, "q75")
    matrix = class_cluster_matrix(sample_labels, labels)
    distances = np.zeros((shape_masks.shape[0], k), dtype=np.float32)
    flat = features
    centers = np.stack([flat[labels == cluster].mean(axis=0) for cluster in range(k)], axis=0)
    for cluster in range(k):
        distances[:, cluster] = np.linalg.norm(flat - centers[cluster], axis=1)
    representatives = {}
    cluster_rows = []
    for cluster in range(k):
        members = np.flatnonzero(labels == cluster)
        order = members[np.argsort(distances[members, cluster], kind="mergesort")]
        representatives[str(cluster)] = [int(item) for item in order[: int(representatives_per_cluster)]]
        row = {"cluster": int(cluster), "size": int(members.size), **_mask_stats(mean_proto[cluster])}
        if members.size and faithfulness:
            row["necessity_js"] = float(np.mean(faithfulness.get("necessity_js", np.zeros(len(labels)))[members]))
            row["sufficiency_js"] = float(np.mean(faithfulness.get("sufficiency_js", np.zeros(len(labels)))[members]))
            row["mean_top1_drop"] = float(np.mean(faithfulness.get("top1_drop", np.zeros(len(labels)))[members]))
            row["top1_preservation"] = float(np.mean(faithfulness.get("top1_preservation", np.zeros(len(labels)))[members]))
        cluster_rows.append(row)
        _plot_audit_card(
            mean_proto[cluster],
            median_proto[cluster],
            q25[cluster],
            q75[cluster],
            labels,
            cluster,
            row,
            figure_dir / "audit_cards" / f"cluster_{cluster}.png",
        )
    faithfulness_payload = {f"faithfulness_{key}": value for key, value in faithfulness.items() if isinstance(value, np.ndarray)}
    np.savez_compressed(
        target / "cluster_result.npz",
        sample_ids=sample_ids,
        labels=sample_labels,
        cluster_labels=labels,
        pca_features=features,
        pca_2d=pca2,
        shape_masks=shape_masks,
        mean_proto_masks=mean_proto,
        median_proto_masks=median_proto,
        q25_proto_masks=q25,
        q75_proto_masks=q75,
        class_cluster_matrix=matrix,
        distances=distances,
        **faithfulness_payload,
    )
    write_csv(target / "cluster_rows.csv", cluster_rows)
    write_json(target / "cluster_summary.json", {"k": int(k), "cluster_rows": cluster_rows, "representatives": representatives})
    plot_class_cluster_matrix(matrix, figure_dir / f"class_cluster_matrix_k{k}.png")
    plot_cluster_scatter(pca2, labels, figure_dir / f"pca_scatter_k{k}.png")
    plot_cluster_sizes(np.bincount(labels, minlength=k), figure_dir / f"cluster_sizes_k{k}.png")
    return {"k": int(k), "cluster_rows": cluster_rows, "representatives": representatives, "path": str(target)}


def _faithfulness_sample_arrays(path: str, ratio: float, sample_ids: np.ndarray) -> dict[str, np.ndarray]:
    if not path:
        return {}
    with np.load(path, allow_pickle=False) as arrays:
        payload = {key: arrays[key] for key in arrays.files}
    if "ratios" not in payload:
        raise ValueError(f"Missing ratios in faithfulness sample file: {path}")
    archived_ids = payload.get("sample_ids")
    if archived_ids is not None:
        archived_ids = np.asarray(archived_ids).astype(str)
        expected_ids = np.asarray(sample_ids).astype(str)
        if archived_ids.shape != expected_ids.shape or not np.array_equal(archived_ids, expected_ids):
            raise ValueError("Faithfulness sample file sample_ids do not match clustering archive order")
    ratio_values = np.asarray(payload["ratios"], dtype=np.float32)
    selected = int(np.argmin(np.abs(ratio_values - float(ratio))))
    return {
        "ratio": np.asarray(float(ratio_values[selected]), dtype=np.float32),
        "necessity_js": np.asarray(payload["necessity_js_div"][selected], dtype=np.float32),
        "sufficiency_js": np.asarray(payload["sufficiency_js_div"][selected], dtype=np.float32),
        "top1_drop": np.asarray(payload["necessity_top1_drop"][selected], dtype=np.float32),
        "top1_preservation": np.asarray(payload["sufficiency_top1_preservation"][selected], dtype=np.float32),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output dir: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(args.archive, allow_pickle=False) as arrays:
        archive = {key: arrays[key] for key in arrays.files}
    sample_ids = np.asarray(archive["sample_ids"]).astype(str)
    faithfulness = _faithfulness_sample_arrays(str(args.faithfulness_sample_npz), float(args.faithfulness_ratio), sample_ids)
    pooled = sum_pool_mask(np.asarray(archive["mask"], dtype=np.float32), int(args.pooled_width))
    shape_masks = l1_normalize(pooled)
    flat = flatten(shape_masks)
    features, pca2 = _fit_features(flat, int(args.pca_components), int(args.base_seed))
    k_values = parse_csv_ints(args.k_values)
    stability_rows = []
    labels_by_k: dict[int, np.ndarray] = {}
    for k in k_values:
        if int(k) < 2 or int(k) >= features.shape[0]:
            continue
        labels_list, run_rows = _cluster_stability(features, shape_masks, int(k), args)
        ari_mean, ari_std = _pairwise(labels_list, adjusted_rand_score)
        proto_mean, proto_std = _pairwise(labels_list, lambda a, b: _prototype_similarity(shape_masks, a, b))
        labels_full = _cluster_once(features, int(k), int(args.base_seed) + int(k), None)
        labels_by_k[int(k)] = labels_full
        sizes = np.bincount(labels_full, minlength=int(k))
        row = {
            "k": int(k),
            "silhouette_mean": float(np.mean([item["silhouette"] for item in run_rows])),
            "silhouette_std": float(np.std([item["silhouette"] for item in run_rows])),
            "davies_bouldin_mean": float(np.mean([item["davies_bouldin"] for item in run_rows])),
            "calinski_harabasz_mean": float(np.mean([item["calinski_harabasz"] for item in run_rows])),
            "min_cluster_size_mean": float(np.mean([item["min_cluster_size"] for item in run_rows])),
            "full_min_cluster_size": int(sizes.min()),
            "full_max_cluster_size": int(sizes.max()),
            "singleton_run_rate": float(np.mean([item["singleton_clusters"] > 0 for item in run_rows])),
            "ari_mean": ari_mean,
            "ari_std": ari_std,
            "prototype_similarity_mean": proto_mean,
            "prototype_similarity_std": proto_std,
        }
        stability_rows.append(row)
    auto_k = choose_auto_k(stability_rows, int(args.min_cluster_size))
    macro_k = int(args.macro_k)
    if macro_k not in labels_by_k:
        labels_by_k[macro_k] = _cluster_once(features, macro_k, int(args.base_seed) + macro_k, None)
    if auto_k not in labels_by_k:
        labels_by_k[auto_k] = _cluster_once(features, auto_k, int(args.base_seed) + auto_k, None)
    macro = _save_cluster_result(
        output_dir,
        f"macro_k{macro_k}",
        shape_masks,
        features,
        pca2,
        labels_by_k[macro_k],
        np.asarray(archive["labels"], dtype=np.int64),
        sample_ids,
        faithfulness,
        int(args.representatives_per_cluster),
    )
    fine = _save_cluster_result(
        output_dir,
        f"auto_k{auto_k}",
        shape_masks,
        features,
        pca2,
        labels_by_k[auto_k],
        np.asarray(archive["labels"], dtype=np.int64),
        sample_ids,
        faithfulness,
        int(args.representatives_per_cluster),
    )
    np.savez_compressed(
        output_dir / "pooled_shape_masks.npz",
        sample_ids=sample_ids,
        labels=np.asarray(archive["labels"], dtype=np.int64),
        pooled_masks=pooled,
        shape_masks=shape_masks,
        pca_features=features,
        pca_2d=pca2,
    )
    write_csv(output_dir / "cluster_stability.csv", stability_rows)
    write_json(
        output_dir / "cluster_stability_summary.json",
        {
            "archive": str(args.archive),
            "samples": int(shape_masks.shape[0]),
            "source_width": int(archive["mask"].shape[-1]),
            "pooled_width": int(args.pooled_width),
            "faithfulness_sample_npz": str(args.faithfulness_sample_npz),
            "faithfulness_ratio": float(faithfulness["ratio"]) if faithfulness else None,
            "k_values": k_values,
            "macro_k": int(macro_k),
            "auto_k": int(auto_k),
            "stability": stability_rows,
            "macro": macro,
            "auto": fine,
        },
    )
    print(f"Stage A cluster stability complete: {output_dir}")
    print(f"macro_k={macro_k} auto_k={auto_k}")


if __name__ == "__main__":
    main()
