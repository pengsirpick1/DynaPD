"""Cluster Stage A keypoint masks and generate review figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dmmp.stage_a.clustering import class_cluster_matrix, load_mask_archive, scan_kmeans
from dmmp.stage_a.viz import (
    plot_class_cluster_matrix,
    plot_cluster_prototypes,
    plot_cluster_scatter,
    plot_cluster_selection,
    plot_cluster_sizes,
    plot_sample_mask,
)
from dmmp.utils import write_csv, write_json
from dmmp.utils.config import parse_csv_ints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, help="Path to stage_a_masks_*/all_masks.npz")
    parser.add_argument("--output_dir", default="", help="Defaults to sibling stage_a_clustering_<attacker-or-mask>")
    parser.add_argument("--k_values", default="4,5,6,7,8,9,10,12")
    parser.add_argument("--pca_components", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--representatives_per_cluster", type=int, default=4)
    return parser.parse_args()


def _output_dir(args: argparse.Namespace, archive: Path) -> Path:
    if args.output_dir:
        target = Path(args.output_dir)
    else:
        run_dir = archive.parents[1]
        target = run_dir / "stage_a_clustering"
    target.mkdir(parents=True, exist_ok=True)
    return target


def main() -> None:
    args = parse_args()
    archive_path = Path(args.archive)
    payload = load_mask_archive(archive_path)
    masks = np.asarray(payload["mask"], dtype=np.float32)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    result = scan_kmeans(
        masks,
        k_values=parse_csv_ints(args.k_values),
        pca_components=int(args.pca_components),
        seed=int(args.seed),
        representatives_per_cluster=int(args.representatives_per_cluster),
    )
    out_dir = _output_dir(args, archive_path)
    figure_dir = out_dir / "figures"
    matrix = class_cluster_matrix(labels, result.best_labels)
    np.savez_compressed(
        out_dir / "cluster_result.npz",
        sample_ids=payload["sample_ids"],
        labels=labels,
        cluster_labels=result.best_labels,
        pca_features=result.pca_features,
        pca_2d=result.pca_2d,
        proto_masks=result.proto_masks,
        cluster_sizes=result.cluster_sizes,
        distances=result.distances,
        class_cluster_matrix=matrix,
        masks=masks,
    )
    write_csv(out_dir / "cluster_metrics.csv", result.rows)
    write_json(
        out_dir / "cluster_summary.json",
        {
            "archive": str(archive_path),
            "samples": int(masks.shape[0]),
            "width": int(masks.shape[2]),
            "best_k": int(result.best_k),
            "cluster_sizes": result.cluster_sizes.tolist(),
            "representative_indices": {str(key): value for key, value in result.representative_indices.items()},
            "metrics": result.rows,
        },
    )
    plot_cluster_selection(result.rows, figure_dir / "cluster_selection.png")
    plot_cluster_prototypes(result.proto_masks, figure_dir / f"prototype_k{result.best_k}.png")
    plot_class_cluster_matrix(matrix, figure_dir / f"class_cluster_matrix_k{result.best_k}.png")
    plot_cluster_scatter(result.pca_2d, result.best_labels, figure_dir / f"pca_scatter_k{result.best_k}.png")
    plot_cluster_sizes(result.cluster_sizes, figure_dir / f"cluster_sizes_k{result.best_k}.png")
    rep_dir = figure_dir / "representatives"
    for cluster, indices in result.representative_indices.items():
        for rank, idx in enumerate(indices):
            plot_sample_mask(
                payload["tam"][idx],
                payload["mask"][idx],
                payload["tam_masked"][idx],
                payload["tam_keypoint_only"][idx],
                payload["pred_prob"][idx],
                payload["masked_prob"][idx],
                rep_dir / f"cluster{cluster}_rank{rank}_sample{idx}.png",
            )
    print(f"Stage A clustering complete: {out_dir}")
    print(f"best_k={result.best_k}")


if __name__ == "__main__":
    main()
