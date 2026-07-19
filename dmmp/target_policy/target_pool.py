"""Versioned target-policy pool persistence.

The first implementation uses only standard project dependencies.  It writes
``policies.npz`` instead of zarr and ``index.csv`` instead of parquet while
keeping the requested directory contract and metadata names stable.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .candidate_generator import CandidatePolicy


FORBIDDEN_LABEL_KEYS = frozenset(
    {
        "y",
        "label",
        "labels",
        "true_label",
        "true_labels",
        "class",
        "classes",
        "class_id",
        "site",
        "site_id",
        "target_class",
        "target_label",
    }
)

ARRAY_KEYS = (
    "x0_star",
    "policy_allocation",
    "policy_counts",
    "allowed_mask",
    "family_weights",
    "primitive_weights",
    "effect_map",
)


def _is_label_key(key: str) -> bool:
    return str(key).strip().lower().replace("-", "_") in FORBIDDEN_LABEL_KEYS


class TargetPolicyPool:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.policies_path = self.root / "policies.npz"
        self.metadata_path = self.root / "metadata.json"
        self.index_path = self.root / "index.csv"

    def exists(self) -> bool:
        return self.policies_path.is_file() and self.metadata_path.is_file() and self.index_path.is_file()

    def load_arrays(self) -> dict[str, np.ndarray]:
        with np.load(self.policies_path, allow_pickle=False) as payload:
            return {key: np.asarray(payload[key]) for key in payload.files}

    def load_metadata(self) -> dict[str, Any]:
        return json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def load_training_arrays(self) -> dict[str, np.ndarray]:
        arrays = self.load_arrays()
        # This loader intentionally omits labels; target diffusion must remain
        # class-agnostic even when the original dataset has labels.
        return {key: value for key, value in arrays.items() if not _is_label_key(key)}


def _pad_indices(rows: list[np.ndarray], width: int, fill: int = -1) -> np.ndarray:
    out = np.full((len(rows), int(width)), int(fill), dtype=np.int64)
    for index, row in enumerate(rows):
        take = min(int(width), int(len(row)))
        if take:
            out[index, :take] = np.asarray(row[:take], dtype=np.int64)
    return out


def _pad_float(rows: list[np.ndarray], width: int) -> np.ndarray:
    out = np.zeros((len(rows), int(width)), dtype=np.float32)
    for index, row in enumerate(rows):
        take = min(int(width), int(len(row)))
        if take:
            out[index, :take] = np.asarray(row[:take], dtype=np.float32)
    return out


def write_target_policy_pool(
    root: str | Path,
    records: Iterable[tuple[int, int, CandidatePolicy] | tuple[int, int, CandidatePolicy, np.ndarray]],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    target = TargetPolicyPool(root)
    target.root.mkdir(parents=True, exist_ok=True)
    raw_records = list(records)
    record_list: list[tuple[int, int, CandidatePolicy]] = []
    prefix_vectors: list[np.ndarray] = []
    for record in raw_records:
        if len(record) == 4:
            clean_index, target_id, candidate, prefix_vector = record
            prefix_vectors.append(np.asarray(prefix_vector, dtype=np.float32).reshape(-1))
        else:
            clean_index, target_id, candidate = record  # type: ignore[misc]
            prefix_vectors.append(np.zeros((0,), dtype=np.float32))
        record_list.append((int(clean_index), int(target_id), candidate))
    if not record_list:
        raise ValueError("Cannot write an empty target policy pool")

    clean_indices = np.asarray([clean for clean, _, _ in record_list], dtype=np.int64)
    target_ids = np.asarray([target_id for _, target_id, _ in record_list], dtype=np.int64)
    candidates = [candidate for _, _, candidate in record_list]
    max_actions = max((len(candidate.marginal_gain) for candidate in candidates), default=0)
    max_actions = max(max_actions, 1)

    arrays = {
        "x0_star": np.stack([candidate.x0_star for candidate in candidates]).astype(np.float32),
        "policy_allocation": np.stack([candidate.allocation for candidate in candidates]).astype(np.float32),
        "policy_counts": np.stack([candidate.counts for candidate in candidates]).astype(np.int32),
        "allowed_mask": np.stack([candidate.allowed_mask for candidate in candidates]).astype(np.float32),
        "family_weights": np.stack([candidate.family_weights for candidate in candidates]).astype(np.float32),
        "primitive_weights": np.stack([candidate.primitive_weights for candidate in candidates]).astype(np.float32),
        "effect_map": np.stack([candidate.effect_map for candidate in candidates]).astype(np.float32),
        "prefix_vector": _pad_float(prefix_vectors, max((len(row) for row in prefix_vectors), default=0)),
        "clean_index": clean_indices,
        "target_id": target_ids,
        "budget_ratio": np.asarray([candidate.budget_ratio for candidate in candidates], dtype=np.float32),
        "budget_count": np.asarray([candidate.budget_count for candidate in candidates], dtype=np.int32),
        "proxy_score_df": np.asarray([candidate.proxy_score_df for candidate in candidates], dtype=np.float32),
        "proxy_score_rf": np.asarray([candidate.proxy_score_rf for candidate in candidates], dtype=np.float32),
        "proxy_score_attack": np.asarray([candidate.proxy_score_attack for candidate in candidates], dtype=np.float32),
        "quality_score": np.asarray([candidate.quality_score for candidate in candidates], dtype=np.float32),
        "latency_cost": np.asarray([candidate.latency_cost for candidate in candidates], dtype=np.float32),
        "fallback_flag": np.asarray([candidate.fallback_flag for candidate in candidates], dtype=np.uint8),
        "family_indices": _pad_indices([candidate.family_indices for candidate in candidates], 5),
        "primitive_indices": _pad_indices([candidate.primitive_indices for candidate in candidates], 5),
        "action_rank": _pad_indices([candidate.action_rank.reshape(-1) for candidate in candidates], max_actions * 2),
        "marginal_gain": _pad_float([candidate.marginal_gain for candidate in candidates], max_actions),
        "construction_seed": np.asarray([candidate.construction_seed for candidate in candidates], dtype=np.int64),
    }
    np.savez_compressed(target.policies_path, **arrays)

    index_fields = [
        "row",
        "clean_index",
        "target_id",
        "budget_ratio",
        "budget_count",
        "quality_score",
        "proxy_score_df",
        "proxy_score_rf",
        "proxy_score_attack",
        "fallback_flag",
        "construction_seed",
    ]
    with target.index_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=index_fields)
        writer.writeheader()
        for row, (clean_index, target_id, candidate) in enumerate(record_list):
            writer.writerow(
                {
                    "row": row,
                    "clean_index": int(clean_index),
                    "target_id": int(target_id),
                    "budget_ratio": f"{float(candidate.budget_ratio):.6f}",
                    "budget_count": int(candidate.budget_count),
                    "quality_score": f"{float(candidate.quality_score):.8f}",
                    "proxy_score_df": f"{float(candidate.proxy_score_df):.8f}",
                    "proxy_score_rf": f"{float(candidate.proxy_score_rf):.8f}",
                    "proxy_score_attack": f"{float(candidate.proxy_score_attack):.8f}",
                    "fallback_flag": int(candidate.fallback_flag),
                    "construction_seed": int(candidate.construction_seed),
                }
            )

    reports = [candidate.constraint_report for candidate in candidates]
    summary = {
        **metadata,
        "pool_version": "target_policy_pool_v1",
        "storage_format": "npz_csv",
        "target_count": int(len(candidates)),
        "unique_clean_traces": int(np.unique(clean_indices).size),
        "budget_violation_count": int(sum(report.budget_error != 0 for report in reports)),
        "mask_violation_count": int(sum(report.allowed_violation_count != 0 for report in reports)),
        "negative_count": int(sum(report.negative_count != 0 for report in reports)),
        "fallback_count": int(sum(candidate.fallback_flag for candidate in candidates)),
        "mean_quality_score": float(np.mean([candidate.quality_score for candidate in candidates])),
        "score_source": str(metadata.get("score_source", "heuristic_proxy_not_df_rf_teacher")),
        "mean_proxy_score_df": float(np.mean([candidate.proxy_score_df for candidate in candidates])),
        "mean_proxy_score_rf": float(np.mean([candidate.proxy_score_rf for candidate in candidates])),
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
    }
    target.metadata_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (target.root / "build_summary_zh.md").write_text(_summary_markdown(summary), encoding="utf-8")
    return summary


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# target_policy_pool_v1 构建摘要",
        "",
        f"- target 数量: {summary.get('target_count')}",
        f"- 覆盖 clean trace 数量: {summary.get('unique_clean_traces')}",
        f"- budget violation: {summary.get('budget_violation_count')}",
        f"- mask violation: {summary.get('mask_violation_count')}",
        f"- fallback count: {summary.get('fallback_count')}",
        f"- mean quality score: {float(summary.get('mean_quality_score', 0.0)):.6f}",
        f"- score source: {summary.get('score_source')}",
        "",
        "注意: v1 存储使用 npz/csv，不向 diffusion loader 暴露真实类别标签。",
    ]
    return "\n".join(lines) + "\n"


def validate_target_policy_pool(root: str | Path) -> dict[str, Any]:
    pool = TargetPolicyPool(root)
    if not pool.exists():
        raise FileNotFoundError(f"Target policy pool is incomplete: {pool.root}")
    arrays = pool.load_arrays()
    required = {
        "x0_star",
        "policy_allocation",
        "policy_counts",
        "allowed_mask",
        "clean_index",
        "budget_count",
        "family_weights",
        "primitive_weights",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError(f"Missing required arrays: {missing}")
    counts = np.asarray(arrays["policy_counts"], dtype=np.int64)
    mask = np.asarray(arrays["allowed_mask"], dtype=np.float32)
    budgets = np.asarray(arrays["budget_count"], dtype=np.int64)
    actual = counts.sum(axis=(1, 2))
    mask_violation = np.maximum(counts, 0)[mask <= 0].sum()
    negative = np.abs(np.minimum(counts, 0)).sum()
    budget_errors = actual - budgets
    leaked_keys = sorted(key for key in arrays if _is_label_key(key))
    result = {
        "valid": bool(mask_violation == 0 and negative == 0 and np.all(budget_errors == 0) and not leaked_keys),
        "rows": int(counts.shape[0]),
        "mask_violation_count": int(mask_violation),
        "negative_count": int(negative),
        "budget_violation_count": int(np.count_nonzero(budget_errors)),
        "leaked_label_keys": leaked_keys,
        "x0_shape": list(np.asarray(arrays["x0_star"]).shape),
    }
    return result
