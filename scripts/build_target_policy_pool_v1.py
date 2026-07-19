from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data.cw import load_cw_data
from dmmp.encoders.prefix import extract_prefix_condition
from dmmp.target_policy.candidate_generator import generate_candidates_for_trace
from dmmp.target_policy.config import load_target_policy_config
from dmmp.target_policy.target_pool import write_target_policy_pool
from dmmp.target_policy.target_selector import select_targets
from dmmp.utils.config import DefenseConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build class-agnostic x0* target policy pool v1.")
    parser.add_argument("--config", default="configs/x0_target_diffusion_v1.yaml")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_traces", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _default_output_dir(smoke: bool) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "target_policy_pool_v1_smoke" if smoke else "target_policy_pool_v1"
    return PROJECT_ROOT / "results" / f"{stamp}_{suffix}"


def _resolve_output_dir(value: str, *, smoke: bool, overwrite: bool) -> Path:
    output = Path(value) if value else _default_output_dir(smoke)
    sentinel_names = {"policies.npz", "metadata.json", "index.csv", "trace_summary.json"}
    existing = [name for name in sentinel_names if (output / name).exists()]
    if existing and not bool(overwrite):
        raise SystemExit(
            f"Refusing to overwrite existing target-pool artifacts in {output}. "
            f"Existing files: {', '.join(sorted(existing))}. Pass --overwrite to replace them."
        )
    return output


def main() -> None:
    args = parse_args()
    cfg = load_target_policy_config(args.config)
    seed = int(cfg.seed if args.seed is None else args.seed)
    output_dir = _resolve_output_dir(args.output_dir, smoke=bool(args.smoke), overwrite=bool(args.overwrite))
    if args.smoke:
        cfg.num_candidates = min(int(cfg.num_candidates), 4)
        cfg.target_count = min(int(cfg.target_count), 2)
        cfg.quality_target_count = min(int(cfg.quality_target_count), 1)
        cfg.diverse_target_count = min(int(cfg.diverse_target_count), 1)
        args.max_traces = args.max_traces or 4
        args.max_samples = args.max_samples or 80
        args.max_classes = args.max_classes or 4

    data_cfg = DefenseConfig(
        data_root=args.data_root or DefenseConfig.data_root,
        seed=seed,
        max_samples=int(args.max_samples),
        max_classes=int(args.max_classes),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        prefix_n=int(cfg.prefix_length),
        patch_num=int(cfg.strategy_horizon),
    )
    raw, labels, trace_ids, splits, data_path = load_cw_data(data_cfg)
    del labels
    train_indices = np.asarray(splits["train"], dtype=np.int64)
    if int(args.max_traces) > 0:
        train_indices = train_indices[: int(args.max_traces)]
    rng = np.random.default_rng(seed)
    records = []
    per_trace_summaries = []
    next_target_id = 0
    for ordinal, clean_index in enumerate(train_indices.tolist()):
        clean = raw[int(clean_index)]
        condition = extract_prefix_condition(
            clean,
            prefix_n=int(cfg.prefix_length),
            patch_num=int(cfg.strategy_horizon),
        )
        candidates = generate_candidates_for_trace(
            clean,
            cfg=cfg,
            prefix_condition=condition,
            clean_index=int(clean_index),
            rng=rng,
        )
        selected_total = 0
        fallback_total = 0
        for budget in cfg.budgets:
            group = [candidate for candidate in candidates if abs(candidate.budget_ratio - float(budget)) < 1.0e-8]
            selected, fallback_count = select_targets(
                group,
                target_count=int(cfg.target_count),
                quality_target_count=int(cfg.quality_target_count),
                diverse_target_count=int(cfg.diverse_target_count),
                allocation_l1_weight=float(cfg.allocation_l1_weight),
                allocation_cosine_weight=float(cfg.allocation_cosine_weight),
            )
            fallback_total += int(fallback_count)
            for candidate in selected:
                records.append((int(clean_index), next_target_id, candidate, condition.vector))
                next_target_id += 1
            selected_total += len(selected)
        per_trace_summaries.append(
            {
                "ordinal": int(ordinal),
                "clean_index": int(clean_index),
                "trace_id": str(trace_ids[int(clean_index)]),
                "selected_targets": int(selected_total),
                "fallback_count": int(fallback_total),
            }
        )

    metadata = {
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "data_path": str(data_path),
        "split": "train",
        "seed": seed,
        "max_traces": int(args.max_traces),
        "train_trace_count": int(len(train_indices)),
        "budgets": [float(value) for value in cfg.budgets],
        "num_candidates": int(cfg.num_candidates),
        "target_count_per_budget": int(cfg.target_count),
        "label_policy": "labels used only for train/val/test split; target-pool arrays do not store labels",
        "score_source": "heuristic_proxy_not_df_rf_teacher",
    }
    summary = write_target_policy_pool(output_dir, records, metadata=metadata)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trace_summary.json").write_text(
        json.dumps(per_trace_summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
