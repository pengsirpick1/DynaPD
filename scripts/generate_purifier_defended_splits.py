"""Generate defended train/validation splits for purifier training.

The script reuses an existing audited DMMPv3 defense run. It does not train
defense components or redo Stage 3 Pareto selection; it loads the frozen run
artifacts and selected budget/keep ratio, then renders defended variants for
the requested original clean splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.constraints.user_profiles import load_profiles  # noqa: E402
from dmmp.data import load_cw_data  # noqa: E402
from dmmp.diffusion.profile_pipeline import generate_v4_ragged_dataset  # noqa: E402
from dmmp.evaluation.profile_attacks import _defense_artifact_signature  # noqa: E402
from dmmp.utils import resolve_device, write_json  # noqa: E402
from dmmp.utils.config import DefenseConfig  # noqa: E402


DEFAULT_RUN_DIR = PROJECT_ROOT / "results" / "dmmpv3_rf_tam_shape_v2_fullcw_seed0_20260719_201036"


@dataclass(frozen=True)
class ShardPlan:
    split: str
    profile_id: str
    shard_index: int
    shard_count: int
    start_local: int
    end_local: int
    source_count: int
    variant_count: int
    output_path: str
    metrics_path: str


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config(run_dir: Path, device: str, progress: bool, log_every: int) -> DefenseConfig:
    payload = _read_json(run_dir / "run_config.json")
    values = {key: payload[key] for key in DefenseConfig.__dataclass_fields__ if key in payload}
    cfg = DefenseConfig(**values)
    cfg.stage = "3"
    cfg.run_name = run_dir.name
    cfg.output_dir = str(run_dir.parent)
    cfg.device = str(device)
    cfg.progress = bool(progress)
    cfg.log_every = int(log_every)
    return cfg


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    payload = _read_json(run_dir / "split_indices.json")
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _selected_budget_keep(run_dir: Path, cfg: DefenseConfig) -> tuple[float, float]:
    selected_path = run_dir / "stage3_guided_refinement" / "selected_policy.json"
    if selected_path.is_file():
        selected = _read_json(selected_path)
        return float(selected.get("budget", cfg.budget_values[-1])), float(selected.get("keep_ratio", 1.0))
    return float(cfg.budget_values[-1]), 1.0


def _cache_key(indices: np.ndarray) -> str:
    return hashlib.sha1(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()[:10]


def _split_profile_id(split: str) -> str:
    if split == "train":
        return "train_000"
    if split in {"validation", "val"}:
        return "validation_000"
    if split == "test":
        return "test_000"
    raise ValueError(f"Unsupported purifier split {split!r}")


def _normal_split_name(split: str) -> str:
    return "validation" if split == "val" else str(split)


def _split_indices_key(split: str) -> str:
    return "val" if split == "validation" else str(split)


def _artifact_digest(run_dir: Path) -> str:
    return str(_defense_artifact_signature(run_dir).get("digest", "unknown"))


def _make_plan(
    run_dir: Path,
    splits: dict[str, np.ndarray],
    requested_splits: list[str],
    shard_size: int,
    budget: float,
    keep_ratio: float,
    digest: str,
    repeat_count: int,
) -> list[ShardPlan]:
    plans: list[ShardPlan] = []
    for requested in requested_splits:
        split = _normal_split_name(requested)
        split_key = _split_indices_key(split)
        if split_key not in splits:
            raise ValueError(f"Split {split!r} is not present in split_indices.json")
        indices = np.asarray(splits[split_key], dtype=np.int64)
        profile_id = _split_profile_id(split)
        size = int(shard_size) if int(shard_size) > 0 else len(indices)
        shard_count = int(np.ceil(len(indices) / max(size, 1)))
        for shard_index in range(shard_count):
            start = shard_index * size
            end = min((shard_index + 1) * size, len(indices))
            shard_indices = indices[start:end]
            key = _cache_key(shard_indices)
            output_dir = run_dir / "defended_datasets" / "purifier" / split / profile_id
            name = (
                f"purifier_{split}_b{budget:.2f}_k{keep_ratio:.2f}_"
                f"shard{shard_index:04d}-of-{shard_count:04d}_"
                f"n{len(shard_indices)}_{key}_d{digest}.npz"
            )
            output = output_dir / name
            plans.append(
                ShardPlan(
                    split=split,
                    profile_id=profile_id,
                    shard_index=shard_index,
                    shard_count=shard_count,
                    start_local=start,
                    end_local=end,
                    source_count=len(shard_indices),
                    variant_count=len(shard_indices) * repeat_count,
                    output_path=str(output),
                    metrics_path=str(output.with_name(output.stem + "_metrics.json")),
                )
            )
    return plans


def _write_plan(run_dir: Path, plans: list[ShardPlan], args: argparse.Namespace, cfg: DefenseConfig, budget: float, keep_ratio: float) -> Path:
    path = run_dir / "manifests" / "purifier_defended_generation_plan.json"
    payload = {
        "run_dir": str(run_dir.resolve()),
        "source": "generate_purifier_defended_splits.py",
        "frozen_defense": True,
        "reuse_stage3_selected_policy": True,
        "budget": float(budget),
        "keep_ratio": float(keep_ratio),
        "deployment_repeats": int(cfg.deployment_repeats),
        "shard_size": int(args.shard_size),
        "requested_splits": [_normal_split_name(item) for item in str(args.splits).split(",") if item.strip()],
        "plans": [asdict(plan) for plan in plans],
    }
    write_json(path, payload)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate DMMPv3 defended train/validation shards for purifier training.")
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR), help="Existing audited DMMPv3 run directory.")
    parser.add_argument("--splits", default="train,validation", help="Comma-separated source splits to generate.")
    parser.add_argument("--shard-size", type=int, default=10000, help="Clean sources per shard; <=0 writes one file per split.")
    parser.add_argument("--device", default="auto", help="Device passed to DMMPv3 model loading/generation.")
    parser.add_argument("--only-missing", action=argparse.BooleanOptionalAction, default=True, help="Skip shards whose NPZ and metrics already exist.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing shards.")
    parser.add_argument("--plan-only", action="store_true", help="Only write the deterministic shard plan.")
    parser.add_argument("--max-shards", type=int, default=0, help="Debug guard: generate at most this many shards after planning.")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    cfg = _load_config(run_dir, str(args.device), bool(args.progress), int(args.log_every))
    splits = _load_splits(run_dir)
    budget, keep_ratio = _selected_budget_keep(run_dir, cfg)
    digest = _artifact_digest(run_dir)
    requested_splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    plans = _make_plan(
        run_dir,
        splits,
        requested_splits,
        int(args.shard_size),
        budget,
        keep_ratio,
        digest,
        int(cfg.deployment_repeats),
    )
    plan_path = _write_plan(run_dir, plans, args, cfg, budget, keep_ratio)
    print(f"[purifier generation] plan written: {plan_path}")
    print(f"[purifier generation] planned shards={len(plans)}, planned variants={sum(plan.variant_count for plan in plans)}")
    if bool(args.plan_only):
        return

    raw, labels, trace_ids, _, _ = load_cw_data(cfg)
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    device = resolve_device(str(args.device))
    generated = 0
    for plan in plans:
        if int(args.max_shards) > 0 and generated >= int(args.max_shards):
            print(f"[purifier generation] max-shards={args.max_shards} reached; remaining shards not generated")
            break
        output_path = Path(plan.output_path)
        metrics_path = Path(plan.metrics_path)
        if bool(args.only_missing) and not bool(args.force) and output_path.is_file() and metrics_path.is_file():
            print(f"[purifier generation] skip existing shard: {output_path}")
            continue
        split_key = _split_indices_key(plan.split)
        source_indices = np.asarray(splits[split_key], dtype=np.int64)[plan.start_local : plan.end_local]
        profile = next(row for row in profiles[plan.split] if row.profile_id == plan.profile_id)
        print(
            f"[purifier generation] split={plan.split} shard={plan.shard_index + 1}/{plan.shard_count} "
            f"sources={len(source_indices)} variants={len(source_indices) * int(cfg.deployment_repeats)} output={output_path}",
            flush=True,
        )
        generate_v4_ragged_dataset(
            raw,
            labels,
            trace_ids,
            source_indices,
            run_dir,
            cfg,
            profile=profile,
            visit_namespace=f"purifier_{plan.split}:{plan.profile_id}:shard{plan.shard_index:04d}",
            budget=float(budget),
            keep_ratio=float(keep_ratio),
            output_npz=output_path,
            device=device,
        )
        generated += 1
    print(f"[purifier generation] generated_shards={generated}")


if __name__ == "__main__":
    main()
