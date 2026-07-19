"""Adaptive re-defense audit for DMMPv3.

Protocol:
1. Train a full clean-base DF attacker A0.
2. Fine-tune A0 on a few-shot old defended support set to obtain A1.
3. Generate new defense candidates after seeing A1, select the candidate that
   minimizes A1 accuracy on a held-out defense-selection split.
4. Generate a full fresh defended test set with the selected new strategy and
   evaluate A1 on it.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.constraints.user_profiles import load_profiles
from dmmp.data import load_cw_data
from dmmp.evaluation.profile_attacks import (
    _defended_input,
    _defense_artifact_signature,
    _defense_config_from_run,
    _find_profile,
    _get_profile_dataset,
    _load_splits,
    _selected_budget_and_keep,
)
from dmmp.projection.padding import load_ragged_npz
from dmmp.utils import log, resolve_device, set_seed, write_csv, write_json
from scripts.few_shot_adaptive_df import (
    _base_quality_report,
    _class_counts,
    _df_input_indexed,
    _filter_indices,
    _finetune_df,
    _load_base_model,
    _make_attack_cfg,
    _ragged_overhead,
    _sample_base_indices_per_class,
    _select_classes,
    _select_unique_clean_rows_per_class,
    _subsample_absolute,
    _train_base_model,
    _unique_clean_counts,
    _evaluate_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Few-shot adaptive re-defense audit: fine-tune DF on old defended support, "
            "then select/generate a new defense strategy against that adapted DF."
        )
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="", help="Defaults to <run_dir>/attack_eval/adaptive_redefense_df.")
    parser.add_argument("--base_checkpoint", default="")
    parser.add_argument("--target_profile_id", default="")
    parser.add_argument("--profile_split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--profile_index", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0, help="Debug class cap; 0 means all run classes.")
    parser.add_argument("--max_samples", type=int, default=0, help="Legacy debug cap; 0 means no cap.")
    parser.add_argument("--base_max_train_traces", type=int, default=0)
    parser.add_argument("--base_max_val_traces", type=int, default=0)
    parser.add_argument("--fresh_max_test_traces", type=int, default=0)
    parser.add_argument("--selection_max_traces", type=int, default=0, help="Defense-selection clean trace cap; 0 means full split.")
    parser.add_argument("--selection_split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--few_shot_per_class", type=int, default=20)
    parser.add_argument("--candidate_count", type=int, default=8, help="Number of fresh namespaces per budget/keep pair.")
    parser.add_argument("--candidate_budgets", default="", help="Comma-separated budgets; default uses selected_policy budget.")
    parser.add_argument("--candidate_keep_ratios", default="", help="Comma-separated keep ratios; default uses selected_policy keep ratio.")
    parser.add_argument("--candidate_overhead_weight", type=float, default=0.0)
    parser.add_argument("--base_epochs", type=int, default=20)
    parser.add_argument("--base_patience", type=int, default=8)
    parser.add_argument("--base_lr", type=float, default=2e-3)
    parser.add_argument("--base_min_val_accuracy", type=float, default=0.70)
    parser.add_argument("--base_min_clean_accuracy", type=float, default=0.70)
    parser.add_argument("--require_qualified_base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--finetune_epochs", type=int, default=5)
    parser.add_argument("--finetune_lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--df_architecture", choices=["project", "wflib"], default="project")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--force_retrain_base", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_surrogate_base_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--trust_base_checkpoint", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def _parse_float_list(text: str, default: list[float]) -> list[float]:
    if not str(text).strip():
        return list(default)
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError(f"Empty float list: {text!r}")
    return values


def _safe_float_label(value: float) -> str:
    return f"{float(value):.4f}".rstrip("0").rstrip(".").replace("-", "m").replace(".", "p")


def _selection_indices(splits: dict[str, np.ndarray], split_name: str) -> np.ndarray:
    key = "val" if str(split_name) == "val" else str(split_name)
    if key not in splits:
        raise ValueError(f"Unknown selection split {split_name!r}; available keys={sorted(splits)}")
    return np.asarray(splits[key], dtype=np.int64)


def _load_clean_indices(path: Path) -> np.ndarray:
    _, _, metadata = load_ragged_npz(path)
    return np.asarray(metadata.get("clean_index", []), dtype=np.int64)


def _prepare_old_support(
    *,
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    train_pool: np.ndarray,
    selected_classes: np.ndarray,
    run_dir: Path,
    cfg,
    target_profile,
    budget: float,
    keep_ratio: float,
    defense_signature: dict[str, Any],
    device: torch.device,
    args: argparse.Namespace,
):
    adaptation_base_idx = _sample_base_indices_per_class(
        train_pool,
        labels,
        selected_classes,
        int(args.few_shot_per_class),
        int(args.seed) + 101,
        purpose="few-shot adaptation",
    )
    old_role = f"ard_old_s{int(args.seed)}_k{int(args.few_shot_per_class)}"
    old_traces_all, old_origins_all, old_y_all, old_metrics, old_path = _get_profile_dataset(
        raw,
        labels,
        trace_ids,
        adaptation_base_idx,
        run_dir,
        cfg,
        target_profile,
        old_role,
        float(budget),
        float(keep_ratio),
        defense_signature,
        device,
    )
    old_clean_indices_all = _load_clean_indices(old_path)
    old_traces, old_origins, old_y, old_rows = _select_unique_clean_rows_per_class(
        old_traces_all,
        old_origins_all,
        old_y_all,
        old_clean_indices_all,
        selected_classes,
        int(args.few_shot_per_class),
        int(args.seed) + 301,
        purpose="adaptation",
    )
    old_clean_idx = old_clean_indices_all[np.asarray(old_rows, dtype=np.int64)]
    old_defended_x, old_adapter_stats = _defended_input("df", old_traces, old_origins, cfg, args.attack_cfg)
    return {
        "role": old_role,
        "path": old_path,
        "metrics": old_metrics,
        "adapter_stats": old_adapter_stats,
        "base_indices": adaptation_base_idx,
        "selected_rows": old_rows,
        "selected_clean_indices": old_clean_idx,
        "traces": old_traces,
        "origins": old_origins,
        "y": old_y,
        "x": old_defended_x,
        "overhead": _ragged_overhead(old_traces, old_origins, int(cfg.max_trace_length)),
    }


def _generate_eval_dataset(
    *,
    raw: np.ndarray,
    labels: np.ndarray,
    trace_ids: np.ndarray,
    indices: np.ndarray,
    run_dir: Path,
    cfg,
    target_profile,
    role: str,
    budget: float,
    keep_ratio: float,
    defense_signature: dict[str, Any],
    device: torch.device,
    attack_cfg,
):
    traces, origins, y, metrics, path = _get_profile_dataset(
        raw,
        labels,
        trace_ids,
        indices,
        run_dir,
        cfg,
        target_profile,
        role,
        float(budget),
        float(keep_ratio),
        defense_signature,
        device,
    )
    x, adapter_stats = _defended_input("df", traces, origins, cfg, attack_cfg)
    clean_indices = _load_clean_indices(path)
    return {
        "role": role,
        "path": path,
        "metrics": metrics,
        "adapter_stats": adapter_stats,
        "traces": traces,
        "origins": origins,
        "y": np.asarray(y, dtype=np.int64),
        "x": x,
        "clean_indices": clean_indices,
        "overhead": _ragged_overhead(traces, origins, int(cfg.max_trace_length)),
    }


def main() -> None:
    args = parse_args()
    started_at = time.perf_counter()
    set_seed(int(args.seed))
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if str(args.output_dir).strip() else run_dir / "attack_eval" / "adaptive_redefense_df"
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_cfg = _make_attack_cfg(args, output_dir)
    args.attack_cfg = attack_cfg
    cfg = _defense_config_from_run(run_dir, attack_cfg)
    cfg.progress = bool(args.progress)
    cfg.device = str(args.device)
    cfg.batch_size = min(int(cfg.batch_size), max(1, int(args.batch_size)))
    device = resolve_device(str(args.device))

    log(f"[adaptive re-defense] loading run/data: run_dir={run_dir}, device={device}", args.progress)
    raw, labels, trace_ids, _, data_source = load_cw_data(cfg)
    splits = _load_splits(run_dir)
    selected_classes = _select_classes(labels, splits["train"], int(args.max_classes))
    train_pool = _filter_indices(splits["train"], labels, selected_classes)
    val_pool = _filter_indices(splits["val"], labels, selected_classes)
    test_pool = _filter_indices(splits["test"], labels, selected_classes)
    selection_pool = _filter_indices(_selection_indices(splits, str(args.selection_split)), labels, selected_classes)

    base_train_cap = int(args.base_max_train_traces) if int(args.base_max_train_traces) > 0 else int(args.max_samples)
    base_val_cap = int(args.base_max_val_traces) if int(args.base_max_val_traces) > 0 else (
        max(int(args.max_samples) // 4, len(selected_classes)) if int(args.max_samples) > 0 else 0
    )
    fresh_test_cap = int(args.fresh_max_test_traces) if int(args.fresh_max_test_traces) > 0 else int(args.max_samples)
    base_train_idx = _subsample_absolute(train_pool, labels, base_train_cap, int(args.seed) + 11, required_classes=selected_classes)
    base_val_idx = _subsample_absolute(val_pool, labels, base_val_cap, int(args.seed) + 12, required_classes=selected_classes)
    fresh_test_idx = _subsample_absolute(test_pool, labels, fresh_test_cap, int(args.seed) + 202, required_classes=selected_classes)
    selection_idx = _subsample_absolute(selection_pool, labels, int(args.selection_max_traces), int(args.seed) + 303, required_classes=selected_classes)

    if len(base_train_idx) == 0 or len(base_val_idx) == 0 or len(fresh_test_idx) == 0 or len(selection_idx) == 0:
        raise ValueError("A required split is empty after filtering/capping")

    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    target_profile = _find_profile(profiles, str(args.target_profile_id), str(args.profile_split), int(args.profile_index))
    selected_budget, selected_keep_ratio = _selected_budget_and_keep(run_dir, cfg)
    candidate_budgets = _parse_float_list(str(args.candidate_budgets), [float(selected_budget)])
    candidate_keep_ratios = _parse_float_list(str(args.candidate_keep_ratios), [float(selected_keep_ratio)])
    defense_signature = _defense_artifact_signature(run_dir)
    repeat_count = max(1, int(getattr(cfg, "deployment_repeats", 1)))

    old_support = _prepare_old_support(
        raw=raw,
        labels=labels,
        trace_ids=trace_ids,
        train_pool=train_pool,
        selected_classes=selected_classes,
        run_dir=run_dir,
        cfg=cfg,
        target_profile=target_profile,
        budget=float(selected_budget),
        keep_ratio=float(selected_keep_ratio),
        defense_signature=defense_signature,
        device=device,
        args=args,
    )
    if len(np.intersect1d(np.unique(old_support["selected_clean_indices"]), np.unique(fresh_test_idx))):
        raise RuntimeError("Old support and fresh test clean traces overlap")

    base_model, attacker_classes, base_info = _load_base_model(run_dir, args, cfg, selected_classes, device)
    if base_model is None or attacker_classes is None:
        base_model, attacker_classes, base_info = _train_base_model(
            raw,
            labels,
            base_train_idx,
            base_val_idx,
            cfg,
            attack_cfg,
            output_dir,
            device,
            args,
        )

    clean_test_x = _df_input_indexed(raw, fresh_test_idx, int(cfg.max_trace_length))
    clean_test_y = np.asarray(labels, dtype=np.int64)[fresh_test_idx]
    clean_base_metrics = _evaluate_df(base_model, clean_test_x, clean_test_y, attacker_classes, device, int(args.batch_size))
    base_quality = _base_quality_report(base_info, clean_base_metrics, args)
    if bool(args.require_qualified_base) and not bool(base_quality["qualified"]):
        write_json(
            output_dir / "invalid_base_df_result.json",
            {
                "status": "invalid_base_df",
                "base_quality": base_quality,
                "base_info": base_info,
                "split_protocol": {
                    "base_train_count": int(len(base_train_idx)),
                    "base_val_count": int(len(base_val_idx)),
                    "fresh_test_count": int(len(fresh_test_idx)),
                    "selection_count": int(len(selection_idx)),
                },
            },
        )
        raise RuntimeError(
            "Clean-base DF is not qualified for adaptive re-defense audit: "
            f"val_acc={base_quality['base_best_val_accuracy']}, clean_acc={base_quality['clean_base_accuracy']:.4f}"
        )

    log(
        f"[adaptive re-defense] fine-tuning attacker on old support only: samples={len(old_support['y'])}",
        args.progress,
    )
    adapted_model, finetune_history = _finetune_df(
        base_model,
        attacker_classes,
        old_support["x"],
        old_support["y"],
        epochs=int(args.finetune_epochs),
        lr=float(args.finetune_lr),
        batch_size=int(args.batch_size),
        device=device,
        seed=int(args.seed) + 401,
        progress=bool(args.progress),
        log_every=int(args.log_every),
    )
    before_old_metrics = _evaluate_df(base_model, old_support["x"], old_support["y"], attacker_classes, device, int(args.batch_size))
    after_old_metrics = _evaluate_df(adapted_model, old_support["x"], old_support["y"], attacker_classes, device, int(args.batch_size))
    after_clean_metrics = _evaluate_df(adapted_model, clean_test_x, clean_test_y, attacker_classes, device, int(args.batch_size))

    attacker_checkpoint = output_dir / "adaptive_attacker_finetuned_df.pt"
    torch.save(
        {
            "model_state": adapted_model.state_dict(),
            "classes": attacker_classes,
            "selected_probe_classes": selected_classes,
            "base_info": base_info,
            "base_quality": base_quality,
            "finetune_history": finetune_history,
            "df_architecture": str(base_info.get("df_architecture", args.df_architecture)),
        },
        attacker_checkpoint,
    )

    baseline_role = f"ard_base_s{int(args.seed)}_n{len(fresh_test_idx)}"
    baseline_test = _generate_eval_dataset(
        raw=raw,
        labels=labels,
        trace_ids=trace_ids,
        indices=fresh_test_idx,
        run_dir=run_dir,
        cfg=cfg,
        target_profile=target_profile,
        role=baseline_role,
        budget=float(selected_budget),
        keep_ratio=float(selected_keep_ratio),
        defense_signature=defense_signature,
        device=device,
        attack_cfg=attack_cfg,
    )
    baseline_metrics = _evaluate_df(adapted_model, baseline_test["x"], baseline_test["y"], attacker_classes, device, int(args.batch_size))

    log(
        f"[adaptive re-defense] selecting new defense strategy with A1 on {len(selection_idx)} clean traces...",
        args.progress,
    )
    candidate_rows: list[dict[str, Any]] = []
    candidate_details: list[dict[str, Any]] = []
    candidate_serial = 0
    for budget in candidate_budgets:
        for keep_ratio in candidate_keep_ratios:
            for repeat_index in range(max(1, int(args.candidate_count))):
                candidate_serial += 1
                role = f"ard_sel_s{int(args.seed)}_c{candidate_serial:03d}_b{_safe_float_label(budget)}_k{_safe_float_label(keep_ratio)}"
                dataset = _generate_eval_dataset(
                    raw=raw,
                    labels=labels,
                    trace_ids=trace_ids,
                    indices=selection_idx,
                    run_dir=run_dir,
                    cfg=cfg,
                    target_profile=target_profile,
                    role=role,
                    budget=float(budget),
                    keep_ratio=float(keep_ratio),
                    defense_signature=defense_signature,
                    device=device,
                    attack_cfg=attack_cfg,
                )
                metrics = _evaluate_df(adapted_model, dataset["x"], dataset["y"], attacker_classes, device, int(args.batch_size))
                visible_overhead = float(dataset["overhead"]["visible_dummy_overhead"])
                score = float(metrics["accuracy"]) + float(args.candidate_overhead_weight) * visible_overhead
                row = {
                    "candidate_id": candidate_serial,
                    "role": role,
                    "budget": float(budget),
                    "keep_ratio": float(keep_ratio),
                    "selection_accuracy": float(metrics["accuracy"]),
                    "selection_true_label_confidence": float(metrics["true_label_confidence"]),
                    "selection_entropy": float(metrics["prediction_entropy"]),
                    "visible_dummy_overhead": visible_overhead,
                    "score": score,
                    "dataset_path": str(dataset["path"]),
                }
                candidate_rows.append(row)
                candidate_details.append({"row": row, "metrics": metrics, "dataset": {
                    "generation_metrics": dataset["metrics"],
                    "adapter_stats": dataset["adapter_stats"],
                    "overhead": dataset["overhead"],
                }})
                log(
                    f"[adaptive re-defense candidate] id={candidate_serial}, acc={row['selection_accuracy']:.4f}, "
                    f"overhead={visible_overhead:.4f}, score={score:.4f}, role={role}",
                    args.progress,
                )

    selected_row = min(candidate_rows, key=lambda row: (float(row["score"]), float(row["selection_accuracy"])))
    selected_role = f"ard_new_c{int(selected_row['candidate_id']):03d}_s{int(args.seed)}_n{len(fresh_test_idx)}"
    selected_test = _generate_eval_dataset(
        raw=raw,
        labels=labels,
        trace_ids=trace_ids,
        indices=fresh_test_idx,
        run_dir=run_dir,
        cfg=cfg,
        target_profile=target_profile,
        role=selected_role,
        budget=float(selected_row["budget"]),
        keep_ratio=float(selected_row["keep_ratio"]),
        defense_signature=defense_signature,
        device=device,
        attack_cfg=attack_cfg,
    )
    selected_test_metrics = _evaluate_df(adapted_model, selected_test["x"], selected_test["y"], attacker_classes, device, int(args.batch_size))

    selected_policy = {
        "candidate_id": int(selected_row["candidate_id"]),
        "selection_role": str(selected_row["role"]),
        "test_role": selected_role,
        "budget": float(selected_row["budget"]),
        "keep_ratio": float(selected_row["keep_ratio"]),
        "selection_score": float(selected_row["score"]),
        "selection_accuracy": float(selected_row["selection_accuracy"]),
        "selected_by": "min adapted DF accuracy on defense-selection split",
    }
    write_json(output_dir / "adaptive_redefense_selected_policy.json", selected_policy)

    per_class_rows = []
    baseline_by_class = {int(row["class"]): row for row in baseline_metrics.get("per_class", [])}
    selected_by_class = {int(row["class"]): row for row in selected_test_metrics.get("per_class", [])}
    for label in selected_classes.astype(int).tolist():
        base_row = baseline_by_class.get(int(label), {})
        selected_cls_row = selected_by_class.get(int(label), {})
        per_class_rows.append(
            {
                "class": int(label),
                "old_support_count": int(_class_counts(old_support["y"], selected_classes).get(str(int(label)), 0)),
                "fresh_test_count": int(_class_counts(selected_test["y"], selected_classes).get(str(int(label)), 0)),
                "baseline_fresh_accuracy": float(base_row.get("accuracy", 0.0)),
                "adaptive_redefense_fresh_accuracy": float(selected_cls_row.get("accuracy", 0.0)),
            }
        )

    summary_row = {
        "run_dir": str(run_dir),
        "profile_id": target_profile.profile_id,
        "seed": int(args.seed),
        "selected_classes": int(len(selected_classes)),
        "train_pool_count": int(len(train_pool)),
        "val_pool_count": int(len(val_pool)),
        "test_pool_count": int(len(test_pool)),
        "base_train_count": int(len(base_train_idx)),
        "base_val_count": int(len(base_val_idx)),
        "selection_clean_trace_count": int(len(selection_idx)),
        "fresh_test_clean_trace_count": int(len(fresh_test_idx)),
        "fresh_test_defended_rows": int(len(selected_test["y"])),
        "few_shot_per_class": int(args.few_shot_per_class),
        "old_support_samples": int(len(old_support["y"])),
        "base_clean_accuracy": float(clean_base_metrics["accuracy"]),
        "base_best_val_accuracy": float(base_quality["base_best_val_accuracy"] or 0.0),
        "before_finetune_old_support_accuracy": float(before_old_metrics["accuracy"]),
        "after_finetune_old_support_accuracy": float(after_old_metrics["accuracy"]),
        "after_finetune_clean_accuracy": float(after_clean_metrics["accuracy"]),
        "adapted_baseline_fresh_defended_accuracy": float(baseline_metrics["accuracy"]),
        "adaptive_redefense_selection_accuracy": float(selected_row["selection_accuracy"]),
        "adaptive_redefense_fresh_defended_accuracy": float(selected_test_metrics["accuracy"]),
        "baseline_fresh_visible_dummy_overhead": float(baseline_test["overhead"]["visible_dummy_overhead"]),
        "adaptive_redefense_visible_dummy_overhead": float(selected_test["overhead"]["visible_dummy_overhead"]),
        "selected_candidate_id": int(selected_row["candidate_id"]),
        "selected_budget": float(selected_row["budget"]),
        "selected_keep_ratio": float(selected_row["keep_ratio"]),
        "attacker_checkpoint": str(attacker_checkpoint),
        "baseline_dataset": str(baseline_test["path"]),
        "adaptive_redefense_dataset": str(selected_test["path"]),
        "selected_clean_trace_overlap": int(
            len(np.intersect1d(np.unique(old_support["selected_clean_indices"]), np.unique(selected_test["clean_indices"])))
        ),
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }
    summary = {
        "protocol": "few_shot_adaptive_redefense_df",
        "question": (
            "After a DF attacker adapts to old defended support, can DMMPv3 select/generate "
            "a new defense strategy that reduces the adapted attacker's full fresh-test accuracy?"
        ),
        "data_source": data_source,
        "run_dir": str(run_dir),
        "output_dir": str(output_dir),
        "device": str(device),
        "deployment_repeats": int(repeat_count),
        "selected_probe_classes": selected_classes.astype(int).tolist(),
        "attacker_classes": attacker_classes.astype(int).tolist(),
        "base_info": base_info,
        "base_quality": base_quality,
        "split_protocol": {
            "base_train_full_selected_split": bool(len(base_train_idx) == len(train_pool)),
            "base_val_full_selected_split": bool(len(base_val_idx) == len(val_pool)),
            "fresh_test_full_selected_split": bool(len(fresh_test_idx) == len(test_pool)),
            "selection_split": str(args.selection_split),
            "selection_count": int(len(selection_idx)),
            "fresh_test_count": int(len(fresh_test_idx)),
        },
        "old_support": {
            "path": str(old_support["path"]),
            "role": old_support["role"],
            "base_clean_indices": old_support["base_indices"].astype(int).tolist(),
            "selected_clean_indices": old_support["selected_clean_indices"].astype(int).tolist(),
            "selected_generated_rows": old_support["selected_rows"].astype(int).tolist(),
            "class_counts": _class_counts(old_support["y"], selected_classes),
            "unique_clean_trace_class_counts": _unique_clean_counts(old_support["selected_clean_indices"], labels, selected_classes),
            "generation_metrics": old_support["metrics"],
            "adapter_stats": old_support["adapter_stats"],
            "overhead": old_support["overhead"],
        },
        "baseline_fresh_test": {
            "path": str(baseline_test["path"]),
            "role": baseline_test["role"],
            "metrics": baseline_metrics,
            "generation_metrics": baseline_test["metrics"],
            "adapter_stats": baseline_test["adapter_stats"],
            "overhead": baseline_test["overhead"],
        },
        "candidate_selection": {
            "budgets": [float(value) for value in candidate_budgets],
            "keep_ratios": [float(value) for value in candidate_keep_ratios],
            "candidate_count_per_pair": int(args.candidate_count),
            "overhead_weight": float(args.candidate_overhead_weight),
            "selected_policy": selected_policy,
            "rows": candidate_rows,
            "details": candidate_details,
        },
        "adaptive_redefense_fresh_test": {
            "path": str(selected_test["path"]),
            "role": selected_test["role"],
            "metrics": selected_test_metrics,
            "generation_metrics": selected_test["metrics"],
            "adapter_stats": selected_test["adapter_stats"],
            "overhead": selected_test["overhead"],
        },
        "metrics": {
            "clean_base": clean_base_metrics,
            "before_finetune_old_support": before_old_metrics,
            "after_finetune_old_support": after_old_metrics,
            "after_finetune_clean": after_clean_metrics,
            "adapted_baseline_fresh_defended": baseline_metrics,
            "adaptive_redefense_fresh_defended": selected_test_metrics,
        },
        "finetune_history": finetune_history,
        "summary_row": summary_row,
        "per_class_rows": per_class_rows,
    }
    write_json(output_dir / "adaptive_redefense_df_summary.json", summary)
    write_csv(output_dir / "adaptive_redefense_df_summary.csv", [summary_row])
    write_csv(output_dir / "adaptive_redefense_df_candidates.csv", candidate_rows)
    write_csv(output_dir / "adaptive_redefense_df_per_class.csv", per_class_rows)
    config_payload = {key: value for key, value in vars(args).items() if key != "attack_cfg"}
    write_json(output_dir / "adaptive_redefense_df_config.json", config_payload)
    log(
        f"[adaptive re-defense result] base_clean={summary_row['base_clean_accuracy']:.4f}, "
        f"baseline_A1_fresh={summary_row['adapted_baseline_fresh_defended_accuracy']:.4f}, "
        f"redefense_A1_fresh={summary_row['adaptive_redefense_fresh_defended_accuracy']:.4f}, "
        f"selected_candidate={summary_row['selected_candidate_id']}, saved={output_dir}",
        True,
    )


if __name__ == "__main__":
    main()
