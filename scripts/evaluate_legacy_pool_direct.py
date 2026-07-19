from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data import load_cw_data
from dmmp.encoders.prefix import extract_prefix_condition
from dmmp.projection.padding import PaddingTemplate, normalized_template_entropy, project_policy_to_template
from dmmp.projection.padding import render_batch_variable, renderer_options_from_config
from dmmp.utils import resolve_device
from dmmp.utils.config import DefenseConfig

from evaluate_target_policy_direct_v1 import (
    _accumulate_metrics,
    _attack_metric_row,
    _dataset_config,
    _defense_config_from_run,
    _apply_renderer_overrides,
    _evaluate_attack_checkpoint,
    _evaluate_attack_checkpoint_from_traces,
    _finalize_metrics,
    _find_attack_checkpoint,
    _load_attack_checkpoint,
    _new_metric_totals,
)


LEGACY_MODES = (
    "gap-adaptive-padding",
    "burst-obfuscation",
    "direction-regularization",
    "rate-smoothing",
    "public-prototype-shaping",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate legacy five-pool direct templates without diffusion training or sampling."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument("--full_dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--max_test_traces", type=int, default=0)
    parser.add_argument("--generation_batch_size", type=int, default=256)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--mode_strategy", choices=("random_uniform", "fixed"), default="random_uniform")
    parser.add_argument("--fixed_mode", choices=LEGACY_MODES, default="burst-obfuscation")
    parser.add_argument("--attack_checkpoint_dir", default="")
    parser.add_argument("--df_checkpoint", default="")
    parser.add_argument("--rf_checkpoint", default="")
    parser.add_argument("--render_coordinate", choices=("rf_tam", "trace_index", "tam_obfuscation", "multi_view"), default="")
    parser.add_argument(
        "--tam_obfuscation_strategy",
        choices=("rayleigh_in_slot", "edge_clustered", "hybrid_clustered"),
        default="",
    )
    parser.add_argument("--tam_slot_jitter", type=float, default=None)
    parser.add_argument("--tam_cluster_ratio", type=float, default=None)
    parser.add_argument("--tam_local_run_max", type=int, default=None)
    parser.add_argument("--tam_preserve_real_timestamps", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--multi_view_mode", choices=("fused", "split"), default="")
    parser.add_argument("--multi_view_df_share", type=float, default=None)
    parser.add_argument("--multi_view_awf_share", type=float, default=None)
    parser.add_argument("--multi_view_rf_share", type=float, default=None)
    parser.add_argument("--fixed_batch_size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "results" / f"{stamp}_legacy_five_pool_direct_eval"


def _resolve_output_dir(value: str, overwrite: bool) -> Path:
    output = Path(value) if value else _default_output_dir()
    existing = [name for name in ("metrics.json", "summary_zh.md") if (output / name).exists()]
    if existing and not overwrite:
        raise SystemExit(f"Refusing to overwrite {output}; pass --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _normalize_peak(values: np.ndarray) -> np.ndarray:
    result = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    peak = float(result.max()) if result.size else 0.0
    if peak > 1.0e-8:
        result = result / peak
    return result.astype(np.float32)


def _center_logits(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.asarray(logits, dtype=np.float32).copy()
    valid = np.asarray(mask, dtype=bool)
    if np.any(valid):
        result[valid] -= float(result[valid].mean())
    else:
        result -= float(result.mean())
    return np.clip(result, -8.0, 8.0).astype(np.float32)


def _legacy_mode_prior(condition, mode: str, rng: np.random.Generator) -> np.ndarray:
    mask = np.asarray(condition.allowed_mask, dtype=np.float32)
    patch_num = int(mask.shape[1])
    allowed_1d = mask[0]
    logits = np.full((2, patch_num), -6.0, dtype=np.float32)
    logits += mask * 0.25

    observed_patch = int(condition.metadata.get("observed_patch", 0))
    out_ratio = float(condition.metadata.get("out_ratio", 0.5))
    in_ratio = float(condition.metadata.get("in_ratio", 0.5))

    burst = np.asarray(condition.burst_saliency, dtype=np.float32)
    if float((burst * allowed_1d).sum()) <= 1.0e-8:
        burst = burst.copy()
        burst[observed_patch : min(patch_num, observed_patch + 4)] += 1.0
        burst = burst / max(float(burst.max()), 1.0e-8)
    gap = _normalize_peak(np.asarray(condition.gap_saliency, dtype=np.float32)) * allowed_1d
    if float(gap.sum()) <= 1.0e-8:
        gap = burst.copy()
    rate = _normalize_peak(np.asarray(condition.rate_saliency, dtype=np.float32)) * allowed_1d
    low_rate = _normalize_peak((1.0 - rate) * allowed_1d)
    prototype = _normalize_peak(np.asarray(condition.public_prototype, dtype=np.float32)) * allowed_1d
    if float(prototype.sum()) <= 1.0e-8:
        prototype = allowed_1d.copy()

    if mode == "gap-adaptive-padding":
        logits += mask * (0.65 + 2.8 * gap.reshape(1, -1))
    elif mode == "burst-obfuscation":
        logits += mask * (0.75 + 2.6 * burst.reshape(1, -1))
    elif mode == "direction-regularization":
        weak_out = in_ratio > out_ratio
        direction_weight = np.asarray([1.45 if weak_out else 0.65, 0.65 if weak_out else 1.45], dtype=np.float32)
        logits += mask * direction_weight.reshape(2, 1) * 1.8
    elif mode == "rate-smoothing":
        logits += mask * (0.85 + 2.2 * low_rate.reshape(1, -1))
        logits += mask * (0.35 * np.asarray(condition.saliency, dtype=np.float32))
    elif mode == "public-prototype-shaping":
        logits += mask * (0.95 + 2.4 * prototype.reshape(1, -1))
        logits += mask * 0.2
    else:
        raise ValueError(f"Unsupported legacy mode: {mode}")

    noise = rng.normal(0.0, 0.08, size=logits.shape).astype(np.float32)
    logits = np.where(mask > 0, logits + noise, -6.0)
    return _center_logits(logits, mask)


def _choose_modes(count: int, args: argparse.Namespace, rng: np.random.Generator) -> list[str]:
    if str(args.mode_strategy) == "fixed":
        return [str(args.fixed_mode)] * int(count)
    indices = rng.integers(0, len(LEGACY_MODES), size=int(count))
    return [LEGACY_MODES[int(index)] for index in indices.tolist()]


def _generate_legacy_templates(
    clean_rows: np.ndarray,
    *,
    budget: float,
    defense_cfg: DefenseConfig,
    modes: list[str],
    seed: int,
) -> list[PaddingTemplate]:
    templates: list[PaddingTemplate] = []
    for local_index, clean in enumerate(np.asarray(clean_rows, dtype=np.float32)):
        item_seed = int(seed) + int(local_index)
        rng = np.random.default_rng(item_seed)
        condition = extract_prefix_condition(
            clean,
            prefix_n=int(defense_cfg.prefix_n),
            patch_num=int(defense_cfg.patch_num),
            max_trace_length=int(defense_cfg.max_trace_length),
            max_load_time=float(defense_cfg.surrogate_rf_max_load_time),
            early_fraction=float(defense_cfg.early_fraction),
        )
        mode = str(modes[local_index])
        logits = _legacy_mode_prior(condition, mode, rng)
        templates.append(
            project_policy_to_template(
                logits,
                condition,
                clean,
                float(budget),
                method="legacy_five_pool_direct",
                metadata={"mode": mode, "seed": item_seed},
                use_causal_mask=True,
                logit_temperature=1.0,
                logit_noise_std=0.0,
                rng=rng,
            )
        )
    return templates


def _mode_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return 0.0
    probs = np.asarray(list(counter.values()), dtype=np.float64) / float(total)
    return float(-(probs * np.log(probs + 1.0e-12)).sum() / math.log(len(LEGACY_MODES)))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = _resolve_output_dir(str(args.output_dir), bool(args.overwrite))
    device = resolve_device(str(args.device))
    defense_cfg = _apply_renderer_overrides(_defense_config_from_run(run_dir), args)
    data_cfg = _dataset_config(defense_cfg, bool(args.full_dataset))
    print(
        f"[legacy-direct] started: run_dir={run_dir}, output_dir={output_dir}, "
        f"budget={float(args.budget):.3f}, split={args.eval_split}, "
        f"render_coordinate={defense_cfg.render_coordinate}, multi_view_mode={defense_cfg.multi_view_mode}, "
        f"device={device}",
        flush=True,
    )
    raw, labels, trace_ids, splits, data_source = load_cw_data(data_cfg)
    del trace_ids
    if str(args.eval_split) == "all":
        test_indices = np.arange(len(labels), dtype=np.int64)
    else:
        test_indices = np.asarray(splits[str(args.eval_split)], dtype=np.int64)
    if int(args.max_test_traces) > 0:
        test_indices = test_indices[: int(args.max_test_traces)]
    print(
        f"[legacy-direct] data loaded: samples={len(labels)}, classes={len(np.unique(labels))}, "
        f"test_traces={len(test_indices)}, source={data_source}",
        flush=True,
    )

    attack_models: dict[str, tuple[torch.nn.Module, np.ndarray, float | None, Path]] = {}
    for kind, explicit in (("df", str(args.df_checkpoint)), ("rf", str(args.rf_checkpoint))):
        checkpoint = _find_attack_checkpoint(run_dir, kind, explicit, str(args.attack_checkpoint_dir))
        model, classes, best_val = _load_attack_checkpoint(checkpoint, kind, defense_cfg, device)
        attack_models[kind] = (model, classes, best_val, checkpoint)
        print(
            f"[legacy-direct] loaded {kind.upper()} checkpoint: classes={len(classes)}, "
            f"best_val={best_val}, path={checkpoint}",
            flush=True,
        )

    clean_totals = {kind: _new_metric_totals() for kind in attack_models}
    defended_totals = {kind: _new_metric_totals() for kind in attack_models}
    mode_counts: Counter = Counter()
    entropy_sum = 0.0
    raw_bandwidth_sum = 0.0
    raw_retention_sum = 0.0
    rendered_count = 0

    batch_size = max(1, int(args.generation_batch_size))
    total_batches = int(np.ceil(len(test_indices) / float(batch_size))) if len(test_indices) else 0
    mode_rng = np.random.default_rng(int(args.seed))
    print(f"[legacy-direct] generating/evaluating batches={total_batches}, batch_size={batch_size}", flush=True)
    for batch_id, start in enumerate(range(0, len(test_indices), batch_size), start=1):
        end = min(start + batch_size, len(test_indices))
        batch_indices = test_indices[start:end]
        clean_rows = np.asarray(raw[batch_indices], dtype=np.float32)
        y = labels[batch_indices].astype(np.int64)
        modes = _choose_modes(len(batch_indices), args, mode_rng)
        mode_counts.update(modes)
        templates = _generate_legacy_templates(
            clean_rows,
            budget=float(args.budget),
            defense_cfg=defense_cfg,
            modes=modes,
            seed=int(args.seed) + int(start),
        )
        traces, _, raw_stats = render_batch_variable(
            clean_rows,
            templates,
            seeds=[int(args.seed) + int(start) + idx for idx in range(len(templates))],
            coordinate_length=int(defense_cfg.max_trace_length),
            **renderer_options_from_config(defense_cfg),
        )
        for kind, (model, classes, _, _) in attack_models.items():
            clean_metrics = _evaluate_attack_checkpoint(
                kind,
                model,
                classes,
                clean_rows,
                y,
                defense_cfg,
                int(args.fixed_batch_size),
                device,
            )
            defended_metrics = _evaluate_attack_checkpoint_from_traces(
                kind,
                model,
                classes,
                traces,
                y,
                defense_cfg,
                int(args.fixed_batch_size),
                device,
            )
            _accumulate_metrics(clean_totals[kind], clean_metrics, len(y))
            _accumulate_metrics(defended_totals[kind], defended_metrics, len(y))
        raw_bandwidth_sum += float(np.sum(raw_stats["raw_bandwidth"]))
        raw_retention_sum += float(np.sum(raw_stats["raw_real_packet_retention"]))
        entropy_sum += float(sum(normalized_template_entropy(template.counts) for template in templates))
        rendered_count += int(len(templates))
        if int(args.progress_every) > 0 and (
            batch_id == 1 or batch_id % int(args.progress_every) == 0 or batch_id == total_batches
        ):
            print(
                f"[legacy-direct] batch {batch_id}/{total_batches}: rows={end}/{len(test_indices)}, "
                f"bandwidth_mean={raw_bandwidth_sum / max(rendered_count, 1):.6f}",
                flush=True,
            )

    fixed_metrics = {}
    for kind, (_, _, best_val, checkpoint) in attack_models.items():
        clean_metrics = _finalize_metrics(clean_totals[kind])
        defended_metrics = _finalize_metrics(defended_totals[kind])
        fixed_metrics[kind] = _attack_metric_row(clean_metrics, defended_metrics, best_val, checkpoint)

    metrics = {
        "method": "legacy_five_pool_direct",
        "note": "Direct legacy five-pool templates; no diffusion training, no diffusion sampling.",
        "run_dir": str(run_dir.resolve()),
        "data_source": str(data_source),
        "output_dir": str(output_dir.resolve()),
        "budget": float(args.budget),
        "render_coordinate": str(defense_cfg.render_coordinate),
        "multi_view_mode": str(defense_cfg.multi_view_mode),
        "tam_obfuscation_strategy": str(defense_cfg.tam_obfuscation_strategy),
        "tam_slot_jitter": float(defense_cfg.tam_slot_jitter),
        "tam_cluster_ratio": float(defense_cfg.tam_cluster_ratio),
        "tam_local_run_max": int(defense_cfg.tam_local_run_max),
        "tam_preserve_real_timestamps": bool(defense_cfg.tam_preserve_real_timestamps),
        "multi_view_df_share": float(defense_cfg.multi_view_df_share),
        "multi_view_awf_share": float(defense_cfg.multi_view_awf_share),
        "multi_view_rf_share": float(defense_cfg.multi_view_rf_share),
        "full_dataset": bool(args.full_dataset),
        "eval_split": str(args.eval_split),
        "loaded_samples": int(len(labels)),
        "loaded_classes": int(len(np.unique(labels))),
        "test_traces": int(len(test_indices)),
        "mode_strategy": str(args.mode_strategy),
        "fixed_mode": str(args.fixed_mode),
        "legacy_modes": list(LEGACY_MODES),
        "mode_counts": {key: int(value) for key, value in sorted(mode_counts.items())},
        "mode_usage_entropy": _mode_entropy(mode_counts),
        "raw_bandwidth_mean": float(raw_bandwidth_sum / max(rendered_count, 1)),
        "raw_retention_mean": float(raw_retention_sum / max(rendered_count, 1)),
        "template_entropy_mean": float(entropy_sum / max(rendered_count, 1)),
        "evaluator_checkpoint_paths": {
            kind: str(values[3].resolve()) for kind, values in attack_models.items()
        },
        "fixed_attackers": fixed_metrics,
        "drops": {
            f"{kind}_accuracy": float(values["accuracy_drop"])
            for kind, values in fixed_metrics.items()
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# legacy_five_pool_direct evaluation",
        "",
        f"- note: {metrics['note']}",
        f"- run_dir: {run_dir}",
        f"- data source: {data_source}",
        f"- full dataset: {bool(args.full_dataset)}",
        f"- eval split: {args.eval_split}",
        f"- loaded samples/classes: {len(labels)}/{len(np.unique(labels))}",
        f"- test traces: {len(test_indices)}",
        f"- budget: {float(args.budget):.4f}",
        f"- render coordinate: {metrics['render_coordinate']}",
        f"- multi-view mode: {metrics['multi_view_mode']}",
        f"- multi-view shares DF/AWF/RF: {metrics['multi_view_df_share']:.3f}/{metrics['multi_view_awf_share']:.3f}/{metrics['multi_view_rf_share']:.3f}",
        f"- TAM obfuscation strategy: {metrics['tam_obfuscation_strategy']}",
        f"- raw bandwidth mean: {metrics['raw_bandwidth_mean']:.6f}",
        f"- raw retention mean: {metrics['raw_retention_mean']:.6f}",
        f"- template entropy mean: {metrics['template_entropy_mean']:.6f}",
        f"- mode usage entropy: {metrics['mode_usage_entropy']:.6f}",
        f"- mode counts: {metrics['mode_counts']}",
        "",
        "| attacker | clean acc | defended acc | drop pp | clean entropy | defended entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, values in fixed_metrics.items():
        clean_acc = float(values["clean_accuracy"])
        defended_acc = float(values["defended_accuracy"])
        lines.append(
            f"| {name.upper()} | {clean_acc:.6f} | {defended_acc:.6f} | "
            f"{100.0 * (clean_acc - defended_acc):.2f} | "
            f"{float(values['clean_entropy']):.6f} | {float(values['defended_entropy']):.6f} |"
        )
    (output_dir / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
