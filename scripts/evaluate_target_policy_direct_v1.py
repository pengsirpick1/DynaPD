from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import json
import sys
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.data import load_cw_data
from dmmp.encoders.prefix import extract_prefix_condition, nonzero_trace
from dmmp.evaluation.attack_models import build_df_input, build_rf_tam_input
from dmmp.evaluation.attack_models import make_attack_model
from dmmp.evaluation.attacks import _eval_torch, train_df_model
from dmmp.guidance.strong_surrogates import ensemble_metrics_from_rendered, load_strong_surrogates
from dmmp.projection.padding import PaddingTemplate, render_batch_variable, renderer_options_from_config
from dmmp.target_policy.candidate_generator import generate_candidates_for_trace
from dmmp.target_policy.config import load_target_policy_config
from dmmp.target_policy.low_level_primitives import PRIMITIVES
from dmmp.target_policy.strategy_families import FAMILIES
from dmmp.target_policy.target_selector import select_targets
from dmmp.utils.config import AttackConfig, DefenseConfig
from dmmp.utils import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Directly evaluate x0* target-policy candidates with frozen DF/RF surrogate.")
    parser.add_argument("--run_dir", required=True, help="Existing DMMPv3 run with strong_surrogate_ensemble.pt.")
    parser.add_argument("--config", default="configs/x0_target_diffusion_v1.yaml")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--budget", type=float, default=0.30)
    parser.add_argument("--full_dataset", action=argparse.BooleanOptionalAction, default=True, help="Ignore run_config max_samples/max_classes and load the whole CW dataset by default.")
    parser.add_argument("--eval_split", choices=("test", "val", "train", "all"), default="test")
    parser.add_argument("--max_test_traces", type=int, default=0)
    parser.add_argument("--generation_batch_size", type=int, default=0, help="Process defense generation in batches; full_dataset defaults to 256 when this is not set.")
    parser.add_argument("--record_preview_limit", type=int, default=200)
    parser.add_argument("--progress_every", type=int, default=10)
    parser.add_argument("--attack_eval_mode", choices=("checkpoint", "strong_surrogate"), default="checkpoint")
    parser.add_argument("--attack_checkpoint_dir", default="", help="Directory containing df/rf checkpoint subdirs; auto-detected from run_dir when omitted.")
    parser.add_argument("--df_checkpoint", default="")
    parser.add_argument("--rf_checkpoint", default="")
    parser.add_argument("--teacher_eval_mode", choices=("none", "checkpoint"), default="none", help="Use frozen DF/RF checkpoints to score deployable x0* candidates before selection.")
    parser.add_argument("--teacher_checkpoint_dir", default="", help="Directory containing teacher df/rf checkpoints; auto-detected from run_dir when omitted.")
    parser.add_argument("--teacher_df_checkpoint", default="")
    parser.add_argument("--teacher_rf_checkpoint", default="")
    parser.add_argument("--teacher_batch_size", type=int, default=0, help="Batch size for checkpoint teacher scoring; <=0 uses --fixed_batch_size.")
    parser.add_argument(
        "--teacher_target_mode",
        choices=("clean_pseudo_label", "true_label"),
        default="clean_pseudo_label",
        help=(
            "Class suppressed by checkpoint teacher scoring. clean_pseudo_label keeps the previous "
            "label-free behavior; true_label is an oracle/class-conditional upper-bound experiment."
        ),
    )
    parser.add_argument(
        "--candidate_class_condition_mode",
        choices=("none", "train_saliency"),
        default="none",
        help="Optionally inject per-class train-split saliency priors into x0* candidate generation.",
    )
    parser.add_argument("--class_condition_weight", type=float, default=0.0, help="Mixing weight for candidate class-condition priors.")
    parser.add_argument("--class_condition_max_train_per_class", type=int, default=80, help="Max train traces per class used to build saliency priors; <=0 uses all train traces.")
    parser.add_argument("--num_candidates", type=int, default=0, help="Override config num_candidates; <=0 keeps the YAML value.")
    parser.add_argument("--target_count", type=int, default=0, help="Override config target_count; <=0 keeps the YAML value.")
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train_fixed_attackers", action="store_true")
    parser.add_argument("--fixed_epochs", type=int, default=10)
    parser.add_argument("--fixed_patience", type=int, default=3)
    parser.add_argument("--fixed_lr", type=float, default=2.0e-3)
    parser.add_argument("--fixed_batch_size", type=int, default=64)
    return parser.parse_args()


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    return {key: np.asarray(value, dtype=np.int64) for key, value in payload.items()}


def _defense_config_from_run(run_dir: Path) -> DefenseConfig:
    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {key: payload[key] for key in DefenseConfig.__dataclass_fields__ if key in payload}
    return DefenseConfig(**values)


def _dataset_config(cfg: DefenseConfig, full_dataset: bool) -> DefenseConfig:
    if not bool(full_dataset):
        return cfg
    return replace(cfg, max_samples=0, max_classes=0)


def _apply_renderer_overrides(cfg: DefenseConfig, args: argparse.Namespace) -> DefenseConfig:
    updates = {}
    if str(getattr(args, "render_coordinate", "")).strip():
        updates["render_coordinate"] = str(args.render_coordinate)
    if str(getattr(args, "tam_obfuscation_strategy", "")).strip():
        updates["tam_obfuscation_strategy"] = str(args.tam_obfuscation_strategy)
    if getattr(args, "tam_slot_jitter", None) is not None:
        updates["tam_slot_jitter"] = float(args.tam_slot_jitter)
    if getattr(args, "tam_cluster_ratio", None) is not None:
        updates["tam_cluster_ratio"] = float(args.tam_cluster_ratio)
    if getattr(args, "tam_local_run_max", None) is not None:
        updates["tam_local_run_max"] = int(args.tam_local_run_max)
    if getattr(args, "tam_preserve_real_timestamps", None) is not None:
        updates["tam_preserve_real_timestamps"] = bool(args.tam_preserve_real_timestamps)
    if str(getattr(args, "multi_view_mode", "")).strip():
        updates["multi_view_mode"] = str(args.multi_view_mode)
    if getattr(args, "multi_view_df_share", None) is not None:
        updates["multi_view_df_share"] = float(args.multi_view_df_share)
    if getattr(args, "multi_view_awf_share", None) is not None:
        updates["multi_view_awf_share"] = float(args.multi_view_awf_share)
    if getattr(args, "multi_view_rf_share", None) is not None:
        updates["multi_view_rf_share"] = float(args.multi_view_rf_share)
    return replace(cfg, **updates) if updates else cfg


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "results" / f"{stamp}_target_policy_direct_v1_eval"


def _resolve_output_dir(value: str, overwrite: bool) -> Path:
    output = Path(value) if value else _default_output_dir()
    sentinels = {"metrics.json", "summary_zh.md"}
    existing = [name for name in sentinels if (output / name).exists()]
    if existing and not overwrite:
        raise SystemExit(f"Refusing to overwrite existing evaluation artifacts in {output}; pass --overwrite.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _clean_trace_list(raw_rows: np.ndarray) -> list[np.ndarray]:
    return [nonzero_trace(row).astype(np.float32) for row in raw_rows]


def _df_input(raw_rows: np.ndarray, cfg: DefenseConfig) -> np.ndarray:
    return build_df_input(raw_rows, max_len=int(cfg.max_trace_length))


def _rf_input(raw_rows: np.ndarray, cfg: DefenseConfig) -> np.ndarray:
    return build_rf_tam_input(
        raw_rows,
        max_len=int(cfg.max_trace_length),
        max_load_time=float(cfg.surrogate_rf_max_load_time),
        num_slots=int(cfg.surrogate_rf_num_slots),
    )


def _ragged_to_padded(traces: list[np.ndarray], max_len: int) -> np.ndarray:
    rows = np.zeros((len(traces), int(max_len)), dtype=np.float32)
    for index, trace in enumerate(traces):
        values = np.asarray(trace, dtype=np.float32)[: int(max_len)]
        rows[index, : len(values)] = values
    return rows


def _indexed_trace_list(traces: list[np.ndarray], mask: np.ndarray) -> list[np.ndarray]:
    return [trace for trace, keep in zip(traces, mask.tolist()) if bool(keep)]


def _normalize_prior_map(values: np.ndarray) -> np.ndarray:
    arr = np.maximum(np.asarray(values, dtype=np.float32), 0.0)
    peak = float(arr.max(initial=0.0))
    if peak <= 1.0e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / peak).astype(np.float32)


def _build_class_condition_priors(
    raw: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    target_cfg,
    *,
    max_train_per_class: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    limit = int(max_train_per_class)
    counts: Counter = Counter()
    sums: dict[int, np.ndarray] = {}
    for clean_index in np.asarray(train_indices, dtype=np.int64).tolist():
        label = int(labels[int(clean_index)])
        if limit > 0 and int(counts[label]) >= limit:
            continue
        condition = extract_prefix_condition(
            raw[int(clean_index)],
            prefix_n=int(target_cfg.prefix_length),
            patch_num=int(target_cfg.strategy_horizon),
        )
        saliency = np.asarray(condition.saliency, dtype=np.float32)
        if label not in sums:
            sums[label] = np.zeros_like(saliency, dtype=np.float32)
        sums[label] += saliency
        counts[label] += 1
    priors = {
        int(label): _normalize_prior_map(total / max(float(counts[int(label)]), 1.0))
        for label, total in sums.items()
        if int(counts[int(label)]) > 0
    }
    summary = {
        "mode": "train_saliency",
        "classes": int(len(priors)),
        "max_train_per_class": int(limit),
        "min_traces_per_class": int(min(counts.values())) if counts else 0,
        "max_traces_per_class": int(max(counts.values())) if counts else 0,
        "mean_traces_per_class": float(np.mean(list(counts.values()))) if counts else 0.0,
    }
    return priors, summary


def _find_attack_checkpoint(run_dir: Path, kind: str, explicit: str, checkpoint_dir: str) -> Path:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"{kind.upper()} checkpoint does not exist: {path}")
        return path
    roots: list[Path] = []
    if checkpoint_dir:
        roots.append(Path(checkpoint_dir))
    roots.extend(
        [
            run_dir / "attack_eval",
            run_dir,
        ]
    )
    patterns = [
        f"**/*{kind}*checkpoint.pt",
        f"**/*{kind.upper()}*checkpoint.pt",
        f"**/{kind}/*checkpoint.pt",
    ]
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in patterns:
            matches.extend(path for path in root.glob(pattern) if path.is_file())
    filtered = [
        path
        for path in matches
        if "target_policy" not in path.name.lower()
        and "diffusion" not in path.name.lower()
        and "encoder" not in path.name.lower()
        and "candidate" not in path.name.lower()
    ]
    if not filtered:
        raise FileNotFoundError(f"Cannot auto-detect {kind.upper()} attack checkpoint under {roots}")
    preferred = [
        path
        for path in filtered
        if "same_user" in str(path).lower() or "fixed" in str(path).lower() or "clean" in str(path).lower()
    ]
    candidates = preferred if preferred else filtered
    return sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True)[0]


def _load_attack_checkpoint(path: Path, kind: str, cfg: DefenseConfig, device: torch.device) -> tuple[torch.nn.Module, np.ndarray, float | None]:
    payload = torch.load(path, map_location=device, weights_only=False)
    classes = np.asarray(payload.get("classes"), dtype=np.int64)
    if classes.size <= 0:
        raise ValueError(f"Checkpoint has no classes array: {path}")
    model = make_attack_model(
        kind.upper(),
        int(len(classes)),
        max_trace_length=int(cfg.max_trace_length),
        df_architecture=str(getattr(cfg, "surrogate_df_architecture", "project")),
    ).to(device)
    state = payload.get("model_state", payload)
    model.load_state_dict(state)
    model.eval()
    return model, classes, float(payload["best_val"]) if "best_val" in payload else None


@dataclass
class TeacherAttack:
    kind: str
    model: torch.nn.Module
    classes: np.ndarray
    checkpoint: Path


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -60.0, 60.0))
    return (exp / np.maximum(exp.sum(axis=1, keepdims=True), 1.0e-12)).astype(np.float32)


def _normalized_entropy_np(probs: np.ndarray) -> np.ndarray:
    values = np.asarray(probs, dtype=np.float64)
    entropy = -np.sum(values * np.log(np.maximum(values, 1.0e-12)), axis=1)
    if values.shape[1] > 1:
        entropy = entropy / np.log(values.shape[1])
    return entropy.astype(np.float32)


def _logits_for_attack_rows(
    kind: str,
    model: torch.nn.Module,
    rows: np.ndarray,
    cfg: DefenseConfig,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    values = _df_input(rows, cfg) if kind == "df" else _rf_input(rows, cfg)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), max(1, int(batch_size))):
            end = min(start + max(1, int(batch_size)), len(values))
            xb = torch.as_tensor(values[start:end], dtype=torch.float32, device=device)
            outputs.append(model(xb).detach().cpu().numpy())
    if outputs:
        return np.concatenate(outputs, axis=0).astype(np.float32)
    return np.zeros((0, int(len(getattr(model, "classes", [])))), dtype=np.float32)


def _logits_for_attack_traces(
    kind: str,
    model: torch.nn.Module,
    traces: list[np.ndarray],
    cfg: DefenseConfig,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    rows = _ragged_to_padded(traces, int(cfg.max_trace_length))
    return _logits_for_attack_rows(kind, model, rows, cfg, int(batch_size), device)


def _target_index_from_label(classes: np.ndarray, label: int) -> int | None:
    matches = np.flatnonzero(np.asarray(classes, dtype=np.int64) == int(label))
    if matches.size == 0:
        return None
    return int(matches[0])


def _teacher_component_arrays(
    clean_logits: np.ndarray,
    defended_logits: np.ndarray,
    *,
    target_index: int | None = None,
) -> dict[str, np.ndarray | float | int | str]:
    clean_probs = _softmax_np(clean_logits.reshape(1, -1))[0]
    defended_probs = _softmax_np(defended_logits)
    pseudo_index = int(np.argmax(clean_probs))
    selected_index = pseudo_index if target_index is None else int(target_index)
    if selected_index < 0 or selected_index >= clean_probs.shape[0]:
        raise ValueError(f"Teacher target index is out of range: target_index={selected_index}, classes={clean_probs.shape[0]}")
    target_source = "clean_pseudo_label" if target_index is None else "true_label"
    clean_target_conf = float(clean_probs[selected_index])
    defended_target_conf = defended_probs[:, selected_index]
    clean_other = clean_probs.copy()
    clean_other[selected_index] = -np.inf
    defended_other = defended_probs.copy()
    defended_other[:, selected_index] = -np.inf
    clean_margin = clean_target_conf - float(np.max(clean_other))
    defended_margin = defended_target_conf - np.max(defended_other, axis=1)
    clean_entropy = float(_normalized_entropy_np(clean_probs.reshape(1, -1))[0])
    defended_entropy = _normalized_entropy_np(defended_probs)
    clean_max_confidence = float(np.max(clean_probs))
    defended_max_confidence = np.max(defended_probs, axis=1)
    target_conf_drop = (clean_target_conf - defended_target_conf).astype(np.float32)
    return {
        "pseudo_index": pseudo_index,
        "target_index": selected_index,
        "target_source": target_source,
        "clean_target_confidence": clean_target_conf,
        "clean_pseudo_confidence": clean_target_conf,
        "clean_margin": float(clean_margin),
        "clean_entropy": clean_entropy,
        "clean_max_confidence": clean_max_confidence,
        "target_conf_drop": target_conf_drop,
        "pseudo_conf_drop": target_conf_drop,
        "margin_drop": (clean_margin - defended_margin).astype(np.float32),
        "entropy_gain": (defended_entropy - clean_entropy).astype(np.float32),
        "max_confidence_drop": (clean_max_confidence - defended_max_confidence).astype(np.float32),
    }


def _score_from_teacher_components(components: dict[str, np.ndarray | float | int], index: int, target_cfg) -> tuple[float, dict[str, float]]:
    values = {
        "pseudo_conf_drop": float(np.asarray(components["pseudo_conf_drop"])[int(index)]),
        "target_conf_drop": float(np.asarray(components["target_conf_drop"])[int(index)]),
        "margin_drop": float(np.asarray(components["margin_drop"])[int(index)]),
        "entropy_gain": float(np.asarray(components["entropy_gain"])[int(index)]),
        "max_confidence_drop": float(np.asarray(components["max_confidence_drop"])[int(index)]),
    }
    score = (
        float(target_cfg.score_entropy_weight) * values["entropy_gain"]
        + float(target_cfg.score_pseudo_weight) * values["pseudo_conf_drop"]
        + float(target_cfg.score_margin_weight) * values["margin_drop"]
        + float(target_cfg.score_max_weight) * values["max_confidence_drop"]
    )
    return float(score), values


def _combine_df_rf_teacher_scores(score_by_kind: dict[str, float], target_cfg) -> float:
    if not score_by_kind:
        return 0.0
    if len(score_by_kind) == 1:
        return float(next(iter(score_by_kind.values())))
    df_score = float(score_by_kind.get("df", np.mean(list(score_by_kind.values()))))
    rf_score = float(score_by_kind.get("rf", np.mean(list(score_by_kind.values()))))
    combined = (
        0.5 * (df_score + rf_score)
        + float(target_cfg.robust_min_weight) * min(df_score, rf_score)
        - float(target_cfg.attacker_gap_weight) * abs(df_score - rf_score)
    )
    gate_penalty = 0.0
    if "df" in score_by_kind and df_score < float(target_cfg.min_df_gain):
        gate_penalty += float(target_cfg.min_df_gain) - df_score
    if "rf" in score_by_kind and rf_score < float(target_cfg.min_rf_gain):
        gate_penalty += float(target_cfg.min_rf_gain) - rf_score
    return float(combined - gate_penalty)


def _new_teacher_score_totals() -> dict[str, object]:
    return {"n": 0, "sums": {}, "mins": {}, "maxs": {}}


def _accumulate_teacher_score(totals: dict[str, object], values: dict[str, float | int | None]) -> None:
    sums = totals["sums"]
    mins = totals["mins"]
    maxs = totals["maxs"]
    assert isinstance(sums, dict) and isinstance(mins, dict) and isinstance(maxs, dict)
    wrote = False
    for key, value in values.items():
        if value is None:
            continue
        numeric = float(value)
        if not np.isfinite(numeric):
            continue
        sums[key] = float(sums.get(key, 0.0)) + numeric
        mins[key] = numeric if key not in mins else min(float(mins[key]), numeric)
        maxs[key] = numeric if key not in maxs else max(float(maxs[key]), numeric)
        wrote = True
    if wrote:
        totals["n"] = int(totals["n"]) + 1


def _finalize_teacher_score_summary(totals: dict[str, object]) -> dict[str, object]:
    n = int(totals["n"])
    if n <= 0:
        return {"n": 0}
    sums = totals["sums"]
    mins = totals["mins"]
    maxs = totals["maxs"]
    assert isinstance(sums, dict) and isinstance(mins, dict) and isinstance(maxs, dict)
    return {
        "n": n,
        "mean": {key: float(value) / float(n) for key, value in sorted(sums.items())},
        "min": {key: float(value) for key, value in sorted(mins.items())},
        "max": {key: float(value) for key, value in sorted(maxs.items())},
    }


class CheckpointTeacherScorer:
    def __init__(
        self,
        attacks: dict[str, TeacherAttack],
        *,
        defense_cfg: DefenseConfig,
        target_cfg,
        device: torch.device,
        batch_size: int,
        render_strategy: str,
        target_mode: str,
    ) -> None:
        self.attacks = dict(attacks)
        self.defense_cfg = defense_cfg
        self.target_cfg = target_cfg
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.render_strategy = str(render_strategy)
        self.target_mode = str(target_mode)

    @property
    def checkpoint_paths(self) -> dict[str, str]:
        return {kind: str(attack.checkpoint.resolve()) for kind, attack in self.attacks.items()}

    def score_candidates(
        self,
        clean_trace: np.ndarray,
        candidates,
        *,
        render_seed: int,
        true_label: int | None = None,
        totals: dict[str, object] | None = None,
    ) -> dict[str, int]:
        deployable = [
            candidate
            for candidate in candidates
            if bool(candidate.constraint_report.valid) and bool(candidate.constraint_report.deployable)
        ]
        if not deployable:
            return {"scored_candidate_count": 0, "scored_trace_count": 0, "unscored_trace_count": 1}
        clean_row = np.asarray(clean_trace, dtype=np.float32).reshape(1, -1)
        clean_matrix = np.repeat(clean_row, len(deployable), axis=0)
        templates = [
            PaddingTemplate(
                counts=candidate.counts,
                target_n_pad=int(candidate.budget_count),
                actual_n_pad=int(candidate.counts.sum()),
                target_bandwidth=float(candidate.budget_ratio),
                metadata={"method": "target_policy_teacher_candidate"},
            )
            for candidate in deployable
        ]
        defended_traces, _, _ = render_batch_variable(
            clean_matrix,
            templates,
            seeds=[int(render_seed)] * len(templates),
            coordinate_length=int(self.defense_cfg.max_trace_length),
            **{**renderer_options_from_config(self.defense_cfg), "strategy": self.render_strategy},
        )
        per_kind_scores: dict[str, np.ndarray] = {}
        per_kind_components: dict[str, dict[str, np.ndarray | float | int | str]] = {}
        for kind, attack in self.attacks.items():
            target_index = None
            if self.target_mode == "true_label":
                if true_label is None:
                    continue
                target_index = _target_index_from_label(attack.classes, int(true_label))
                if target_index is None:
                    continue
            clean_logits = _logits_for_attack_rows(
                kind,
                attack.model,
                clean_row,
                self.defense_cfg,
                self.batch_size,
                self.device,
            )[0]
            defended_logits = _logits_for_attack_traces(
                kind,
                attack.model,
                defended_traces,
                self.defense_cfg,
                self.batch_size,
                self.device,
            )
            components = _teacher_component_arrays(clean_logits, defended_logits, target_index=target_index)
            scores = []
            for index in range(len(deployable)):
                score, _ = _score_from_teacher_components(components, index, self.target_cfg)
                scores.append(float(score))
            per_kind_scores[kind] = np.asarray(scores, dtype=np.float32)
            per_kind_components[kind] = components
        if not per_kind_scores:
            return {"scored_candidate_count": 0, "scored_trace_count": 0, "unscored_trace_count": 1}
        for index, candidate in enumerate(deployable):
            score_by_kind = {kind: float(scores[index]) for kind, scores in per_kind_scores.items()}
            combined = _combine_df_rf_teacher_scores(score_by_kind, self.target_cfg)
            components_flat: dict[str, float] = {}
            for kind, components in per_kind_components.items():
                _, values = _score_from_teacher_components(components, index, self.target_cfg)
                for name, value in values.items():
                    components_flat[f"{kind}_{name}"] = float(value)
                components_flat[f"{kind}_score"] = float(score_by_kind[kind])
                components_flat[f"{kind}_clean_pseudo_confidence"] = float(components["clean_pseudo_confidence"])
                components_flat[f"{kind}_clean_margin"] = float(components["clean_margin"])
                components_flat[f"{kind}_clean_entropy"] = float(components["clean_entropy"])
                components_flat[f"{kind}_clean_max_confidence"] = float(components["clean_max_confidence"])
                components_flat[f"{kind}_pseudo_index"] = float(components["pseudo_index"])
                components_flat[f"{kind}_target_index"] = float(components["target_index"])
                components_flat[f"{kind}_clean_target_confidence"] = float(components["clean_target_confidence"])
            candidate.selection_score_attack = float(combined)
            candidate.selection_score_df = float(score_by_kind["df"]) if "df" in score_by_kind else None
            candidate.selection_score_rf = float(score_by_kind["rf"]) if "rf" in score_by_kind else None
            candidate.teacher_scored = True
            candidate.teacher_score_source = f"checkpoint_{self.target_mode}_logit_feedback"
            candidate.teacher_score_attack = float(combined)
            candidate.teacher_score_df = candidate.selection_score_df
            candidate.teacher_score_rf = candidate.selection_score_rf
            candidate.teacher_score_components = components_flat
            if totals is not None:
                _accumulate_teacher_score(
                    totals,
                    {
                        "score_attack": float(combined),
                        "score_df": candidate.teacher_score_df,
                        "score_rf": candidate.teacher_score_rf,
                        **components_flat,
                    },
                )
        return {
            "scored_candidate_count": int(len(deployable)),
            "scored_trace_count": 1,
            "unscored_trace_count": 0,
        }


def _load_teacher_scorer(
    args: argparse.Namespace,
    run_dir: Path,
    defense_cfg: DefenseConfig,
    target_cfg,
    device: torch.device,
) -> CheckpointTeacherScorer | None:
    if str(args.teacher_eval_mode) != "checkpoint":
        return None
    attacks: dict[str, TeacherAttack] = {}
    for kind, explicit in (("df", str(args.teacher_df_checkpoint)), ("rf", str(args.teacher_rf_checkpoint))):
        try:
            checkpoint = _find_attack_checkpoint(run_dir, kind, explicit, str(args.teacher_checkpoint_dir))
        except FileNotFoundError as exc:
            if explicit:
                raise
            print(f"[target-policy teacher] skip {kind.upper()} checkpoint: {exc}", flush=True)
            continue
        model, classes, _ = _load_attack_checkpoint(checkpoint, kind, defense_cfg, device)
        attacks[kind] = TeacherAttack(kind=kind, model=model, classes=classes, checkpoint=checkpoint)
    if not attacks:
        raise SystemExit("--teacher_eval_mode checkpoint requested, but no DF/RF teacher checkpoint could be loaded.")
    batch_size = int(args.teacher_batch_size) if int(args.teacher_batch_size) > 0 else int(args.fixed_batch_size)
    return CheckpointTeacherScorer(
        attacks,
        defense_cfg=defense_cfg,
        target_cfg=target_cfg,
        device=device,
        batch_size=max(1, int(batch_size)),
        render_strategy=str(defense_cfg.insertion_strategy),
        target_mode=str(args.teacher_target_mode),
    )


def _teacher_evaluator_relation(teacher_paths: dict[str, str], evaluator_paths: dict[str, str], attack_eval_mode: str) -> str:
    if not teacher_paths:
        return "none; direct_x0_candidate_scoring_uses_heuristic_proxy"
    if str(attack_eval_mode) != "checkpoint" or not evaluator_paths:
        return "teacher_checkpoint; evaluator_is_not_checkpoint"
    overlap = sorted(set(teacher_paths) & set(evaluator_paths))
    if not overlap:
        return "teacher_checkpoint; no_overlapping_checkpoint_evaluator"
    same = [
        str(Path(teacher_paths[kind]).resolve()).lower() == str(Path(evaluator_paths[kind]).resolve()).lower()
        for kind in overlap
    ]
    if all(same):
        return "same_checkpoint_for_overlapping_attackers"
    if any(same):
        return "mixed_same_and_different_checkpoints"
    return "different_teacher_and_evaluator_checkpoints"


def _evaluate_attack_checkpoint(
    kind: str,
    model: torch.nn.Module,
    classes: np.ndarray,
    rows: np.ndarray,
    labels: np.ndarray,
    cfg: DefenseConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    missing = sorted(set(int(value) for value in np.unique(labels).tolist()) - set(int(value) for value in classes.tolist()))
    if missing:
        raise ValueError(
            f"{kind.upper()} checkpoint classes do not cover evaluation labels: missing={missing[:10]}, "
            f"checkpoint_classes={len(classes)}, eval_classes={len(np.unique(labels))}"
        )
    values = _df_input(rows, cfg) if kind == "df" else _rf_input(rows, cfg)
    return _eval_torch(model, values, labels, classes, device, int(batch_size))


def _evaluate_attack_checkpoint_from_traces(
    kind: str,
    model: torch.nn.Module,
    classes: np.ndarray,
    traces: list[np.ndarray],
    labels: np.ndarray,
    cfg: DefenseConfig,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    rows = _ragged_to_padded(traces, int(cfg.max_trace_length))
    return _evaluate_attack_checkpoint(kind, model, classes, rows, labels, cfg, batch_size, device)


def _attack_metric_row(clean_metrics: dict[str, float], defended_metrics: dict[str, float], best_val: float | None, checkpoint: Path) -> dict[str, float | str | None]:
    return {
        "checkpoint": str(checkpoint.resolve()),
        "best_val_accuracy": None if best_val is None else float(best_val),
        "clean_accuracy": float(clean_metrics["defended_accuracy"]),
        "defended_accuracy": float(defended_metrics["defended_accuracy"]),
        "accuracy_drop": float(clean_metrics["defended_accuracy"] - defended_metrics["defended_accuracy"]),
        "clean_entropy": float(clean_metrics["prediction_entropy"]),
        "defended_entropy": float(defended_metrics["prediction_entropy"]),
        "clean_max_confidence": float(clean_metrics["max_confidence"]),
        "defended_max_confidence": float(defended_metrics["max_confidence"]),
    }


def _metric_row_from_clean_defended(clean_metrics: dict[str, float], defended_metrics: dict[str, float]) -> dict[str, float]:
    if not clean_metrics or not defended_metrics:
        return {}
    return {
        "clean_accuracy": float(clean_metrics["defended_accuracy"]),
        "defended_accuracy": float(defended_metrics["defended_accuracy"]),
        "accuracy_drop": float(clean_metrics["defended_accuracy"] - defended_metrics["defended_accuracy"]),
        "clean_entropy": float(clean_metrics["prediction_entropy"]),
        "defended_entropy": float(defended_metrics["prediction_entropy"]),
        "clean_max_confidence": float(clean_metrics["max_confidence"]),
        "defended_max_confidence": float(defended_metrics["max_confidence"]),
    }


def _counter_add(counter: Counter, values) -> None:
    for value in values:
        counter[str(value)] += 1


def _length_bin(clean_trace: np.ndarray) -> str:
    length = int(nonzero_trace(clean_trace).size)
    if length < 500:
        return "lt_500"
    if length < 1000:
        return "500_999"
    if length < 2000:
        return "1000_1999"
    if length < 4000:
        return "2000_3999"
    if length < 8000:
        return "4000_7999"
    return "ge_8000"


def _candidate_diagnostic_counters(candidates, trace_length_bin: str) -> dict[str, Counter]:
    counters = {
        "non_deployable_reason_counts": Counter(),
        "non_deployable_reason_by_budget": Counter(),
        "non_deployable_reason_by_family": Counter(),
        "non_deployable_reason_by_primitive": Counter(),
        "non_deployable_reason_by_trace_length": Counter(),
    }
    for candidate in candidates:
        if bool(candidate.constraint_report.deployable):
            continue
        reasons = tuple(candidate.constraint_report.reasons) or ("other",)
        _counter_add(counters["non_deployable_reason_counts"], reasons)
        for reason in reasons:
            counters["non_deployable_reason_by_budget"][f"{candidate.budget_ratio:.2f}:{reason}"] += 1
            counters["non_deployable_reason_by_trace_length"][f"{trace_length_bin}:{reason}"] += 1
            for family_index in candidate.family_indices.tolist():
                if 0 <= int(family_index) < len(FAMILIES):
                    counters["non_deployable_reason_by_family"][f"{FAMILIES[int(family_index)]}:{reason}"] += 1
            for primitive_index in candidate.primitive_indices.tolist():
                if 0 <= int(primitive_index) < len(PRIMITIVES):
                    counters["non_deployable_reason_by_primitive"][f"{PRIMITIVES[int(primitive_index)]}:{reason}"] += 1
    return counters


def _merge_counter_maps(total: dict[str, Counter], update: dict[str, Counter]) -> None:
    for key, value in update.items():
        total.setdefault(key, Counter()).update(value)


def _serialize_counter(counter: Counter, limit: int | None = None) -> dict[str, int]:
    items = counter.most_common(None if limit is None or int(limit) <= 0 else int(limit))
    return {str(key): int(value) for key, value in items}


def _serialize_counter_map(counters: dict[str, Counter], limit: int = 200) -> dict[str, dict[str, int]]:
    return {key: _serialize_counter(value, limit=limit) for key, value in counters.items()}


def _new_surrogate_totals() -> dict[str, object]:
    return {"n": 0, "sums": {}}


def _accumulate_surrogate(totals: dict[str, object], metrics: dict[str, float], count: int) -> None:
    if int(count) <= 0:
        return
    sums = totals["sums"]
    assert isinstance(sums, dict)
    for key, value in metrics.items():
        if "_ensemble_" in key or key == "surrogate_label_free_attack_pressure":
            continue
        if key.startswith("surrogate_"):
            sums[key] = float(sums.get(key, 0.0)) + float(value) * int(count)
    totals["n"] = int(totals["n"]) + int(count)


def _new_metric_totals() -> dict[str, object]:
    return {"n": 0, "sums": {}}


def _accumulate_metrics(totals: dict[str, object], metrics: dict[str, float], count: int) -> None:
    if int(count) <= 0:
        return
    sums = totals["sums"]
    assert isinstance(sums, dict)
    for key, value in metrics.items():
        sums[key] = float(sums.get(key, 0.0)) + float(value) * int(count)
    totals["n"] = int(totals["n"]) + int(count)


def _finalize_metrics(totals: dict[str, object]) -> dict[str, float]:
    n = int(totals["n"])
    if n <= 0:
        return {}
    sums = totals["sums"]
    assert isinstance(sums, dict)
    return {key: float(value) / float(n) for key, value in sums.items()}


def _finalize_surrogate(totals: dict[str, object], attacker_names: tuple[str, ...]) -> dict[str, float]:
    n = int(totals["n"])
    if n <= 0:
        return {}
    sums = totals["sums"]
    assert isinstance(sums, dict)
    result: dict[str, float] = {key: float(value) / float(n) for key, value in sums.items()}
    accuracies = []
    entropies = []
    confidences = []
    margins = []
    pressures = []
    for name in attacker_names:
        accuracy = float(result.get(f"surrogate_{name}_accuracy", 0.0))
        entropy = float(result.get(f"surrogate_{name}_entropy", 0.0))
        max_confidence = float(result.get(f"surrogate_{name}_max_confidence", 0.0))
        margin = float(result.get(f"surrogate_{name}_margin", 0.0))
        pressure = max_confidence + 0.50 * margin - 0.50 * entropy
        result[f"surrogate_{name}_label_free_pressure"] = float(pressure)
        accuracies.append(accuracy)
        entropies.append(entropy)
        confidences.append(max_confidence)
        margins.append(margin)
        pressures.append(pressure)
    result["surrogate_ensemble_mean_accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
    result["surrogate_ensemble_worst_accuracy"] = float(np.max(accuracies)) if accuracies else 0.0
    result["surrogate_ensemble_mean_entropy"] = float(np.mean(entropies)) if entropies else 0.0
    result["surrogate_ensemble_worst_max_confidence"] = float(np.max(confidences)) if confidences else 0.0
    result["surrogate_ensemble_worst_margin"] = float(np.max(margins)) if margins else 0.0
    result["surrogate_ensemble_worst_label_free_pressure"] = float(np.max(pressures)) if pressures else 1.0
    result["surrogate_label_free_attack_pressure"] = float(result["surrogate_ensemble_worst_label_free_pressure"])
    return result


def _optional_float(value) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


def _selected_record(row_id: int, clean_index: int, trace_id: str, chosen) -> dict[str, object]:
    report = chosen.constraint_report
    return {
        "row": int(row_id),
        "clean_index": int(clean_index),
        "trace_id": str(trace_id),
        "budget_count": int(chosen.budget_count),
        "actual_count": int(chosen.counts.sum()),
        "proxy_score_attack": float(chosen.proxy_score_attack),
        "proxy_score_df": float(chosen.proxy_score_df),
        "proxy_score_rf": float(chosen.proxy_score_rf),
        "selection_score_attack": _optional_float(getattr(chosen, "selection_score_attack", None)),
        "selection_score_df": _optional_float(getattr(chosen, "selection_score_df", None)),
        "selection_score_rf": _optional_float(getattr(chosen, "selection_score_rf", None)),
        "teacher_scored": bool(getattr(chosen, "teacher_scored", False)),
        "teacher_score_source": str(getattr(chosen, "teacher_score_source", "heuristic_proxy")),
        "teacher_score_attack": _optional_float(getattr(chosen, "teacher_score_attack", None)),
        "teacher_score_df": _optional_float(getattr(chosen, "teacher_score_df", None)),
        "teacher_score_rf": _optional_float(getattr(chosen, "teacher_score_rf", None)),
        "teacher_score_components": {
            str(key): float(value)
            for key, value in getattr(chosen, "teacher_score_components", {}).items()
            if np.isfinite(float(value))
        },
        "deployable": bool(report.deployable),
        "fallback": bool(chosen.fallback_flag),
        "non_deployable_reasons": list(report.reasons),
        "allowed_violation_count": int(report.allowed_violation_count),
        "negative_count": int(report.negative_count),
        "budget_error": int(report.budget_error),
        "max_slot_count": int(report.max_slot_count),
        "max_local_density": float(report.max_local_density),
        "tail_extension_ratio": float(report.tail_extension_ratio),
        "family_names": [FAMILIES[int(index)] for index in chosen.family_indices.tolist() if 0 <= int(index) < len(FAMILIES)],
        "primitive_names": [PRIMITIVES[int(index)] for index in chosen.primitive_indices.tolist() if 0 <= int(index) < len(PRIMITIVES)],
    }


def _generate_templates_for_batch(
    clean_rows: np.ndarray,
    labels: np.ndarray,
    batch_indices: np.ndarray,
    row_offset: int,
    trace_ids: np.ndarray,
    target_cfg,
    rng: np.random.Generator,
    seed: int,
    teacher_scorer: CheckpointTeacherScorer | None = None,
    teacher_totals: dict[str, object] | None = None,
    class_condition_priors: dict[int, np.ndarray] | None = None,
    class_condition_weight: float = 0.0,
) -> tuple[list[PaddingTemplate], int, int, int, list[dict[str, object]], dict[str, Counter], dict[str, Counter], dict[str, int]]:
    templates: list[PaddingTemplate] = []
    records: list[dict[str, object]] = []
    candidate_counters: dict[str, Counter] = {}
    selected_counters: dict[str, Counter] = {
        "selected_non_deployable_reason_counts": Counter(),
        "selected_non_deployable_reason_by_budget": Counter(),
        "selected_non_deployable_reason_by_family": Counter(),
        "selected_non_deployable_reason_by_primitive": Counter(),
        "selected_non_deployable_reason_by_trace_length": Counter(),
    }
    fallback_count = 0
    chosen_fallback_count = 0
    chosen_non_deployable_count = 0
    teacher_stats = {"scored_candidate_count": 0, "scored_trace_count": 0, "unscored_trace_count": 0}
    for local_row, clean_index in enumerate(batch_indices.tolist()):
        clean = clean_rows[local_row]
        condition = extract_prefix_condition(
            clean,
            prefix_n=int(target_cfg.prefix_length),
            patch_num=int(target_cfg.strategy_horizon),
        )
        class_prior = None
        if class_condition_priors is not None:
            class_prior = class_condition_priors.get(int(labels[local_row]))
        candidates = generate_candidates_for_trace(
            clean,
            cfg=target_cfg,
            prefix_condition=condition,
            clean_index=int(clean_index),
            class_condition_prior=class_prior,
            class_condition_weight=float(class_condition_weight),
            rng=rng,
        )
        trace_length_bin = _length_bin(clean)
        _merge_counter_maps(candidate_counters, _candidate_diagnostic_counters(candidates, trace_length_bin))
        if teacher_scorer is not None:
            stats = teacher_scorer.score_candidates(
                clean,
                candidates,
                render_seed=int(seed) + int(row_offset) + int(local_row),
                true_label=int(labels[local_row]),
                totals=teacher_totals,
            )
            for key, value in stats.items():
                teacher_stats[key] = int(teacher_stats.get(key, 0)) + int(value)
        selected, fallback = select_targets(
            candidates,
            target_count=int(target_cfg.target_count),
            quality_target_count=int(target_cfg.quality_target_count),
            diverse_target_count=int(target_cfg.diverse_target_count),
            allocation_l1_weight=float(target_cfg.allocation_l1_weight),
            allocation_cosine_weight=float(target_cfg.allocation_cosine_weight),
        )
        fallback_count += int(fallback)
        chosen = selected[0]
        chosen_fallback_count += int(bool(chosen.fallback_flag))
        chosen_non_deployable_count += int(not bool(chosen.constraint_report.deployable))
        if not bool(chosen.constraint_report.deployable):
            reasons = tuple(chosen.constraint_report.reasons) or ("other",)
            _counter_add(selected_counters["selected_non_deployable_reason_counts"], reasons)
            for reason in reasons:
                selected_counters["selected_non_deployable_reason_by_budget"][f"{chosen.budget_ratio:.2f}:{reason}"] += 1
                selected_counters["selected_non_deployable_reason_by_trace_length"][f"{trace_length_bin}:{reason}"] += 1
                for family_index in chosen.family_indices.tolist():
                    if 0 <= int(family_index) < len(FAMILIES):
                        selected_counters["selected_non_deployable_reason_by_family"][f"{FAMILIES[int(family_index)]}:{reason}"] += 1
                for primitive_index in chosen.primitive_indices.tolist():
                    if 0 <= int(primitive_index) < len(PRIMITIVES):
                        selected_counters["selected_non_deployable_reason_by_primitive"][f"{PRIMITIVES[int(primitive_index)]}:{reason}"] += 1
        templates.append(
            PaddingTemplate(
                counts=chosen.counts,
                target_n_pad=int(chosen.budget_count),
                actual_n_pad=int(chosen.counts.sum()),
                target_bandwidth=float(chosen.budget_ratio),
                metadata={
                    "method": "target_policy_direct_v1",
                    "clean_index": int(clean_index),
                    "trace_id": str(trace_ids[int(clean_index)]),
                    "proxy_score_attack": float(chosen.proxy_score_attack),
                    "selection_score_attack": _optional_float(getattr(chosen, "selection_score_attack", None)),
                    "teacher_scored": bool(getattr(chosen, "teacher_scored", False)),
                    "teacher_score_source": str(getattr(chosen, "teacher_score_source", "heuristic_proxy")),
                    "teacher_target_mode": "none" if teacher_scorer is None else str(teacher_scorer.target_mode),
                    "candidate_class_condition_mode": "none" if class_condition_priors is None else "train_saliency",
                    "class_condition_weight": float(class_condition_weight),
                    "fallback": bool(chosen.fallback_flag),
                },
            )
        )
        record = _selected_record(int(row_offset) + int(local_row), int(clean_index), str(trace_ids[int(clean_index)]), chosen)
        record["true_label"] = int(labels[local_row])
        record["teacher_target_mode"] = "none" if teacher_scorer is None else str(teacher_scorer.target_mode)
        record["candidate_class_condition_mode"] = "none" if class_condition_priors is None else "train_saliency"
        record["class_condition_weight"] = float(class_condition_weight)
        records.append(record)
    return templates, int(fallback_count), int(chosen_fallback_count), int(chosen_non_deployable_count), records, candidate_counters, selected_counters, teacher_stats


def _train_and_eval_fixed(
    kind: str,
    train_rows: np.ndarray,
    train_y: np.ndarray,
    val_rows: np.ndarray,
    val_y: np.ndarray,
    clean_test_rows: np.ndarray,
    defended_test_rows: np.ndarray,
    test_y: np.ndarray,
    defense_cfg: DefenseConfig,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, float]:
    attack_cfg = AttackConfig(
        run_dir=str(args.run_dir),
        clean_df_epochs=int(args.fixed_epochs),
        clean_df_patience=int(args.fixed_patience),
        df_batch_size=int(args.fixed_batch_size),
        max_load_time=float(defense_cfg.surrogate_rf_max_load_time),
        rf_tam_num_slots=int(defense_cfg.surrogate_rf_num_slots),
    )
    if kind == "df":
        train_x = _df_input(train_rows, defense_cfg)
        val_x = _df_input(val_rows, defense_cfg)
        clean_test_x = _df_input(clean_test_rows, defense_cfg)
        defended_test_x = _df_input(defended_test_rows, defense_cfg)
    else:
        train_x = _rf_input(train_rows, defense_cfg)
        val_x = _rf_input(val_rows, defense_cfg)
        clean_test_x = _rf_input(clean_test_rows, defense_cfg)
        defended_test_x = _rf_input(defended_test_rows, defense_cfg)
    model, classes, best_val = train_df_model(
        train_x,
        train_y,
        val_x,
        val_y,
        attacker_kind=kind.upper(),
        defense_cfg=defense_cfg,
        attack_cfg=attack_cfg,
        epochs=int(args.fixed_epochs),
        patience=int(args.fixed_patience),
        lr=float(args.fixed_lr),
        batch_size=int(args.fixed_batch_size),
        device=device,
        seed=int(args.seed),
        progress=True,
    )
    clean_metrics = _eval_torch(model, clean_test_x, test_y, classes, device, int(args.fixed_batch_size))
    defended_metrics = _eval_torch(model, defended_test_x, test_y, classes, device, int(args.fixed_batch_size))
    return {
        "best_val_accuracy": float(best_val),
        "clean_accuracy": float(clean_metrics["defended_accuracy"]),
        "defended_accuracy": float(defended_metrics["defended_accuracy"]),
        "accuracy_drop": float(clean_metrics["defended_accuracy"] - defended_metrics["defended_accuracy"]),
    }


def _run_batched_evaluation(
    args: argparse.Namespace,
    run_dir: Path,
    output_dir: Path,
    defense_cfg: DefenseConfig,
    data_cfg: DefenseConfig,
    target_cfg,
    device: torch.device,
) -> None:
    if bool(args.train_fixed_attackers):
        raise SystemExit("Batched/full-dataset mode does not train fixed attackers; run without --train_fixed_attackers.")
    print(
        f"[target-policy full] loading CW data; full_dataset={bool(args.full_dataset)}, "
        f"eval_split={args.eval_split}, output_dir={output_dir}",
        flush=True,
    )
    raw, labels, trace_ids, data_splits, data_source = load_cw_data(data_cfg)
    print(
        f"[target-policy full] data loaded: samples={len(labels)}, classes={len(np.unique(labels))}, "
        f"source={data_source}",
        flush=True,
    )
    splits = data_splits if bool(args.full_dataset) else _load_splits(run_dir)
    class_condition_priors: dict[int, np.ndarray] | None = None
    class_condition_summary: dict[str, object] = {"mode": str(args.candidate_class_condition_mode), "classes": 0}
    if str(args.candidate_class_condition_mode) == "train_saliency" and float(args.class_condition_weight) > 0.0:
        train_for_priors = np.asarray(splits["train"], dtype=np.int64)
        print(
            f"[target-policy full] building class-condition train saliency priors: "
            f"train_rows={len(train_for_priors)}, max_per_class={int(args.class_condition_max_train_per_class)}, "
            f"weight={float(args.class_condition_weight):.3f}",
            flush=True,
        )
        class_condition_priors, class_condition_summary = _build_class_condition_priors(
            raw,
            labels,
            train_for_priors,
            target_cfg,
            max_train_per_class=int(args.class_condition_max_train_per_class),
        )
        class_condition_summary["weight"] = float(args.class_condition_weight)
        print(f"[target-policy full] class-condition priors ready: {class_condition_summary}", flush=True)
    if str(args.eval_split) == "all":
        test_indices = np.arange(len(labels), dtype=np.int64)
    else:
        test_indices = np.asarray(splits[str(args.eval_split)], dtype=np.int64)
    if int(args.max_test_traces) > 0:
        test_indices = test_indices[: int(args.max_test_traces)]
    print(
        f"[target-policy full] split ready: test_traces={len(test_indices)}, "
        f"max_test_traces={int(args.max_test_traces)}, attack_eval_mode={args.attack_eval_mode}",
        flush=True,
    )
    bundle = None
    attack_models: dict[str, tuple[torch.nn.Module, np.ndarray, float | None, Path]] = {}
    if str(args.attack_eval_mode) == "strong_surrogate":
        print("[target-policy full] loading strong surrogate ensemble...", flush=True)
        bundle = load_strong_surrogates(run_dir, defense_cfg, device)
        print("[target-policy full] strong surrogate ensemble loaded", flush=True)
    else:
        for kind, explicit in (("df", str(args.df_checkpoint)), ("rf", str(args.rf_checkpoint))):
            print(f"[target-policy full] loading {kind.upper()} evaluator checkpoint...", flush=True)
            checkpoint = _find_attack_checkpoint(run_dir, kind, explicit, str(args.attack_checkpoint_dir))
            model, classes, best_val = _load_attack_checkpoint(checkpoint, kind, defense_cfg, device)
            attack_models[kind] = (model, classes, best_val, checkpoint)
            print(
                f"[target-policy full] {kind.upper()} evaluator loaded: classes={len(classes)}, "
                f"best_val={best_val}, path={checkpoint}",
                flush=True,
            )
    evaluator_checkpoint_paths = {
        kind: str(values[3].resolve())
        for kind, values in attack_models.items()
    }
    print(f"[target-policy full] teacher_eval_mode={args.teacher_eval_mode}; loading teacher scorer...", flush=True)
    teacher_scorer = _load_teacher_scorer(args, run_dir, defense_cfg, target_cfg, device)
    teacher_checkpoint_paths = {} if teacher_scorer is None else teacher_scorer.checkpoint_paths
    print(
        f"[target-policy full] teacher scorer ready: "
        f"{teacher_checkpoint_paths if teacher_checkpoint_paths else 'heuristic proxy only'}",
        flush=True,
    )
    teacher_score_totals = _new_teacher_score_totals()
    teacher_scored_candidate_count = 0
    teacher_scored_trace_count = 0
    teacher_unscored_trace_count = 0
    batch_size = int(args.generation_batch_size)
    if batch_size <= 0:
        batch_size = 256 if bool(args.full_dataset) else len(test_indices)
    batch_size = max(1, int(batch_size))
    rng = np.random.default_rng(int(args.seed))
    clean_totals = _new_surrogate_totals()
    defended_totals = _new_surrogate_totals()
    attack_clean_totals = {kind: _new_metric_totals() for kind in attack_models}
    attack_defended_totals = {kind: _new_metric_totals() for kind in attack_models}
    subset_names = ("deployable", "non_deployable")
    attack_subset_clean_totals = {
        kind: {subset: _new_metric_totals() for subset in subset_names}
        for kind in attack_models
    }
    attack_subset_defended_totals = {
        kind: {subset: _new_metric_totals() for subset in subset_names}
        for kind in attack_models
    }
    subset_counts = Counter()
    candidate_reason_counters: dict[str, Counter] = {}
    selected_reason_counters: dict[str, Counter] = {}
    fallback_count = 0
    chosen_fallback_count = 0
    chosen_non_deployable_count = 0
    raw_bandwidth_sum = 0.0
    raw_retention_sum = 0.0
    rendered_count = 0
    surrogate_eval_traces = 0
    selected_preview: list[dict[str, object]] = []
    records_path = output_dir / "selected_records.jsonl"
    total_batches = int(np.ceil(len(test_indices) / float(batch_size))) if len(test_indices) else 0
    print(
        f"[target-policy full] starting generation: batches={total_batches}, batch_size={batch_size}, "
        f"num_candidates={int(target_cfg.num_candidates)}, target_count={int(target_cfg.target_count)}, "
        f"progress_every={int(args.progress_every)}",
        flush=True,
    )
    with records_path.open("w", encoding="utf-8") as record_file:
        for batch_id, start in enumerate(range(0, len(test_indices), batch_size), start=1):
            end = min(start + batch_size, len(test_indices))
            if int(args.progress_every) > 0 and (
                batch_id == 1 or batch_id % int(args.progress_every) == 0 or batch_id == total_batches
            ):
                print(
                    f"[target-policy full] batch {batch_id}/{total_batches} start: "
                    f"rows={start}-{end - 1}, size={end - start}",
                    flush=True,
                )
            batch_indices = test_indices[start:end]
            clean_rows = np.asarray(raw[batch_indices], dtype=np.float32)
            y = labels[batch_indices].astype(np.int64)
            supported_count = 0
            surrogate_mask = np.zeros_like(y, dtype=bool)
            if bundle is not None:
                surrogate_mask = np.isin(y, bundle.classes)
                supported_count = int(np.sum(surrogate_mask))
                clean_batch_metrics = ensemble_metrics_from_rendered(
                    _clean_trace_list(clean_rows[surrogate_mask]),
                    y[surrogate_mask],
                    bundle,
                    defense_cfg,
                    device,
                )
                _accumulate_surrogate(clean_totals, clean_batch_metrics, supported_count)
            for kind, (model, classes, _, _) in attack_models.items():
                clean_batch_metrics = _evaluate_attack_checkpoint(
                    kind,
                    model,
                    classes,
                    clean_rows,
                    y,
                    defense_cfg,
                    int(args.fixed_batch_size),
                    device,
                )
                _accumulate_metrics(attack_clean_totals[kind], clean_batch_metrics, len(y))
            templates, fallback, chosen_fallback, chosen_non_deployable, records, batch_candidate_counters, batch_selected_counters, batch_teacher_stats = _generate_templates_for_batch(
                clean_rows,
                y,
                batch_indices,
                start,
                trace_ids,
                target_cfg,
                rng,
                int(args.seed),
                teacher_scorer=teacher_scorer,
                teacher_totals=teacher_score_totals,
                class_condition_priors=class_condition_priors,
                class_condition_weight=float(args.class_condition_weight),
            )
            teacher_scored_candidate_count += int(batch_teacher_stats.get("scored_candidate_count", 0))
            teacher_scored_trace_count += int(batch_teacher_stats.get("scored_trace_count", 0))
            teacher_unscored_trace_count += int(batch_teacher_stats.get("unscored_trace_count", 0))
            _merge_counter_maps(candidate_reason_counters, batch_candidate_counters)
            _merge_counter_maps(selected_reason_counters, batch_selected_counters)
            traces, _, raw_stats = render_batch_variable(
                clean_rows,
                templates,
                seeds=[int(args.seed) + int(start) + idx for idx in range(len(templates))],
                coordinate_length=int(defense_cfg.max_trace_length),
                **renderer_options_from_config(defense_cfg),
            )
            if bundle is not None and supported_count:
                defended_batch_metrics = ensemble_metrics_from_rendered(
                    _indexed_trace_list(traces, surrogate_mask),
                    y[surrogate_mask],
                    bundle,
                    defense_cfg,
                    device,
                )
                _accumulate_surrogate(defended_totals, defended_batch_metrics, supported_count)
                surrogate_eval_traces += supported_count
            for kind, (model, classes, _, _) in attack_models.items():
                defended_batch_metrics = _evaluate_attack_checkpoint_from_traces(
                    kind,
                    model,
                    classes,
                    traces,
                    y,
                    defense_cfg,
                    int(args.fixed_batch_size),
                    device,
                )
                _accumulate_metrics(attack_defended_totals[kind], defended_batch_metrics, len(y))
            deployable_mask = np.asarray([bool(record["deployable"]) for record in records], dtype=bool)
            subset_masks = {
                "deployable": deployable_mask,
                "non_deployable": ~deployable_mask,
            }
            for subset, mask in subset_masks.items():
                count = int(np.sum(mask))
                subset_counts[subset] += count
                if count <= 0:
                    continue
                subset_traces = _indexed_trace_list(traces, mask)
                for kind, (model, classes, _, _) in attack_models.items():
                    clean_subset_metrics = _evaluate_attack_checkpoint(
                        kind,
                        model,
                        classes,
                        clean_rows[mask],
                        y[mask],
                        defense_cfg,
                        int(args.fixed_batch_size),
                        device,
                    )
                    defended_subset_metrics = _evaluate_attack_checkpoint_from_traces(
                        kind,
                        model,
                        classes,
                        subset_traces,
                        y[mask],
                        defense_cfg,
                        int(args.fixed_batch_size),
                        device,
                    )
                    _accumulate_metrics(attack_subset_clean_totals[kind][subset], clean_subset_metrics, count)
                    _accumulate_metrics(attack_subset_defended_totals[kind][subset], defended_subset_metrics, count)
            fallback_count += int(fallback)
            chosen_fallback_count += int(chosen_fallback)
            chosen_non_deployable_count += int(chosen_non_deployable)
            raw_bandwidth_sum += float(np.sum(raw_stats["raw_bandwidth"]))
            raw_retention_sum += float(np.sum(raw_stats["raw_real_packet_retention"]))
            rendered_count += int(len(templates))
            for record in records:
                record_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                if len(selected_preview) < int(args.record_preview_limit):
                    selected_preview.append(record)
            if int(args.progress_every) > 0 and (batch_id == 1 or batch_id % int(args.progress_every) == 0 or batch_id == total_batches):
                print(
                    f"[target-policy full] batch {batch_id}/{total_batches}, "
                    f"rows={end}/{len(test_indices)}, eval_mode={args.attack_eval_mode}, surrogate_supported={surrogate_eval_traces}, "
                    f"teacher_scored_candidates={teacher_scored_candidate_count}, "
                    f"bandwidth_mean={raw_bandwidth_sum / max(rendered_count, 1):.4f}",
                    flush=True,
                )
    clean_metrics = _finalize_surrogate(clean_totals, bundle.attacker_names) if bundle is not None else {}
    defended_metrics = _finalize_surrogate(defended_totals, bundle.attacker_names) if bundle is not None else {}
    fixed_metrics = {}
    subset_metrics: dict[str, dict[str, dict[str, float]]] = {subset: {} for subset in subset_names}
    for kind, (_, _, best_val, checkpoint) in attack_models.items():
        clean_attack_metrics = _finalize_metrics(attack_clean_totals[kind])
        defended_attack_metrics = _finalize_metrics(attack_defended_totals[kind])
        fixed_metrics[kind] = _attack_metric_row(clean_attack_metrics, defended_attack_metrics, best_val, checkpoint)
        for subset in subset_names:
            clean_subset = _finalize_metrics(attack_subset_clean_totals[kind][subset])
            defended_subset = _finalize_metrics(attack_subset_defended_totals[kind][subset])
            subset_metrics[subset][kind] = _metric_row_from_clean_defended(clean_subset, defended_subset)
    attacker_drops = {
        f"{kind}_accuracy": float(values["accuracy_drop"])
        for kind, values in fixed_metrics.items()
    }
    metrics = {
        "method": "target_policy_direct_v1_candidate_generator",
        "note": "This evaluates the direct x0* candidate generator, not a fully trained target diffusion sampler.",
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
        "attack_eval_mode": str(args.attack_eval_mode),
        "teacher_eval_mode": str(args.teacher_eval_mode),
        "target_scoring_source": f"frozen_checkpoint_{args.teacher_target_mode}_logit_feedback" if teacher_scorer is not None else "heuristic_proxy_not_df_rf_checkpoint",
        "teacher_scoring_source": f"checkpoint_{args.teacher_target_mode}_logit_feedback" if teacher_scorer is not None else "heuristic_proxy",
        "teacher_target_mode": str(args.teacher_target_mode),
        "candidate_class_condition_mode": str(args.candidate_class_condition_mode),
        "class_condition_weight": float(args.class_condition_weight),
        "class_condition_summary": class_condition_summary,
        "teacher_checkpoint_paths": teacher_checkpoint_paths,
        "evaluator_checkpoint_paths": evaluator_checkpoint_paths,
        "teacher_evaluator_checkpoint_relation": _teacher_evaluator_relation(teacher_checkpoint_paths, evaluator_checkpoint_paths, str(args.attack_eval_mode)),
        "teacher_scored_candidate_count": int(teacher_scored_candidate_count),
        "teacher_scored_trace_count": int(teacher_scored_trace_count),
        "teacher_unscored_trace_count": int(teacher_unscored_trace_count),
        "teacher_score_summary": _finalize_teacher_score_summary(teacher_score_totals),
        "full_dataset": bool(args.full_dataset),
        "eval_split": str(args.eval_split),
        "loaded_samples": int(len(labels)),
        "loaded_classes": int(len(np.unique(labels))),
        "test_traces": int(len(test_indices)),
        "surrogate_eval_traces": int(surrogate_eval_traces),
        "surrogate_skipped_traces": int(len(test_indices) - int(surrogate_eval_traces)),
        "surrogate_supported_classes": [] if bundle is None else [int(value) for value in bundle.classes.tolist()],
        "attack_checkpoint_classes": {
            kind: int(len(values[1]))
            for kind, values in attack_models.items()
        },
        "fallback_count": int(fallback_count),
        "chosen_fallback_count": int(chosen_fallback_count),
        "chosen_non_deployable_count": int(chosen_non_deployable_count),
        "subset_counts": {key: int(value) for key, value in subset_counts.items()},
        "subset_attack_metrics": subset_metrics,
        "non_deployable_diagnostics": {
            **_serialize_counter_map(candidate_reason_counters, limit=300),
            **_serialize_counter_map(selected_reason_counters, limit=300),
        },
        "raw_bandwidth_mean": float(raw_bandwidth_sum / max(rendered_count, 1)),
        "raw_retention_mean": float(raw_retention_sum / max(rendered_count, 1)),
        "clean": clean_metrics,
        "defended": defended_metrics,
        "fixed_attackers": fixed_metrics,
        "drops": attacker_drops if fixed_metrics else {
            key.replace("surrogate_", ""): float(clean_metrics[key] - defended_metrics.get(key, 0.0))
            for key in clean_metrics
            if key.endswith("_accuracy") and key in defended_metrics
        },
        "selected_records_path": str(records_path.resolve()),
        "selected_records_preview": selected_preview,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# target_policy_direct_v1 full CW defense evaluation",
        "",
        f"- run_dir: {run_dir}",
        f"- data source: {data_source}",
        f"- full dataset: {bool(args.full_dataset)}",
        f"- eval split: {args.eval_split}",
        f"- loaded samples/classes: {len(labels)}/{len(np.unique(labels))}",
        f"- test traces: {len(test_indices)}",
        f"- attack eval mode: {args.attack_eval_mode}",
        f"- teacher eval mode: {args.teacher_eval_mode}",
        f"- teacher target mode: {args.teacher_target_mode}",
        f"- candidate class condition mode: {args.candidate_class_condition_mode}",
        f"- class condition weight: {float(args.class_condition_weight):.4f}",
        f"- teacher scoring source: {metrics['teacher_scoring_source']}",
        f"- teacher/evaluator checkpoint relation: {metrics['teacher_evaluator_checkpoint_relation']}",
        f"- teacher checkpoints: {teacher_checkpoint_paths}",
        f"- evaluator checkpoints: {evaluator_checkpoint_paths}",
        f"- teacher scored candidates/traces/unscored traces: {teacher_scored_candidate_count}/{teacher_scored_trace_count}/{teacher_unscored_trace_count}",
        f"- surrogate eval/skipped traces: {int(surrogate_eval_traces)}/{len(test_indices) - int(surrogate_eval_traces)}",
        f"- budget: {float(args.budget):.4f}",
        f"- render coordinate: {metrics['render_coordinate']}",
        f"- multi-view mode: {metrics['multi_view_mode']}",
        f"- multi-view shares DF/AWF/RF: {metrics['multi_view_df_share']:.3f}/{metrics['multi_view_awf_share']:.3f}/{metrics['multi_view_rf_share']:.3f}",
        f"- TAM obfuscation strategy: {metrics['tam_obfuscation_strategy']}",
        f"- raw bandwidth mean: {metrics['raw_bandwidth_mean']:.6f}",
        f"- raw retention mean: {metrics['raw_retention_mean']:.6f}",
        f"- fallback count: {fallback_count}",
        f"- chosen fallback/non-deployable count: {chosen_fallback_count}/{chosen_non_deployable_count}",
        f"- subset counts: deployable={int(subset_counts['deployable'])}, non_deployable={int(subset_counts['non_deployable'])}",
        f"- selected records: {records_path}",
        "",
        "| attacker | clean acc | defended acc | drop pp | clean entropy | defended entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    if fixed_metrics:
        for name, values in fixed_metrics.items():
            clean_acc = float(values["clean_accuracy"])
            def_acc = float(values["defended_accuracy"])
            clean_entropy = float(values["clean_entropy"])
            def_entropy = float(values["defended_entropy"])
            lines.append(f"| {name.upper()} | {clean_acc:.6f} | {def_acc:.6f} | {(clean_acc - def_acc) * 100.0:.2f} | {clean_entropy:.6f} | {def_entropy:.6f} |")
    elif bundle is not None:
        for name in bundle.attacker_names:
            clean_acc = float(clean_metrics.get(f"surrogate_{name}_accuracy", 0.0))
            def_acc = float(defended_metrics.get(f"surrogate_{name}_accuracy", 0.0))
            clean_entropy = float(clean_metrics.get(f"surrogate_{name}_entropy", 0.0))
            def_entropy = float(defended_metrics.get(f"surrogate_{name}_entropy", 0.0))
            lines.append(f"| {name.upper()} | {clean_acc:.6f} | {def_acc:.6f} | {(clean_acc - def_acc) * 100.0:.2f} | {clean_entropy:.6f} | {def_entropy:.6f} |")
    if subset_metrics:
        lines.extend(["", "## Subset Attack Metrics", ""])
        for subset in subset_names:
            lines.extend([f"### {subset}", "", "| attacker | clean acc | defended acc | drop pp | clean entropy | defended entropy |", "|---|---:|---:|---:|---:|---:|"])
            for name, values in subset_metrics[subset].items():
                if not values:
                    continue
                lines.append(
                    f"| {name.upper()} | {values['clean_accuracy']:.6f} | {values['defended_accuracy']:.6f} | "
                    f"{100.0 * values['accuracy_drop']:.2f} | {values['clean_entropy']:.6f} | {values['defended_entropy']:.6f} |"
                )
            lines.append("")
    if candidate_reason_counters or selected_reason_counters:
        lines.extend(["", "## Non-Deployable Reasons", "", "### Selected"])
        for key, value in _serialize_counter_map(selected_reason_counters, limit=30).items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "### All Candidates"])
        for key, value in _serialize_counter_map(candidate_reason_counters, limit=30).items():
            lines.append(f"- {key}: {value}")
    if teacher_scorer is not None:
        lines.extend(["", "## Teacher Score Summary", ""])
        summary = metrics["teacher_score_summary"]
        lines.append(json.dumps(summary, indent=2, ensure_ascii=False))
    (output_dir / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ["output_dir", "budget", "render_coordinate", "multi_view_mode", "multi_view_df_share", "multi_view_awf_share", "multi_view_rf_share", "tam_obfuscation_strategy", "tam_slot_jitter", "tam_cluster_ratio", "tam_local_run_max", "tam_preserve_real_timestamps", "attack_eval_mode", "teacher_eval_mode", "teacher_target_mode", "candidate_class_condition_mode", "class_condition_weight", "class_condition_summary", "teacher_scoring_source", "teacher_checkpoint_paths", "evaluator_checkpoint_paths", "teacher_evaluator_checkpoint_relation", "teacher_scored_candidate_count", "teacher_scored_trace_count", "teacher_unscored_trace_count", "teacher_score_summary", "full_dataset", "eval_split", "loaded_samples", "loaded_classes", "test_traces", "surrogate_eval_traces", "surrogate_skipped_traces", "attack_checkpoint_classes", "fallback_count", "chosen_fallback_count", "chosen_non_deployable_count", "subset_counts", "raw_bandwidth_mean", "raw_retention_mean", "fixed_attackers", "subset_attack_metrics", "drops", "non_deployable_diagnostics"]}, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    output_dir = _resolve_output_dir(args.output_dir, bool(args.overwrite))
    print(
        f"[target-policy] started: cwd={Path.cwd()}, run_dir={run_dir}, config={args.config}, "
        f"output_dir={output_dir}, device={args.device}",
        flush=True,
    )
    defense_cfg = _apply_renderer_overrides(_defense_config_from_run(run_dir), args)
    data_cfg = _dataset_config(defense_cfg, bool(args.full_dataset))
    target_cfg = load_target_policy_config(args.config)
    target_cfg.budgets = (float(args.budget),)
    target_cfg.max_budget = max(float(target_cfg.max_budget), float(args.budget))
    if int(args.num_candidates) > 0:
        target_cfg.num_candidates = int(args.num_candidates)
    if int(args.target_count) > 0:
        target_cfg.target_count = int(args.target_count)
    target_cfg.quality_target_count = max(1, min(int(target_cfg.quality_target_count), int(target_cfg.target_count)))
    target_cfg.diverse_target_count = max(0, int(target_cfg.target_count) - int(target_cfg.quality_target_count))
    device = resolve_device(str(args.device))
    print(
        f"[target-policy] config ready: full_dataset={bool(args.full_dataset)}, "
        f"budget={float(args.budget):.4f}, num_candidates={int(target_cfg.num_candidates)}, "
        f"target_count={int(target_cfg.target_count)}, teacher_eval_mode={args.teacher_eval_mode}, "
        f"attack_eval_mode={args.attack_eval_mode}, render_coordinate={defense_cfg.render_coordinate}, "
        f"multi_view_mode={defense_cfg.multi_view_mode}, "
        f"multi_view_shares={defense_cfg.multi_view_df_share:.3f}/{defense_cfg.multi_view_awf_share:.3f}/{defense_cfg.multi_view_rf_share:.3f}, "
        f"tam_obfuscation_strategy={defense_cfg.tam_obfuscation_strategy}, resolved_device={device}",
        flush=True,
    )
    if bool(args.full_dataset) or int(args.generation_batch_size) > 0:
        _run_batched_evaluation(args, run_dir, output_dir, defense_cfg, data_cfg, target_cfg, device)
        return
    print("[target-policy small] loading CW data...", flush=True)
    raw, labels, trace_ids, data_splits, data_source = load_cw_data(data_cfg)
    print(
        f"[target-policy small] data loaded: samples={len(labels)}, classes={len(np.unique(labels))}, "
        f"source={data_source}",
        flush=True,
    )
    splits = _load_splits(run_dir)
    if bool(args.full_dataset):
        splits = data_splits
    class_condition_priors: dict[int, np.ndarray] | None = None
    class_condition_summary: dict[str, object] = {"mode": str(args.candidate_class_condition_mode), "classes": 0}
    if str(args.candidate_class_condition_mode) == "train_saliency" and float(args.class_condition_weight) > 0.0:
        train_for_priors = np.asarray(splits["train"], dtype=np.int64)
        print(
            f"[target-policy small] building class-condition train saliency priors: "
            f"train_rows={len(train_for_priors)}, max_per_class={int(args.class_condition_max_train_per_class)}, "
            f"weight={float(args.class_condition_weight):.3f}",
            flush=True,
        )
        class_condition_priors, class_condition_summary = _build_class_condition_priors(
            raw,
            labels,
            train_for_priors,
            target_cfg,
            max_train_per_class=int(args.class_condition_max_train_per_class),
        )
        class_condition_summary["weight"] = float(args.class_condition_weight)
        print(f"[target-policy small] class-condition priors ready: {class_condition_summary}", flush=True)
    if str(args.eval_split) == "all":
        test_indices = np.arange(len(labels), dtype=np.int64)
    else:
        test_indices = np.asarray(splits[str(args.eval_split)], dtype=np.int64)
    if int(args.max_test_traces) > 0:
        test_indices = test_indices[: int(args.max_test_traces)]
    clean_rows = np.asarray(raw[test_indices], dtype=np.float32)
    y = labels[test_indices].astype(np.int64)
    bundle = load_strong_surrogates(run_dir, defense_cfg, device)
    surrogate_mask = np.isin(y, bundle.classes)
    clean_metrics = ensemble_metrics_from_rendered(_clean_trace_list(clean_rows[surrogate_mask]), y[surrogate_mask], bundle, defense_cfg, device)
    rng = np.random.default_rng(int(args.seed))
    teacher_scorer = _load_teacher_scorer(args, run_dir, defense_cfg, target_cfg, device)
    teacher_checkpoint_paths = {} if teacher_scorer is None else teacher_scorer.checkpoint_paths
    evaluator_checkpoint_paths: dict[str, str] = {}
    teacher_score_totals = _new_teacher_score_totals()
    teacher_scored_candidate_count = 0
    teacher_scored_trace_count = 0
    teacher_unscored_trace_count = 0
    templates: list[PaddingTemplate] = []
    fallback_count = 0
    selected_records = []
    for row_id, clean_index in enumerate(test_indices.tolist()):
        clean = clean_rows[row_id]
        condition = extract_prefix_condition(
            clean,
            prefix_n=int(target_cfg.prefix_length),
            patch_num=int(target_cfg.strategy_horizon),
        )
        class_prior = None
        if class_condition_priors is not None:
            class_prior = class_condition_priors.get(int(y[row_id]))
        candidates = generate_candidates_for_trace(
            clean,
            cfg=target_cfg,
            prefix_condition=condition,
            clean_index=int(clean_index),
            class_condition_prior=class_prior,
            class_condition_weight=float(args.class_condition_weight),
            rng=rng,
        )
        if teacher_scorer is not None:
            stats = teacher_scorer.score_candidates(
                clean,
                candidates,
                render_seed=int(args.seed) + int(row_id),
                true_label=int(y[row_id]),
                totals=teacher_score_totals,
            )
            teacher_scored_candidate_count += int(stats.get("scored_candidate_count", 0))
            teacher_scored_trace_count += int(stats.get("scored_trace_count", 0))
            teacher_unscored_trace_count += int(stats.get("unscored_trace_count", 0))
        selected, fallback = select_targets(
            candidates,
            target_count=int(target_cfg.target_count),
            quality_target_count=int(target_cfg.quality_target_count),
            diverse_target_count=int(target_cfg.diverse_target_count),
            allocation_l1_weight=float(target_cfg.allocation_l1_weight),
            allocation_cosine_weight=float(target_cfg.allocation_cosine_weight),
        )
        fallback_count += int(fallback)
        chosen = selected[0]
        templates.append(
            PaddingTemplate(
                counts=chosen.counts,
                target_n_pad=int(chosen.budget_count),
                actual_n_pad=int(chosen.counts.sum()),
                target_bandwidth=float(chosen.budget_ratio),
                metadata={
                    "method": "target_policy_direct_v1",
                    "clean_index": int(clean_index),
                    "trace_id": str(trace_ids[int(clean_index)]),
                    "proxy_score_attack": float(chosen.proxy_score_attack),
                    "selection_score_attack": _optional_float(getattr(chosen, "selection_score_attack", None)),
                    "teacher_scored": bool(getattr(chosen, "teacher_scored", False)),
                    "teacher_score_source": str(getattr(chosen, "teacher_score_source", "heuristic_proxy")),
                    "teacher_target_mode": "none" if teacher_scorer is None else str(teacher_scorer.target_mode),
                    "candidate_class_condition_mode": "none" if class_condition_priors is None else "train_saliency",
                    "class_condition_weight": float(args.class_condition_weight),
                    "fallback": bool(chosen.fallback_flag),
                },
            )
        )
        record = _selected_record(int(row_id), int(clean_index), str(trace_ids[int(clean_index)]), chosen)
        record["true_label"] = int(y[row_id])
        record["teacher_target_mode"] = "none" if teacher_scorer is None else str(teacher_scorer.target_mode)
        record["candidate_class_condition_mode"] = "none" if class_condition_priors is None else "train_saliency"
        record["class_condition_weight"] = float(args.class_condition_weight)
        selected_records.append(record)
    traces, origins, raw_stats = render_batch_variable(
        clean_rows,
        templates,
        seeds=[int(args.seed) + idx for idx in range(len(templates))],
        coordinate_length=int(defense_cfg.max_trace_length),
        **renderer_options_from_config(defense_cfg),
    )
    defended_metrics = ensemble_metrics_from_rendered(_indexed_trace_list(traces, surrogate_mask), y[surrogate_mask], bundle, defense_cfg, device)
    defended_padded = _ragged_to_padded(traces, int(defense_cfg.max_trace_length))
    fixed_metrics = {}
    if bool(args.train_fixed_attackers):
        train_indices = np.asarray(splits["train"], dtype=np.int64)
        val_indices = np.asarray(splits["val"], dtype=np.int64)
        train_rows = np.asarray(raw[train_indices], dtype=np.float32)
        val_rows = np.asarray(raw[val_indices], dtype=np.float32)
        for kind in ("df", "rf"):
            fixed_metrics[kind] = _train_and_eval_fixed(
                kind,
                train_rows,
                labels[train_indices].astype(np.int64),
                val_rows,
                labels[val_indices].astype(np.int64),
                clean_rows,
                defended_padded,
                y,
                defense_cfg,
                args,
                device,
            )
    metrics = {
        "method": "target_policy_direct_v1_candidate_generator",
        "note": "This evaluates the direct x0* candidate generator, not a fully trained target diffusion sampler.",
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
        "teacher_eval_mode": str(args.teacher_eval_mode),
        "target_scoring_source": f"frozen_checkpoint_{args.teacher_target_mode}_logit_feedback" if teacher_scorer is not None else "heuristic_proxy_not_df_rf_checkpoint",
        "teacher_scoring_source": f"checkpoint_{args.teacher_target_mode}_logit_feedback" if teacher_scorer is not None else "heuristic_proxy",
        "teacher_target_mode": str(args.teacher_target_mode),
        "candidate_class_condition_mode": str(args.candidate_class_condition_mode),
        "class_condition_weight": float(args.class_condition_weight),
        "class_condition_summary": class_condition_summary,
        "teacher_checkpoint_paths": teacher_checkpoint_paths,
        "evaluator_checkpoint_paths": evaluator_checkpoint_paths,
        "teacher_evaluator_checkpoint_relation": _teacher_evaluator_relation(teacher_checkpoint_paths, evaluator_checkpoint_paths, str(args.attack_eval_mode)),
        "teacher_scored_candidate_count": int(teacher_scored_candidate_count),
        "teacher_scored_trace_count": int(teacher_scored_trace_count),
        "teacher_unscored_trace_count": int(teacher_unscored_trace_count),
        "teacher_score_summary": _finalize_teacher_score_summary(teacher_score_totals),
        "full_dataset": bool(args.full_dataset),
        "eval_split": str(args.eval_split),
        "loaded_samples": int(len(labels)),
        "loaded_classes": int(len(np.unique(labels))),
        "test_traces": int(len(test_indices)),
        "surrogate_eval_traces": int(np.sum(surrogate_mask)),
        "surrogate_skipped_traces": int(len(y) - int(np.sum(surrogate_mask))),
        "surrogate_supported_classes": [int(value) for value in bundle.classes.tolist()],
        "fallback_count": int(fallback_count),
        "raw_bandwidth_mean": float(np.mean(raw_stats["raw_bandwidth"])) if len(templates) else 0.0,
        "raw_retention_mean": float(np.mean(raw_stats["raw_real_packet_retention"])) if len(templates) else 1.0,
        "clean": clean_metrics,
        "defended": defended_metrics,
        "fixed_attackers": fixed_metrics,
        "drops": {
            key.replace("surrogate_", ""): float(clean_metrics[key] - defended_metrics.get(key, 0.0))
            for key in clean_metrics
            if key.endswith("_accuracy") and key in defended_metrics
        },
        "selected_records": selected_records,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# target_policy_direct_v1 defense evaluation",
        "",
        f"- run_dir: {run_dir}",
        f"- test traces: {len(test_indices)}",
        f"- full dataset: {bool(args.full_dataset)}",
        f"- eval split: {args.eval_split}",
        f"- loaded samples/classes: {len(labels)}/{len(np.unique(labels))}",
        f"- teacher eval mode: {args.teacher_eval_mode}",
        f"- teacher target mode: {args.teacher_target_mode}",
        f"- candidate class condition mode: {args.candidate_class_condition_mode}",
        f"- class condition weight: {float(args.class_condition_weight):.4f}",
        f"- teacher scoring source: {metrics['teacher_scoring_source']}",
        f"- teacher checkpoints: {teacher_checkpoint_paths}",
        f"- teacher scored candidates/traces/unscored traces: {teacher_scored_candidate_count}/{teacher_scored_trace_count}/{teacher_unscored_trace_count}",
        f"- surrogate eval/skipped traces: {int(np.sum(surrogate_mask))}/{len(y) - int(np.sum(surrogate_mask))}",
        f"- budget: {float(args.budget):.4f}",
        f"- render coordinate: {metrics['render_coordinate']}",
        f"- multi-view mode: {metrics['multi_view_mode']}",
        f"- multi-view shares DF/AWF/RF: {metrics['multi_view_df_share']:.3f}/{metrics['multi_view_awf_share']:.3f}/{metrics['multi_view_rf_share']:.3f}",
        f"- TAM obfuscation strategy: {metrics['tam_obfuscation_strategy']}",
        f"- raw bandwidth mean: {metrics['raw_bandwidth_mean']:.6f}",
        f"- raw retention mean: {metrics['raw_retention_mean']:.6f}",
        f"- fallback count: {fallback_count}",
        "",
        "| attacker | clean acc | defended acc | drop pp | clean entropy | defended entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in bundle.attacker_names:
        clean_acc = float(clean_metrics.get(f"surrogate_{name}_accuracy", 0.0))
        def_acc = float(defended_metrics.get(f"surrogate_{name}_accuracy", 0.0))
        clean_entropy = float(clean_metrics.get(f"surrogate_{name}_entropy", 0.0))
        def_entropy = float(defended_metrics.get(f"surrogate_{name}_entropy", 0.0))
        lines.append(f"| {name.upper()} | {clean_acc:.6f} | {def_acc:.6f} | {(clean_acc - def_acc) * 100.0:.2f} | {clean_entropy:.6f} | {def_entropy:.6f} |")
    if fixed_metrics:
        lines.extend(["", "## Clean-trained fixed attackers", "", "| attacker | val acc | clean test acc | defended acc | drop pp |", "|---|---:|---:|---:|---:|"])
        for name, values in fixed_metrics.items():
            lines.append(
                f"| {name.upper()} | {values['best_val_accuracy']:.6f} | {values['clean_accuracy']:.6f} | {values['defended_accuracy']:.6f} | {100.0 * values['accuracy_drop']:.2f} |"
            )
    if teacher_scorer is not None:
        lines.extend(["", "## Teacher Score Summary", "", json.dumps(metrics["teacher_score_summary"], indent=2, ensure_ascii=False)])
    (output_dir / "summary_zh.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: metrics[key] for key in ["output_dir", "budget", "render_coordinate", "multi_view_mode", "multi_view_df_share", "multi_view_awf_share", "multi_view_rf_share", "tam_obfuscation_strategy", "tam_slot_jitter", "tam_cluster_ratio", "tam_local_run_max", "tam_preserve_real_timestamps", "teacher_eval_mode", "teacher_target_mode", "candidate_class_condition_mode", "class_condition_weight", "class_condition_summary", "teacher_scoring_source", "teacher_checkpoint_paths", "teacher_scored_candidate_count", "teacher_scored_trace_count", "teacher_unscored_trace_count", "teacher_score_summary", "full_dataset", "eval_split", "loaded_samples", "loaded_classes", "test_traces", "surrogate_eval_traces", "surrogate_skipped_traces", "fallback_count", "raw_bandwidth_mean", "raw_retention_mean", "clean", "defended", "drops"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
