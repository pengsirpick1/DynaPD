"""Isolation experiment helpers for diagnosing DMMPv3 fixed-attacker regressions.

The commands in this file are intentionally orchestration-only. They create new
diagnostic outputs, reuse existing DMMPv3 modules, and never overwrite historical
DMMP/DMMP2 result directories.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
RESULTS_ROOT = PROJECT_ROOT / "results"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FIXED_DF = (
    REPO_ROOT
    / "results"
    / "dmmp2_v5_fixed_oriented_seed0_bwo30"
    / "attack_eval"
    / "fixed"
    / "df"
    / "fixed_df_checkpoint.pt"
)
DEFAULT_FIXED_RF = (
    REPO_ROOT
    / "results"
    / "dmmp2_v5_fixed_oriented_seed0_bwo30"
    / "attack_eval"
    / "fixed"
    / "rf"
    / "fixed_rf_checkpoint.pt"
)
REGISTERED_CLEAN_ACC = {"df": 0.973684, "rf": 0.976619}


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _project_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _ensure_under_results(path: Path) -> Path:
    resolved = path.resolve()
    root = RESULTS_ROOT.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"DMMPv3 diagnostic outputs must stay under {root}; got {resolved}") from exc
    return resolved


def _results_run_dir(path: str | Path, *, must_exist: bool = True) -> Path:
    value = Path(path)
    if value.is_absolute():
        candidate = value
    elif len(value.parts) == 1:
        candidate = RESULTS_ROOT / value
    else:
        candidate = PROJECT_ROOT / value
    resolved = _ensure_under_results(candidate)
    if must_exist and not resolved.is_dir():
        raise FileNotFoundError(f"DMMPv3 results run does not exist: {resolved}")
    return resolved


def _safe_run_name(run_name: str) -> str:
    value = Path(run_name)
    if not run_name or value.is_absolute() or len(value.parts) != 1 or run_name in {".", ".."}:
        raise ValueError(f"run_name must be a single new directory name under {RESULTS_ROOT}: {run_name!r}")
    return run_name


def _new_results_run_dir(run_name: str) -> Path:
    return _ensure_under_results(RESULTS_ROOT / _safe_run_name(run_name))


def _prepare_new_output_dir(path: Path, *, auto_suffix: bool = True) -> Path:
    output_dir = _ensure_under_results(path)
    if auto_suffix and output_dir.exists() and any(output_dir.iterdir()):
        output_dir = output_dir.with_name(f"{output_dir.name}_{_timestamp()}")
        output_dir = _ensure_under_results(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Diagnostic output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _array_digest(values: Any) -> str:
    import numpy as np

    arr = np.ascontiguousarray(values)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _load_defense_config(run_dir: Path, *, device: str = "auto", progress: bool = True) -> DefenseConfig:
    from dmmp.utils.config import DefenseConfig

    payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    values = {name: payload[name] for name in DefenseConfig.__dataclass_fields__ if name in payload}
    cfg = DefenseConfig(**values)
    data_root = Path(cfg.data_root)
    if not data_root.is_absolute():
        cfg.data_root = str((REPO_ROOT / data_root).resolve())
    cfg.output_dir = str(RESULTS_ROOT)
    cfg.run_name = run_dir.name
    cfg.device = str(device)
    cfg.progress = bool(progress)
    return cfg


def _load_splits(run_dir: Path) -> dict[str, np.ndarray]:
    import numpy as np

    payload = json.loads((run_dir / "split_indices.json").read_text(encoding="utf-8"))
    return {name: np.asarray(indices, dtype=np.int64) for name, indices in payload.items()}


def _attack_config(args: argparse.Namespace, run_dir: Path) -> AttackConfig:
    from dmmp.utils.config import AttackConfig

    return AttackConfig(
        run_dir=str(run_dir),
        attackers="fixed_df,fixed_rf",
        adaptive_protocol="fixed",
        attack_require_quality_gate=False,
        device=str(args.device),
        df_batch_size=int(args.batch_size),
        df_architecture=str(args.df_architecture),
        max_load_time=float(args.max_load_time),
        rf_tam_num_slots=int(args.rf_tam_num_slots),
        progress=bool(args.progress),
    )


def _load_fixed_model(
    kind: str,
    checkpoint_path: Path,
    cfg: DefenseConfig,
    attack_cfg: AttackConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, np.ndarray, dict[str, Any]]:
    import numpy as np
    import torch

    from dmmp.evaluation.attack_models import make_attack_model

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing fixed {kind.upper()} checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = payload.get("model_state") if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint does not contain a model_state dict: {checkpoint_path}")
    classes = np.asarray(payload.get("classes", np.arange(95)), dtype=np.int64) if isinstance(payload, dict) else np.arange(95)
    model = make_attack_model(
        kind.upper(),
        len(classes),
        max_trace_length=int(cfg.max_trace_length),
        df_architecture=str(attack_cfg.df_architecture),
    ).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    evidence = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_exists": True,
        "checkpoint_classes": int(len(classes)),
        "checkpoint_classes_digest": _array_digest(classes),
        "checkpoint_classes_are_0_to_n_minus_1": bool(np.array_equal(classes, np.arange(len(classes)))),
        "best_val": None if not isinstance(payload, dict) else payload.get("best_val"),
        "strict_state_load": True,
        "state_key_count": int(len(state)),
        "model_class": model.__class__.__name__,
        "model_parameter_count": int(sum(param.numel() for param in model.parameters())),
        "input_representation": {
            "attacker": kind.upper(),
            "max_trace_length": int(cfg.max_trace_length),
            "df_architecture": str(attack_cfg.df_architecture),
            "rf_tam_num_slots": int(attack_cfg.rf_tam_num_slots),
            "max_load_time": float(attack_cfg.max_load_time),
        },
    }
    return model, classes, evidence


def _metric_row(
    *,
    kind: str,
    checkpoint_path: Path,
    clean: dict[str, float],
    defended: dict[str, float],
    visible_bandwidth: float,
    raw_retention: float,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    clean_acc = float(clean["defended_accuracy"])
    defended_acc = float(defended["defended_accuracy"])
    return {
        "protocol": "fixed_checkpoint_reuse",
        "attacker": kind.upper(),
        "checkpoint_path": str(checkpoint_path),
        "clean_acc": clean_acc,
        "fresh_defended_acc": defended_acc,
        "accuracy_drop": clean_acc - defended_acc,
        "true_label_confidence": float(defended.get("true_label_confidence", 0.0)),
        "prediction_entropy": float(defended.get("prediction_entropy", 0.0)),
        "max_confidence": float(defended.get("max_confidence", 0.0)),
        "visible_bandwidth": float(visible_bandwidth),
        "raw_retention": float(raw_retention),
        "registered_clean_acc": REGISTERED_CLEAN_ACC.get(kind),
        "clean_acc_abs_delta_vs_registry": abs(clean_acc - REGISTERED_CLEAN_ACC.get(kind, clean_acc)),
        "compatibility_passed": int(bool(compatibility.get("compatibility_passed", False))),
    }


def run_reuse_fixed_eval(args: argparse.Namespace) -> Path:
    import numpy as np

    from dmmp.constraints.user_profiles import load_profiles
    from dmmp.data import load_cw_data
    from dmmp.evaluation.attacks import _eval_torch
    from dmmp.evaluation.profile_attacks import (
        _clean_input_indexed,
        _defended_input,
        _defense_artifact_signature,
        _find_profile,
        _get_profile_dataset,
        _selected_budget_and_keep,
    )
    from dmmp.utils import resolve_device, set_seed, write_csv, write_json

    run_dir = _results_run_dir(args.run_dir)
    output_dir = _prepare_new_output_dir(run_dir / "isolation" / "reuse_fixed_eval")
    cfg = _load_defense_config(run_dir, device=args.device, progress=args.progress)
    attack_cfg = _attack_config(args, run_dir)
    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    raw, labels, trace_ids, _, data_source = load_cw_data(cfg)
    splits = _load_splits(run_dir)
    test_idx = np.asarray(splits["test"], dtype=np.int64)
    if int(args.max_test_traces) > 0 and len(test_idx) > int(args.max_test_traces):
        rng = np.random.default_rng(int(args.seed) + 13)
        test_idx = np.sort(rng.choice(test_idx, size=int(args.max_test_traces), replace=False)).astype(np.int64)
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    target = _find_profile(profiles, str(args.target_profile_id), "test")
    budget, keep_ratio = _selected_budget_and_keep(run_dir, cfg)
    defense_signature = _defense_artifact_signature(run_dir)
    defended_role = str(args.defended_role)
    defended_traces, origins, defended_y, defended_metrics, defended_path = _get_profile_dataset(
        raw,
        labels,
        trace_ids,
        test_idx,
        run_dir,
        cfg,
        target,
        defended_role,
        budget,
        keep_ratio,
        defense_signature,
        device,
    )
    checkpoint_paths = {
        "df": _project_path(args.fixed_df_checkpoint),
        "rf": _project_path(args.fixed_rf_checkpoint),
    }
    unique_labels = np.asarray(np.unique(labels), dtype=np.int64)
    split_evidence = {
        name: {"count": int(len(values)), "sha256": _array_digest(np.asarray(values, dtype=np.int64))}
        for name, values in splits.items()
    }
    rows = []
    details: dict[str, Any] = {
        "experiment": "reuse-fixed-eval",
        "theory": "Hold the fixed attacker constant to separate defense policy effects from fixed-attacker retraining variance.",
        "run_dir": str(run_dir),
        "data_source": str(data_source),
        "test_samples": int(len(test_idx)),
        "target_profile": target.profile_id,
        "budget": float(budget),
        "keep_ratio": float(keep_ratio),
        "defended_role": defended_role,
        "defended_dataset": str(defended_path),
        "defended_dataset_metrics": defended_metrics,
        "defense_artifact_signature_digest": defense_signature.get("digest"),
        "split_evidence": split_evidence,
        "evaluated_test_indices_sha256": _array_digest(test_idx),
        "unique_labels": unique_labels.astype(int).tolist(),
        "unique_labels_sha256": _array_digest(unique_labels),
        "compatibility_clean_acc_tolerance": float(args.clean_acc_tolerance),
        "attackers": {},
    }
    for kind, checkpoint_path in checkpoint_paths.items():
        model, classes, compatibility = _load_fixed_model(kind, checkpoint_path, cfg, attack_cfg, device)
        clean_x = _clean_input_indexed(kind, raw, test_idx, cfg, attack_cfg)
        defended_x, adapter_stats = _defended_input(kind, defended_traces, origins, cfg, attack_cfg)
        clean_metrics = _eval_torch(model, clean_x, labels[test_idx], classes, device, int(args.batch_size))
        defended_eval = _eval_torch(model, defended_x, defended_y, classes, device, int(args.batch_size))
        registered_clean_acc = REGISTERED_CLEAN_ACC.get(kind)
        clean_acc_delta = abs(float(clean_metrics["defended_accuracy"]) - float(registered_clean_acc))
        compatibility_checks = {
            "checkpoint_exists": bool(compatibility.get("checkpoint_exists")),
            "strict_state_load": bool(compatibility.get("strict_state_load")),
            "checkpoint_class_count_is_95": int(compatibility.get("checkpoint_classes", 0)) == 95,
            "checkpoint_classes_are_0_to_n_minus_1": bool(compatibility.get("checkpoint_classes_are_0_to_n_minus_1")),
            "dataset_label_count_is_95": int(len(unique_labels)) == 95,
            "checkpoint_classes_match_dataset_labels": bool(np.array_equal(classes, unique_labels)),
            "clean_acc_matches_registry": clean_acc_delta <= float(args.clean_acc_tolerance),
        }
        compatibility.update(
            {
                "data_source": str(data_source),
                "split_evidence": split_evidence,
                "evaluated_test_count": int(len(test_idx)),
                "evaluated_test_indices_sha256": _array_digest(test_idx),
                "label_count": int(len(np.unique(labels))),
                "unique_labels_sha256": _array_digest(unique_labels),
                "clean_acc": float(clean_metrics["defended_accuracy"]),
                "registered_clean_acc": registered_clean_acc,
                "clean_acc_abs_delta_vs_registry": clean_acc_delta,
                "clean_acc_tolerance": float(args.clean_acc_tolerance),
                "compatibility_checks": compatibility_checks,
                "compatibility_passed": bool(all(compatibility_checks.values())),
                "adapter_stats": adapter_stats,
            }
        )
        row = _metric_row(
            kind=kind,
            checkpoint_path=checkpoint_path,
            clean=clean_metrics,
            defended=defended_eval,
            visible_bandwidth=float(defended_metrics.get("visible_dummy_overhead", 0.0)),
            raw_retention=float(defended_metrics.get("raw_real_packet_retention", 0.0)),
            compatibility=compatibility,
        )
        rows.append(row)
        details["attackers"][kind] = {
            "compatibility": compatibility,
            "clean_metrics": clean_metrics,
            "defended_metrics": defended_eval,
            "summary_row": row,
        }
    write_csv(output_dir / "attack_summary.csv", rows)
    write_json(output_dir / "attack_summary.json", details)
    lines = [
        "# Isolation: Reused Fixed Checkpoints",
        "",
        f"- run: `{run_dir}`",
        f"- target profile: `{target.profile_id}`",
        f"- budget / keep ratio: `{budget:.4f}` / `{keep_ratio:.4f}`",
        f"- defended dataset: `{defended_path}`",
        "",
        "| attacker | clean acc | defended acc | drop | visible overhead | checkpoint |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['attacker']} | {row['clean_acc']:.6f} | {row['fresh_defended_acc']:.6f} | "
            f"{row['accuracy_drop']:.6f} | {row['visible_bandwidth']:.6f} | `{row['checkpoint_path']}` |"
        )
    lines.extend(
        [
            "",
            "This evaluation does not retrain fixed attackers. It is only valid when the compatibility fields in `attack_summary.json` remain true.",
        ]
    )
    (output_dir / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[reuse-fixed-eval] saved to {output_dir}", flush=True)
    return output_dir


def _copy_run_for_stage3(source: Path, run_name: str) -> Path:
    from dmmp.utils import write_json

    dest = _new_results_run_dir(run_name)
    if dest.exists():
        raise FileExistsError(f"Destination run already exists; choose a new --run_name: {dest}")
    if source.resolve() == dest.resolve():
        raise ValueError(f"Source and destination run directories must differ: {source}")

    def ignore(_dir: str, names: list[str]) -> set[str]:
        return {name for name in names if name in {"attack_eval", "defended_datasets", "isolation"}}

    shutil.copytree(source, dest, ignore=ignore)
    stage3 = dest / "stage3_guided_refinement"
    if stage3.exists():
        shutil.rmtree(stage3)
    payload_path = dest / "run_config.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    payload["output_dir"] = str(RESULTS_ROOT)
    payload["run_name"] = dest.name
    write_json(payload_path, payload)
    return dest


def run_reliable_stage3_probe(args: argparse.Namespace) -> Path:
    from dmmp.constraints.user_profiles import load_profiles
    from dmmp.data import load_cw_data
    from dmmp.diffusion.profile_pipeline import run_v4_stage3
    from dmmp.utils import resolve_device, set_seed, write_json

    source = _results_run_dir(args.source_run_dir)
    run_name = args.run_name or f"{source.name}_reliable_probe_{_timestamp()}"
    run_dir = _copy_run_for_stage3(source, run_name)
    cfg = _load_defense_config(run_dir, device=args.device, progress=args.progress)
    cfg.stage3_fixed_probe_train_samples = int(args.probe_train_samples)
    cfg.stage3_fixed_probe_val_samples = int(args.probe_val_samples)
    cfg.stage3_fixed_probe_samples = int(args.probe_eval_samples)
    cfg.stage3_fixed_probe_epochs = int(args.probe_epochs)
    cfg.stage3_fixed_probe_min_clean_accuracy = float(args.probe_min_clean_accuracy)
    cfg.stage3_fixed_probe_attackers = str(args.probe_attackers)
    cfg.stage3_use_diagnostic_accuracy_gate = True
    cfg.stage3_max_attack_accuracy = float(args.max_attack_accuracy)
    cfg.stage3_max_rendered_rf_accuracy = float(args.max_rendered_rf_accuracy)
    cfg.stage3_max_reliable_fixed_probe_accuracy = float(args.max_reliable_fixed_probe_accuracy)
    cfg.stage3_require_quality_gate = not bool(args.allow_diagnostic_fallback)
    cfg.progress = bool(args.progress)
    set_seed(int(cfg.seed))
    device = resolve_device(str(args.device))
    raw, labels, trace_ids, _, data_source = load_cw_data(cfg)
    splits = _load_splits(run_dir)
    profiles = load_profiles(run_dir / "stage2_user_diffusion" / "user_profiles")
    selected: dict[str, Any] = {}
    try:
        result = run_v4_stage3(raw, labels, trace_ids, splits, run_dir, cfg, device, profiles)
        selected = dict(result.get("selected", {}))
        status = "ok"
    except RuntimeError as exc:
        status = "failed_hard_gate"
        fallback = run_dir / "stage3_guided_refinement" / "best_diagnostic_fallback.json"
        if fallback.is_file():
            selected = json.loads(fallback.read_text(encoding="utf-8"))
        else:
            selected = {"error": str(exc)}
        if not bool(args.allow_diagnostic_fallback):
            raise
    used_fallback = bool(int(selected.get("selection_used_quality_fallback", 0) or 0))
    policy_valid = bool(int(selected.get("selection_policy_valid", 0) or 0))
    if status == "ok" and used_fallback:
        status = "diagnostic_fallback"
    elif status == "ok" and not policy_valid:
        status = "diagnostic_no_valid_policy"
    formal_gate_passed = bool(policy_valid and not used_fallback)
    metadata = {
        "experiment": "reliable-stage3-probe",
        "theory": "Make the Stage 3 diagnostic fixed probes reliable enough that RF/DF accuracy can participate in selection/gates.",
        "source_run_dir": str(source),
        "run_dir": str(run_dir),
        "data_source": str(data_source),
        "status": status,
        "formal_gate_passed": formal_gate_passed,
        "diagnostic_fallback": used_fallback,
        "overrides": {
            "stage3_fixed_probe_samples": cfg.stage3_fixed_probe_samples,
            "stage3_fixed_probe_train_samples": cfg.stage3_fixed_probe_train_samples,
            "stage3_fixed_probe_val_samples": cfg.stage3_fixed_probe_val_samples,
            "stage3_fixed_probe_epochs": cfg.stage3_fixed_probe_epochs,
            "stage3_fixed_probe_min_clean_accuracy": cfg.stage3_fixed_probe_min_clean_accuracy,
            "stage3_fixed_probe_attackers": cfg.stage3_fixed_probe_attackers,
            "stage3_max_attack_accuracy": cfg.stage3_max_attack_accuracy,
            "stage3_max_rendered_rf_accuracy": cfg.stage3_max_rendered_rf_accuracy,
            "stage3_max_reliable_fixed_probe_accuracy": cfg.stage3_max_reliable_fixed_probe_accuracy,
            "stage3_require_quality_gate": cfg.stage3_require_quality_gate,
        },
        "selected_policy": selected,
    }
    output_dir = run_dir / "isolation" / "reliable_stage3_probe"
    write_json(output_dir / "summary.json", metadata)
    lines = [
        "# Isolation: Reliable Stage 3 Fixed Probe",
        "",
        f"- source run: `{source}`",
        f"- diagnostic run: `{run_dir}`",
        f"- status: `{status}`",
        f"- formal gate passed: `{int(formal_gate_passed)}`",
        f"- selected policy valid: `{selected.get('selection_policy_valid', 0)}`",
        f"- used fallback: `{selected.get('selection_used_quality_fallback', 0)}`",
        f"- selected attack pressure: `{selected.get('selection_attack_pressure', 0.0)}`",
        f"- selected rendered RF accuracy: `{selected.get('selection_rendered_rf_accuracy', 0.0)}`",
        f"- reliable fixed probe count: `{selected.get('fixed_probe_reliable_count', 0)}`",
        "",
        "A valid policy should pass the label-free pressure gate and, when reliable probes exist, the fixed-probe/RF diagnostic gates.",
    ]
    (output_dir / "summary_zh.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[reliable-stage3-probe] saved to {output_dir}", flush=True)
    return output_dir


def _minimal_command(args: argparse.Namespace, run_name: str) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_defense.py"),
        "--run_name",
        run_name,
        "--seed",
        str(args.seed),
        "--budgets",
        str(args.budget),
        "--pareto_budgets",
        str(args.budget),
        "--guidance_attackers",
        "both",
        "--no-condition_profile_mask",
        "--no-condition_selected_mask",
        "--no-condition_preference_weights",
        "--preference_weight",
        "0",
        "--profile_weight",
        "0",
        "--direction_target",
        "none",
        "--direction_correction_strength",
        "0",
        "--min_incoming_dummy_share",
        "0",
        "--policy_logit_noise_std",
        "0",
    ]
    if not bool(args.progress):
        cmd.append("--no-progress")
    if args.extra_args:
        cmd.extend(args.extra_args)
    return cmd


def run_minimal_ablation(args: argparse.Namespace) -> Path:
    run_name = _safe_run_name(args.run_name or f"dmmpv3_isolation_minimal_seed{int(args.seed)}_{_timestamp()}")
    run_dir = _new_results_run_dir(run_name)
    cmd = _minimal_command(args, run_name)
    payload = {
        "experiment": "minimal-ablation",
        "theory": "Disable V3 profile/preference/direction/noise constraints while keeping DF/RF guidance to isolate which upgrade layer hurts fixed RF defense.",
        "run_dir": str(run_dir),
        "command": cmd,
        "disabled_features": [
            "condition_profile_mask",
            "condition_selected_mask",
            "condition_preference_weights",
            "preference_weight",
            "profile_weight",
            "direction_correction",
            "policy_logit_noise",
        ],
        "kept_features": ["guidance_attackers=both", "budget=0.30 default"],
    }
    if bool(args.dry_run):
        print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
        return run_dir
    if run_dir.exists():
        raise FileExistsError(f"Destination run already exists; choose a new --run_name: {run_dir}")
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    output_dir = run_dir / "isolation" / "minimal_ablation"
    from dmmp.utils import write_json

    write_json(output_dir / "command.json", payload)
    if bool(args.reuse_fixed_eval):
        eval_args = argparse.Namespace(
            run_dir=str(run_dir),
            fixed_df_checkpoint=str(args.fixed_df_checkpoint),
            fixed_rf_checkpoint=str(args.fixed_rf_checkpoint),
            target_profile_id=str(args.target_profile_id),
            seed=int(args.seed),
            device=str(args.device),
            batch_size=int(args.batch_size),
            max_load_time=float(args.max_load_time),
            rf_tam_num_slots=int(args.rf_tam_num_slots),
            df_architecture=str(args.df_architecture),
            max_test_traces=int(args.max_test_traces),
            clean_acc_tolerance=float(args.clean_acc_tolerance),
            defended_role=str(args.defended_role),
            progress=bool(args.progress),
        )
        run_reuse_fixed_eval(eval_args)
    print(f"[minimal-ablation] run directory: {run_dir}", flush=True)
    return run_dir


def print_suite(args: argparse.Namespace) -> None:
    source = _results_run_dir(args.source_run_dir)
    stamp = _timestamp()
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "reuse-fixed-eval",
            "--run_dir",
            str(source),
        ],
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "reliable-stage3-probe",
            "--source_run_dir",
            str(source),
            "--run_name",
            f"{source.name}_reliable_probe_{stamp}",
        ],
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "minimal-ablation",
            "--run_name",
            f"dmmpv3_isolation_minimal_seed{int(args.seed)}_{stamp}",
            "--dry_run",
        ],
    ]
    print("# Suggested isolation commands", flush=True)
    for command in commands:
        print(" ".join(f'"{item}"' if " " in item else item for item in command), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DMMPv3 isolation experiments for fixed-attacker regression diagnosis.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reuse = subparsers.add_parser("reuse-fixed-eval", help="Evaluate a DMMPv3 run with registered fixed DF/RF checkpoints without retraining.")
    reuse.add_argument("--run_dir", required=True)
    reuse.add_argument("--fixed_df_checkpoint", default=str(DEFAULT_FIXED_DF))
    reuse.add_argument("--fixed_rf_checkpoint", default=str(DEFAULT_FIXED_RF))
    reuse.add_argument("--target_profile_id", default="")
    reuse.add_argument("--seed", type=int, default=0)
    reuse.add_argument("--device", default="auto")
    reuse.add_argument("--batch_size", type=int, default=256)
    reuse.add_argument("--max_load_time", type=float, default=80.0)
    reuse.add_argument("--rf_tam_num_slots", type=int, default=1800)
    reuse.add_argument("--df_architecture", choices=["project", "wflib"], default="project")
    reuse.add_argument("--max_test_traces", type=int, default=0)
    reuse.add_argument("--clean_acc_tolerance", type=float, default=0.005)
    reuse.add_argument("--defended_role", default="fresh_deployment_test")
    reuse.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    reuse.set_defaults(func=run_reuse_fixed_eval)

    probe = subparsers.add_parser("reliable-stage3-probe", help="Clone an existing run and rerun Stage 3 with reliable fixed probes.")
    probe.add_argument("--source_run_dir", required=True)
    probe.add_argument("--run_name", required=True)
    probe.add_argument("--device", default="auto")
    probe.add_argument("--probe_train_samples", type=int, default=30000)
    probe.add_argument("--probe_val_samples", type=int, default=5000)
    probe.add_argument("--probe_eval_samples", type=int, default=5000)
    probe.add_argument("--probe_epochs", type=int, default=10)
    probe.add_argument("--probe_min_clean_accuracy", type=float, default=0.85)
    probe.add_argument("--probe_attackers", default="df,rf")
    probe.add_argument("--max_attack_accuracy", type=float, default=0.40)
    probe.add_argument("--max_rendered_rf_accuracy", type=float, default=0.40)
    probe.add_argument("--max_reliable_fixed_probe_accuracy", type=float, default=0.40)
    probe.add_argument("--allow_diagnostic_fallback", action=argparse.BooleanOptionalAction, default=True)
    probe.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    probe.set_defaults(func=run_reliable_stage3_probe)

    minimal = subparsers.add_parser("minimal-ablation", help="Run or print a minimal V3 ablation command.")
    minimal.add_argument("--run_name", default="")
    minimal.add_argument("--seed", type=int, default=0)
    minimal.add_argument("--budget", type=float, default=0.30)
    minimal.add_argument("--dry_run", action=argparse.BooleanOptionalAction, default=False)
    minimal.add_argument("--reuse_fixed_eval", action=argparse.BooleanOptionalAction, default=False)
    minimal.add_argument("--fixed_df_checkpoint", default=str(DEFAULT_FIXED_DF))
    minimal.add_argument("--fixed_rf_checkpoint", default=str(DEFAULT_FIXED_RF))
    minimal.add_argument("--target_profile_id", default="")
    minimal.add_argument("--device", default="auto")
    minimal.add_argument("--batch_size", type=int, default=256)
    minimal.add_argument("--max_load_time", type=float, default=80.0)
    minimal.add_argument("--rf_tam_num_slots", type=int, default=1800)
    minimal.add_argument("--df_architecture", choices=["project", "wflib"], default="project")
    minimal.add_argument("--max_test_traces", type=int, default=0)
    minimal.add_argument("--clean_acc_tolerance", type=float, default=0.005)
    minimal.add_argument("--defended_role", default="fresh_deployment_test")
    minimal.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    minimal.add_argument("extra_args", nargs=argparse.REMAINDER)
    minimal.set_defaults(func=run_minimal_ablation)

    suite = subparsers.add_parser("print-suite", help="Print the three isolation commands for a source run.")
    suite.add_argument("--source_run_dir", required=True)
    suite.add_argument("--seed", type=int, default=0)
    suite.set_defaults(func=print_suite)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None


if __name__ == "__main__":
    main()
