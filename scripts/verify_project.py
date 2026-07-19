"""Verify the DMMPv3 harness structure.

This script is intentionally lightweight: it checks required directories,
required documents, and guarded entrypoints. It does not train or evaluate
models.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "AGENTS.md",
    "README.md",
    "requirements.txt",
    "docs/method_spec.md",
    "docs/experiment_protocol.md",
    "docs/project_map.md",
    "docs/implementation_index.md",
    "docs/runbook.md",
    "docs/decisions.md",
    "docs/model_registry.md",
    "tasks/current_task.md",
    "dmmp/__init__.py",
    "dmmp/constraints/__init__.py",
    "dmmp/constraints/combination_catalogue.py",
    "dmmp/constraints/preferences.py",
    "dmmp/constraints/user_profiles.py",
    "dmmp/data/__init__.py",
    "dmmp/data/cw.py",
    "dmmp/diffusion/__init__.py",
    "dmmp/diffusion/models.py",
    "dmmp/diffusion/pipeline.py",
    "dmmp/diffusion/policy.py",
    "dmmp/diffusion/profile_pipeline.py",
    "dmmp/encoders/__init__.py",
    "dmmp/encoders/condition_encoders.py",
    "dmmp/encoders/leakage.py",
    "dmmp/encoders/prefix.py",
    "dmmp/evaluation/__init__.py",
    "dmmp/evaluation/attack_models.py",
    "dmmp/evaluation/attacks.py",
    "dmmp/evaluation/profile_attacks.py",
    "dmmp/guidance/__init__.py",
    "dmmp/guidance/candidate_scorer.py",
    "dmmp/guidance/diffusion_guidance.py",
    "dmmp/guidance/strong_surrogates.py",
    "dmmp/losses/__init__.py",
    "dmmp/projection/__init__.py",
    "dmmp/projection/padding.py",
    "dmmp/renderer/__init__.py",
    "dmmp/utils/__init__.py",
    "dmmp/utils/common.py",
    "dmmp/utils/config.py",
    "scripts/train_defense.py",
    "scripts/run_defense.py",
    "scripts/run_attack_eval.py",
    "scripts/evaluate_fixed.py",
    "scripts/train_mixed_attackers.py",
    "scripts/evaluate_mixed.py",
    "scripts/resume_stage3.py",
    "scripts/sweep_guidance.py",
    "scripts/validate_strong_surrogates.py",
    "scripts/validate_v4.py",
    "scripts/verify_project.py",
]

REQUIRED_DIRS = [
    "dmmp/encoders",
    "dmmp/diffusion",
    "dmmp/guidance",
    "dmmp/projection",
    "dmmp/renderer",
    "dmmp/constraints",
    "dmmp/losses",
    "dmmp/data",
    "dmmp/evaluation",
    "dmmp/utils",
    "configs/defense",
    "configs/attackers",
    "configs/experiments",
    "models/defense",
    "models/attackers/fixed/df",
    "models/attackers/fixed/rf",
    "models/attackers/mixed/df",
    "models/attackers/mixed/rf",
    "results/training",
    "results/fixed_evaluation",
    "results/mixed_evaluation",
    "results/ablation",
    "results/failed_runs",
    "logs",
    "tests",
]


def main() -> None:
    missing_files = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    missing_dirs = [path for path in REQUIRED_DIRS if not (ROOT / path).is_dir()]
    config_errors: list[str] = []
    from dmmp.utils.config import DefenseConfig

    cfg = DefenseConfig()
    if str(cfg.version).lower() != "v3":
        config_errors.append(f"DefenseConfig.version should default to v3, got {cfg.version!r}")
    expected_output = (ROOT / "results").resolve()
    if Path(cfg.output_dir).resolve() != expected_output:
        config_errors.append(f"DefenseConfig.output_dir should default to {expected_output}, got {Path(cfg.output_dir).resolve()}")
    if missing_files or missing_dirs or config_errors:
        if missing_files:
            print("Missing files:")
            for path in missing_files:
                print(f"  - {path}")
        if missing_dirs:
            print("Missing directories:")
            for path in missing_dirs:
                print(f"  - {path}")
        if config_errors:
            print("Configuration errors:")
            for error in config_errors:
                print(f"  - {error}")
        raise SystemExit(1)
    print(f"DMMPv3 harness verification passed: {ROOT}")


if __name__ == "__main__":
    main()
