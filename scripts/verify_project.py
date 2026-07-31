"""Verify the slim DynaPD Stage B release structure."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "pyproject.toml",
    "dynapd/__init__.py",
    "dynapd/data/cw.py",
    "dynapd/evaluation/attack_models.py",
    "dynapd/projection/padding.py",
    "dynapd/stage_a/additive_probe.py",
    "dynapd/stage_a/faithfulness.py",
    "dynapd/stage_a/mask_ops.py",
    "dynapd/stage_a/modeling.py",
    "dynapd/stage_b/action_selector.py",
    "dynapd/stage_b/expanded_generator.py",
    "dynapd/stage_b/objectives.py",
    "dynapd/stage_b/policy_data.py",
    "dynapd/stage_b/policy_model.py",
    "dynapd/stage_b/smoothing.py",
    "dynapd/utils/common.py",
    "dynapd/utils/config.py",
    "docs/stage_b2d_strategy_versions.md",
    "docs/stage_b_vectorized_candidate_selection_20260731.md",
    "docs/stage_b_candidate_frontend_parallel_20260731.md",
    "docs/stage_b_latency_overhead_audit_20260731.md",
    "docs/stage_b_pyinstrument_profile_20260730.md",
    "scripts/stage_b_prepare_fast_keypoint_archive.py",
    "scripts/stage_b_build_policy_dataset.py",
    "scripts/stage_b_export_teacher_trajectories.py",
    "scripts/stage_b_launch_teacher_shards_parallel.py",
    "scripts/stage_b_probe_parallel_teacher_workers.py",
    "scripts/stage_b_train_candidate_policy.py",
    "scripts/stage_b_eval_candidate_policy_offline.py",
    "scripts/stage_b_run_student_policy_controller.py",
]

REQUIRED_DIRS = [
    "dynapd/data",
    "dynapd/evaluation",
    "dynapd/projection",
    "dynapd/stage_a",
    "dynapd/stage_b",
    "dynapd/utils",
    "scripts",
    "docs",
]

FORBIDDEN_PATHS = [
    "dynapd/diffusion",
    "dynapd/purifier",
    "dynapd/target_policy",
    "configs",
]


def main() -> None:
    missing_files = [path for path in REQUIRED_FILES if not (ROOT / path).is_file()]
    missing_dirs = [path for path in REQUIRED_DIRS if not (ROOT / path).is_dir()]
    forbidden = [path for path in FORBIDDEN_PATHS if (ROOT / path).exists()]

    from dynapd.utils.config import DEFAULT_OUTPUT_DIR

    config_errors: list[str] = []
    expected_output = (ROOT / "results").resolve()
    if Path(DEFAULT_OUTPUT_DIR).resolve() != expected_output:
        config_errors.append(f"DEFAULT_OUTPUT_DIR should be {expected_output}, got {Path(DEFAULT_OUTPUT_DIR).resolve()}")

    if missing_files or missing_dirs or forbidden or config_errors:
        if missing_files:
            print("Missing files:")
            for path in missing_files:
                print(f"  - {path}")
        if missing_dirs:
            print("Missing directories:")
            for path in missing_dirs:
                print(f"  - {path}")
        if forbidden:
            print("Forbidden release paths exist:")
            for path in forbidden:
                print(f"  - {path}")
        if config_errors:
            print("Configuration errors:")
            for error in config_errors:
                print(f"  - {error}")
        raise SystemExit(1)
    print(f"DynaPD slim release verification passed: {ROOT}")


if __name__ == "__main__":
    main()
