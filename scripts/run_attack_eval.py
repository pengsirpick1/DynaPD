"""CLI entry for self-contained DMMPv3 attack evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.utils.config import AttackConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate fixed and mixed attackers for a DMMPv3 run.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--data_root", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--attackers", nargs="+", default="fixed_df,fixed_rf,mixed_df,mixed_rf")
    parser.add_argument("--policy_variant", choices=["stage2", "stage3"], default="stage3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_train_traces", type=int, default=0)
    parser.add_argument("--max_val_traces", type=int, default=0)
    parser.add_argument("--max_test_traces", type=int, default=0)
    parser.add_argument("--clean_df_epochs", "--df_epochs", dest="clean_df_epochs", type=int, default=10)
    parser.add_argument("--clean_df_patience", type=int, default=3)
    parser.add_argument("--clean_df_lr", type=float, default=2e-3)
    parser.add_argument("--adaptive_epochs", type=int, default=10)
    parser.add_argument("--adaptive_patience", "--patience", dest="adaptive_patience", type=int, default=3)
    parser.add_argument("--adaptive_lr", type=float, default=1e-3)
    parser.add_argument("--adaptive_init", choices=["checkpoint", "scratch"], default="checkpoint")
    parser.add_argument("--df_batch_size", "--batch_size", dest="df_batch_size", type=int, default=256)
    parser.add_argument("--df_architecture", choices=["project", "wflib"], default="project")
    parser.add_argument("--max_load_time", type=float, default=80.0)
    parser.add_argument("--rf_tam_num_slots", type=int, default=1800)
    parser.add_argument("--fixed_eval_defense_seed", type=int, default=41000)
    parser.add_argument("--mixed_train_defense_seed", type=int, default=42000)
    parser.add_argument("--mixed_val_defense_seed", type=int, default=43000)
    parser.add_argument("--mixed_test_defense_seed", type=int, default=44000)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument(
        "--adaptive_protocol",
        choices=["fixed", "same_user", "cross_user", "multi_source", "profile_known", "full_catalogue"],
        default="fixed",
    )
    parser.add_argument("--adaptive_source_run_dir", default="", help="Optional old defense run used to generate adaptive train/val defended pools.")
    parser.add_argument("--adaptive_source_label", default="", help="Short label for the adaptive source run in transfer-mixed outputs.")
    parser.add_argument("--source_user_count", type=int, default=1)
    parser.add_argument("--source_profile_ids", default="")
    parser.add_argument("--target_profile_id", default="")
    parser.add_argument("--fixed_total_adaptive_samples", type=int, default=0)
    parser.add_argument("--fixed_per_user_adaptive_samples", type=int, default=0)
    parser.add_argument("--attack_min_clean_accuracy", type=float, default=0.85)
    parser.add_argument("--attack_max_defended_accuracy", type=float, default=0.40)
    parser.add_argument("--attack_require_quality_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_retrain", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    cfg = AttackConfig.from_namespace(parse_args())
    run_config = Path(cfg.run_dir) / "run_config.json"
    uses_guided_profile_engine = False
    if run_config.is_file():
        import json

        uses_guided_profile_engine = str(json.loads(run_config.read_text(encoding="utf-8")).get("version", "v3")).lower().startswith("v")
    if uses_guided_profile_engine:
        from dmmp.evaluation.profile_attacks import run_v4_attack_evaluation

        run_v4_attack_evaluation(cfg)
    else:
        from dmmp.evaluation.attacks import run_attack_evaluation

        run_attack_evaluation(cfg)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "hard quality gate failed" not in str(exc):
            raise
        print(f"[DMMPv3 ATTACK HARD GATE FAILED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)

