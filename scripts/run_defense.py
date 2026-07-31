"""CLI entry for the self-contained DMMPv3 defense pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dmmp.utils.config import DefenseConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run self-contained DMMPv3 random-preference diffusion defense.")
    parser.add_argument("--version", choices=["v3"], default="v3")
    parser.add_argument("--data_root", default=str(REPO_ROOT / "datasets" / "CW"))
    parser.add_argument("--output_dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--run_name", default="")
    parser.add_argument(
        "--stage",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Run all stages, run a new Stage 1 only, or resume Stage 2/3 from an existing --run_name.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prefix_n", type=int, default=500)
    parser.add_argument("--patch_num", type=int, default=200)
    parser.add_argument("--budgets", default="0.30")
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.10)
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--max_classes", type=int, default=0)
    parser.add_argument("--max_generation_traces", type=int, default=0)
    parser.add_argument("--generation_split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--max_trace_length", type=int, default=5000)
    parser.add_argument("--early_fraction", type=float, default=0.40)
    parser.add_argument("--topk_cells", type=int, default=80)
    parser.add_argument("--mi_bins", type=int, default=8)
    parser.add_argument("--masking_max_samples", type=int, default=10000)
    parser.add_argument("--preference_pool", default="interval,early,burst,direction,shape")
    parser.add_argument("--combination_sizes", default="2")
    parser.add_argument("--hidden_dim", type=int, default=384)
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--sampling_steps", type=int, default=20)
    parser.add_argument("--encoder_epochs", type=int, default=10)
    parser.add_argument("--encoder_train_samples", type=int, default=20000)
    parser.add_argument("--encoder_lr", type=float, default=1e-3)
    parser.add_argument("--diffusion_train_steps", type=int, default=30000)
    parser.add_argument("--diffusion_lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--insertion_strategy", choices=["uniform_in_patch", "random_jitter_in_patch", "burst_aware"], default="uniform_in_patch")
    parser.add_argument("--render_coordinate", choices=["rf_tam", "trace_index", "tam_obfuscation", "multi_view"], default="rf_tam")
    parser.add_argument(
        "--tam_obfuscation_strategy",
        choices=["rayleigh_in_slot", "edge_clustered", "hybrid_clustered"],
        default="hybrid_clustered",
    )
    parser.add_argument("--tam_slot_jitter", type=float, default=0.03)
    parser.add_argument("--tam_cluster_ratio", type=float, default=0.70)
    parser.add_argument("--tam_local_run_max", type=int, default=8)
    parser.add_argument("--tam_preserve_real_timestamps", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--multi_view_mode", choices=["fused", "split"], default="fused")
    parser.add_argument("--multi_view_df_share", type=float, default=0.40)
    parser.add_argument("--multi_view_awf_share", type=float, default=0.30)
    parser.add_argument("--multi_view_rf_share", type=float, default=0.30)
    parser.add_argument("--shrink_method", choices=["weighted", "greedy"], default="weighted")
    parser.add_argument("--shrink_keep_ratio", type=float, default=0.75)
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument("--candidate_mode", choices=["observed", "executable"], default="executable")
    parser.add_argument("--candidate_topk", type=int, default=80)
    parser.add_argument("--candidate_soft_topk", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--candidate_temperature", type=float, default=0.15)
    parser.add_argument("--candidate_epochs", type=int, default=10)
    parser.add_argument("--probe_samples", type=int, default=10000)
    parser.add_argument("--probe_exact_samples", type=int, default=256)
    parser.add_argument("--probe_dummy_count", type=int, default=4)
    parser.add_argument("--probe_attacker", choices=["df", "rf", "both"], default="both")
    parser.add_argument("--view_profile_samples", type=int, default=10000)
    parser.add_argument("--view_cv_folds", type=int, default=3)
    parser.add_argument("--num_train_profiles", type=int, default=32)
    parser.add_argument("--num_val_profiles", type=int, default=8)
    parser.add_argument("--num_test_profiles", type=int, default=8)
    parser.add_argument("--profile_seed", type=int, default=17000)
    parser.add_argument("--profile_secret", default="dmmpv3-private-experiment-key")
    parser.add_argument("--profile_combination_mode", choices=["fixed_pair", "legacy_pool"], default="fixed_pair")
    parser.add_argument("--active_pair_count", type=int, default=1)
    parser.add_argument("--active_triple_count", type=int, default=0)
    parser.add_argument("--pair_probability", type=float, default=1.0)
    parser.add_argument("--dirichlet_alpha", type=float, default=1.0)
    parser.add_argument("--profile_pair_weight_min", type=float, default=0.0)
    parser.add_argument("--profile_pair_weight_max", type=float, default=1.0)
    parser.add_argument("--visit_selector", choices=["prng", "hash"], default="hash")
    parser.add_argument("--profile_overlap_target", choices=["low", "medium", "high"], default="medium")
    parser.add_argument("--condition_profile_mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--condition_selected_mask", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--condition_preference_map", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--condition_preference_weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--policy_generator", choices=["diffusion", "heuristic_prior_direct"], default="diffusion")
    parser.add_argument("--guidance_attackers", choices=["df", "rf", "both"], default="both")
    parser.add_argument(
        "--guidance_label_mode",
        choices=["pseudo", "true"],
        default="pseudo",
        help="Use label-free frozen-surrogate pseudo labels or true site labels as Stage 2/3 guidance targets.",
    )
    parser.add_argument("--surrogate_train_samples", type=int, default=30000)
    parser.add_argument("--surrogate_val_samples", type=int, default=5000)
    parser.add_argument("--surrogate_epochs", type=int, default=10)
    parser.add_argument("--surrogate_patience", type=int, default=3)
    parser.add_argument("--surrogate_lr", type=float, default=2e-3)
    parser.add_argument("--surrogate_batch_size", type=int, default=128)
    parser.add_argument("--surrogate_gradient_batch_size", type=int, default=16)
    parser.add_argument("--surrogate_min_val_accuracy", type=float, default=0.85)
    parser.add_argument("--surrogate_df_architecture", choices=["project", "wflib"], default="project")
    parser.add_argument("--surrogate_df_weight", type=float, default=0.5)
    parser.add_argument("--surrogate_rf_weight", type=float, default=0.5)
    parser.add_argument("--surrogate_robust_weight", type=float, default=0.35)
    parser.add_argument("--surrogate_rf_num_slots", type=int, default=1800)
    parser.add_argument("--surrogate_rf_max_load_time", type=float, default=80.0)
    parser.add_argument("--guidance_weight", type=float, default=0.10)
    parser.add_argument("--guidance_last_steps", type=int, default=4)
    parser.add_argument("--guidance_train_steps", type=int, default=30000)
    parser.add_argument("--defense_hard_weight", type=float, default=1.0)
    parser.add_argument("--defense_soft_objective_scale", type=float, default=0.05)
    parser.add_argument("--defense_soft_utility_weight", type=float, default=0.05)
    parser.add_argument("--defense_risk_tolerance", type=float, default=0.0)
    parser.add_argument("--prefix_hidden_align_weight", type=float, default=0.03)
    parser.add_argument("--prefix_hidden_align_dim", type=int, default=128)
    parser.add_argument("--prefix_hidden_align_temperature", type=float, default=0.10)
    parser.add_argument("--v1_mode_pool", choices=["early", "legacy_direct"], default="early")
    parser.add_argument("--v1_mode_prior_weight", type=float, default=0.50)
    parser.add_argument("--prior_leak_weight", type=float, default=1.50)
    parser.add_argument("--prior_preference_weight", type=float, default=0.15)
    parser.add_argument("--prior_noise_std", type=float, default=0.0)
    parser.add_argument("--full_sample_guidance_interval", type=int, default=10)
    parser.add_argument("--full_sample_guidance_steps", type=int, default=20)
    parser.add_argument("--diversity_weight", type=float, default=0.02)
    parser.add_argument("--preference_weight", type=float, default=0.01)
    parser.add_argument("--preference_attack_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preference_attack_gate_margin", type=float, default=0.02)
    parser.add_argument("--constraint_weight", type=float, default=0.02)
    parser.add_argument("--profile_weight", type=float, default=0.0)
    parser.add_argument("--refine_method", choices=["continuous", "greedy"], default="continuous")
    parser.add_argument("--refine_keep_ratios", default="1.0")
    parser.add_argument("--refine_steps", type=int, default=6)
    parser.add_argument("--pareto_budgets", default="0.30")
    parser.add_argument("--pareto_samples", type=int, default=500)
    parser.add_argument("--direction_target", choices=["none", "clean", "balanced", "incoming"], default="none")
    parser.add_argument("--direction_correction_strength", type=float, default=0.0)
    parser.add_argument("--min_incoming_dummy_share", type=float, default=0.0)
    parser.add_argument("--policy_logit_temperature", type=float, default=1.50)
    parser.add_argument("--policy_logit_noise_std", type=float, default=0.0)
    parser.add_argument("--deployment_repeats", type=int, default=3)
    parser.add_argument("--stage3_repeats", type=int, default=1)
    parser.add_argument("--stage3_fixed_probe_samples", type=int, default=512)
    parser.add_argument("--stage3_fixed_probe_train_samples", type=int, default=3000)
    parser.add_argument("--stage3_fixed_probe_val_samples", type=int, default=1000)
    parser.add_argument("--stage3_fixed_probe_epochs", type=int, default=3)
    parser.add_argument("--stage3_fixed_probe_attackers", default="df,rf")
    parser.add_argument("--stage3_fixed_probe_weight", type=float, default=1.0)
    parser.add_argument("--stage3_fixed_probe_min_clean_accuracy", type=float, default=0.70)
    parser.add_argument("--stage3_accuracy_guard_margin", type=float, default=0.02)
    parser.add_argument("--stage3_max_label_free_attack_pressure", type=float, default=0.45)
    parser.add_argument("--stage3_max_attack_accuracy", type=float, default=0.40)
    parser.add_argument("--stage3_use_diagnostic_accuracy_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage3_max_rendered_rf_accuracy", type=float, default=0.40)
    parser.add_argument("--stage3_max_reliable_fixed_probe_accuracy", type=float, default=0.40)
    parser.add_argument("--stage3_require_quality_gate", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve_variable_length_traces", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug_sample_records", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    cfg = DefenseConfig.from_namespace(parse_args())
    if str(cfg.version).lower().startswith("v"):
        from dmmp.diffusion.profile_pipeline import run_v4_pipeline

        run_v4_pipeline(cfg)
    else:
        from dmmp.diffusion.pipeline import run_defense_pipeline

        run_defense_pipeline(cfg)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        if "hard defense gate failed" not in str(exc):
            raise
        print(f"[DMMPv3 HARD GATE FAILED] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)


