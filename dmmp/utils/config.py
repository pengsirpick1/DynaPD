"""Configuration objects for the self-contained DMMPv3 experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DATA_ROOT = str(REPO_ROOT / "datasets" / "CW")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "results")


def parse_csv_floats(value: str | list[float] | tuple[float, ...]) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(item) for item in str(value).replace(";", ",").split(",") if item.strip()]


def parse_csv_ints(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(item) for item in str(value).replace(";", ",").split(",") if item.strip()]


def parse_csv_strings(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(part.strip() for part in str(item).replace(";", ",").split(",") if part.strip())
        return result
    return [item.strip() for item in str(value).replace(";", ",").split(",") if item.strip()]


@dataclass
class DefenseConfig:
    version: str = "v3"
    data_root: str = DEFAULT_DATA_ROOT
    output_dir: str = DEFAULT_OUTPUT_DIR
    run_name: str = ""
    stage: str = "all"
    seed: int = 0
    prefix_n: int = 500
    patch_num: int = 200
    budgets: str = "0.30"
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    max_samples: int = 0
    max_classes: int = 0
    max_generation_traces: int = 0
    generation_split: str = "test"
    max_trace_length: int = 5000
    early_fraction: float = 0.40
    topk_cells: int = 80
    mi_bins: int = 8
    masking_max_samples: int = 10000
    preference_pool: str = "interval,early,burst,direction,shape"
    combination_sizes: str = "2"
    hidden_dim: int = 384
    diffusion_steps: int = 100
    sampling_steps: int = 20
    encoder_epochs: int = 10
    encoder_train_samples: int = 20000
    encoder_lr: float = 1e-3
    diffusion_train_steps: int = 30000
    diffusion_lr: float = 1e-4
    batch_size: int = 128
    device: str = "auto"
    insertion_strategy: str = "uniform_in_patch"
    render_coordinate: str = "rf_tam"
    tam_obfuscation_strategy: str = "hybrid_clustered"
    tam_slot_jitter: float = 0.03
    tam_cluster_ratio: float = 0.70
    tam_local_run_max: int = 8
    tam_preserve_real_timestamps: bool = True
    tam_flatten_strength: float = 0.0
    tam_flatten_floor: float = 1.0
    multi_view_mode: str = "fused"
    multi_view_df_share: float = 0.40
    multi_view_awf_share: float = 0.30
    multi_view_rf_share: float = 0.30
    shrink_method: str = "weighted"
    shrink_keep_ratio: float = 0.75
    progress: bool = True
    log_every: int = 500
    candidate_mode: str = "executable"
    candidate_topk: int = 80
    candidate_soft_topk: bool = True
    candidate_temperature: float = 0.15
    candidate_epochs: int = 10
    probe_samples: int = 10000
    probe_exact_samples: int = 256
    probe_dummy_count: int = 4
    probe_attacker: str = "both"
    view_profile_samples: int = 10000
    view_cv_folds: int = 3
    num_train_profiles: int = 32
    num_val_profiles: int = 8
    num_test_profiles: int = 8
    profile_seed: int = 17000
    profile_secret: str = "dmmpv3-private-experiment-key"
    profile_combination_mode: str = "fixed_pair"
    active_pair_count: int = 1
    active_triple_count: int = 0
    pair_probability: float = 1.0
    dirichlet_alpha: float = 1.0
    profile_pair_weight_min: float = 0.0
    profile_pair_weight_max: float = 1.0
    visit_selector: str = "hash"
    profile_overlap_target: str = "medium"
    condition_profile_mask: bool = False
    condition_selected_mask: bool = False
    condition_preference_map: bool = False
    condition_preference_weights: bool = False
    policy_generator: str = "diffusion"
    guidance_attackers: str = "both"
    guidance_label_mode: str = "pseudo"
    surrogate_train_samples: int = 30000
    surrogate_val_samples: int = 5000
    surrogate_epochs: int = 10
    surrogate_patience: int = 3
    surrogate_lr: float = 2e-3
    surrogate_batch_size: int = 128
    surrogate_gradient_batch_size: int = 16
    surrogate_min_val_accuracy: float = 0.85
    surrogate_df_architecture: str = "project"
    surrogate_df_weight: float = 0.5
    surrogate_rf_weight: float = 0.5
    surrogate_robust_weight: float = 0.35
    surrogate_rf_num_slots: int = 1800
    surrogate_rf_max_load_time: float = 80.0
    guidance_weight: float = 0.10
    guidance_last_steps: int = 4
    guidance_train_steps: int = 30000
    defense_hard_weight: float = 1.0
    defense_soft_objective_scale: float = 0.05
    defense_soft_utility_weight: float = 0.05
    defense_risk_tolerance: float = 0.0
    prefix_hidden_align_weight: float = 0.03
    prefix_hidden_align_dim: int = 128
    prefix_hidden_align_temperature: float = 0.10
    v1_mode_pool: str = "early"
    v1_mode_prior_weight: float = 0.50
    prior_leak_weight: float = 1.50
    prior_preference_weight: float = 0.15
    prior_noise_std: float = 0.0
    full_sample_guidance_interval: int = 10
    full_sample_guidance_steps: int = 20
    diversity_weight: float = 0.02
    preference_weight: float = 0.01
    preference_attack_gate: bool = True
    preference_attack_gate_margin: float = 0.02
    constraint_weight: float = 0.02
    profile_weight: float = 0.0
    refine_method: str = "continuous"
    refine_keep_ratios: str = "1.0"
    refine_steps: int = 6
    pareto_budgets: str = "0.30"
    pareto_samples: int = 500
    direction_target: str = "none"
    direction_correction_strength: float = 0.0
    min_incoming_dummy_share: float = 0.0
    policy_logit_temperature: float = 1.50
    policy_logit_noise_std: float = 0.0
    deployment_repeats: int = 3
    stage3_repeats: int = 1
    stage3_fixed_probe_samples: int = 512
    stage3_fixed_probe_train_samples: int = 3000
    stage3_fixed_probe_val_samples: int = 1000
    stage3_fixed_probe_epochs: int = 3
    stage3_fixed_probe_attackers: str = "df,rf"
    stage3_fixed_probe_weight: float = 1.0
    stage3_fixed_probe_min_clean_accuracy: float = 0.70
    stage3_accuracy_guard_margin: float = 0.02
    stage3_max_label_free_attack_pressure: float = 0.45
    stage3_max_attack_accuracy: float = 0.40
    stage3_use_diagnostic_accuracy_gate: bool = True
    stage3_max_rendered_rf_accuracy: float = 0.40
    stage3_max_reliable_fixed_probe_accuracy: float = 0.40
    stage3_min_dummy_incoming_share: float = 0.0
    stage3_max_dummy_incoming_share: float = 1.0
    stage3_min_tam_incoming_l1_shift: float = 0.0
    stage3_incoming_metric_weight: float = 0.0
    stage3_require_quality_gate: bool = True
    preserve_variable_length_traces: bool = True
    debug_sample_records: bool = False

    @classmethod
    def from_namespace(cls, args) -> "DefenseConfig":
        values = {field: getattr(args, field) for field in cls.__dataclass_fields__ if hasattr(args, field)}
        return cls(**values)

    @property
    def budget_values(self) -> list[float]:
        return parse_csv_floats(self.budgets)

    @property
    def combination_size_values(self) -> list[int]:
        return parse_csv_ints(self.combination_sizes)

    @property
    def preference_values(self) -> list[str]:
        return parse_csv_strings(self.preference_pool)

    @property
    def refine_keep_ratio_values(self) -> list[float]:
        return parse_csv_floats(self.refine_keep_ratios)

    @property
    def pareto_budget_values(self) -> list[float]:
        return parse_csv_floats(self.pareto_budgets)

    def output_root(self) -> Path:
        return Path(self.output_dir)


@dataclass
class AttackConfig:
    run_dir: str
    data_root: str = ""
    output_dir: str = ""
    attackers: str = "fixed_df,fixed_rf,mixed_df,mixed_rf"
    policy_variant: str = "stage3"
    seed: int = 0
    device: str = "auto"
    max_train_traces: int = 0
    max_val_traces: int = 0
    max_test_traces: int = 0
    clean_df_epochs: int = 10
    clean_df_patience: int = 3
    clean_df_lr: float = 2e-3
    df_epochs: int = 10
    adaptive_epochs: int = 10
    adaptive_patience: int = 3
    adaptive_lr: float = 1e-3
    adaptive_init: str = "checkpoint"
    patience: int = 5
    lr: float = 2e-3
    df_batch_size: int = 256
    df_architecture: str = "project"
    batch_size: int = 256
    max_load_time: float = 80.0
    rf_tam_num_slots: int = 1800
    progress: bool = True
    log_every: int = 500
    fixed_eval_defense_seed: int = 41000
    mixed_train_defense_seed: int = 42000
    mixed_val_defense_seed: int = 43000
    mixed_test_defense_seed: int = 44000
    adaptive_protocol: str = "fixed"
    adaptive_source_run_dir: str = ""
    adaptive_source_label: str = ""
    source_user_count: int = 1
    source_profile_ids: str = ""
    target_profile_id: str = ""
    fixed_total_adaptive_samples: int = 0
    fixed_per_user_adaptive_samples: int = 0
    attack_min_clean_accuracy: float = 0.85
    attack_max_defended_accuracy: float = 0.40
    attack_require_quality_gate: bool = True
    force_retrain: bool = False

    @property
    def source_profile_values(self) -> list[str]:
        return parse_csv_strings(self.source_profile_ids)

    @classmethod
    def from_namespace(cls, args) -> "AttackConfig":
        values = {field: getattr(args, field) for field in cls.__dataclass_fields__ if hasattr(args, field)}
        return cls(**values)

    @property
    def attacker_values(self) -> list[str]:
        return parse_csv_strings(self.attackers)

