# DMMPv3 Isolation Experiments

This note defines three diagnostic scripts for explaining why the current DMMPv3 fixed-attacker result can be worse than the original DMMP run. The scripts are wrappers around existing DMMPv3 modules; they do not move or overwrite historical DMMP/DMMP2 checkpoints.

The script enforces that new diagnostic outputs stay under `D:\learning\TOR\defence\DMMPv3\results`. Existing run directories are not reused for write-heavy experiments; choose a fresh `--run_name` whenever a command creates a new run.

All commands are run from:

```powershell
cd D:\learning\TOR\defence\DMMPv3
```

The local `llm` conda environment can be used directly:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\run_isolation_experiments.py <command>
```

## 1. Reuse Fixed Checkpoints

Purpose: hold the fixed DF/RF attackers constant so the measured difference comes from the defended traces, not from retraining variance.

Command:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\run_isolation_experiments.py reuse-fixed-eval --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

By default this uses the run's `fresh_deployment_test` defended cache. If that cache already exists, it will be reused instead of regenerating defended traces. If the cache is missing, generating the full defended test can take a long time; use `--max_test_traces 20` only for a quick smoke check.

Default checkpoints come from the verified fixed evaluator entries in `docs/model_registry.md`:

- DF: `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt`
- RF: `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt`

Output:

```text
<run_dir>\isolation\reuse_fixed_eval\attack_summary.csv
<run_dir>\isolation\reuse_fixed_eval\attack_summary.json
<run_dir>\isolation\reuse_fixed_eval\summary_zh.md
```

If this output directory already contains files, the command writes a timestamp-suffixed sibling directory instead of overwriting the previous diagnostic record.

Interpretation:

- If reused fixed RF remains high while reused fixed DF drops, V3 is failing to disrupt TAM/RF-visible features.
- If both attackers improve versus the run's retrained fixed attackers, part of the original gap came from attacker retraining differences.
- Compatibility fields in `attack_summary.json` must remain true before using the result. They include strict checkpoint loading, 95-class label mapping, split hashes, input representation settings, and clean accuracy within the registry tolerance.

## 2. Reliable Stage 3 Probe

Purpose: rerun only Stage 3 with larger fixed-probe train/validation budgets so the fixed RF/DF diagnostic accuracy can participate in Stage 3 gate/selection instead of being ignored as unreliable.

Command:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\run_isolation_experiments.py reliable-stage3-probe `
  --source_run_dir D:\learning\TOR\defence\DMMPv3\results\<source_run> `
  --run_name <source_run>_reliable_probe
```

Default overrides:

- `stage3_fixed_probe_train_samples = 30000`
- `stage3_fixed_probe_val_samples = 5000`
- `stage3_fixed_probe_samples = 5000`
- `stage3_fixed_probe_epochs = 10`
- `stage3_fixed_probe_min_clean_accuracy = 0.85`
- `stage3_fixed_probe_attackers = df,rf`
- diagnostic accuracy gate enabled
- rendered RF/fixed probe accuracy gate = `0.40`
- diagnostic fallback is allowed by default and marked as fallback, not as a passed policy. Use `formal_gate_passed = 1` in `summary.json` before treating the result as a gate-passing policy.

Output:

```text
D:\learning\TOR\defence\DMMPv3\results\<run_name>\stage3_guided_refinement\
D:\learning\TOR\defence\DMMPv3\results\<run_name>\isolation\reliable_stage3_probe\summary.json
D:\learning\TOR\defence\DMMPv3\results\<run_name>\isolation\reliable_stage3_probe\summary_zh.md
```

Interpretation:

- `selection_policy_valid = 1` means Stage 3 found a candidate that passed the configured gates.
- `selection_used_quality_fallback = 1` means the artifact is diagnostic only.
- `formal_gate_passed = 1` means the selected artifact is valid and did not use diagnostic fallback.
- RF-specific fields such as `selection_rendered_rf_accuracy`, `selection_fixed_rf_probe_reliable`, and `selection_fixed_rf_probe_reliable_accuracy` show whether the stronger probe actually constrained RF.

## 3. Minimal V3 Ablation

Purpose: keep DF/RF guidance but remove V3 additions that may overconstrain the policy search: profile masks, selected/preference weights, direction correction, and logit noise.

Dry run:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\run_isolation_experiments.py minimal-ablation --run_name dmmpv3_minimal_probe --dry_run
```

Full run:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\run_isolation_experiments.py minimal-ablation --run_name dmmpv3_minimal_probe --reuse_fixed_eval
```

The generated defense command includes:

```text
--no-condition_profile_mask
--no-condition_selected_mask
--no-condition_preference_weights
--preference_weight 0
--profile_weight 0
--direction_target none
--direction_correction_strength 0
--min_incoming_dummy_share 0
--policy_logit_noise_std 0
--guidance_attackers both
```

Interpretation:

- If minimal V3 improves fixed RF substantially, the regression is probably caused by profile/preference/direction/noise constraints rather than the core DF/RF guidance idea.
- If minimal V3 still fails against RF, the main problem is likely the surrogate-to-rendered TAM gap or candidate utility quality.

## Print The Suite

To print all three commands for an existing source run:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\run_isolation_experiments.py print-suite --source_run_dir D:\learning\TOR\defence\DMMPv3\results\<source_run>
```
