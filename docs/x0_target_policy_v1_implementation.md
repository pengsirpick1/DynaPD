# x0* Target-Policy Diffusion v1 Implementation Notes

## Scope

This v1 implementation adds an independent target-policy path beside legacy
DMMPv3. It does not remove `make_prior_logits`, does not modify existing
checkpoints, and does not replace `dmmp/diffusion/profile_pipeline.py`.

Implemented now:

- Offline `x0*` target-policy representation with masked CLR.
- Exact largest-remainder budget rounding.
- Label-free prefix-based candidate generation.
- Five high-level strategy families built from five low-level primitives.
- Versioned target-policy pool persistence using `policies.npz`, `index.csv`,
  `metadata.json`, and `build_summary_zh.md`.
- Training loader that filters label-like arrays.
- Smoke target-diffusion training with `L_eps`, `L_x0`, `L_alloc`, `L_effect`,
  `L_family`, `L_primitive`, `L_struct`, `L_fusion`, and `L_smooth`.
- Unit tests for hard invariants and loader label filtering.
- Timestamped default output directories and overwrite protection.
- Deployable-aware target selection.

Not implemented as full production yet:

- Real DF/RF teacher scoring inside the pool builder.
- Completion-set retrieval across prefix-neighbor traces.
- Full replacement of the legacy stage2/stage3 pipeline.
- Full fixed/adaptive/mixed/open-world experiments.
- zarr/parquet storage; v1 uses project-existing dependencies only.

## Label Boundary

`scripts/build_target_policy_pool_v1.py` may load dataset labels only because
the existing CW loader returns labels and uses them for split/subset utilities.
The builder deletes `labels` immediately and candidate generation receives only
trace-derived prefix features.

`TargetPolicyPool.load_training_arrays()` filters label-like keys such as:

- `y`
- `label`, `labels`
- `true_label`, `true_labels`
- `class`, `classes`, `class_id`
- `site`, `site_id`
- `target_class`, `target_label`

The target-diffusion training script reads through this training loader.

## Current Smoke Commands

```powershell
conda run -n llm python -m unittest discover -s tests -p "test*.py"

conda run -n llm python scripts\build_target_policy_pool_v1.py `
  --config configs\x0_target_diffusion_v1.yaml `
  --smoke --max_samples 80 --max_classes 4 --max_traces 4

# Use the timestamped pool path printed by the build command.
conda run -n llm python scripts\validate_target_policy_pool_v1.py `
  --pool_dir results\<timestamp>_target_policy_pool_v1_smoke `
  --strict

conda run -n llm python scripts\train_target_diffusion_v1.py `
  --pool_dir results\<timestamp>_target_policy_pool_v1_smoke `
  --config configs\x0_target_diffusion_v1.yaml `
  --smoke --epochs 1 --batch_size 4 --max_steps 1
```

If an explicit output directory already contains result sentinels, the scripts
fail unless `--overwrite` is supplied.

## Smoke Results

Latest smoke pool:

- rows: 24
- x0 shape: `[24, 2, 200]`
- budget violations: 0
- mask violations: 0
- negative counts: 0
- leaked label keys: none
- fallback count: 0
- score source: `heuristic_proxy_not_df_rf_teacher`

Latest smoke training:

- rows: 24
- batch size: 4
- steps: 1
- last loss includes `eps`, `x0`, `alloc`, `effect`, `family`, `primitive`,
  `struct`, `fusion`, and `smooth`.

## Next Full Implementation Steps

1. Replace proxy quality with frozen DF/RF teacher scoring.
2. Add completion-set retrieval using train-only prefix embeddings.
3. Store completion indices and pseudo-labels from DF/RF without returning true
   labels to diffusion.
4. Integrate target-pool diffusion into the existing stage2/stage3 pipeline as
   a new explicit mode, keeping legacy prior as baseline.
5. Run Review Agent approval again before full experiments.
