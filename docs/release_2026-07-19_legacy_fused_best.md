# DMMPv3 Legacy-Fused Best Fixed Evaluation Snapshot

Date: 2026-07-19

This snapshot preserves the best fixed-attack DMMPv3 result currently available in local experiments.

## Best Run

- Run name: `dmmpv3_legacydirect_fused_fullcw_seed0_20260719_134324`
- Run directory at experiment time: `results/dmmpv3_legacydirect_fused_fullcw_seed0_20260719_134324`
- Dataset: external CW dataset at `D:\learning\TOR\datasets\CW\CW.npz`
- Dataset is intentionally not included in this repository snapshot.

## Method

- Full CW split: train 84602, validation 10564, test 10564
- Defense pipeline: full Stage 1 + Stage 2 guided diffusion + Stage 3 selection
- `profile_combination_mode`: `legacy_pool`
- `v1_mode_pool`: `legacy_direct`
- Renderer: `multi_view`
- `multi_view_mode`: `fused`
- Multi-view shares: DF 0.50, AWF 0.25, RF 0.25
- Budget: 0.30
- Deployment repeats: 3
- Stage 3 fixed probe was disabled in this run.

## Fixed Attack Result

| Attacker | Clean Accuracy | Defended Accuracy | Gate |
|---|---:|---:|---:|
| DF | 0.973968 | 0.184495 | pass |
| RF | 0.978607 | 0.448252 | fail |

Visible dummy overhead: 0.289779

Raw real-packet retention: 1.000000

This is the strongest saved fixed DF/RF result so far because it has the lowest saved RF defended accuracy while also strongly suppressing DF. It does not pass the RF target gate of 0.40.

## Reproduce

```powershell
cd D:\learning\TOR\defence\DMMPv3
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1
```

To inspect the expanded command without running:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1 -DryRun -RunName dryrun_legacy_fused_fullcw
```

The script now uses real-time subprocess output through `conda run --no-capture-output` and `python -u`.

## Lightweight Artifacts

The following lightweight files are preserved in `docs/best_legacy_fused_20260719/`:

- `run_config.json`
- `fixed_attack_summary.json`
- `fixed_attack_summary_zh.md`
- `stage2_metrics.json`
- `stage3_selected_policy.json`
- `stage3_summary_zh.md`
- `stage3_pareto_results.csv`

Heavy artifacts such as `.npz` defended datasets and `.pt` checkpoints are excluded from git.
