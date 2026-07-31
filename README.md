# DynaPD

DynaPD is a research codebase for dynamic padding defenses against website
fingerprinting attacks. The repository contains the current DMMPv3 harness, with
the most active line of work in Stage B: teacher/oracle action search, compact
candidate selection, GPU TAM candidate evaluation, and student policy training.

This public snapshot intentionally excludes datasets, checkpoints, generated
archives, logs, and experiment outputs.

## Current Research Anchor

The strongest current method is the Stage B2-E `stratified_top128` teacher with
D64 delay support under the `bidirectional_cooperative` protocol.

Reference results from the local CW/RF evaluation:

| method | clean RF acc | defended RF acc | mean bandwidth | mean delay |
|---|---:|---:|---:|---:|
| Stage B2-E Teacher, `stratified_top128`, D64 | 96.84% | 4.21% | 6.17% | 5.03 bins |
| Student `top4_verify` | n/a | 8.42% | 5.90% | n/a |

Latency audit:

| method | mean completion delay | p95 completion delay |
|---|---:|---:|
| Teacher | 0.0706 s | 0.4202 s |
| Student `top4_verify` | 0.0664 s | 0.4198 s |

Throughput notes:

| parallel workers | throughput | GPU memory |
|---:|---:|---:|
| 1 | 10520 traces/hour | 2.49 GB |
| 2 | 19124 traces/hour | 3.05 GB |
| 3 | 27421 traces/hour | 4.11 GB |
| 4 | 32109 traces/hour | 4.55 GB |
| 8 | 24561 traces/hour | 8.04 GB |

The recommended local shard setting is currently 4 workers. The active-state
batch exporter is kept as an experimental probe; the canonical teacher data path
should use the vectorized exporter or the shard launcher unless a later change
strictly preserves the teacher trajectory.

## Repository Layout

```text
configs/              Experiment configuration files.
dmmp/                 Python package for data loading, attacks, defense models,
                      rendering, purifier prototypes, Stage A, and Stage B.
scripts/              Reproducible experiment, audit, plotting, and training entrypoints.
docs/                 Method notes, runbooks, audit reports, and experiment logs.
stage_a/README.md     Stage A research notes.
stage_b/README.md     Stage B research notes.
tasks/                Local task memory for the research harness.
```

Important Stage B files:

```text
dmmp/stage_b/expanded_generator.py
scripts/stage_b_prepare_fast_keypoint_archive.py
scripts/stage_b_build_policy_dataset.py
scripts/stage_b_export_teacher_trajectories.py
scripts/stage_b_launch_teacher_shards_parallel.py
scripts/stage_b_train_candidate_policy.py
scripts/stage_b_run_student_policy_controller.py
```

## Installation

The code is developed on Windows with a CUDA PyTorch environment, but the package
itself is plain Python.

```powershell
cd D:\learning\TOR\defence\DMMPv3
python -m pip install -r requirements.txt
python -m pip install -e .
```

For GPU runs, install the PyTorch build that matches your CUDA driver first, then
install the remaining requirements.

## Data And Checkpoint Policy

No dataset, model checkpoint, teacher archive, policy checkpoint, result CSV,
profile report, or log file is committed.

Expected local inputs are:

```text
CW dataset: pass with --data_root, or place under D:\learning\TOR\datasets\CW
RF/DF checkpoints: pass with --checkpoint when a script needs a fixed attacker
Generated archives/results: written under results/ by default
```

The `.gitignore` excludes `datasets/`, `data/`, `results/`, `logs/`, `models/`,
`*.pt`, `*.pth`, `*.ckpt`, `*.npz`, `*.npy`, and related temporary artifacts.

## Quick Checks

```powershell
python scripts\verify_project.py
python -m compileall dmmp scripts
```

## Stage B2-E Reproduction Skeleton

Prepare a full CW fast-keypoint archive:

```powershell
python scripts\stage_b_prepare_fast_keypoint_archive.py `
  --data_root D:\learning\TOR\datasets\CW `
  --attacker rf `
  --checkpoint <path-to-rf-checkpoint.pt> `
  --split_name all `
  --run_name stage_b_fast_keypoint_full_cw_all_seed0 `
  --progress
```

Build train/validation/test policy splits:

```powershell
python scripts\stage_b_build_policy_dataset.py `
  --archive results\stage_b_fast_keypoint_full_cw_all_seed0\fast_keypoint_archive.npz `
  --full_cw `
  --run_name stage_b_policy_dataset_full_cw_seed0
```

Export canonical teacher trajectories for one shard:

```powershell
python scripts\stage_b_export_teacher_trajectories.py `
  --archive results\stage_b_fast_keypoint_full_cw_all_seed0\fast_keypoint_archive.npz `
  --split_file results\stage_b_policy_dataset_full_cw_seed0\policy_splits.npz `
  --split_name train `
  --num_shards 256 `
  --shard_id 0 `
  --max_samples -1 `
  --budget_mode fixed `
  --fixed_budget 0.10 `
  --method stratified_top128 `
  --protocol bidirectional_cooperative `
  --max_delay 64 `
  --rounds 3 `
  --max_dummy_steps 8 `
  --max_action_budget 0.10 `
  --compact_candidate_generation `
  --candidate_batch_size 4096 `
  --candidate_device cuda `
  --candidate_eval_mode gpu_tam `
  --storage_mode sparse `
  --resume `
  --run_name stage_b2e_teacher_full_cw_train_shard000_of256
```

Launch multiple teacher shards:

```powershell
python scripts\stage_b_launch_teacher_shards_parallel.py `
  --archive results\stage_b_fast_keypoint_full_cw_all_seed0\fast_keypoint_archive.npz `
  --split_file results\stage_b_policy_dataset_full_cw_seed0\policy_splits.npz `
  --split_name train `
  --num_shards 256 `
  --start_shard 0 `
  --end_shard 256 `
  --max_parallel_workers 4 `
  --budget 0.10 `
  --method stratified_top128 `
  --candidate_batch_size 4096 `
  --candidate_device cuda `
  --candidate_score_device cpu `
  --candidate_eval_device cuda `
  --candidate_eval_mode gpu_tam `
  --storage_mode sparse `
  --resume `
  --progress
```

Train a student policy from exported teacher records:

```powershell
python scripts\stage_b_train_candidate_policy.py `
  --records_csv <teacher-train-records.csv> `
  --val_records_csv <teacher-val-records.csv> `
  --run_name stage_b_candidate_policy_top4 `
  --epochs 5 `
  --batch_size 8
```

Evaluate a student controller:

```powershell
python scripts\stage_b_run_student_policy_controller.py `
  --archive results\stage_b_fast_keypoint_full_cw_all_seed0\fast_keypoint_archive.npz `
  --policy_checkpoint <path-to-policy-checkpoint.pt> `
  --split_file results\stage_b_policy_dataset_full_cw_seed0\policy_splits.npz `
  --split_name test `
  --methods student_top4_verify `
  --dummy_budgets 0.10 `
  --max_delay 64 `
  --compact_candidate_generation `
  --candidate_batch_size 4096 `
  --candidate_device cuda `
  --progress
```

## Documentation

The most useful local notes for this snapshot are:

```text
docs/method_spec.md
docs/experiment_protocol.md
docs/runbook.md
docs/stage_b2d_strategy_versions.md
docs/stage_b_vectorized_candidate_selection_20260731.md
docs/stage_b_candidate_frontend_parallel_20260731.md
docs/stage_b_active_batch_exporter_20260731.md
docs/stage_b_latency_overhead_audit_20260731.md
```

## Status

This is research code, not a polished library release. The default focus is
reproducible local experimentation rather than a stable public API.
