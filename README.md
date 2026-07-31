# DynaPD

DynaPD is a compact research release for the current Stage B dynamic padding
defense against website fingerprinting attacks.

The current method is not a diffusion defense. This repository contains the
Stage B2-E teacher/search pipeline, vectorized candidate selection, GPU TAM
candidate evaluation, shard-based teacher export, and student candidate-policy
training/evaluation.

Datasets, checkpoints, generated archives, logs, and experiment outputs are not
included.

## Current Method

Main point:

```text
Teacher: Stage B2-E stratified_top128
Protocol: bidirectional_cooperative
Budget: fixed 10% dummy bandwidth
Delay: D64
Attacker metric: RF accuracy
```

Reference local CW/RF results:

| method | clean RF acc | defended RF acc | mean bandwidth | mean delay |
|---|---:|---:|---:|---:|
| Teacher `stratified_top128`, D64 | 96.84% | 4.21% | 6.17% | 5.03 bins |
| Student `top4_verify` | n/a | 8.42% | 5.90% | n/a |

Latency audit:

| method | mean completion delay | p95 completion delay |
|---|---:|---:|
| Teacher | 0.0706 s | 0.4202 s |
| Student `top4_verify` | 0.0664 s | 0.4198 s |

Parallel worker probe on the local machine:

| workers | throughput | GPU memory |
|---:|---:|---:|
| 1 | 10520 traces/hour | 2.49 GB |
| 2 | 19124 traces/hour | 3.05 GB |
| 3 | 27421 traces/hour | 4.11 GB |
| 4 | 32109 traces/hour | 4.55 GB |
| 8 | 24561 traces/hour | 8.04 GB |

The practical default is 4 workers per GPU, then tune on the target server.

## Layout

```text
dynapd/
  data/              CW loading and split helpers.
  evaluation/        RF/DF attacker model adapters and TAM input builders.
  projection/        Trace rendering/padding utilities.
  stage_a/           Minimal attacker/keypoint helpers reused by Stage B.
  stage_b/           Candidate actions, objectives, teacher search, policy data/model.
  utils/             Runtime config and serialization helpers.

scripts/
  stage_b_prepare_fast_keypoint_archive.py
  stage_b_build_policy_dataset.py
  stage_b_export_teacher_trajectories.py
  stage_b_launch_teacher_shards_parallel.py
  stage_b_probe_parallel_teacher_workers.py
  stage_b_train_candidate_policy.py
  stage_b_eval_candidate_policy_offline.py
  stage_b_run_student_policy_controller.py
```

## Install

Install a CUDA-compatible PyTorch build first if you plan to run on GPU.

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
python scripts/verify_project.py
```

## Inputs

Pass local paths explicitly:

```text
--data_root      CW dataset directory or npz path
--checkpoint     fixed RF/DF attacker checkpoint
--archive        generated fast-keypoint archive
--split_file     generated policy split npz
```

The repository ignores:

```text
datasets/ data/ results/ logs/ models/
*.pt *.pth *.ckpt *.npz *.npy *.pkl *.pickle
```

## Build Inputs

Prepare the fast-keypoint archive:

```bash
python scripts/stage_b_prepare_fast_keypoint_archive.py \
  --data_root /path/to/CW \
  --attacker rf \
  --checkpoint /path/to/rf_checkpoint.pt \
  --split_name all \
  --batch_size 512 \
  --device cuda \
  --run_name stage_b_fast_keypoint_full_cw_all_seed0 \
  --progress
```

Build policy splits:

```bash
python scripts/stage_b_build_policy_dataset.py \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --full_cw \
  --run_name stage_b_policy_dataset_full_cw_seed0
```

## Probe Server Throughput

Run a short probe before the full export:

```bash
python scripts/stage_b_probe_parallel_teacher_workers.py \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --split_file results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz \
  --split_name train \
  --worker_counts 1,2,3,4,6,8 \
  --start_shard 0 \
  --num_shards 256 \
  --samples_per_worker 16 \
  --target_gpu_mem_gb 999 \
  --continue_after_target \
  --budget 0.10 \
  --method stratified_top128 \
  --protocol bidirectional_cooperative \
  --max_delay 64 \
  --rounds 3 \
  --max_dummy_steps 8 \
  --max_action_budget 0.10 \
  --compact_candidate_generation \
  --candidate_batch_size 4096 \
  --candidate_device cuda \
  --candidate_score_device cpu \
  --candidate_eval_device cuda \
  --candidate_eval_mode gpu_tam \
  --storage_mode sparse \
  --device cuda
```

Read:

```text
results/<probe_name>/parallel_probe_summary.csv
results/<probe_name>/parallel_probe_monitor.csv
```

Pick the highest `traces_per_hour` setting that still leaves stable GPU memory
and acceptable CPU load.

## Full Teacher Export

Dry-run first:

```bash
python scripts/stage_b_launch_teacher_shards_parallel.py \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --split_file results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz \
  --split_name train \
  --num_shards 256 \
  --start_shard 0 \
  --end_shard 256 \
  --max_parallel_workers 4 \
  --gpu_ids 0 \
  --per_worker_threads 2 \
  --budget 0.10 \
  --method stratified_top128 \
  --protocol bidirectional_cooperative \
  --max_delay 64 \
  --rounds 3 \
  --max_dummy_steps 8 \
  --max_action_budget 0.10 \
  --candidate_batch_size 4096 \
  --candidate_device cuda \
  --candidate_score_device cpu \
  --candidate_eval_device cuda \
  --candidate_eval_mode gpu_tam \
  --storage_mode sparse \
  --device cuda \
  --resume \
  --dry_run
```

Single-GPU full export:

```bash
python scripts/stage_b_launch_teacher_shards_parallel.py \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --split_file results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz \
  --split_name train \
  --num_shards 256 \
  --start_shard 0 \
  --end_shard 256 \
  --max_parallel_workers 4 \
  --gpu_ids 0 \
  --per_worker_threads 2 \
  --budget 0.10 \
  --method stratified_top128 \
  --protocol bidirectional_cooperative \
  --max_delay 64 \
  --rounds 3 \
  --max_dummy_steps 8 \
  --max_action_budget 0.10 \
  --candidate_batch_size 4096 \
  --candidate_device cuda \
  --candidate_score_device cpu \
  --candidate_eval_device cuda \
  --candidate_eval_mode gpu_tam \
  --storage_mode sparse \
  --device cuda \
  --resume \
  --fail_fast \
  --progress
```

Two-GPU server example:

```bash
python scripts/stage_b_launch_teacher_shards_parallel.py \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --split_file results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz \
  --split_name train \
  --num_shards 256 \
  --start_shard 0 \
  --end_shard 256 \
  --max_parallel_workers 8 \
  --gpu_ids 0,1 \
  --per_worker_threads 2 \
  --budget 0.10 \
  --method stratified_top128 \
  --protocol bidirectional_cooperative \
  --max_delay 64 \
  --rounds 3 \
  --max_dummy_steps 8 \
  --max_action_budget 0.10 \
  --candidate_batch_size 4096 \
  --candidate_device cuda \
  --candidate_score_device cpu \
  --candidate_eval_device cuda \
  --candidate_eval_mode gpu_tam \
  --storage_mode sparse \
  --device cuda \
  --resume \
  --fail_fast \
  --progress
```

Multiple servers can split the shard range:

```text
server A: --start_shard 0   --end_shard 64
server B: --start_shard 64  --end_shard 128
server C: --start_shard 128 --end_shard 192
server D: --start_shard 192 --end_shard 256
```

The launcher writes:

```text
results/<launcher_name>/teacher_parallel_run.json
results/<launcher_name>/teacher_parallel_shards.csv
results/<launcher_name>/teacher_parallel_monitor.csv
results/<launcher_name>/logs/*.stdout.log
results/<launcher_name>/logs/*.stderr.log
```

## Student Policy

Train from exported teacher CSV records:

```bash
python scripts/stage_b_train_candidate_policy.py \
  --records_csv /path/to/teacher_train_records.csv \
  --val_records_csv /path/to/teacher_val_records.csv \
  --run_name stage_b_candidate_policy_top4 \
  --epochs 5 \
  --batch_size 8 \
  --device cuda
```

Evaluate `student_top4_verify`:

```bash
python scripts/stage_b_run_student_policy_controller.py \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --policy_checkpoint /path/to/policy_checkpoint.pt \
  --split_file results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz \
  --split_name test \
  --methods student_top4_verify \
  --dummy_budgets 0.10 \
  --max_delay 64 \
  --compact_candidate_generation \
  --candidate_batch_size 4096 \
  --candidate_device cuda \
  --device cuda \
  --progress
```

## Notes

This is a research release. The public tree is intentionally slim and excludes
legacy diffusion, purifier, target-policy, plotting, and local scratch modules
that are not part of the current Stage B2-E method.
