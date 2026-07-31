# Stage B Active-Batch Teacher Exporter

Date: 2026-07-31

## Change

Added an experimental active-state Teacher exporter:

```text
scripts/stage_b_export_teacher_trajectories_active_batch.py
```

The script keeps multiple trace states active and exposes:

```text
--active_states
--candidate_batch_size
--rf_candidate_batch_size
```

It currently supports the compact single-action B2-E route, e.g. `stratified_top128`.

## Validation

Small aligned smoke:

```text
Split: test
Samples: 4
Canonical: results/stage_b2e_teacher_canonical_for_active_batch_cmp_test_n4
Active batch: results/stage_b2e_teacher_active_batch_smoke_test_n4_v2
```

Matched sample-level outcomes:

```text
final_pred, accuracy, stop_reason, records, action_records, stop_records,
accepted_action_count, bandwidth, delay, candidate_total_count
```

The only differences were tiny GPU batch floating-point changes in `selected_gain`.

## Throughput Probe

Aligned `n=128`, train shard 3:

```text
Archive: results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz
Split: train
Shard: 3 / 256
Budget: fixed 10%
Method: stratified_top128
Protocol: bidirectional_cooperative
Max delay: 64
Storage: sparse
```

| run | active_states | candidate_batch_size | rf_candidate_batch_size | wall sec | records | defended RF acc | mean BW | mean delay bins | internal GPU peak |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage_b2e_teacher_canonical_current_train_shard3_n128 | 1 | 4096 | 4096 | 12.84 | 339 | 6.25% | 5.66% | 5.21 | n/a |
| stage_b2e_teacher_active_batch_a8_b4096_train_shard3_n128 | 8 | 4096 | 4096 | 14.67 | 339 | 6.25% | 5.66% | 5.21 | 1094 MB |
| stage_b2e_teacher_active_batch_a32_b4096_train_shard3_n128 | 32 | 4096 | 4096 | 13.46 | 339 | 6.25% | 5.66% | 5.21 | 3688 MB |
| stage_b2e_teacher_active_batch_a32_b8192_train_shard3_n128 | 32 | 8192 | 8192 | 13.96 | 327 | 5.47% | 5.66% | 5.22 | 4075 MB |

## Interpretation

- After NumPy vectorized candidate selection, active-state batching is not a net speedup on this shard.
- Larger active states can increase GPU memory use, but the formal exporter is now bottlenecked by per-state candidate generation, keypoint refresh, rendering, serialization, and scheduling overhead more than by RF candidate forward.
- `active_states=32, candidate_batch_size=8192` changes Teacher trajectories on this probe, likely from larger-batch floating-point differences around near-tie action choices. It should not be used as a canonical training-data default.
- For full Teacher trajectory generation, keep the current canonical vectorized exporter as the default unless a later change batchifies candidate generation itself or adds a strict canonical re-verify pass.

## Recommended Use

Canonical full-shard training data:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\stage_b_export_teacher_trajectories.py `
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

Experimental active-batch probe only:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\stage_b_export_teacher_trajectories_active_batch.py `
  --archive results\stage_b_fast_keypoint_full_cw_all_seed0\fast_keypoint_archive.npz `
  --split_file results\stage_b_policy_dataset_full_cw_seed0\policy_splits.npz `
  --split_name train `
  --num_shards 256 `
  --shard_id 0 `
  --max_samples 128 `
  --budget_mode fixed `
  --fixed_budget 0.10 `
  --active_states 8 `
  --candidate_batch_size 4096 `
  --rf_candidate_batch_size 4096 `
  --candidate_device cuda `
  --candidate_eval_mode gpu_tam `
  --storage_mode sparse `
  --run_name stage_b2e_teacher_active_batch_probe
```
