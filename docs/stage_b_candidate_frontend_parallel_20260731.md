# Stage B Candidate Frontend And Parallel Export Probe

Date: 2026-07-31

## Code Changes

Updated candidate materialization and scheduling utilities:

```text
dynapd/stage_b/expanded_generator.py
scripts/stage_b_run_b2e_diverse_search.py
scripts/stage_b_export_teacher_trajectories.py
scripts/stage_b_export_teacher_trajectories_active_batch.py
scripts/stage_b_run_student_policy_controller.py
scripts/stage_b_probe_parallel_teacher_workers.py
scripts/stage_b_launch_teacher_shards_parallel.py
```

Changes:

- Added a fast descriptor materialization path that reuses descriptor metadata and only builds exact sparse counts.
- Split candidate devices:
  - `--candidate_score_device`: descriptor score/cost generation.
  - `--candidate_eval_device`: candidate TAM / RF evaluation.
  - `--candidate_device` remains as a backward-compatible fallback.
- Added a NumPy CPU branch for descriptor score/cost computation.
- Updated the parallel worker probe so it can run the current compact B2-E Teacher exporter rather than the older default path.
- Added a bounded parallel shard launcher for full Teacher export runs.

## Equivalence Checks

Descriptor materialization:

```text
materialization equivalence ok, checked=600
```

Teacher smoke:

```text
Canonical old n=4: results/stage_b2e_teacher_canonical_for_active_batch_cmp_test_n4
Fast materialize n=4: results/stage_b2e_teacher_fast_materialize_smoke_test_n4
Diff count: 0
```

Train shard 3, n=128:

```text
Base: results/stage_b2e_teacher_canonical_current_train_shard3_n128
Score CPU / eval CUDA: results/stage_b2e_teacher_scorecpu_numpy_evalcuda_train_shard3_n128
sampleDiffCount = 0
recordStructuralDiffCount = 0
records = 339 / 339
```

## Single-Process Timing

Aligned train shard 3, n=128:

| run | wall sec | candidate gen sec | descriptor gen sec | materialization sec | RF forward sec | result |
|---|---:|---:|---:|---:|---:|---|
| stage_b2e_teacher_canonical_current_train_shard3_n128 | 12.84 | 5.65 | 2.31 | 1.48 | 2.61 | canonical |
| stage_b2e_teacher_fast_materialize_train_shard3_n128 | 13.64 | 5.78 | 2.46 | 1.38 | 2.87 | equivalent |
| stage_b2e_teacher_scorecpu_numpy_evalcuda_train_shard3_n128 | 12.95 | 5.20 | 2.14 | 1.26 | 2.90 | equivalent |

Interpretation:

- Fast materialization reduces the materialization subcomponent, but single-process wall time is dominated by mixed CPU/GPU scheduling variance.
- NumPy score on CPU avoids the earlier Torch-CPU device mismatch and keeps RF candidate eval on CUDA, but it is not a major wall-time win by itself.

## Parallel Worker Probe

Current B2-E compact Teacher, fixed budget 10%, sparse records:

```text
--compact_candidate_generation
--candidate_batch_size 4096
--candidate_score_device cpu
--candidate_eval_device cuda
--candidate_eval_mode gpu_tam
```

Probe outputs:

```text
results/stage_b_parallel_worker_current_b2e_probe_1_2w_n32
results/stage_b_parallel_worker_current_b2e_probe_3_4w_n32
results/stage_b_parallel_worker_current_b2e_probe_6_8w_n24
```

| workers | samples/worker | completed samples | wall sec | traces/hour | max GPU mem | mean GPU util |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 32 | 10.95 | 10520 | 2.49 GB | 10.29% |
| 2 | 32 | 64 | 12.05 | 19124 | 3.05 GB | 14.13% |
| 3 | 32 | 96 | 12.60 | 27421 | 4.11 GB | 20.67% |
| 4 | 32 | 128 | 14.35 | 32109 | 4.55 GB | 23.59% |
| 6 | 24 | 144 | 18.85 | 27506 | 6.20 GB | 23.29% |
| 8 | 24 | 192 | 28.14 | 24561 | 8.04 GB | 17.52% |

Interpretation:

- Multi-process shard parallelism is more effective than single-process active-state batching for the current implementation.
- On this RTX 5070 run, 4 workers is the best observed point in the small probe.
- 8 workers fills about 8 GB VRAM but lowers throughput, so filling VRAM is not the same as faster Teacher export.

## Parallel Launcher Smoke

Smoke:

```text
Run: results/stage_b_teacher_parallel_launcher_smoke_2shards_n4
Shards: 90-91
Max parallel workers: 2
Samples/shard: 4
Completed shards: 2
Failed shards: 0
Completed samples: 8
Records: 17
```

## Recommended Full-Scale Command Shape

Use the bounded launcher with 4 workers as the first full-train attempt, then audit failed shards and retry with `--resume`:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\stage_b_launch_teacher_shards_parallel.py `
  --python_exe D:\Miniconda3\envs\llm\python.exe `
  --archive results\stage_b_fast_keypoint_full_cw_all_seed0\fast_keypoint_archive.npz `
  --split_file results\stage_b_policy_dataset_full_cw_seed0\policy_splits.npz `
  --split_name train `
  --start_shard 0 `
  --end_shard 256 `
  --num_shards 256 `
  --max_parallel_workers 4 `
  --max_samples_per_shard -1 `
  --candidate_batch_size 4096 `
  --candidate_score_device cpu `
  --candidate_eval_device cuda `
  --candidate_eval_mode gpu_tam `
  --method stratified_top128 `
  --protocol bidirectional_cooperative `
  --max_delay 64 `
  --rounds 3 `
  --max_dummy_steps 8 `
  --max_action_budget 0.10 `
  --budget 0.10 `
  --storage_mode sparse `
  --device auto `
  --resume `
  --launcher_name stage_b2e_teacher_full_cw_train_parallel4
```

The launcher writes:

```text
teacher_parallel_shards.csv
teacher_parallel_monitor.csv
teacher_parallel_run.json
```
