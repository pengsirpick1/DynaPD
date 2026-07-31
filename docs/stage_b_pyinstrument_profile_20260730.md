# Stage B Pyinstrument Profile, 2026-07-30

Profiler:

```text
pyinstrument 5.1.3
Installed in D:\Miniconda3\envs\llm
```

Profile outputs:

```text
results/pyinstrument_profiles_current/teacher950_all_tensorized.html
results/pyinstrument_profiles_current/teacher950_all_tensorized.txt
results/pyinstrument_profiles_current/teacher950_all_tensorized_flat.txt

results/pyinstrument_profiles_current/active_state16_950_all.html
results/pyinstrument_profiles_current/active_state16_950_all.txt
results/pyinstrument_profiles_current/active_state16_950_all_flat.txt

results/pyinstrument_profiles_current/student_top4_verify_950_test.html
results/pyinstrument_profiles_current/student_top4_verify_950_test.txt
results/pyinstrument_profiles_current/student_top4_verify_950_test_flat.txt
```

## Formal Teacher 950 All

Command target:

```text
scripts/stage_b_export_teacher_trajectories.py
split_name = all
samples = 950
method = stratified_top128
protocol = bidirectional_cooperative
B = 10%
compact_candidate_generation = true
candidate_eval_mode = gpu_tam
candidate_batch_size = 4096
```

Result:

```text
Run: results/stage_b2e_teacher_950_all_pyinstrument_tensorized_seed0
Wall time under pyinstrument: 306.11 sec
Clean RF acc: 98.11%
Defended RF acc: 9.79%
Mean actual dummy bandwidth: 6.12%
Candidate RF eval/sample: 168.39
GPU candidate peak allocated: 2507 MB
```

Manual component timers:

| component | sec | share of wall |
|---|---:|---:|
| candidate_generation_time | 228.48 | 74.64% |
| compact_descriptor_generation_total | 173.25 | 56.60% |
| rf_forward_time | 21.49 | 7.02% |
| candidate_gpu_tam_eval_time | 18.88 | 6.17% |
| deferred_materialization_time | 18.23 | 5.96% |
| renderer_time | 13.71 | 4.48% |
| delay_trace_time | 10.05 | 3.28% |
| keypoint_refresh_time | 6.41 | 2.09% |
| serialization_time | 5.80 | 1.89% |
| queue_or_wait_time | 14.70 | 4.80% |

Pytorch/pyinstrument call stack hotspots:

```text
_run_controller -> _select_dummy_b2e
  -> generate_compact_action_descriptors
  -> _generate_compact_action_descriptors_batched
  -> _select_descriptor_indices
  -> cost / estimated_gain / composite / identity
```

Interpretation:

- The dominant bottleneck is still descriptor selection and candidate scoring in Python/Numpy logic.
- RF forward and GPU TAM candidate evaluation are not the main bottleneck.
- More GPU batch size alone cannot fix this path unless descriptor selection is also vectorized or reduced.

## Active-State 16 Probe, 950 All

Command target:

```text
scripts/stage_b_probe_active_state_gpu_batch.py
split_name = all
samples = 950
active_states = 16
candidate_batch_size = 8192
```

Result:

```text
Run: results/stage_b_active_state16_950_all_pyinstrument_seed0
Wall time under pyinstrument: 519.13 sec
Throughput under pyinstrument: 6587.94 traces/hour
Internal GPU peak allocated: 3291 MB
Descriptor count: 2772430
Action objects built: 235552
```

Manual component timers:

| component | sec | share of wall |
|---|---:|---:|
| candidate_generation_time | 465.76 | 89.72% |
| rf_forward_time | 32.06 | 6.18% |
| candidate_gpu_tam_eval_time | 31.72 | 6.11% |
| renderer_time | 15.18 | 2.92% |
| candidate_tam_gpu_build_time | 3.09 | 0.59% |
| tam_rebuild_time | 0.58 | 0.11% |

Pytorch/pyinstrument call stack hotspots:

```text
main -> _build_pack
  -> generate_compact_action_descriptors
  -> _generate_compact_action_descriptors_batched
  -> _select_descriptor_indices
  -> cost / estimated_gain / composite / identity
```

Interpretation:

- `active_states=16` does create a batched execution structure, but it still spends most wall time before RF evaluation.
- The GPU peak is only about 3.29 GB under profile, so it still does not approach an 8 GB or 12 GB target.
- Increasing active states further may fill memory, but the profile says the next useful optimization is reducing `_select_descriptor_indices` cost, not simply adding more active states.

## Student Top4 Verify, 950 Test

Command target:

```text
scripts/stage_b_run_student_policy_controller.py
split_name = test
samples = 95
method = student_top4_verify
B = 10%
compact_candidate_generation = true
candidate_batch_size = 4096
```

Result:

```text
Run: results/stage_b_student_top4_verify_950_test_pyinstrument_seed0
Wall time under pyinstrument: 38.17 sec
Mean runtime/sample under pyinstrument: 0.308 sec
Defended RF acc: 8.42%
Mean actual dummy bandwidth: 5.90%
Candidate RF eval/sample: 6.32
Scored candidates/sample: 125.55
```

Pytorch/pyinstrument call stack hotspots:

```text
_run_student_controller
  -> _generate_student_actions
  -> generate_compact_action_descriptors
  -> _generate_compact_action_descriptors_batched
  -> _select_descriptor_indices
```

Interpretation:

- Student verification successfully reduces RF evaluation to about 6.32 per sample.
- Even in Student mode, action generation dominates over RF verification.
- The policy network itself is not visible as a primary runtime bottleneck; candidate enumeration still is.

## Optimization Priority

The profile result is consistent across Teacher, active-state probe, and Student:

```text
The bottleneck is not RF forward.
The bottleneck is candidate descriptor scoring/selection before RF forward.
```

Priority order:

1. Replace Python lambda/list/sorted-heavy `_select_descriptor_indices` with array or torch top-k selection.
2. Cache action `cost`, `identity`, `estimated_gain`, and bucket fields inside the descriptor table instead of recomputing them through Python callables.
3. Avoid constructing large Python `CandidateDescriptor` objects until after top-k selection.
4. Move candidate table filtering and top-k fully to GPU only if the host-side descriptor enumeration is first reduced.
5. Integrate active-state scheduling into the formal sparse Teacher exporter only after descriptor selection is vectorized; otherwise it mainly increases waiting and memory pressure.
