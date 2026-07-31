# Stage B2-D Strategy Version Registry

This file is the local memory anchor for Stage B2-D strategy iterations.

## Version 1 / V1 Baseline

Recorded on: 2026-07-29

User-facing name:

```text
First version / V1
```

Scope:

```text
Fixed-budget curve, D=64, margin <= 0
Protocol: bidirectional
Attacker metric: RF accuracy
Bandwidth metric: actual dummy bandwidth, not the configured upper bound
```

Canonical V1 result table from the user-provided screenshot:

| protocol | B bound | RF acc | mean actual bw | p95 bw |
|---|---:|---:|---:|---:|
| bidirectional | 0% | 63.54% | 0% | 0% |
| bidirectional | 2% | 40.63% | 0.61% | 1.98% |
| bidirectional | 5% | 37.50% | 0.78% | 3.25% |
| bidirectional | 10% | 37.50% | 0.92% | 3.95% |
| bidirectional | 20% | 36.46% | 1.02% | 3.95% |
| bidirectional | 30% | 36.46% | 1.00% | 3.95% |

Implementation anchors:

```text
scripts/stage_b_run_target_min_cost.py
scripts/stage_b_run_dual_actuator.py
dmmp/stage_b/expanded_generator.py
dmmp/stage_b/smoothing.py
```

Related audit and visualization helpers created during the V1 review:

```text
scripts/stage_b_plot_same_flow_tam.py
scripts/stage_b_audit_candidate_pool.py
```

Important V1 behavior:

- `B bound` is a maximum allowed dummy bandwidth, not a target that must be filled.
- The measured bandwidth is actual rendered dummy overhead.
- Delay cost is separate from dummy bandwidth cost.
- V1 uses strict single-action greedy acceptance for dummy selection: a selected action must have positive marginal utility.
- Candidate generation uses `ExpandedAction`, so the data structure can represent a local template, but the current V1 prefilter often evaluates mostly 1-dummy / 1-bin actions.
- Higher-dose actions exist in deeper candidate pools but are often ranked too low to be evaluated under the current V1 selector settings.
- This V1 baseline should be treated as the comparison point for all following Stage B2-D strategy changes.

## Version 2 / V2 Work Area

Status:

```text
Implemented as Stage B2-E preliminary V2.
```

Default meaning after this record:

```text
When the user says "second version" or "V2", it means a new Stage B2-D strategy implemented after V1, compared against the V1 table above.
```

Candidate V2 directions discussed before implementation:

- Keep a hard dummy bandwidth upper bound, typically 10%.
- Improve candidate prefiltering so higher-dose and multi-bin actions can enter evaluation.
- Add a relaxed two-action search: allow the first action to be slightly negative only when the submitted pair has positive combined utility.
- Track stop reasons separately: `no_positive_single`, `no_positive_pair`, `candidate_pool_exhausted`, `budget_reached`, and `target_reached`.

Implementation anchors:

```text
scripts/stage_b_run_b2e_diverse_search.py
dmmp/stage_b/expanded_generator.py
```

Preliminary V2 n=16 screening:

```text
Run: results/stage_b2e_screen_n16_b10_v1
Setting: n=16, bidirectional, D=64, margin<=0, B=10%
```

| method | RF acc | mean actual bw | p95 bw | note |
|---|---:|---:|---:|---|
| current_v1 | 18.75% | 0.97% | 2.75% | V1-like selector |
| score_hint_top128 | 12.50% | 0.70% | 1.93% | still low-dose biased |
| stratified_top64 | 0.00% | 6.50% | 10.03% | diverse strict single |
| stratified_top128 | 0.00% | 5.99% | 10.03% | diverse strict single |
| stratified_pair64_e0.01_t0 | 12.50% | 6.56% | 10.03% | relaxed pair, worse here |
| stratified_pair128_e0.01_t0 | 0.00% | 6.54% | 10.06% | relaxed pair, not better than strict |

Preliminary V2 n=96 B=10% confirmation:

| method | RF acc | mean actual bw | p95 bw | max bw | note |
|---|---:|---:|---:|---:|---|
| V1 target curve | 37.50% | 0.92% | 3.95% | 9.65% | V1 baseline at B=10% |
| stratified_top64 | 14.58% | 5.49% | 10.00% | 10.14% | ran before hard-floor budget fix |
| stratified_top128 | 11.46% | 5.04% | 9.94% | 10.00% | current V2 recommended point |
| stratified_pair128_e0.01_t0 | 18.75% | 5.79% | 9.97% | 10.00% | pair worsened accuracy |

Current V2 recommendation:

```text
Use stratified_top128 strict single as the V2 main point.
The main improvement comes from exposing high-dose and multi-bin actions to RF evaluation.
Relaxed pair search is implemented, but this first pair configuration is not the best route.
```

Important V2 finding:

- V1's 36.46%-37.50% plateau is not the observed action-space ceiling.
- Once diverse candidate exposure is enabled, n=96 RF accuracy at B=10% drops to 11.46%.
- Actual bandwidth rises from about 0.92% to about 5.04%, still under the 10% hard budget.
- The accepted V2 actions are mostly multi-dummy / multi-bin actions, confirming the V1 prefilter bias diagnosis.

### V2 Budget Trajectory Reuse Audit

Recorded on: 2026-07-29

Implementation anchors:

```text
scripts/stage_b_validate_b2e_budget_trajectory.py
scripts/stage_b_run_b2e_diverse_search.py
```

Run:

```text
results/stage_b2e_budget_trajectory_n96_top128
Setting: n=96, bidirectional, D=64, margin<=0, method=stratified_top128
Independent budgets: 1%, 2%, 5%, 8%, 10%
Trajectory: one Bmax=10% run, snapshotted by the latest prefix staying within each budget
```

Summary:

| B bound | independent RF acc | trajectory-prefix RF acc | independent mean bw | prefix mean bw | prediction match | action-sequence match |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | 44.79% | 83.33% | 0.76% | 0.16% | 61.46% | 16.67% |
| 2% | 40.63% | 71.88% | 1.48% | 0.62% | 67.71% | 30.21% |
| 5% | 15.63% | 52.08% | 3.14% | 2.00% | 59.38% | 50.00% |
| 8% | 10.42% | 34.38% | 4.38% | 3.36% | 75.00% | 62.50% |
| 10% | 11.46% | 11.46% | 5.04% | 5.04% | 100.00% | 100.00% |

Conclusion:

```text
Under the current V2 policy, a single B=10% trajectory cannot be treated as an exact replacement
for independently optimized 1/2/5/8% budget runs.
```

Reason:

- `dummy_budget_bound` is not used in RF scoring, but it is used before evaluation to filter candidate actions by remaining dummy budget.
- The B=10% run can accept a large first action that is illegal for lower budgets.
- In the n=96 audit, 82 samples accepted at least one dummy action in the B=10% trajectory.
- Among those 82 samples, the first dummy action already exceeded 1% in 62 samples, 2% in 44 samples, 5% in 21 samples, and 8% in 11 samples.
- Therefore, low-budget independent runs often choose different smaller actions, while the trajectory-prefix snapshot often remains delay-only or under-filled.

Practical implication:

```text
Budget-curve reuse is valid only for the Bmax endpoint in the current implementation.
For exact lower-budget curves, either keep independent budget runs or implement a new
multi-budget-aware selector that evaluates actions against all budget frontiers during search.
```

### V2 Teacher Throughput Work: V1-A Deferred Materialization

Recorded on: 2026-07-29

Implementation anchors:

```text
dmmp/stage_b/expanded_generator.py
scripts/stage_b_run_b2e_diverse_search.py
scripts/stage_b_export_teacher_trajectories.py
scripts/stage_b_audit_teacher_shard.py
```

New optional flags:

```text
--compact_candidate_generation
--deferred_materialize_oversample
```

Audit sequence:

| run | n | compact | result |
|---|---:|---:|---|
| stage_b2e_teacher_deferred_materialization_smoke_n8 | 8 | no, SparseCounts only | exact sample-level match to cached baseline, but slower |
| stage_b2e_teacher_compact_deferred_o1_smoke_n8 | 8 | yes | same accuracy success, not action-trajectory equivalent |
| stage_b2e_teacher_compact_deferred_o8_smoke_n8 | 8 | yes | same as O1; oversample did not restore old trajectory |
| stage_b2e_teacher_compact_deferred_o1_probe_n32 | 32 | yes | fast-path calibration |

Key n=8 comparison against `stage_b2e_teacher_candidate_cache_noprofile_smoke_n8`:

| variant | runtime sum | candidate generation sum | final accuracy | exact trajectory match |
|---|---:|---:|---:|---|
| cached baseline | 41.92s | 32.59s | 0.00% | reference |
| SparseCounts-only deferred | 55.23s | 44.57s | 0.00% | yes |
| compact deferred O1 | 11.58s | 6.91s | 0.00% | no |
| compact deferred O8 | 10.42s | 6.33s | 0.00% | no |

Compact O1/O8 n=8 preserved `stop_reason` and final accuracy on these samples, but differed from the baseline in:

```text
final_pred: 2 / 8
actual_dummy_bandwidth: 6 / 8
accepted_action_count: 5 / 8
records: 5 / 8
candidate_total_count: 7 / 8
```

Compact O1 n=32 calibration:

| metric | value |
|---|---:|
| clean RF accuracy | 96.88% |
| defended RF accuracy | 15.63% |
| flip rate | 84.38% |
| mean actual bandwidth | 5.41% |
| p95 actual bandwidth | 9.99% |
| max actual bandwidth | 10.00% |
| mean average delay | 5.30 bins |
| traces/hour from per-sample runtime | 3176.94 |
| candidate generation mean | 0.734s/trace |
| deferred materialization mean | 0.011s/trace |
| renderer mean | 0.319s/trace |
| RF forward mean | 0.022s/trace |
| descriptor count mean | 1835.34/trace |
| materialized action/counts mean | 168.75/trace |
| GPU max memory during probe | 3507 MB |
| GPU average utilization during probe | 3.71% |

Conclusion:

```text
Compact/deferred generation is a valid fast Teacher variant, but it is not yet an exact
drop-in replacement for the canonical stratified_top128 Teacher trajectory. Keep the
canonical path as the exact Teacher baseline; use compact/deferred for throughput
experiments and for V1-B active-state batching once its scientific delta is explicitly
reported.
```

## Teacher-Student Policy Work Area After V2

Status:

```text
Stage B2-E stratified_top128 is now positioned as Teacher/Oracle, not the final online method.
The final scalable method should be a learned candidate-scoring policy.
```

Dataset definition:

```text
CW classes: 95
Balanced policy dataset: 95 x 100 = 9500 traces
Policy train: 95 x 70 = 6650
Policy validation: 95 x 10 = 950
Policy test: 95 x 20 = 1900
```

Implementation anchors:

```text
scripts/stage_b_build_policy_dataset.py
scripts/stage_b_export_teacher_trajectories.py
scripts/stage_b_train_candidate_policy.py
scripts/stage_b_eval_candidate_policy_offline.py
dmmp/stage_b/policy_data.py
dmmp/stage_b/policy_model.py
```

Policy split audit:

```text
Run: results/stage_b_policy_dataset_9500_seed0
Source: D:\learning\TOR\datasets\CW\CW.npz
Policy total: 9500
Policy train/val/test: 6650 / 950 / 1900
```

Pilot n=96 audit:

```text
Archive: results/stage_a_rf_native_w1800_n96_s60_seed0/stage_a_masks_rf/all_masks.npz
Total samples: 96
Observed classes: 71
Strict one-per-class: false
Duplicate classes: 25
Missing classes: 24
```

Teacher export smoke test:

```text
Run: results/stage_b2e_teacher_stop_smoke_n4
Samples: 4
Teacher records: 17
Action records: 15
Stop records: 2
Method: stratified_top128, strict single, B=10%, D=64
```

Student policy smoke test:

```text
Training run: results/stage_b_candidate_policy_smoke_n4
Offline eval run: results/stage_b_candidate_policy_offline_eval_smoke_n4
Teacher records: 15
Total candidates scored offline: 1503
Candidates/sec: about 4652 on this tiny smoke run
```

Mini balanced-budget policy smoke:

```text
Teacher run: results/stage_b2e_teacher_mini_n16_balanced
Samples: 16
Budgets: balanced over 1%, 2%, 5%, 8%, 10%
Teacher records: 58
Training run: results/stage_b_candidate_policy_mini_n16
Offline eval: results/stage_b_candidate_policy_offline_eval_mini_n16
Total candidates scored offline: 5446
Oracle action recall@1/4/8/16: 29.31% / 32.76% / 39.66% / 62.07%
Mean teacher-student utility gap: 0.0186
Candidates/sec: about 11413 on this tiny offline eval
```

Regret and near-optimal audit:

```text
Run: results/stage_b_candidate_policy_offline_eval_mini_n16_regret
Teacher records: 58
Mean regret@1/4/8/16: 0.0186 / 0.0127 / 0.0102 / 0.0028
NearOptimalRecall@4, epsilon=0.005/0.01/0.02: 53.45% / 67.24% / 86.21%
NearOptimalRecall@8, epsilon=0.005/0.01/0.02: 56.90% / 72.41% / 91.38%
NearOptimalRecall@16, epsilon=0.005/0.01/0.02: 75.86% / 87.93% / 98.28%
```

Closed-loop Student controller smoke:

```text
Run: results/stage_b_student_policy_controller_smoke_n4
Samples: 4
Budget: B=10%
Methods: oracle_top128, student_only, student_top4_verify, student_top4_to8_verify
```

| method | RF acc | mean bw | mean candidate RF eval/sample | mean scored candidates/sample | note |
|---|---:|---:|---:|---:|---|
| oracle_top128 | 0.00% | 5.64% | 375.75 | 375.75 | Teacher upper-bound smoke |
| student_only | 25.00% | 6.70% | 0.00 | 298.00 | Fastest, one sample failed |
| student_top4_verify | 0.00% | 5.45% | 14.00 | 348.00 | Main verification smoke |
| student_top4_to8_verify | 0.00% | 3.84% | 14.00 | 349.25 | Adaptive path smoke |

Interpretation:

- This n=4 closed-loop result is only a functionality smoke test.
- It confirms that candidate RF queries can be reduced from hundreds per sample to about top-k-level verification while preserving the smoke-set outcome.
- Student-only should not be trusted before validation chooses `student_threshold`.
- Student top4/top8 must be evaluated on an independent validation subset after generating a larger fixed-B=10 Teacher set.

Important interpretation:

- `stratified_top128` remains the high-quality Teacher/Oracle for supervision and upper-bound analysis.
- The policy model must not consume true website labels as inputs or training targets.
- The current implemented policy receives state tensors, state summary features, action tensors, action metadata, budget, and delay condition.
- Offline recall/gap metrics are only a first policy-quality gate; final evaluation still requires running `student_only` and `student_top4_verify` controllers on the 1900-trace policy test split.
- Teacher records now include stop examples for target reached, no-positive-action, bandwidth exhaustion, and candidate-pool exhaustion when those conditions occur.

Full CW Expansion

```text
Run: results/stage_b_policy_dataset_full_cw_seed0
Mode: full_cw
CW total: 105730
Full train/val/test: 84602 / 10564 / 10564
Classes: 95
```

Full fast-keypoint archive:

```text
Run: results/stage_b_fast_keypoint_full_cw_all_seed0
Archive: results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz
Samples: 105730
Arrays: tam, mask, pred_prob, pred_labels, labels, source_indices, sample_ids
TAM/mask dtype: float16
Archive size: about 334 MB
Generation runtime: about 92.7 seconds
Clean RF accuracy over all 105730 traces: 98.28%
```

Full archive controller smoke:

```text
Run: results/stage_b_student_policy_full_archive_test_offset_smoke_n4
Archive: full_cw_all fast-keypoint archive
Offset: 95166
Samples: 4
This selects the first 4 traces of the full CW test split.
Method: student_top4_verify, B=10%
RF acc on smoke: 0.00%
Mean actual bandwidth: 5.04%
```

Important boundary:

- The project now has Stage B-compatible full-CW TAM/prob/fast-keypoint inputs.
- This is not the same as generating full-CW Teacher Oracle trajectories.
- Full Teacher trajectories remain expensive and should be generated in fixed-B=10 shards after the controller and labels are frozen.

Full CW Protocol Freeze Before Teacher Expansion

Split audit:

```text
Run: results/stage_b_full_archive_split_audit_seed0
Archive: results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz
Split file: results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz
```

| split | samples | RF clean acc | mean confidence | archive rows |
|---|---:|---:|---:|---|
| train | 84602 | 98.41% | 93.81% | 0-84601 |
| val | 10564 | 97.85% | 92.71% | 84602-95165 |
| test | 10564 | 97.66% | 92.92% | 95166-105729 |

Important correction:

- The earlier "98.28%" is the clean RF accuracy over all 105730 CW traces.
- The held-out test clean baseline for final reporting is 97.66%.

Split-aware archive access:

```text
Student controller smoke:
results/stage_b_student_policy_full_archive_split_test_smoke_n4

Teacher split smoke:
results/stage_b2e_teacher_split_test_sparse_smoke_n4
```

Both controller and Teacher export now support:

```text
--split_file results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz
--split_name train|val|test|all
```

The old hard-coded test offset `95166` should no longer be used in protocol commands. In the split smoke, `--split_name test --max_samples 4` selected archive rows `95166-95169` through explicit `source_indices` mapping.

Teacher sharding:

```text
Train samples: 84602
Command shape: --split_name train --num_shards 256 --shard_id k --max_samples -1
Shard sizes: 330-331 traces/shard
Smoke: results/stage_b2e_teacher_train_shard000_smoke_n1
Resume smoke: results/stage_b2e_teacher_resume_smoke_n1
```

Teacher export now supports `--resume`. Sample summaries are flushed after each completed trace, and resumed runs skip completed archive rows. In the resume smoke, the first run wrote 9 records for 1 trace; the second run reported `new_records=0`, `new_samples=0`, and `skipped_samples=1`.

Sparse Teacher storage:

```text
Run: results/stage_b2e_teacher_sparse_smoke_n4
Samples: 4
Records: 13
Mean record size: about 13.8 KB/state
Dense action_counts stored: no
Sparse action encoding: action index, direction, bin, count
```

The policy training loader reconstructs dense action tensors from sparse records at batch time. `current_prob` and `original_prob` are stored for Teacher-decision auditing, but the training manifest now explicitly records that policy inputs are only:

```text
state_tensor
state_features
action_counts
action_features
candidate_mask
```

Training targets are:

```text
candidate_gains
selected_index
stop_target
```

Labels and identifiers are traceability/evaluation-only fields and must not be consumed by the Student policy.

Fast-keypoint vs DynaMask audit:

```text
Run: results/stage_b_fast_vs_dynamask_audit_n96_seed0
DynaMask archive: results/stage_a_rf_native_w1800_n96_s60_seed0/stage_a_masks_rf/all_masks.npz
Aligned fast archive: results/stage_b_fast_vs_dynamask_audit_n96_seed0/aligned_fast_keypoint_archive.npz
Matched samples: 96
```

Top-location overlap is moderate globally but low for very small top ratios:

| top ratio | Jaccard | DynaMask-in-Fast recall | cosine | pearson |
|---:|---:|---:|---:|---:|
| 1% | 1.82% | 3.36% | 54.24% | 46.33% |
| 2% | 4.92% | 8.78% | 54.24% | 46.33% |
| 5% | 14.84% | 24.48% | 54.24% | 46.33% |
| 10% | 30.82% | 45.12% | 54.24% | 46.33% |

Deletion intervention remains strong:

| mask | delete 1% | delete 2% | delete 5% | delete 10% |
|---|---:|---:|---:|---:|
| DynaMask | 86.46% | 78.13% | 43.75% | 11.46% |
| Fast-keypoint | 91.67% | 73.96% | 32.29% | 15.63% |

Interpretation:

- Fast-keypoint does not merely copy DynaMask's exact top-k pixels.
- It is still causally useful: at 2% and 5% deletion it is stronger than DynaMask on the aligned n=96 audit, while DynaMask is stronger at 10%.
- This supports using Fast-keypoint for scalable full-CW Teacher input, with DynaMask retained as the expensive Oracle reference.

Oracle top128 comparison on aligned n=16:

| archive | RF acc | flip | mean bw | p95 bw | mean candidate RF eval/sample |
|---|---:|---:|---:|---:|---:|
| DynaMask | 0.00% | 100.00% | 5.96% | 9.93% | 359.63 |
| Aligned Fast-keypoint | 0.00% | 100.00% | 5.91% | 9.93% | 359.13 |

Prediction equality between the two runs was 93.75%; accuracy equality was 100.00%. This is a small comparison, but it is a useful sanity check that the scalable keypoint archive has not broken the Teacher action space.

Frozen V2 full-CW Teacher protocol:

```text
Teacher/Oracle:
  split = train only
  B = fixed 10%
  method = stratified_top128
  protocol = bidirectional_cooperative
  max_delay = 64
  margin_target = 0
  storage_mode = sparse
  shards = 256

Validation:
  use val split for policy selection and threshold/top-k tuning

Final test:
  use test split only after policy and protocol are frozen
```

First full-shard command shape:

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
  --storage_mode sparse `
  --resume `
  --run_name stage_b2e_teacher_full_cw_train_shard000_of256
```

Calibration Shard 0 Result

```text
Run: results/stage_b2e_teacher_calibration_train_shard000_of256
Audit: results/stage_b2e_teacher_calibration_train_shard000_of256/teacher_shard_audit.json
Split: train
Shard: 0 / 256
Samples: 331
Budget: fixed B=10%
Method: stratified_top128
Storage: sparse
```

Completeness:

| metric | value |
|---|---:|
| expected samples | 331 |
| completed samples | 331 |
| failed samples | 0 |
| duplicate sample ids | 0 |
| missing sample ids | 0 |
| bandwidth violations | 0 |

Runtime:

| metric | value |
|---|---:|
| total wall time | 2965.53 sec / 49.43 min |
| mean sec/trace | 8.96 |
| median sec/trace | 6.56 |
| p90 sec/trace | 20.32 |
| p95 sec/trace | 25.83 |
| max sec/trace | 53.21 |

Timing totals:

| component | total sec | mean sec/trace |
|---|---:|---:|
| candidate generation | 2406.65 | 7.27 |
| candidate prefilter | 140.13 | 0.42 |
| renderer | 337.05 | 1.02 |
| TAM rebuild | 24.05 | 0.07 |
| RF forward | 22.68 | 0.07 |
| serialization | 3.70 | 0.01 |
| keypoint refresh | 3.23 | 0.01 |
| delay trace | 3.10 | 0.01 |
| residual/wait | 24.63 | 0.07 |

Teacher data scale:

| metric | value |
|---|---:|
| Teacher states | 1560 |
| states / trace mean | 4.71 |
| states / trace p95 | 12 |
| candidates / state mean | 82.99 |
| candidates / trace mean | 424.83 |
| candidates / trace p95 | 1135 |
| action records | 1367 |
| stop records | 193 |
| shard record storage | 21.43 MB |
| projected full train storage at this rate | about 5.49 GB |
| projected full train states at this rate | about 399k |

Teacher behavior:

| metric | value |
|---|---:|
| clean RF acc on shard | 98.49% |
| defended RF acc | 9.06% |
| flip rate | 90.94% |
| margin success rate | 90.94% |
| mean actual bandwidth | 5.21% |
| p95 actual bandwidth | 9.97% |
| max actual bandwidth | 10.00% |
| mean average delay | 5.22 bins |
| mean p95 delay | 14.04 bins |
| p95 of p95 delay | 18.00 bins |
| max delay | 21 bins |
| mean action rounds | 4.26 |
| candidate RF eval / sample | 424.83 |
| candidate positive-gain rate | 72.40% |

Stop reasons:

| stop reason | count | defended acc within group |
|---|---:|---:|
| target_reached | 301 | 0.00% |
| bandwidth_10pct_reached | 22 | 100.00% |
| no_positive_single | 7 | 100.00% |
| max_actions_reached | 1 | 100.00% |

Interpretation:

- Calibration shard 0 passes the mechanical checks: no missing samples, no duplicates, no failed samples, no bandwidth violation, and resume-compatible files were produced.
- Storage is not the bottleneck under sparse records. The shard projects to only about 5.5 GB of train Teacher records if other shards behave similarly.
- Runtime is dominated by candidate generation and prefiltering, not RF GPU forward. This supports optimizing CPU candidate generation/rendering before rewriting the RF forward path.
- The remaining 30 correct-after-defense samples are all boundary cases: bandwidth exhausted, no positive action, or max action rounds. They should be inspected as action-space/controllability limits, not random failures.
- The single-shard time projection is about 8.8 days for all 256 shards on one process, but this must not be treated as final throughput until middle and late calibration shards are run.

GPU Memory Parallelism Probe

```text
Probe 1: results/stage_b_parallel_worker_probe_target8gb_s4
Probe 2: results/stage_b_parallel_worker_probe_target8gb_s2_p3
Target GPU memory: 8 GB
Parallelism tested: independent Teacher shard worker processes
```

| workers | samples total | wall sec | traces/hour | max GPU memory | mean GPU util | result |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 4 | 26.13 | 551.08 | 2.93 GB | 8.08% | below target |
| 2 | 8 | 61.63 | 467.32 | 3.15 GB | 8.10% | below target and slower |
| 3 | 6 | 22.00 | 981.88 | 3.17 GB | 5.55% | below target; tiny sample, not comparable |

Interpretation:

- Independent shard processes do not make GPU memory scale toward the 8 GB target.
- The GPU is still mostly waiting because each process spends most of its time in Python candidate generation before short RF forward bursts.
- More independent CUDA processes would also duplicate host RAM and model state, so this is not the right path to saturate the RTX 5070.
- The next parallelism design should be single-process `active_states` batching: generate candidates for several active Teacher states, render/build candidate TAMs per state, then concatenate candidate TAMs into one larger RF batch.
- The target should be `active_states=4` first, then `8`, while monitoring peak GPU memory, RF batch size, and end-to-end traces/hour.

Candidate-generation profile after cache fields:

```text
Run: results/stage_b2e_teacher_candidate_cache_noprofile_smoke_n8
Detailed profile: results/stage_b2e_teacher_candidate_profile_detail_smoke_n2
```

The detailed profile shows that `action_object_build_time_sec` dominates candidate generation. This confirms that the current bottleneck is repeated dense `counts` materialization and Python `ExpandedAction` object creation, not window extraction or RF inference.

Candidate Batch / Active-State GPU Work

Recorded on: 2026-07-29

Implementation:

```text
dmmp/stage_b/expanded_generator.py
scripts/stage_b_probe_active_state_gpu_batch.py
scripts/stage_b_run_b2e_diverse_search.py
scripts/stage_b_export_teacher_trajectories.py
scripts/stage_b_audit_teacher_shard.py
```

Changes:

- `candidate_batch_size` now reaches candidate generation in the compact path.
- Compact candidate generation has a tensorized descriptor-table route: candidate parameters are assembled as arrays, cost/score filtering is computed in batches on `candidate_device`, and only selected descriptors are materialized.
- GPU TAM candidate evaluation remains available through `--candidate_eval_mode gpu_tam`.
- `scripts/stage_b_probe_active_state_gpu_batch.py` tests single-process active-state batching by concatenating candidates from multiple active traces into shared RF batches.
- Internal active-state GPU peak accounting was changed from accumulated peak to true max-style tracking; external `nvidia_smi_poll.csv` remains the main hardware audit source.

Formal Teacher exporter comparison on aligned `n=128`, train shard 3:

| run | candidate path | wall sec | traces/hour | defended RF acc | mean BW | candidate gen sec | descriptor gen sec | RF forward sec |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| stage_b2e_teacher_gpu_tam_batch2048_nodiag_probe_n128 | compact Python descriptors + GPU TAM eval | 99.72 | 4622.93 | 5.47% | 5.44% | 92.20 | 88.86 | 3.10 |
| stage_b2e_teacher_tensorized_candidate_gpu_tam_probe_n128_175410 | tensorized descriptor table + GPU TAM eval | 31.56 | 14607.44 | 6.25% | 5.66% | 21.39 | 17.41 | 3.88 |

Active-state throughput probe on aligned `n=128`, train shard 3:

| run | active_states | candidate_batch_size | wall sec | traces/hour | max GPU memory | mean GPU memory | mean GPU util | defended RF acc |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage_b_active_state_gpu_batch_a4_b2048_probe_n128_172446 | 4 | 2048 | 136.81 | 3368.23 | 4.02 GB | 3.73 GB | 4.26% | 12.50% |
| stage_b_active_state_gpu_batch_a8_b2048_probe_n128_173202 | 8 | 2048 | 139.75 | 3297.31 | 5.10 GB | 4.85 GB | 5.23% | 12.50% |
| stage_b_active_state_gpu_batch_a16_b4096_probe_n128_173525 | 16 | 4096 | 140.13 | 3288.36 | 7.42 GB | 6.89 GB | 8.46% | 12.50% |
| stage_b_active_state_gpu_batch_a24_b4096_probe_n128_173829 | 24 | 4096 | 139.68 | 3299.00 | 9.69 GB | 8.77 GB | 10.01% | 12.50% |
| stage_b_active_state_tensorized_candidate_a4_b2048_probe_n128_175106 | 4 | 2048 | 38.80 | 11875.03 | 7.08 GB | 5.74 GB | 40.13% | 14.84% |
| stage_b_active_state_tensorized_candidate_a8_b4096_probe_n128_175223 | 8 | 4096 | 38.35 | 12014.11 | 9.64 GB | 7.62 GB | 45.04% | 14.84% |
| stage_b_active_state_tensorized_candidate_a12_b8192_probe_n128_180539 | 12 | 8192 | 59.22 | 7780.91 | 7.20 GB | 6.51 GB | 31.00% | 13.28% |
| stage_b_active_state_tensorized_candidate_a16_b8192_probe_n128_180724 | 16 | 8192 | 58.41 | 7889.49 | 8.21 GB | 7.43 GB | 37.32% | 13.28% |
| stage_b_active_state_tensorized_candidate_top512_a8_b8192_probe_n128_180907 | 8 | 8192 | 70.97 | 6492.56 | 8.21 GB | 6.52 GB | 33.05% | 10.94% |
| stage_b_active_state_tensorized_candidate_a32_b8192_probe_n128_181100 | 32 | 8192 | 61.74 | 7463.54 | 11.50 GB | 9.92 GB | 35.37% | 13.28% |

Interpretation:

- Blindly increasing active states before tensorizing candidate generation fills more VRAM but does not improve total throughput, because the GPU still waits for Python candidate generation.
- After tensorized candidate generation, the n=128 formal Teacher exporter improves from 99.72s to 31.56s, about 3.16x faster, while preserving the main RF degradation behavior.
- The practical hardware point for this RTX 5070 is currently `active_states=8`, `candidate_batch_size=4096`: it crosses the requested 8 GB peak memory target and gives the best observed active-state probe throughput.
- Pushing toward the 12 GB card limit is possible: `active_states=32`, `candidate_batch_size=8192` reached 11.50 GB peak. It is not the throughput optimum, and running closer to the 12.0 GB ceiling risks OOM/driver instability.
- The active-state probe is a throughput experiment, not yet the canonical Teacher exporter. Canonical full-shard generation should use the formal exporter until active-state scheduling is integrated with sparse record writing and resume.

### Balanced 950 8:1:1 Teacher-Student Run

Recorded on: 2026-07-29

Dataset:

```text
Run: results/stage_b_policy_dataset_950_seed0_8_1_1
Source archive: results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz
Classes: 95
Total: 950 traces = 95 x 10
Train/val/test: 760 / 95 / 95 = 8 / 1 / 1 per class
Split file: results/stage_b_policy_dataset_950_seed0_8_1_1/policy_splits.npz
```

Formal Teacher generation:

```text
Method: stratified_top128
Protocol: bidirectional_cooperative
Budget: fixed B=10%
Candidate path: compact tensorized descriptors + gpu_tam candidate evaluation
Storage: sparse
```

| split | traces | records | clean RF acc | defended RF acc | flip | mean BW | p95 BW | mean delay | candidate RF eval/trace | wall sec |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 760 | 2209 | 98.42% | 10.66% | 89.74% | 6.17% | 9.98% | 5.15 bins | 171.76 | 168.57 |
| val | 95 | 258 | 96.84% | 8.42% | 91.58% | 5.67% | 9.99% | 5.19 bins | 159.15 | 23.57 |
| test | 95 | 225 | 96.84% | 4.21% | 95.79% | 6.17% | 9.99% | 5.03 bins | 150.03 | 21.77 |

Merged 950 Teacher defense summary:

```text
Run: results/stage_b2e_teacher_950_8_1_1_aggregate
Summary JSON: results/stage_b2e_teacher_950_8_1_1_aggregate/teacher_950_aggregate_summary.json
Summary CSV: results/stage_b2e_teacher_950_8_1_1_aggregate/teacher_950_aggregate_summary.csv
```

| samples | clean RF acc | defended RF acc | flip | mean actual BW | p95 BW | max BW | global dummy overhead | total dummy packets | mean delay |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 950 | 98.11% | 9.79% | 90.53% | 6.12% | 9.98% | 10.00% | 5.27% | 91913 | 5.14 bins |

Completeness:

```text
Train/val/test all completed with:
failed_samples = 0
duplicate_sample_ids = 0
missing_sample_ids = 0
bandwidth_violation_count = 0
```

Student policy training:

```text
Run: results/stage_b_candidate_policy_950_8_1_1_seed0
Train records: 2209
Validation records: 258
Epochs: 10
Best checkpoint: results/stage_b_candidate_policy_950_8_1_1_seed0/best_policy.pt
```

Best validation metrics:

```text
val_choice_acc = 27.25%
val_recall_at_4 = 52.42%
val_utility_gap = 0.07537
```

Offline policy evaluation on test Teacher records:

```text
Run: results/stage_b_candidate_policy_950_8_1_1_seed0_offline_test
Evaluated action records: 198
Total candidates: 14253
Candidates/sec: 7479.34
```

| metric | top1 | top4 | top8 | top16 |
|---|---:|---:|---:|---:|
| Oracle action recall | 25.25% | 56.57% | 67.68% | 85.86% |
| Mean regret | 0.0594 | 0.0182 | 0.0105 | 0.0033 |
| Near-optimal recall, eps=0.01 | 41.92% | 75.25% | 85.35% | 95.45% |

Closed-loop Student test:

```text
Run: results/stage_b_student_policy_950_8_1_1_seed0_closed_loop_test_tensorized
Split: test
Samples: 95
Budget: fixed B=10%
```

| method | RF acc | flip | mean BW | p95 BW | mean delay | candidate RF eval/trace | scored candidates/trace | mean runtime/trace |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| student_only | 15.79% | 84.21% | 6.75% | 9.98% | 5.09 bins | 0.00 | 170.12 | 0.226s |
| student_top4_verify | 8.42% | 91.58% | 5.90% | 9.97% | 5.06 bins | 6.32 | 125.55 | 0.188s |
| student_top4_to8_verify | 8.42% | 91.58% | 5.99% | 9.97% | 5.06 bins | 6.61 | 127.02 | 0.184s |

Interpretation:

- The 950 split confirms the Teacher still produces a strong upper bound on unseen test traces: 96.84% clean RF accuracy drops to 4.21% under the formal Teacher.
- The trained Student is weaker than the Teacher but preserves much of the effect. `student_top4_verify` reaches 8.42% test RF accuracy while reducing candidate RF evaluations from the Teacher test average of 150.03 per trace to 6.32 per trace.
- `student_top4_to8_verify` did not improve over top4 in this run.
- `student_only` removes candidate RF verification entirely, but its closed-loop accuracy is worse at 15.79%.

Active-state request audit:

```text
Run: results/stage_b_active_state16_tensorized_probe_950_all_seed0
Split: all 950 traces
active_states = 16
candidate_batch_size = 8192
Wall time: 364.29 sec
Throughput: 9388.20 traces/hour
Internal candidate GPU peak allocated: 2645.70 MB
Defended RF accuracy in probe controller: 21.58%
Mean dummy bandwidth in probe controller: 7.55%
```

Important boundary:

- The active-state script is still a throughput probe, not the canonical sparse Teacher exporter.
- The formal 8:1:1 Teacher/Student numbers above are the scientific result for this 950 experiment.
- The active-state=16 probe shows that batching 950 traces works, but its internal CUDA allocation is only about 2.65 GB on this run, so it still does not approach an 8 GB or 12 GB VRAM target.
- Because the probe controller is simplified, its 21.58% defended accuracy should not be compared directly against the formal Teacher's 4.21% test accuracy.
