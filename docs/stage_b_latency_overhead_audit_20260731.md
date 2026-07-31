# Stage B Last-Real Latency Overhead Audit

Date: 2026-07-31

Purpose: report the actual page-load-style latency overhead for the current best Stage B method, using last original real packet completion time rather than packet-level average delay bins.

## Metric

For each trace:

```text
completion_delay = max_t_original_real_after_defense - max_t_original_real_clean
latency_overhead = max(completion_delay, 0) / max_t_original_real_clean
```

Trailing dummy packets are ignored. The audit also tracks every original real packet by id, so model input cropping at `max_trace_length=5000` is not treated as real packet deletion for completion-time accounting.

Bin conversion:

```text
rf_num_slots = 1800
max_load_time = 80.0 sec
bin_width = 80 / 1800 = 0.044444 sec
```

## Run

Script:

```text
scripts/stage_b_latency_overhead_audit.py
```

Command:

```powershell
D:\Miniconda3\envs\llm\python.exe scripts\stage_b_latency_overhead_audit.py `
  --methods teacher,student_top4_verify `
  --run_name stage_b_latency_overhead_audit_950_test_teacher_student_top4_kernel_20260731
```

Output:

```text
results/stage_b_latency_overhead_audit_950_test_teacher_student_top4_kernel_20260731
```

Inputs:

```text
archive = results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz
split_file = results/stage_b_policy_dataset_950_seed0_8_1_1/policy_splits.npz
split = test
samples = 95
budget = B=10%
max_delay = D64
rounds = 3
protocol = bidirectional_cooperative
```

## Results

| method | RF acc | mean BW | mean completion delay | median | p95 | max | macro latency OH | micro latency OH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher stratified_top128 | 4.21% | 6.17% | 0.0706 s | 0.0156 s | 0.4202 s | 0.9021 s | 0.5845% | 0.2300% |
| Student top4 verify | 8.42% | 5.90% | 0.0664 s | 0.0117 s | 0.4198 s | 0.9021 s | 0.5529% | 0.2164% |

Equivalent mean completion-delay bins:

```text
Teacher: 1.59 bins
Student top4 verify: 1.49 bins
```

## Packet-Level Delay Diagnostics

| method | existing mean delayed-event stat | mean cumulative bins over all real packets | mean cumulative bins over delayed real packets | delayed real packet ratio | p95-of-sample-p95 cumulative bins | max cumulative real-packet bins |
|---|---:|---:|---:|---:|---:|---:|
| Teacher stratified_top128 | 5.03 | 3.98 | 5.35 | 68.67% | 18.0 | 38.0 |
| Student top4 verify | 5.06 | 3.69 | 5.06 | 68.25% | 18.0 | 21.0 |

Important: the Teacher run has 4 / 95 samples where at least one original real packet exceeded 21 cumulative bins. This confirms that the previous `maximum_delay_bins = 21` state statistic is a per-event or per-round-style maximum, not a full original-real-packet cumulative maximum.

## Interpretation

The current best Teacher result has strong RF degradation with very small page-load-style latency overhead on this offline trace metric:

```text
clean RF acc 96.84% -> defended RF acc 4.21%
mean completion delay about 70 ms
micro latency overhead about 0.23%
macro latency overhead about 0.58%
```

The scalable Student `top4_verify` result is close:

```text
defended RF acc 8.42%
mean completion delay about 66 ms
micro latency overhead about 0.22%
macro latency overhead about 0.55%
```

This should be reported separately from packet-level delay bins. The page-load metric is governed only by the last original real packet, while many delayed packets are in the middle of the trace and do not move completion time.
