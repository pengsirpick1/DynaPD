# Stage B Vectorized Candidate Selection

Date: 2026-07-31

## Change

Optimized the compact descriptor-table selection hotspot:

```text
dynapd/stage_b/expanded_generator.py
_select_descriptor_indices
```

The old path repeatedly used Python `sorted`, list comprehensions, tuple identity construction, and callable `cost / estimated_gain / composite` calculations.

The new path keeps the same selection policy but precomputes arrays:

```text
cost
estimated_gain
identity_key
composite
```

It then uses NumPy `lexsort` / `unique` / boolean masks for:

```text
score_hint ordering
identity dedupe
dose buckets
action-type buckets
multi-bin bucket
two-window bucket
rest composite ordering
```

Bucket insertion still preserves the original greedy selected-order semantics.

## Validation

Synthetic descriptor-table equivalence:

```text
descriptor selection equivalence ok
```

Real smoke comparison against previous 950 test Teacher first 4 samples:

```text
accuracy, final_pred, actual_dummy_bandwidth, average_delay_bins,
accepted_action_count, candidate_total_count, stop_reason all matched.
```

## Teacher Export Results

Test split, 95 traces:

```text
Run: results/stage_b2e_teacher_vector_select_test_n95
Old reference: results/stage_b2e_teacher_950_test_tensorized_seed0_a16req
```

| metric | old | vectorized |
|---|---:|---:|
| wall time | 21.77 s | 8.98 s |
| defended RF acc | 4.21% | 4.21% |
| mean BW | 6.17% | 6.17% |
| mean delay | 5.03 bins | 5.03 bins |
| candidate eval / sample | 150.03 | 150.03 |

Train split, 760 traces:

```text
Run: results/stage_b2e_teacher_vector_select_train_950_8_1_1
```

| metric | old | vectorized |
|---|---:|---:|
| wall time | 168.57 s | 81.57 s |
| records | 2209 | 2209 |
| defended RF acc | 10.66% | 10.66% |
| mean BW | 6.17% | 6.17% |

Val split, 95 traces:

```text
Run: results/stage_b2e_teacher_vector_select_val_950_8_1_1
```

| metric | old | vectorized |
|---|---:|---:|
| wall time | 23.57 s | 9.79 s |
| records | 258 | 258 |
| defended RF acc | 8.42% | 8.42% |
| mean BW | 5.67% | 5.67% |

## Student Results

Existing policy checkpoint under vectorized selection:

```text
Run: results/stage_b_student_policy_vector_select_top4_verify_test_n95
Policy: results/stage_b_candidate_policy_950_8_1_1_seed0/best_policy.pt
```

| metric | old | vectorized |
|---|---:|---:|
| RF acc | 8.42% | 8.42% |
| mean BW | 5.90% | 5.90% |
| candidate RF eval / sample | 6.32 | 6.32 |
| mean runtime / trace | 0.188 s | 0.083 s |

Retrained policy from vectorized Teacher records:

```text
Training: results/stage_b_candidate_policy_vector_select_950_8_1_1_seed0
Offline test: results/stage_b_candidate_policy_vector_select_950_8_1_1_seed0_offline_test
Closed-loop: results/stage_b_student_policy_vector_select_retrained_top4_verify_test_n95
```

Offline test:

```text
oracle recall@1/4/8/16 = 23.23% / 54.55% / 71.72% / 85.86%
mean regret@1/4/8/16 = 0.0667 / 0.0202 / 0.0085 / 0.0022
```

Closed-loop test:

```text
student_top4_verify RF acc = 9.47%
mean BW = 5.97%
candidate RF eval / sample = 6.40
mean runtime / trace = 0.088 s
```

Interpretation: the optimized selection preserves the Teacher and the existing trained Student result. Retraining on the same 950-scale records gives a nearby but slightly weaker closed-loop result, consistent with small-data policy training variance.

## Next Step

The next full-CW scaling step should use this vectorized selection as the default baseline before integrating active-state batching into the formal sparse Teacher exporter.
