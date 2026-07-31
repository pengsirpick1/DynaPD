# Stage B Cluster Prototype Transfer Audit

This audit tests whether a real train-split cluster medoid can run the formal Teacher once and transfer its action trajectory to other traces.

## Run

- Output dir: `results\stage_b_cluster_prototype_transfer_audit_20260730\fullcw_k300_test_absolute_budget_normalized_D64`
- Archive: `results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz`
- Split file: `results/stage_b_policy_dataset_full_cw_seed0/policy_splits.npz`
- Transfer modes: `absolute_budget_normalized_replay`

## Prototype-Only Teacher Results

| representation | K | prototypes | represented | weighted clean acc | weighted defended acc | weighted BW | weighted delay | weighted actions | runtime |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_flat_tam | 300 | 300 | 105730 | 99.73% | 0.06% | 1.93% | 6.76 | 2.13 | 81.6s |

## Aggregate Results

| representation | K | mode | split | n | clean acc | defended acc | mean BW | p95 BW | mean delay | eval/sample | traces/hour |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normalized_flat_tam | 300 | absolute_budget_normalized_replay | test | 10564 | 97.66% | 47.39% | 1.54% | 5.85% | 4.93 | 3.93 | 95084.7 |

## Interpretation Checklist

- `absolute_replay` checks whether absolute prototype bins are reusable. A weak result means absolute timing is too instance-specific.
- `relative_replay` checks whether keypoint-relative mapping is enough without per-member RF verification.
- `relative_top4_verify` checks whether a tiny exact verification set can approach the per-sample Teacher while cutting candidate evaluations.
- The distance/effect correlations in `audit.json` indicate whether cluster compactness predicts transfer quality.
