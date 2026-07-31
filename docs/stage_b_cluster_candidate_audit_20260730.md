# Stage B Cluster-Aware Candidate Audit, 2026-07-30

Purpose:

```text
Check whether similar traffic can support Cluster-Aware Batched Candidate Retrieval.
```

The audit is deliberately non-invasive. It does not change Teacher or Student behavior.

Implementation:

```text
scripts/stage_b_cluster_candidate_audit.py
```

Main run:

```text
results/stage_b_cluster_candidate_audit_950_k32_seed0
```

Additional sensitivity runs:

```text
results/stage_b_cluster_candidate_audit_950_k16_t16_seed0
results/stage_b_cluster_candidate_audit_950_k16_t8_seed0
results/stage_b_cluster_candidate_audit_950_k8_t8_seed0
```

Comparison table:

```text
results/stage_b_cluster_candidate_audit_comparison_950.csv
```

Dataset:

```text
Archive: results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz
Split: results/stage_b_policy_dataset_950_seed0_8_1_1/policy_splits.npz
Samples: 950
Train/val/test: 760 / 95 / 95
Teacher records:
  results/stage_b2e_teacher_950_train_tensorized_seed0_a16req/teacher_records.csv
  results/stage_b2e_teacher_950_val_tensorized_seed0_a16req/teacher_records.csv
  results/stage_b2e_teacher_950_test_tensorized_seed0_a16req/teacher_records.csv
```

Clustering:

```text
Fit split: train only
Assignment: val/test assigned to train-fitted centers
Labels: not used for clustering, only written as eval-only audit columns
Features: TAM structure + fast-keypoint temporal summaries
```

## Keypoint Similarity

K=32, time signature bins=64:

| pair group | Top-1% Jaccard | Top-2% Jaccard | Top-5% Jaccard | Top-10% Jaccard | Pearson |
|---|---:|---:|---:|---:|---:|
| within cluster | 5.47% | 8.97% | 18.05% | 33.72% | 39.95% |
| different cluster | 3.37% | 5.52% | 11.32% | 21.36% | 26.08% |

Interpretation:

- Clustering does group flows with more similar keypoint maps.
- The lift is meaningful but moderate: about 1.59x for Top-5% keypoint Jaccard and 1.53x for Pearson.

## First Teacher Action Similarity

K=32, time signature bins=64:

| pair group | exact first-action signature agreement | action-type agreement |
|---|---:|---:|
| within cluster | 0.091% | 19.01% |
| different cluster | 0.006% | 16.59% |

Sensitivity:

| run | keypoint Jaccard@5 lift | Pearson lift | action-type lift | exact within agreement |
|---|---:|---:|---:|---:|
| k32_t64 | 1.59x | 1.53x | 1.15x | 0.091% |
| k16_t16 | 1.48x | 1.42x | 1.09x | 0.330% |
| k16_t8 | 1.48x | 1.42x | 1.09x | 0.347% |
| k8_t8 | 1.23x | 1.28x | 1.10x | 0.089% |

Interpretation:

- Same-cluster samples are more likely to share broad action type, but only slightly.
- Exact or coarse first-action signature agreement remains extremely low.
- This confirms that RF decision-boundary effects are strongly instance-level.

## Cluster Template Coverage

Templates are built from train Teacher first-action candidate records only.
Validation and test use assigned train-fitted clusters.

Top128 cluster template vs global template:

| run | split | scope | oracle coverage@128 | near-optimal@0.02 | mean regret | shared candidates/sample |
|---|---|---|---:|---:|---:|---:|
| k32_t64 | val | cluster | 2.41% | 8.43% | 0.3830 | 0.28 |
| k32_t64 | val | global | 1.20% | 6.02% | 0.4115 | 0.02 |
| k32_t64 | test | cluster | 1.19% | 3.57% | 0.4225 | 0.21 |
| k32_t64 | test | global | 0.00% | 1.19% | 0.4537 | 0.00 |
| k16_t16 | val | cluster | 7.23% | 13.25% | 0.3779 | 0.49 |
| k16_t16 | val | global | 0.00% | 6.02% | 0.4111 | 0.10 |
| k16_t16 | test | cluster | 3.57% | 4.76% | 0.4359 | 0.17 |
| k16_t16 | test | global | 1.19% | 2.38% | 0.4435 | 0.04 |
| k16_t8 | val | cluster | 7.23% | 13.25% | 0.3357 | 0.58 |
| k16_t8 | val | global | 2.41% | 8.43% | 0.4072 | 0.11 |
| k16_t8 | test | cluster | 1.19% | 3.57% | 0.4299 | 0.23 |
| k16_t8 | test | global | 1.19% | 2.38% | 0.4473 | 0.01 |
| k8_t8 | val | cluster | 4.82% | 10.84% | 0.4003 | 0.27 |
| k8_t8 | val | global | 2.41% | 8.43% | 0.4072 | 0.11 |
| k8_t8 | test | cluster | 1.19% | 2.38% | 0.4452 | 0.42 |
| k8_t8 | test | global | 1.19% | 2.38% | 0.4473 | 0.01 |

Interpretation:

- Cluster templates are consistently better than global templates, but the absolute coverage is too low.
- The best validation setting reaches 7.23% oracle signature coverage@128, but test coverage remains only 1.19%-3.57%.
- This is not sufficient for replacing per-sample candidate generation with a cluster template library.

## Decision

Current evidence supports:

```text
Cluster Batch V1: use clustering for scheduling and batch regularity.
```

Current evidence does not yet support:

```text
Cluster Batch V2/V3 as a direct shared-template candidate replacement.
```

Reason:

- Keypoint similarity improves within clusters.
- But Teacher first-action agreement and cluster-template coverage remain weak.
- The shared templates can provide a weak prior, not a substitute for per-sample candidate generation.

Recommended next implementation:

```text
Cluster-aware scheduling only:
  fit train-only clusters
  assign val/test by train centers
  process active states grouped by cluster
  keep per-sample candidate generation and exact Teacher selection unchanged
```

This should preserve defense results and may improve batch regularity.

Template reuse should wait until candidate descriptors are redesigned around relative templates rather than current action-feature signatures.
