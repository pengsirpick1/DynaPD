**English** | [中文](README_CN.md)

# DynaPD

Dynamic Padding Defense against Website Fingerprinting with
**Multi-Surrogate Teacher Optimization** and **Stochastic Completion**.

## Overview

DynaPD generates defended traffic traces by iteratively injecting dummy
packets and redistributing delays at RF-identified keypoints. Candidate
actions are evaluated against a panel of surrogate attack models and
selected via weighted utility scoring.

### Key Innovations

| Component | Description |
|------|-------------|
| **Multi-Surrogate Teacher** | RF (TAM/burst), DF (DIR CNN), AWF (burst CNN) provide three distinct attack perspectives for robustness. |
| **Normalized Gain** | `gain_m = (current_margin - candidate_margin) / abs(original_margin)` makes cross-model gain comparable. |
| **Stochastic Action Selection** | ε-greedy randomization (ε=0.5) ensures different defended traces for the same clean sample, resisting adversarial training. |
| **Bandwidth-Aware Completion** | Two-phase: flip all models first, then fill remaining bandwidth budget (`preserve_flip_fill`). |
| **Representation-Family Coverage** | Surrogates chosen to cover DIR, burst, and timing feature families rather than model names. |

### Design Philosophy

```
Clean trace
    ↓
RF keypoints → stratified Top128 candidates (5D cross-product)
    ↓
3-model evaluation (RF / DF / AWF)
    ↓
Weighted score = 0.8×gain_rf + 0.1×gain_df + 0.1×gain_awf - λ×cost
    ↓
ε-greedy select → apply action → loop until flipped or budget exhausted
    ↓
Completion: fill remaining bandwidth with random valid actions
```

## Current Best Results

Completion b30 ε=0.5, Full CW Test (10564 samples), Server T640g0 RTX 4090

| Model | Role | Clean Acc | Defended Acc | Flip |
|------|------|------|------|------|
| RF | Teacher | 97.66% | **13.37%** | 86.59% |
| DF | Teacher | 98.25% | 9.94% | 89.91% |
| AWF | Teacher | 94.86% | 4.95% | 95.05% |
| TF | Held-out | 96.16% | **10.20%** | 89.80% |
| **WC** | — | — | **13.37%** | — |

All 3 surrogate flipped: 78.75% | Mean BW: 24.50% | Mean actions: 12.09 | Mean delay: 5.35 bins

### Overhead

| Metric | Value |
|------|------|
| Mean BW | 24.50% |
| Median BW | 29.99% |
| P90 BW | 30.03% |
| P95 BW | 30.06% |
| Max BW | 30.77% |
| Mean Actions | 12.09 |
| Mean Delay | 5.35 bins |
| Stop: all_flipped | 78.75% (8319/10564) |
| Stop: no_robust | 21.13% (2232/10564) |
| Stop: bw_reached | 0.12% (13/10564) |

### Evolution

| Version | Surrogates | Budget | Random ε | Completion | WC | TF held-out |
|------|------|------|------|------|------|------|
| RF-only Student | RF | 10% | 0 | No | 37.96% | — |
| E2b (512) | RF+DF+AWF | 15% | 0 | No | 17.77% | 17.77% |
| E2b (Full CW) | RF+DF+AWF | 15% | 0 | No | 17.84% | 17.84% |
| **Completion (Full CW)** | **RF+DF+AWF** | **30%** | **0.5** | **Yes** | **13.37%** | **10.20%** |

WC: RF-only baseline 37.96% → Completion 13.37% (-24.59pp)

## Anti-Adversarial Training (512 subset preview)

| Method | BW | DF Adaptive | AWF Adaptive |
|------|------|------|------|
| E2b b20 ε=0.1, no completion | 7.60% | 65.04% | 81.64% |
| Completion b20 ε=0.5 | 17.51% | 17.19% | 18.36% |
| Completion b30 ε=0.5 | 25.13% | 11.72% | 14.84% |

Without randomization + completion, 2 epochs of adversarial fine-tuning
break the defense (65-82% recovery). With both, adaptive recovery drops
to 12-18%.

## Install

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
conda create -n pyda python=3.10
conda activate pyda
pip install torch numpy tqdm
pip install -e .
```

Server environment: Python 3.10.20, PyTorch 2.13.0+cu130, RTX 4090 24GB

## Build Inputs

Prepare the fast-keypoint archive (RF keypoints + TAM for full CW):

```bash
python scripts/stage_b_prepare_fast_keypoint_archive.py \
  --data_root datasets/CW.npz \
  --attacker rf \
  --checkpoint models/attacks/fixed_rf_checkpoint.pt \
  --split_name all \
  --batch_size 512 \
  --device cuda \
  --run_name stage_b_fast_keypoint_full_cw_all_seed0
```

## Run E2b Oracle (RF+DF+AWF)

Single-process 512 subset:

```bash
python scripts/stage_b_run_ensemble_oracle_e2b.py \
  --output_dir results --run_name stage_b_e2b_512 \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --data_root datasets/CW.npz \
  --rf_checkpoint models/attacks/fixed_rf_checkpoint.pt \
  --df_checkpoint wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth \
  --varcnn_checkpoint wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth \
  --methods norm_weighted_r80_d10_v10 \
  --dummy_budgets 0.15 \
  --max_samples 512 --sample_end 512 \
  --compact_candidate_generation \
  --candidate_batch_size 4096 \
  --candidate_device cuda --candidate_score_device cpu \
  --device cuda --batch_size 128 \
  --cost_lambda 0.05 \
  --robust_min_positive_models 1 \
  --robust_epsilon 0.20 \
  --max_dummy_steps 10 \
  --export_defended_npz wflib_copy/datasets/CW/test_e2b_512_defended.npz
```

## Run Completion Oracle (Full CW, 64-shard parallel)

```bash
# Launch 64 shards, 20 workers parallel
samples=10564
for s in $(seq 0 63); do
  start=$((s * samples / 64))
  end=$(((s + 1) * samples / 64))
  python scripts/stage_b_run_ensemble_oracle_e2b_completion.py \
    --output_dir results/$RUN \
    --run_name shard_$(printf '%03d' $s) \
    --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
    --data_root datasets/CW.npz \
    --rf_checkpoint models/attacks/fixed_rf_checkpoint.pt \
    --df_checkpoint wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth \
    --varcnn_checkpoint wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth \
    --methods norm_weighted_r80_d10_v10 \
    --dummy_budgets 0.30 \
    --max_samples $samples --sample_start $start --sample_end $end \
    --random_epsilon 0.5 \
    --completion_mode preserve_flip_fill \
    --completion_target_bandwidth 0.30 --completion_topk 16 \
    --compact_candidate_generation \
    --candidate_batch_size 4096 \
    --candidate_device cuda --candidate_score_device cpu \
    --device cuda --batch_size 128 \
    --cost_lambda 0.05 \
    --robust_min_positive_models 1 \
    --robust_epsilon 0.20 \
    --max_dummy_steps 20 \
    --seed 202 \
    --export_defended_npz wflib_copy/datasets/CW/adapt_e2b_${RUN}_shard$(printf '%03d' $s).npz &
done
wait
```

## Candidate Space

Actions are generated as a 5D cross-product:

| Dimension | Source | Values |
|------|------|------|
| Windows | RF TAM keypoint peaks | up to 8 |
| Anchors | 7 structural types | keypoint offsets, rate peaks, direction transitions, gaps, burst extensions |
| Doses | absolute + relative | {1,2,4,8,16,32} + {10-100% of window count} |
| Widths | injection spread | window length, half-length, 8, 16, 32 |
| Modes | direction strategy | direction_balance, dynamask_causal |

~30k raw combinations → budget-filtered → deduped → Top128 by structural score.

The Top128 selection uses a structural relevance score (RF confidence × tier × dose bonus / sqrt(cost)),
**not** per-model gain. Actual effectiveness is evaluated later by the 3 surrogates.

## Layout

```
dynapd/
  data/               CW loading and split helpers
  stage_a/            Attacker models (RF/DF), keypoint extraction
  stage_b/            Candidate generation, objectives, policy data
  utils/              Runtime config, device helpers

scripts/
  stage_b_prepare_fast_keypoint_archive.py    # RF keypoint archive
  stage_b_run_ensemble_oracle_e2b.py          # E2b base (RF+DF+AWF)
  stage_b_run_ensemble_oracle_e2b_completion.py  # E2b + Completion + Randomization
  stage_b_run_ensemble_oracle_e2b_rand.py     # E2b + Randomization
  stage_b_build_policy_dataset.py             # Student policy data
  stage_b_train_candidate_policy.py           # Student training
  stage_b_run_student_policy_controller.py    # Student inference
  stage_b_eval_candidate_policy_offline.py    # Offline evaluation
```

## Notes

This is a research release. Datasets, checkpoints, and experiment outputs are
not included. The current main line uses `RF + DF + AWF` as the Teacher
surrogate set with completion and stochastic action selection.
