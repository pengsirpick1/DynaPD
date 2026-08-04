# Completion b30 ε=0.5 Full CW Results

Date: 2026-08-04 | Server: T640gpu0 RTX 4090 | Conda: pyda (Python 3.10.20, PyTorch 2.13.0+cu130)

## Experiment ID

`stage_b_e2b_completion_fulltest_b030_e05_seed202`

## Configuration

| Parameter | Value |
|------|------|
| Method | `norm_weighted_r80_d10_v10` |
| Budget | 0.30 (30%) |
| Random Epsilon | 0.50 |
| Robust Epsilon | 0.20 |
| Min Positive Models | 1 |
| Max Dummy Steps | 20 |
| Completion Mode | `preserve_flip_fill` |
| Completion Target BW | 0.30 |
| Completion TopK | 16 |
| Candidate Space | `stratified_top128` |
| Candidate Batch | 4096 |
| Seed | 202 |
| Dataset | Full CW Test (10564 samples) |
| Sharding | 64 shards (~166 samples each) |
| Parallel Workers | 20 |
| Peak VRAM | ~23.9 GB / 24.6 GB |
| Duration | ~2h 20min |

## Key Scripts

```
scripts/
├── stage_b_run_ensemble_oracle_e2b.py                # Base E2b (RF+DF+AWF)
├── stage_b_run_ensemble_oracle_e2b_completion.py     # E2b + Completion mode
└── stage_b_run_ensemble_oracle_e2b_rand.py           # E2b + Randomization
```

## Non-Adaptive Results (Teacher Oracle)

| Model | Role | Clean Acc | Defended Acc | Flip |
|------|------|------|------|------|
| RF | Teacher | 97.66% | **13.37%** | 86.59% |
| DF | Teacher | 98.25% | 9.94% | 89.91% |
| AWF | Teacher | 94.86% | 4.95% | 95.05% |
| TF | Held-out | 96.16% | **10.20%** | 89.80% |
| **WC** | — | — | **13.37%** | — |

## Bandwidth & Action Overhead

| Metric | Value |
|------|------|
| Mean BW | 24.50% |
| Median BW | 29.99% |
| P90 BW | 30.03% |
| P95 BW | 30.06% |
| Max BW | 30.77% |
| Mean Actions | 12.09 |
| Mean Delay | 5.35 bins |

## Stop Reasons

| Stop Reason | Count | % |
|------|------|------|
| all_models_target_reached | 8319 | 78.75% |
| no_robust_positive | 2232 | 21.13% |
| bandwidth_reached | 13 | 0.12% |

## All Flipped

```
All 3 surrogate flipped: 78.75%
At least 2 flipped:       93.83%
```

## E2b Evolution (RF+DF+AWF, r80_d10_v10)

| Version | Budget | Random ε | Completion | WC | TF held-out |
|------|------|------|------|------|------|
| Base 512 | 0.15 | 0.0 | No | 17.77% | 17.77% |
| Base Full CW | 0.15 | 0.0 | No | 17.84% | 17.84% |
| **Completion Full CW** | **0.30** | **0.5** | **Yes** | **13.37%** | **10.20%** |

## Key Improvements

1. **TF held-out: 17.84% → 10.20%** (-7.64pp) — completion broke TF's resistance
2. **DF: 11.20% → 9.94%** (-1.26pp) — modest gain
3. **AWF: 6.07% → 4.95%** (-1.12pp) — already good, slight improvement
4. **All Flip: ~76% → 78.75%** — 2pp more samples fully covered
5. Cost: BW 7% → 24.5% (acceptable for anti-adversarial use case)

## Next Steps

- [ ] Student policy training (seed 202 full + seed 101 512)
- [ ] Dual-seed adaptive adversarial training test
- [ ] Var-CNN held-out full CW eval
- [ ] Compare b20 vs b30 cost/robustness trade-off

## Output Files (Server)

```
/home/xuke/pzy/DynaPD-main/DynaPD-main/
├── wflib_copy/datasets/CW/
│   └── adapt_e2b_completion_fulltest_b030_e05_seed202_merged.npz  (10564, 1, 5000)
└── results/
    └── stage_b_e2b_completion_fulltest_b030_e05_seed202_shard*/
        └── ensemble_oracle_summary.csv
```
