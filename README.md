[中文](README_CN.md) | **English**

# DynaPD

**Dynamic Traffic Perturbation against Website Fingerprinting**

DynaPD is a research framework for defending closed-world website fingerprinting
(WF). It has two complementary branches with deliberately different operating
assumptions:

| Branch | Goal | Information available at decision time |
|---|---|---|
| **Offline DynaPD** | High-effectiveness dynamic defense and robustness studies | Complete trace and multi-surrogate feedback |
| **DynaPD-RT** | Causal, label-free streaming deployment | Current and past packets only |

The repository contains source code, compact public configurations, and
reproducibility records. CW traces, attacker checkpoints, WFlib source,
generated defended traces, and logs are intentionally excluded.

## Method at a glance

```text
Offline DynaPD
complete trace -> candidate generation -> RF / DF / AWF gain evaluation
               -> constrained action selection -> dummy + delay

DynaPD-RT
arriving packets -> running budget + download-burst state -> utility lookup
                 -> burst-tail dummy + bounded causal delay
```

### Offline DynaPD

The offline controller builds a stratified Top-128 action set and scores each
candidate with normalized margin reduction across three complementary WF
surrogates: RF (TAM/burst), DF (direction sequence), and AWF (burst family).
The main deterministic setting is `norm_weighted_r80_d10_v10`, with
RF/DF/AWF weights `0.80/0.10/0.10`.

Randomized selection and bandwidth completion are research variants for
studying robustness under adaptive attackers. They are not used by the RT
deployment path.

### DynaPD-RT

DynaPD-RT distills coarse offline experience into a compact utility table
indexed by `(phase, direction, dose)`. Online execution maintains only packet
count, positive-direction/download-burst state, consumed token budget, and this
table. At a burst ending it makes a bounded perturbation decision without a
website label and without querying RF, DF, AWF, TF, or VarCNN.

```python
from streaming_state_machine import defend_stream

defended_trace = defend_stream(clean_trace, seed=0, rho=0.25)
```

The deployment-oriented protocol is **`tail0`**: a final unresolved burst is
not modified unless a real timeout event is available. `tail1` is an
end-of-trace ablation, while the batch controller is a full-information upper
bound rather than a deployment result.

## Main results

All values below are closed-world attack accuracies, so lower is better.
`WC` is the maximum accuracy among the evaluated attack models.

### Offline DynaPD, 512-trace CW subset

| Configuration | RF | DF | AWF | VarCNN held-out | TF held-out | WC |
|---|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 12.70% | 8.40% | 3.32% | 7.03% | 17.77% | 17.77% |

### DynaPD-RT, full CW evaluation (105,730 traces)

| Variant | RF | DF | TF | AWF | VarCNN | WC | Measured bandwidth |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streaming `tail0` | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | 15.31% | 16.36% |
| Streaming `tail1` (ablation) | 10.13% | 13.68% | 10.43% | 8.18% | 6.07% | 13.68% | 17.34% |
| Batch full-information upper bound | 5.85% | 5.57% | 5.90% | 6.96% | 3.53% | 6.96% | 17.20% |

The `tail0` run records zero future-packet accesses in its causality audit.
Generation used 20 CPU workers and took 1.98 ms/trace amortized; a separate
single-trace state-machine measurement is approximately 12 ms/trace.

The full bandwidth curve, matched causal random baselines, raw manifests, and
evaluation boundaries are documented in [docs/RESULTS.md](docs/RESULTS.md).

## Repository layout

```text
dynapd/                         Core data, padding, objectives, and models
scripts/                        Offline teacher/search and evaluation scripts
streaming_state_machine.py      DynaPD-RT causal state machine
streaming_allcw_mp.py           Full-CW RT evaluation
streaming_allcw_bw_sweep.py     RT bandwidth sweep
random_streaming_baseline_bw_sweep.py
                                 Matched causal random baselines
configs/dynapd_rt_utility.json  Compact public RT utility table
reproducibility/                Tracked manifests and run summaries
docs/RESULTS.md                 Curated experimental record
docs/REPRODUCIBILITY.md         Data, split, and protocol notes
```

## Installation

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
```

Five-model evaluation additionally requires a compatible WFlib checkout, CW
signed-timestamp traces, and clean attacker checkpoints. Place WFlib in
`wflib_copy/` or expose it through `PYTHONPATH`; provide local dataset and
checkpoint paths when running evaluation scripts.

## Entry points

```bash
# Offline multi-surrogate controller
python scripts/stage_b_run_ensemble_oracle_e2b_completion.py --help
python scripts/stage_b_run_ensemble_oracle_e2b_rand.py --help

# Streaming DynaPD-RT evaluation
python streaming_allcw_mp.py --help
python streaming_allcw_bw_sweep.py --help
python random_streaming_baseline_bw_sweep.py --help
```

## Scope and responsible interpretation

- DynaPD-RT results in this release are **non-adaptive** attacker evaluations.
- The RT utility table is **small calibration**, not class-level few-shot
  learning. It is a global 12-cell phase/direction/dose summary.
- The delay bound is 64 bins (about 2.84 s per affected packet under the
  80-second, 1800-bin representation). Page-completion overhead must be
  measured from untruncated traces; see `scripts/measure_page_completion.py`.
- No license has been selected yet.

For full reproduction details, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
