[中文](README_CN.md) | **English**

# DynaPD

**Dynamic Traffic Perturbation against Website Fingerprinting**

DynaPD is a research framework for closed-world website-fingerprinting (WF)
defense. Its deployment path, **DynaPD-RT**, compiles expensive offline
multi-surrogate defense search into a compact burst-event utility table, then
executes that table causally as traffic arrives. No website label, complete
trace, or online attacker-model query is required at deployment time.

CW traces, WFlib source, attacker checkpoints, generated defended traces, and
logs are intentionally excluded from this repository.

## Design

```text
Offline discovery
complete traces + RF/DF/AWF surrogate gains
    -> aggregate action utility for observable outgoing-burst events
    -> (phase, out, duration-bin, packet-volume-bin) -> allocation scale

Online DynaPD-RT
arriving packets -> token budget + outgoing burst state
    -> identify an ended local burst event -> utility lookup
    -> burst-tail dummy injection + bounded causal delay
```

The default controller is [`streaming_state_machine.py`](streaming_state_machine.py).
At an outgoing/download burst end, it derives a local event type from the
already observed burst duration and packet volume, then selects an
offline-calibrated allocation scale. The `tail0` deployment protocol leaves the
last unresolved burst unchanged unless a real network timeout is observed.

```python
from streaming_state_machine import defend_stream

defended_trace = defend_stream(clean_trace, seed=0, rho=0.213)
```

The preceding phase-only implementation is retained as
[`streaming_state_machine_phase_baseline.py`](streaming_state_machine_phase_baseline.py)
for ablation and comparison.

## Results

All values are closed-world attacker accuracies; lower is better. `WC` is the
largest defended accuracy among RF, DF, TF, AWF, and VarCNN.

### Offline DynaPD reference

The offline reference uses complete traces and normalized RF/DF/AWF gain with
weights `0.80/0.10/0.10`; it is not an online deployment claim.

| Configuration | RF | DF | TF | AWF | VarCNN | WC | Measured BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

### DynaPD-RT strict streaming evaluation

| Controller | RF | DF | TF | AWF | VarCNN | WC | Measured BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Event-keypoint RT (`tail0`, default)** | 11.54% | 15.39% | 11.63% | 10.15% | 16.24% | 16.24% | 16.17% |
| Phase-only RT (`tail0`, baseline) | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | **15.31%** | 16.36% |

The event-keypoint controller was evaluated on 105,634 CW traces disjoint from
its 96-trace calibration interval. Its causality audit reports zero future
packet accesses. The event-conditioned design is operationally useful because
it links the online action to an offline-discovered local burst shape; however,
under this duration-and-volume schema it is performance-comparable rather than
superior to the phase-only baseline. See [docs/EVENT_KEYPOINT_RT.md](docs/EVENT_KEYPOINT_RT.md).

## Repository layout

```text
dynapd/                                  Core data, objectives, and models
scripts/build_event_keypoint_utility.py  Offline event-utility calibration
scripts/                                 Offline teacher/search and evaluation tools
streaming_state_machine.py               Default event-keypoint RT controller
streaming_state_machine_phase_baseline.py Phase-only RT baseline
configs/dynapd_rt_event_utility.npy      Compact published event utility table
reproducibility/event_keypoint_rt_fullcw/ Full-CW manifests and five-model result
docs/RESULTS.md                          Curated baseline record
docs/EVENT_KEYPOINT_RT.md                Event-keypoint experiment record
docs/REPRODUCIBILITY.md                  Data, split, and protocol notes
```

## Installation

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
```

Five-model evaluation additionally requires a compatible WFlib checkout, CW
signed-timestamp traces, and clean attacker checkpoints. Put WFlib in
`wflib_copy/` or expose it through `PYTHONPATH`; provide local dataset and
checkpoint paths when evaluating.

## Reproduction entry points

```bash
# Build a new event-conditioned utility artifact from local calibration data.
python scripts/build_event_keypoint_utility.py --help

# Run the causal RT bandwidth sweep with local CW data and model checkpoints.
python streaming_allcw_bw_sweep.py --help

# Offline multi-surrogate reference controller.
python scripts/stage_b_run_ensemble_oracle_e2b_completion.py --help
```

## Scope

- Reported RT results are **non-adaptive attacker** evaluations.
- The public event table is a global **small-calibration** artifact, not a
  class-level few-shot model.
- The maximum delay is 64 bins. Page-completion overhead must be measured from
  untruncated traces; see `scripts/measure_page_completion.py`.
- No license has been selected yet.
