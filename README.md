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
    -> timeout-triggered dummy injection + forward-only bounded delay
```

The default controller is [`streaming_state_machine.py`](streaming_state_machine.py).
At an outgoing/download burst end it schedules a real timeout of five bins.
When that timer fires, it derives a local event type from the already observed
burst duration and packet volume, then selects an offline-calibrated allocation
scale. Dummy injection begins strictly after the timeout; delay rules apply
only to packets arriving after the timer activation.

```python
from streaming_state_machine import defend_stream

defended_trace = defend_stream(clean_trace, seed=0, rho=0.35)
```

Earlier phase-only streaming code is retained only in Git history: its action
timestamp convention has been superseded by the timeout protocol and it is not
a valid strict-causality baseline.

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
| **Timeout event-keypoint RT (default)** | 11.03% | 21.71% | 17.88% | 20.31% | 19.08% | **21.71%** | 15.24% |

The event-keypoint controller was evaluated on 105,634 CW traces disjoint from
its 96-trace calibration interval. Its audit reports zero dummy-before-decision,
delay-before-activation, delay-after-emission, and future-packet-read events.
The event-conditioned design links online actions to offline-discovered local
burst shapes. See [docs/EVENT_KEYPOINT_RT.md](docs/EVENT_KEYPOINT_RT.md) for
the superseded retrospective implementation and the corrected protocol.

## Repository layout

```text
dynapd/                                  Core data, objectives, and models
scripts/build_event_keypoint_utility.py  Offline event-utility calibration
scripts/                                 Offline teacher/search and evaluation tools
streaming_state_machine.py               Timeout-driven event-keypoint RT controller
causal_event_renderer.py                 Explicit timestamp materialization
configs/dynapd_rt_event_utility_timeout.npy Compact timeout event utility table
reproducibility/event_keypoint_timeout_fullcw/ Corrected full-CW record
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
