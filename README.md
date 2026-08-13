[中文](README_CN.md) | **English**

# DynaPD

**Dynamic Traffic Perturbation against Website Fingerprinting**

DynaPD is a research framework for closed-world website-fingerprinting (WF)
defense. Its deployable controller, **DynaPD-RT**, compiles offline
multi-surrogate action evidence into a compact utility table and executes it as
a strictly causal, timeout-driven streaming state machine. At deployment it
uses neither a website label, the complete trace, nor an online attacker-model
query.

CW traces, WFlib source, attacker checkpoints, generated defended traces, and
logs are intentionally excluded from this repository.

## DynaPD-RT

```text
Offline calibration
complete traces + RF / DF / AWF surrogate gains
  -> utility over observable outgoing-burst event states
  -> (traffic phase, direction, duration bin, volume bin) -> action profile

Online controller
packet arrivals -> token budget + outgoing-burst state
  -> idle timeout -> utility lookup
  -> dummy after the timeout + bounded delay for later arrivals only
```

The event state is observable without a label. A profile specifies dummy-dose
scale, spacing, a forward delay window, and a bounded delay cap. The utility
artifact is global aggregate calibration data: it contains no raw trace, site
label, or model checkpoint.

The controller is [`streaming_state_machine.py`](streaming_state_machine.py):

```python
from streaming_state_machine import defend_stream, load_utility

load_utility("configs/dynapd_rt_event_utility.npy")
defended_trace = defend_stream(clean_trace, seed=0, rho=0.35)
```

For each outgoing burst, the controller schedules a timer at
`burst_end + GAP_THRESH + 1`. When it expires, the controller selects a
profile from the utility table. Dummies begin strictly after the decision time;
delay is applied only to packets that arrive after activation and have not yet
been emitted. This action order is audited per trace.

## Full-CW Result

All values are closed-world attacker accuracies; lower is better. `WC` is the
largest defended accuracy among RF, DF, TF, AWF, and VarCNN.

| Controller | RF | DF | TF | AWF | VarCNN | WC | Measured BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DynaPD-RT** | 7.01% | 11.41% | 9.87% | 11.96% | 10.18% | **11.96%** | 19.11% |

The result covers `105,634` CW traces disjoint from the `96`-trace calibration
interval. The controller used `rho=0.35` and the checked-in utility artifact.
Generation throughput was 1.37 ms/trace with 18 workers. The time semantics
audit reports zero dummy-before-decision, delay-before-activation,
delay-after-emission, and future-packet-read events.

The standard evaluation uses each attacker's native fixed-prefix preprocessing
on the repository's 5,000-packet CW representation. The physical defended
stream preserves all input packets and explicit timestamps. A control subset
of 92,940 traces with no known real packet displaced outside the first 5,000
observed packets still yields 10.83% WC. This is a fixed-prefix attacker
result, not a claim about arbitrary length-adaptive attackers.

### Offline Reference

The offline reference is used to discover action utility and characterize a
complete-trace upper-bound controller; it is not a deployment claim.

| Configuration | RF | DF | TF | AWF | VarCNN | WC | Measured BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

## Installation

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
```

Five-model evaluation additionally needs compatible WFlib code, signed-
timestamp CW traces, and clean RF/DF/TF/AWF/VarCNN checkpoints. Supply their
local paths; they are not distributed here.

## Reproduction Entry Points

```bash
# Rebuild a profile utility table from a disjoint calibration interval.
python scripts/build_event_keypoint_utility.py --help

# Export a small strict-streaming evaluation.
python scripts/run_dynapd_rt_eval.py --data /path/to/CW.npz --output-dir results/rt_small

# Export the full CW interval with parallel workers.
python scripts/run_dynapd_rt_fullcw.py --data /path/to/CW.npz --output-dir results/rt_full --workers 18
```

See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for required audits and
[docs/RESULTS.md](docs/RESULTS.md) for the recorded protocol.

## Scope

- Results are against non-adaptive attackers.
- The public utility table is a global small-calibration artifact, not a
  class-level few-shot model.
- The reported attack evaluation is fixed-prefix; it must not be generalized
  to arbitrary unbounded or length-adaptive attacker inputs.
- No license has been selected yet.
