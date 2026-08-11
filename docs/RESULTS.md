# Experimental Record

This page records the results currently backed by the tracked manifests and
summaries. Accuracies are closed-world attack accuracies; lower is better.
`WC` is the maximum accuracy over the evaluated attacker set.

## Default event-keypoint RT controller

The repository default, `streaming_state_machine.py`, is the causal
event-keypoint controller. It maps an ended outgoing burst to an
offline-calibrated `(phase, out, duration-bin, packet-volume-bin)` utility
row, then selects a token-budget-compatible allocation scale. On the full CW
evaluation interval disjoint from its 96-trace calibration interval, it gives:

| Controller | RF | DF | TF | AWF | VarCNN | WC | BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| Event-keypoint RT, `tail0` | 11.54% | 15.39% | 11.63% | 10.15% | 16.24% | 16.24% | 16.17% |
| Phase-only RT, `tail0` baseline | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | **15.31%** | 16.36% |

The event controller has zero future-packet audit violations. Its current
duration-and-volume event definition is an online/offline linkage mechanism,
not a demonstrated accuracy improvement; the full protocol and artifacts are
in [EVENT_KEYPOINT_RT.md](EVENT_KEYPOINT_RT.md) and
`reproducibility/event_keypoint_rt_fullcw/`.

## Offline DynaPD: multi-surrogate teacher

The offline controller uses the complete trace, a stratified Top-128 candidate
space, and normalized gain weighted as RF/DF/AWF = 0.80/0.10/0.10. The
following deterministic low-bandwidth result is reported with its measured
dummy-bandwidth overhead.

| Configuration | RF | DF | TF | AWF | VarCNN | WC | Measured BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

This branch is an offline effectiveness controller, not an online deployment
claim. It evaluates candidate actions against surrogate models and therefore
has a much higher per-trace decision cost than DynaPD-RT.

## DynaPD-RT causal streaming evaluation

The table below is from `reproducibility/streaming_allcw/manifest.json`.
`tail0` is the deployment-oriented protocol: an unresolved final burst is left
untouched because no timeout event has yet been observed. `tail1` is an
ablation that permits an end-of-trace action.

| Variant | RF | DF | TF | AWF | VarCNN | WC | Measured BW |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streaming `tail0` | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | 15.31% | 16.36% |
| Streaming `tail1` (ablation) | 10.13% | 13.68% | 10.43% | 8.18% | 6.07% | 13.68% | 17.34% |
| Batch full-information upper bound | 5.85% | 5.57% | 5.90% | 6.96% | 3.53% | 6.96% | 17.20% |

The `tail0` generation audit reports `audit_future_total = 0` across all
observed delay actions. The full evaluation used 20 CPU workers and measured
1.98 ms/trace amortized generation. A separate single-trace state-machine
measurement is about 12 ms/trace; it is the relevant deployment latency.

## DynaPD-RT bandwidth trade-off

`rho` is the running token-budget coefficient, not the final overhead. The
renderer computes measured bandwidth after trace generation.

| rho | Measured BW | RF | DF | TF | AWF | VarCNN | WC |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 7.55% | 20.66% | 26.76% | 21.98% | 19.71% | 13.06% | 26.76% |
| 0.15 | 10.56% | 15.93% | 21.36% | 17.05% | 14.65% | 10.17% | 21.36% |
| 0.20 | 13.48% | 13.30% | 17.65% | 14.02% | 11.66% | 8.30% | 17.65% |
| 0.23 | 15.19% | 12.40% | 16.21% | 12.61% | 10.38% | 7.57% | 16.21% |
| 0.25 | 16.36% | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | 15.31% |

## Matched causal random baselines

The random controls use the same burst-end trigger, renderer, delay bound and
token-budget rule as DynaPD-RT. `random_recent` randomizes locally near the
current burst; `random_far` samples a later location in the remaining horizon.

| Method | Measured BW | RF | DF | TF | AWF | VarCNN | WC |
|---|---:|---:|---:|---:|---:|---:|---:|
| DynaPD-RT, rho=0.15 | 10.56% | 15.93% | 21.36% | 17.05% | 14.65% | 10.17% | 21.36% |
| Random recent, rho=0.15 | 10.55% | 25.15% | 21.77% | 16.61% | 15.04% | 12.71% | 25.15% |
| Random far, rho=0.15 | 10.52% | 24.26% | 30.52% | 20.66% | 22.66% | 17.80% | 30.52% |
| DynaPD-RT, rho=0.25 | 16.36% | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | 15.31% |
| Random recent, rho=0.25 | 16.35% | 19.30% | 16.11% | 11.92% | 10.52% | 9.45% | 19.30% |

The utility table supplies a measurable refinement over the matched random
controls; the streaming burst-state actuator and bounded delay are also major
contributors and should not be conflated with utility lookup alone.

## Scope and limitations

- The RT results above are **non-adaptive** attacker evaluations.
- The batch controller is an upper bound and must not be presented as a causal
  deployment result.
- The utility calibration table is global and coarse. It should be called
  **small calibration**, not class-level few-shot learning: the 32-trace
  sensitivity result only shows early stabilization of a 12-cell global table,
  not coverage of all 95 websites.
- The maximum per-packet delay is 64 bins (about 2.84 s at 80/1800 s/bin).
  Page-completion overhead must be measured from untruncated traces with the
  script in `scripts/measure_page_completion.py` before making a user-facing
  latency claim.

Raw run metadata is tracked under `reproducibility/`; large trace arrays and
checkpoints are intentionally not published.
