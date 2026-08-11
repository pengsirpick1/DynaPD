# Event-Keypoint DynaPD-RT Pilot Report

## Question
Does adding an offline-discovered, causal outgoing-burst shape to the DynaPD-RT utility table improve strict streaming defense at equal high bandwidth?

## Protocol
- Calibration: `datasets/CW.npz[0:96)`, not used by evaluation.
- Utility ensemble: RF/DF/AWF with weights `0.8/0.1/0.1`.
- Event key: `(phase, out, duration_tercile, volume_tercile)`.
- Action: a learned allocation scale in `{0.50, 0.75, 1.00, 1.25}` multiplied by the causal per-burst token allocation.
- Event table: 1,395 calibration burst events; 5 event types met `min_support=12`.
- Evaluation: `datasets/CW.npz[1024:1536)`, 512 traces, strict streaming `tail0`, no label, no full-trace visibility, and no online attack-model query.
- All attackers consume timestamp-preserving defended traces; VarCNN uses DT2.

## Equal-Bandwidth Result

| Method | BWO | RF | DF | TF | AWF | VarCNN | WC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-only RT | 32.11% | 6.05% | 8.98% | 6.45% | 4.10% | 8.01% | **8.98%** |
| Event-keypoint RT v2 | 32.61% | 5.86% | 8.79% | 5.66% | 4.88% | 9.57% | **9.57%** |

The BWO difference is 0.50 percentage points, favouring the event-keypoint variant. Despite this, its WC is 0.59 percentage points higher.

## Runtime and Causality
- Event-keypoint actions used supported event rows 9,852 times and phase fallback 2,240 times across the 512 traces.
- Future-packet audit violations: 0.
- Parallel generation throughput: about 2.06 ms/trace with 16 workers. This is throughput, not single-trace latency.

## Low-Bandwidth Repeat

The high-bandwidth point was intentionally configured with a large token-bucket rate. To test whether the event mechanism merely depended on extra budget, the experiment was repeated with the low-bandwidth RT operating point.

| Method | BWO | RF | DF | TF | AWF | VarCNN | WC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Phase-only RT | 15.90% | 10.16% | 14.65% | 11.72% | 9.38% | 16.80% | **16.80%** |
| Event-keypoint RT v2 | 16.28% | 9.38% | 14.65% | 10.74% | 8.01% | 17.58% | **17.58%** |

The low-bandwidth table is also a 512-trace holdout using the same 96-trace calibration interval. The event variant uses 0.38 percentage points more BWO and has a 0.78 percentage point higher WC. It therefore remains an unsupported extension at both tested bandwidth points.

## Decision
The current duration-and-volume event prototype does **not** improve the phase-only RT baseline at either the low- or high-bandwidth operating point. It must not be claimed as a keypoint contribution or replace the existing RT mainline.

Allowed wording: the pilot tests an offline-discovered causal event-keypoint extension and finds no improvement under this schema.

Forbidden wording: event-keypoint recognition improves DynaPD-RT, or the current RT mainline already uses validated keypoint detection.

## Full-CW Scaling Check

The corrected v2 controller was scaled to the entire available CW export after
excluding its calibration interval: `datasets/CW.npz[96:105730)` (105,634
traces). It uses the low-bandwidth table, target token rate `rho=0.213`,
18 generation workers, timestamp-preserving export, and the same final RF,
DF, TF, AWF, and VarCNN (DT2) evaluation protocol as the offline experiments.

| Method | Traces | BWO | RF | DF | TF | AWF | VarCNN | WC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase-only RT (`rho=0.25`) | 105,730 | 16.36% | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | **15.31%** |
| Event-keypoint RT v2 (`rho=0.213`) | 105,634 | 16.17% | 11.54% | 15.39% | 11.63% | 10.15% | 16.24% | **16.24%** |

The event-keypoint run preserves strict causality: `0` future-packet audit
violations across all defended traces. It used a supported event row 2,006,395
times and fell back to the phase row 492,264 times. At a 0.19 percentage-point
lower BWO, its WC is 0.93 percentage points higher than the phase-only RT
baseline; the table therefore confirms the small-pilot finding rather than
reversing it. The present duration-and-volume event feature should remain an
ablation, not a claimed accuracy improvement.

## Next Technical Options
1. Replace fixed absolute `early/mid/late` bins with calibration-derived causal time thresholds; the current calibration events were overwhelmingly in `early`.
2. Add a preceding-gap feature only after increasing calibration support, then compare it with a traffic-matched randomized event-type control.
3. Keep phase-only burst-event utility as the paper mainline unless a held-out, equal-BWO experiment reverses this result.
