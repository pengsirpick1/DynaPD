# Experimental Record

All accuracies are closed-world website-fingerprinting attacker accuracies;
lower is better. `WC` is the maximum defended accuracy over RF, DF, TF, AWF,
and VarCNN.

## Current online result: timeout event-keypoint DynaPD-RT

The current deployment controller waits for a five-bin burst-idle timeout,
then performs an offline event-utility lookup using the observed outgoing burst
duration and volume. It emits dummy strictly after the timeout and delays only
packets arriving after activation. This is the only RT result in this release
with explicit action-time auditing.

| Evaluation interval | RF | DF | TF | AWF | VarCNN | WC | BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| CW `[96:105730)`; 105,634 traces | 11.03% | 21.71% | 17.88% | 20.31% | 19.08% | **21.71%** | 15.24% |

Calibration uses the disjoint interval `CW[0:96)`. The full-CW audit records
zero `dummy_before_decision`, `delay_before_activation`,
`delay_after_emission`, and `future_packet_read` events. The tracked evidence
is under `reproducibility/event_keypoint_timeout_fullcw/`.

## Offline DynaPD reference

The offline reference has access to complete traces and evaluates candidate
actions with RF/DF/AWF surrogate gains weighted `0.80/0.10/0.10`. It is useful
for offline experience discovery, but is not an online deployment result.

| Configuration | RF | DF | TF | AWF | VarCNN | WC | BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

## Superseded results

Earlier repository records labelled phase-only or event-keypoint RT as
“strictly causal” used a burst-end decision that could be confirmed only later
but placed dummy immediately after the earlier burst end. Their audit checked
future data reads, not action timestamps. Those records are retained only in
Git history for auditability and must not be used as deployment results or
compared with the timeout-driven protocol.

See [EVENT_KEYPOINT_RT.md](EVENT_KEYPOINT_RT.md) for the failure mode, its
correction, and the protocol boundaries.
