# Experimental Record

All accuracies are closed-world website-fingerprinting attacker accuracies;
lower is better. `WC` is the maximum defended accuracy over RF, DF, TF, AWF,
and VarCNN.

## DynaPD-RT: Strict Streaming Full-CW Evaluation

DynaPD-RT is a timeout-driven event state machine. An outgoing burst schedules
a timer at `burst_end + GAP_THRESH + 1`; at expiration the controller reads an
offline profile utility table and emits dummy only after that decision. Delay
rules apply only to later-arriving, not-yet-emitted packets.

| Evaluation interval | RF | DF | TF | AWF | VarCNN | WC | BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| CW `[96:105730)`; 105,634 traces | 7.01% | 11.41% | 9.87% | 11.96% | 10.18% | **11.96%** | 19.11% |

Protocol:

- Calibration: `CW[0:96)`; evaluation uses the disjoint interval above.
- Offline utility surrogates: RF, DF, and AWF. TF and VarCNN are held out.
- `rho=0.35`; action profiles are `compact`, `spread`, `delay_heavy`, and
  `strong`.
- Mean dummy count: 351.48 packets/trace; mean forward delay rules: 23.05.
- Physical trace export is packet-conserving and timestamp-preserving.
- Fixed-prefix attacker protocol: the repository CW representation and native
  attacker preprocessing use the first 5,000 observed packets.
- Causality audit: `dummy_before_decision=0`, `delay_before_activation=0`,
  `delay_after_emission=0`, and `future_packet_read=0`.
- Generation throughput with 18 workers: 1.37 ms/trace. This is parallel
  throughput, not the latency of an isolated trace.

### Fixed-Prefix Truncation Control

`12,694` traces have one or more known real packets beyond the fixed 5,000-
packet attack input after defense. To isolate this factor, the evaluator also
uses the remaining `92,940` traces for which every known real packet is still
inside the attack input.

| Control subset | RF | DF | TF | AWF | VarCNN | WC |
|---|---:|---:|---:|---:|---:|---:|
| No known real-packet truncation; 92,940 traces | 3.77% | 8.20% | 6.96% | 10.83% | 8.03% | **10.83%** |

This control indicates that the measured defense is not primarily explained by
displacing known real packets outside the fixed prefix. It does **not** claim
security against arbitrary full-stream or length-adaptive attackers.

## Offline DynaPD Reference

The offline controller observes a complete trace and evaluates candidate
actions with normalized RF/DF/AWF surrogate gains weighted `0.80/0.10/0.10`.
It supports offline action discovery but is not an online deployment result.

| Configuration | RF | DF | TF | AWF | VarCNN | WC | BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

## Boundaries

- Results are non-adaptive-attacker evaluations.
- The fixed-prefix protocol is a property of the released CW representation
  and clean attacker checkpoints. It must be reported with every result.
- Historical streaming controllers with backfilled action timestamps are not
  included in this document and must not be compared to DynaPD-RT.
