# DynaPD-RT Event-Conditioned Streaming Controller

This document describes the public DynaPD-RT implementation. It supersedes
earlier experimental controllers that assigned an action timestamp before the
event at which the burst could actually be confirmed.

## Runtime State and Offline Utility

The online state is limited to arrived packet timestamps and directions, the
current outgoing burst, an elapsed idle timer, and consumed token budget. It
does not include a website label, a complete trace, or attacker probabilities.

Offline calibration uses complete traces and RF/DF/AWF surrogate margins. It
aggregates evidence by:

```text
(phase, out, duration bin, packet-volume bin) -> action profile utility
```

An action profile contains a dummy dose scale and spacing, plus a bounded
forward delay window. TF and VarCNN do not contribute to the utility table and
are held-out evaluation attackers.

## Strict Time Semantics

For an outgoing burst ending at time bin `b`, DynaPD-RT schedules a timer at:

```text
decision = b + GAP_THRESH + 1
```

When the timer expires, it selects a profile from the utility table. Dummy
packets begin at `decision + 1` or later. Delay applies only to a packet that
arrives after the timer activation and has not been emitted. The implementation
records four per-trace audit totals:

```text
dummy_before_decision
delay_before_activation
delay_after_emission
future_packet_read
```

All four are zero in the current Full-CW record.

## Recorded Full-CW Result

| Traces | RF | DF | TF | AWF | VarCNN | WC | BWO |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 105,634 | 7.01% | 11.41% | 9.87% | 11.96% | 10.18% | **11.96%** | 19.11% |

The calibration interval is `CW[0:96)` and evaluation is `CW[96:105730)`.
The result uses a fixed-prefix 5,000-packet attacker protocol. See
[RESULTS.md](RESULTS.md) for the no-known-real-packet-truncation control.

## Public Files

- [`../streaming_state_machine.py`](../streaming_state_machine.py): online
  state machine.
- [`../causal_event_renderer.py`](../causal_event_renderer.py): explicit
  timestamp-preserving merge of real and dummy packets.
- [`../scripts/build_event_keypoint_utility.py`](../scripts/build_event_keypoint_utility.py):
  offline utility calibration.
- [`../configs/dynapd_rt_event_utility.npy`](../configs/dynapd_rt_event_utility.npy):
  checked-in global utility artifact.
