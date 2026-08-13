# Reproducibility Notes

## Required Local Inputs

This repository deliberately excludes CW traces, clean attacker checkpoints,
WFlib source, generated defended traces, and logs. Five-model evaluation needs
signed-timestamp CW traces and compatible RF, DF, TF, AWF, and VarCNN
checkpoints.

## DynaPD-RT Protocol

1. Build `configs/dynapd_rt_event_utility.npy` with
   `scripts/build_event_keypoint_utility.py` on a calibration interval disjoint
   from the final evaluation interval. It aggregates RF/DF/AWF gains by
   `(phase, out, burst-duration-bin, burst-volume-bin, action-profile)`.
2. Call `streaming_state_machine.defend_stream` as packets arrive in timestamp
   order. An outgoing server-to-client burst schedules a timer at
   `burst_end + GAP_THRESH + 1`.
3. At timer expiration, select an action profile from the aggregate utility
   table. Dummy starts strictly after that decision. A delay rule can affect
   only a packet that arrives after activation and has not been emitted.
4. Export traces with `scripts/run_dynapd_rt_eval.py` or
   `scripts/run_dynapd_rt_fullcw.py`. Both use repository-relative paths and
   support `--data`, `--utility`, `--workers`, and `--output-dir`.
5. Evaluate with timestamp-preserving traces; VarCNN must use DT2 (direction
   plus consecutive absolute timestamp differences).
6. Require all causality audit totals to be zero:
   `dummy_before_decision`, `delay_before_activation`,
   `delay_after_emission`, and `future_packet_read`.

The checked-in utility artifact is derived from a 96-trace global calibration
interval. It contains aggregate thresholds and gains only, never website
labels, raw traces, or attacker weights.

## Evaluation Boundary

CW data in this project is represented by fixed 5,000-packet traces. RF, DF,
TF, AWF, and VarCNN use their native fixed-prefix preprocessing. Defended
physical traces retain all original packets, but a classifier may observe only
the first 5,000 packets. Report this fixed-prefix threat model explicitly.

For the published Full-CW result, also report the no-known-real-truncation
control subset from `final_5model_no_real_trunc.json` when the corresponding
physical audit is available.

## Evidence Files

- `reproducibility/dynapd_rt_profiles_fullcw/`: full-CW configuration,
  output accounting, physical-trace audit, and five-model performance record.
- `docs/RESULTS.md`: current reportable DynaPD-RT protocol and results.
