# Reproducibility Notes

## Required local inputs

This repository deliberately excludes CW traces, clean attacker checkpoints,
WFlib source, generated defended traces, and logs. A five-model evaluation
requires signed-timestamp CW traces plus compatible RF, DF, TF, AWF, and
VarCNN checkpoints.

## Timeout event-keypoint RT protocol

1. Use `scripts/build_event_keypoint_utility.py` with a calibration interval
   disjoint from the evaluation interval. It aggregates RF/DF/AWF gains by
   `(phase, out, burst-duration-bin, burst-volume-bin, allocation-scale)`.
2. Run `streaming_state_machine.defend_stream` packet-by-packet. A positive
   server-to-client burst creates a timer deadline at
   `burst_end + GAP_THRESH + 1`.
3. At that timer deadline, select an action from the aggregate utility table.
   Dummy starts after the deadline. A delay rule can affect only packets that
   arrive after the action is active and before they are emitted.
4. Generate exports with `scripts/run_timeout_event_keypoint_eval.py`, then
   evaluate using the project five-model evaluator with timestamp-preserving
   traces and VarCNN DT2 features.
5. Require all four audit totals to be zero:
   `dummy_before_decision`, `delay_before_activation`,
   `delay_after_emission`, and `future_packet_read`.

The checked-in `configs/dynapd_rt_event_utility_timeout.npy` is the compact
utility artifact from a 96-trace calibration interval. It contains aggregate
thresholds and gains only, not website labels, raw traces, or model weights.

## Result files

- `reproducibility/event_keypoint_timeout_fullcw/generation_summary.json`:
  controller configuration, measured BWO, and hard causality audit totals.
- `reproducibility/event_keypoint_timeout_fullcw/final_5model_evaluation.json`:
  final timestamp-preserving five-model evaluation.

## Historical code

Older streaming scripts and manifests remain in the Git history. They use an
action-time convention that has been superseded by the timeout protocol above;
do not treat their old “future packet” audits as action-time certification.
