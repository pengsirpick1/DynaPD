# Reproducibility Notes

## Required private/local inputs

The public repository does not distribute the following artifacts:

- Closed-world CW signed-timestamp traces.
- Clean attacker checkpoints for RF, DF, TF, AWF and VarCNN.
- WFlib-compatible model definitions and preprocessing code.
- Generated defended traces, teacher record shards and trained policy weights.

The offline and five-model RT evaluators expect a local `wflib_copy/` directory
or an equivalent WFlib installation. Set dataset and checkpoint paths locally;
do not commit them.

## Offline protocol

1. Train or provide clean surrogate attackers.
2. Build the RF keypoint archive and policy split.
3. Run the E2b multi-surrogate oracle using RF/DF/AWF weights 0.80/0.10/0.10.
4. Export timing-preserving defended traces for any evaluation requiring RF or
   VarCNN. A sign-only export is valid for direction-only attackers but not for
   timing-sensitive RF/VarCNN evaluation.

The primary entry point is
`scripts/stage_b_run_ensemble_oracle_e2b_completion.py`.

## RT protocol

1. Use `scripts/build_event_keypoint_utility.py` to aggregate offline
   RF/DF/AWF gain by `(phase, direction, burst-duration-bin, burst-volume-bin,
   allocation-scale)`. The checked-in `configs/dynapd_rt_event_utility.npy`
   is the compact calibration artifact used by the default controller.
2. Run `streaming_state_machine.defend_stream` packet-by-packet. Budget state
   accumulates over all real packets; burst detection and event recognition use
   only the positive download/server-to-client direction under the CW encoding.
3. Use `streaming_allcw_mp.py` or `streaming_allcw_bw_sweep.py` for a full-CW
   measurement. `streaming_state_machine_phase_baseline.py` is retained for
   the phase-only ablation; use `random_streaming_baseline_bw_sweep.py` for
   matched causal random controls.

## Causality checks

The deployment result is `tail0`. Its action is emitted only when a positive
direction burst has ended and it delays only packets with timestamps at or
before the decision bin. The tracked full-CW manifest reports zero future
packet accesses. `tail1` should be used only as an end-of-trace ablation.

## Result files

- `reproducibility/streaming_allcw/manifest.json`: all-CW RT main result.
- `reproducibility/streaming_allcw_bw_sweep/manifest.json`: bandwidth sweep.
- `reproducibility/random_streaming_baselines/manifest_partial.json`: matched
  causal random controls currently completed.
- `reproducibility/dynapd_rt_fullcw_manifest.json`: frozen earlier RT manifest.
- `reproducibility/event_keypoint_rt_fullcw/`: full-CW event-keypoint RT
  generation manifest, causality summary, and five-model final evaluation.

All reported accuracy values are lower-is-better WF attacker accuracies.
