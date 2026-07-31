# Stage B0: Sequential Budgeted Action Selection

Stage B0 turns Stage A5 additive action rows into executable sequential
padding plans. Each selected action is rendered with `PaddingTemplate` and
`render_batch_variable`, rebuilt as native `2x1800` RF TAM, and re-evaluated by
the fixed RF attacker before the next sequential decision.

## Inputs

- `stage_a_masks_rf/all_masks.npz`: clean TAM, labels, probabilities, and sample
  ids from Stage A.
- `action_results.csv`: candidate additive actions from Stage A5.

## Protocols

- `client_only`: keeps only actions that do not require incoming-side insertion.
- `bidirectional_cooperative`: permits outgoing and incoming insertion actions.

## Methods

- `random`: random static action order.
- `early`: earlier insertion positions first.
- `magnitude`: actions near higher clean TAM magnitude first.
- `static_single_action_efficiency`: static order by Stage A5 single-action
  utility per bandwidth.
- `dynamask_same_sequential`: sequential greedy search over actions inside the
  same DynaMask window.
- `dynamask_causal_sequential`: sequential greedy search over causal shifted
  insertion actions.

The label-free selection utility is:

```text
0.30 * confidence_uncertainty + 0.50 * margin_uncertainty + 0.20 * normalized_entropy
```

Labels are used only for post-hoc accuracy and flip-rate reporting.

## Run

```powershell
python scripts\stage_b_run_sequential_oracle.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --action_table results\stage_a_additive_probe_rf_native_w1800_n96_seed0\action_results.csv --run_name stage_b0_sequential_oracle_rf_native_w1800_n96_seed0 --attacker rf --budgets "0.02,0.05,0.10,0.15" --protocols "client_only,bidirectional_cooperative" --methods "random,early,magnitude,static_single_action_efficiency,dynamask_same_sequential,dynamask_causal_sequential" --max_candidates_per_sample 64 --progress
python scripts\stage_b_summarize_oracle.py --result_dir results\stage_b0_sequential_oracle_rf_native_w1800_n96_seed0
```

## Outputs

- `oracle_filter_counts.csv`: action-count audit after protocol and Pareto
  filtering.
- `oracle_step_results.csv`: selected action sequence and marginal gains.
- `oracle_sample_results.csv`: per-sample, per-budget post-hoc metrics.
- `oracle_summary.csv`: aggregate method and protocol comparison.
- `figures/*.png`: budget curves and marginal gain curves.

## Stage B1 Expanded Oracle

Stage B1 keeps the Stage B0 action table as the primary high-yield pool, then
adds secondary structural actions and exploration actions generated from the
current rendered TAM. The dynamic greedy mode regenerates candidates after each
selected action. Beam search is implemented, but full n=96 beam runs are much
more expensive because every beam expansion goes through the real renderer and
RF model.

```powershell
python scripts\stage_b_run_expanded_oracle.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --action_table results\stage_a_additive_probe_rf_native_w1800_n96_seed0\action_results.csv --run_name stage_b1_expanded_oracle_rf_native_w1800_n96_seed0_greedy --attacker rf --methods "stage_b0_static_efficiency,stage_b0_sequential_causal,expanded_static,expanded_dynamic_greedy" --max_static_candidates 64 --max_dynamic_candidates 20 --max_generated_actions 128 --max_pair_actions 24 --max_steps 12 --progress
python scripts\stage_b_summarize_expanded.py --result_dir results\stage_b1_expanded_oracle_rf_native_w1800_n96_seed0_greedy
```

The Stage B1 objective freezes the original RF prediction class:

```text
0.30 * (p_y0(original) - p_y0(current))
+ 0.50 * original-class margin drop
+ 0.20 * normalized entropy gain
```

Labels remain post-hoc only.

## Stage B2-S Causal Smoothing

Stage B2-S tests keypoint-guided smoothing before continuous budget allocation.
It compares:

- `noncausal_symmetric_oracle`: TAM-space upper bound that redistributes mass
  inside `[t-L, t+L]`.
- `causal_delay_smoothing`: delays real packets only to future bins, with no
  dummy bandwidth.
- `direct_same_position_dummy`: direct keypoint dummy insertion baseline.
- `add_only_future_flattening`: dummy-only future-neighborhood flattening.
- `hybrid_delay_dummy`: causal delay plus dummy future flattening.

```powershell
python scripts\stage_b_run_smoothing_oracle.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --run_name stage_b2s_smoothing_rf_native_w1800_n96_bestgrid_v2 --attacker rf --methods "clean,noncausal_symmetric_oracle,causal_delay_smoothing,direct_same_position_dummy,add_only_future_flattening,hybrid_delay_dummy" --lengths "16,32" --rhos "0.75,1.0" --dummy_budgets "0.02,0.05,0.10" --max_delays "16,32" --progress
python scripts\stage_b_summarize_smoothing.py --result_dir results\stage_b2s_smoothing_rf_native_w1800_n96_bestgrid_v2
```
