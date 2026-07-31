# Stage A: Counterfactual TAM Keypoint Discovery

This folder documents the runnable Stage A loop for DMMPv3. Code lives in
`dmmp/stage_a`, entry scripts live in `scripts`, and new run artifacts are saved
under `results/<run_name>`.

## Outputs

- `stage_a_masks_<attacker>/all_masks.npz`: TAM, deletion keypoint masks,
  baseline TAM, deleted-keypoint TAM, keypoint-only TAM, prediction
  distributions, JS divergence, entropy gain, and top-1 confidence drop.
- `stage_a_figures/sample_masks/*.png`: per-sample audit figures.
- `faithfulness_metrics.csv` / `aopc_summary.csv`: equal-budget deletion and
  keep-only validation over DynaMask, random, random block, TAM magnitude, and
  early-position masks.
- `dynamask_sample_metrics.npz`: per-sample necessity and sufficiency metrics
  for DynaMask top-r masks, used by cluster audit cards.
- `stage_a_clustering/cluster_result.npz`: PCA features, KMeans labels,
  prototype masks, cluster sizes, class-cluster matrix, and distances.
- `stage_a_clustering/figures/*.png`: K-selection curves, prototype heatmaps,
  PCA scatter, cluster sizes, class-cluster matrix, and representative samples.
- `stage_a_cluster_stability/*`: sum-pooled `2x200`, per-sample L1 normalized
  shape masks, K=3..12 bootstrap stability metrics, macro K=4 clusters, and
  an automatically selected fine-grained clustering.

## Quick RF/TAM Run

```powershell
python scripts\stage_a_run_dyn_mask.py --attacker rf --max_samples 32 --steps 120 --width 200 --progress
python scripts\stage_a_cluster_masks.py --archive results\<run_name>\stage_a_masks_rf\all_masks.npz
```

RF is the most faithful first target because the existing fixed RF checkpoint
already consumes TAM. DF support is included through a differentiable TAM-to-DF
adapter, but it should be treated as an approximation until a native TAM-DF
checkpoint is available.

## Formal Scaling

After visual inspection passes, prefer native RF width (`--width 1800`) and a
stratified validation subset:

```powershell
python scripts\stage_a_run_dyn_mask.py --attacker rf --run_name stage_a_rf_native_w1800_perclass10_seed0 --split val --samples_per_class 10 --max_samples 0 --steps 300 --width 1800 --batch_size 4 --progress
python scripts\stage_a_faithfulness.py --archive results\stage_a_rf_native_w1800_perclass10_seed0\stage_a_masks_rf\all_masks.npz --run_name stage_a_rf_native_w1800_perclass10_seed0_faithfulness --attacker rf
python scripts\stage_a_cluster_stability.py --archive results\stage_a_rf_native_w1800_perclass10_seed0\stage_a_masks_rf\all_masks.npz --output_dir results\stage_a_rf_native_w1800_perclass10_seed0\stage_a_cluster_stability --faithfulness_sample_npz results\stage_a_rf_native_w1800_perclass10_seed0_faithfulness\dynamask_sample_metrics.npz
```

`configs/stage_a/native_rf_per_class10.json` records the 95 x 10 sampling
target. Labels are used only for stratified sampling and post-hoc
class-cluster statistics, not for mask optimization.

## Stage A5 Additive Probing

After deletion faithfulness passes, run additive intervention probing before
using Stage A masks as Stage B defense guidance:

```powershell
python scripts\stage_a_additive_probe.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --faithfulness_sample_npz results\stage_a_validation_rf_native_w1800_n96_s60_seed0\dynamask_sample_metrics.npz --run_name stage_a_additive_probe_rf_native_w1800_n96_seed0 --attacker rf --renderer_top_actions 1 --progress
python scripts\stage_a_plot_additive_results.py --result_dir results\stage_a_additive_probe_rf_native_w1800_n96_seed0
```

The additive probe emits sanity audits, candidate keypoint windows,
per-action dose/offset/direction responses, equal-budget insertion baselines,
and sparse maps for additive efficiency, minimum effective budget, best causal
offset, and keypoint-to-insertion mapping. `action_results.csv` and
`budget_results.csv` are TAM-space screening results. `renderer_top_action_results.csv`
checks the top TAM-space action per sample with `PaddingTemplate` +
`render_batch_variable`, rebuilds native `2x1800` TAM, and re-evaluates RF.

Rows with `requires_incoming_capability=1` assume incoming dummy insertion is
deployable through a server/relay-side mechanism; keep them separate from
client-only Stage B actions.
