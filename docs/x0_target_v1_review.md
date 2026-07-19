# x0 Target-Policy v1 Review

Role: Agent B / Review Agent

Scope reviewed:

- `dmmp/target_policy/*.py`
- `configs/x0_target_diffusion_v1.yaml`
- `scripts/build_target_policy_pool_v1.py`
- `scripts/validate_target_policy_pool_v1.py`
- `scripts/train_target_diffusion_v1.py`
- `tests/target_policy_v1/test_core.py`

## Verdict

Re-review verdict: approved.

The four previous must-fix items have been addressed: build/train now use timestamped defaults and refuse explicit output overwrites without `--overwrite`; validation report writes also refuse overwrite by default; training now consumes YAML `epochs`/`batch_size`, `batch_size=auto`, projector `num_layers`, `use_amp`, `beta_schedule` validation, and structural/fusion/smooth loss weights; target selection now prioritizes deployable candidates and counts fallback from deployable shortage; heuristic DF/RF-like scores are renamed to `proxy_score_*` and metadata records `score_source=heuristic_proxy_not_df_rf_teacher`.

The lightweight invariants still pass: target-pool arrays do not contain label-like keys, the builder uses the train split, strict validation reports exact budgets and zero mask violations, renderer order and masked CLR behavior are covered by tests, and train smoke performs a real backward/optimizer step including the added structural/fusion/smooth losses.

## Checks

1. Defense/target pool path reads or returns true labels: no direct true-label leakage found in the pool arrays or training loader. `build_target_policy_pool_v1.py` reads `labels` through `load_cw_data` for stratified splitting, then deletes the local variable before generation. `TargetPolicyPool.load_training_arrays()` filters label-like keys, and both existing and fresh smoke pools reported `leaked_label_keys: []`. Note: `clean_index` and `trace_id` are still persisted; they are not labels, but can be joined back to the source dataset if distributed with label access.

2. Target pool train split only: pass. The builder uses `splits["train"]` and records `metadata["split"] == "train"`. Fresh smoke build also reported `split: train`.

3. Budget largest-remainder exactness: pass in tests and smoke validation. `largest_remainder_rounding()` produces exact totals under tested masks; fresh smoke pool had `budget_violation_count: 0`.

4. Mask violation zero: pass in tests and smoke validation. Existing smoke pool and fresh smoke pool both reported `mask_violation_count: 0` and `negative_count: 0`.

5. Renderer true packet order: pass at unit-test level. `test_renderer_preserves_real_packet_order` checks that real packets retained by `render_trace_variable()` match the nonzero clean trace order.

6. x0* masked CLR forward/reverse: pass at unit-test and smoke-array level. The fresh pool round-trip check reconstructed counts exactly for all 12 smoke rows, with budget error 0 and mask violation 0.

7. Loss backpropagation: pass in smoke. `train_target_diffusion_v1.py` calls `loss.backward()` and `optimizer.step()`, and a 1-step CPU train smoke completed with finite loss.

8. Config external loading: pass for the reviewed v1 scope. Scripts load `configs/x0_target_diffusion_v1.yaml`; train CLI `epochs` and `batch_size` are optional and fall back to YAML, `batch_size=auto` is implemented, projector `num_layers` is wired, `use_amp` is wired for CUDA, non-linear `beta_schedule` is rejected explicitly, and `lambda_struct`/`lambda_fusion`/`lambda_smooth` contribute to the train loss. The proxy scoring weights and min-gain fields are now used in `_proxy_quality()`.

9. Old checkpoint/result overwrite risk: pass. Build and train default to timestamped directories, and explicit output directories refuse existing sentinel artifacts unless `--overwrite` is passed. `validate_target_policy_pool_v1.py --write_report` also refuses to overwrite an existing report unless `--overwrite` is passed.

10. Design consistency: acceptable for v1. The heuristic scores are now explicitly named `proxy_score_df`, `proxy_score_rf`, and `proxy_score_attack`, and pool metadata identifies the score source as `heuristic_proxy_not_df_rf_teacher`. `select_targets()` now prefers deployable candidates and computes fallback count from deployable shortage. The consecutive dummy-run proxy was changed to use maximum per-slot stack rather than adjacent occupied-bin runs, avoiding an overly strict bin-level interpretation.

## Tests Run

- `conda run -n llm python -m compileall dmmp\target_policy scripts\build_target_policy_pool_v1.py scripts\validate_target_policy_pool_v1.py scripts\train_target_diffusion_v1.py tests\target_policy_v1`
  - Result: OK.

- `conda run -n llm python -m unittest discover -s tests -p "test*.py"`
  - Result: 7 tests OK.

- `conda run -n llm python scripts\validate_target_policy_pool_v1.py --pool_dir results\20260717_185916_target_policy_pool_v1_smoke --strict`
  - Result: valid, rows 24, mask violations 0, budget violations 0, leaked label keys empty.

- `conda run -n llm python scripts\train_target_diffusion_v1.py --pool_dir results\20260717_185916_target_policy_pool_v1_smoke --config configs\x0_target_diffusion_v1.yaml --output_dir C:\Users\Pengtor\AppData\Local\Temp\dmmpv3_x0_review_train_recheck_20260717 --epochs 1 --batch_size 4 --max_steps 1 --device cpu --smoke`
  - Result: 1 step completed, finite loss, temp checkpoint written; logged `eps`, `x0`, `alloc`, `effect`, `family`, `primitive`, `struct`, `fusion`, and `smooth`.

- Pool key/metadata inspection on `results\20260717_185916_target_policy_pool_v1_smoke`
  - Result: `score_source=heuristic_proxy_not_df_rf_teacher`, `fallback_count=0`, proxy keys present, legacy `score_df`/`score_rf`/`score_attack` keys absent, label-like keys absent.

- Overwrite guard checks
  - Build refused an existing pool output directory without `--overwrite`; direct env Python reported `LASTEXIT=1`.
  - Train refused an existing training output directory without `--overwrite`.
  - Validate refused an existing `--write_report` path without `--overwrite`.

## Must-Fix Status

1. Overwrite protection: resolved.

2. Config wiring: resolved for v1 training and proxy scoring.

3. Deployable target selection: resolved.

4. Scoring/design naming mismatch: resolved by proxy naming and `score_source` metadata.

APPROVED
