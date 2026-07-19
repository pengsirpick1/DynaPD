# Adaptive re-defense DF audit

Use `scripts\adaptive_redefense_df.py` to test a stronger loop than `few_shot_adaptive_df.py`.

Protocol:

1. Train a full clean-base DF attacker `A0`.
2. Generate old defended support with the existing DMMPv3 deployment policy.
3. Fine-tune `A0` only on the old defended support, producing adapted attacker `A1`.
4. Generate multiple new defense candidates after `A1` exists.
5. Select the new candidate that minimizes `A1` accuracy on the defense-selection split, validation by default.
6. Generate a full fresh defended test set with that selected new defense strategy.
7. Evaluate `A1` on both the baseline fresh defended test and the adaptive re-defense fresh defended test.

Default command:

```powershell
python scripts\adaptive_redefense_df.py `
  --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name> `
  --few_shot_per_class 20
```

Important semantics:

- Only the old support set is few-shot.
- Base DF training uses the full selected clean train/val splits by default.
- Final fresh testing uses the full selected test split by default.
- Re-defense selection uses validation by default, not test.
- This is selection-level adaptive re-defense: it resamples/selects new deployment strategies using `A1` as a probe. It does not retrain the diffusion model weights.

Key outputs:

- `adapted_baseline_fresh_defended_accuracy`: `A1` on the baseline fresh deployment.
- `adaptive_redefense_selection_accuracy`: `A1` on the selected candidate during validation selection.
- `adaptive_redefense_fresh_defended_accuracy`: `A1` on the new selected defense strategy's full fresh test.
- `adaptive_redefense_df_candidates.csv`: all candidate scores.
- `adaptive_redefense_selected_policy.json`: selected new defense strategy metadata.
