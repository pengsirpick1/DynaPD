# Few-shot adaptive DF audit

Use `scripts\few_shot_adaptive_df.py` to check whether a clean-trained DF attacker can adapt from a small old defended support set to a fresh defended deployment.

Default formal protocol:

```powershell
conda run --no-capture-output -n llm python scripts\few_shot_adaptive_df.py `
  --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name> `
  --force_retrain_base `
  --few_shot_per_class 20
```

Protocol:

- Train or load a full clean train/val DF model.
- Generate an old defended support set and a disjoint fresh defended eval set.
- Fine-tune only on the old defended support set.
- Evaluate the same base model before fine-tuning and the adapted model after fine-tuning on the full fresh defended test set.

The support sampler enforces `few_shot_per_class` unique clean traces per class and uses only one defended visit from each clean trace. This prevents deployment repeats from being counted as independent few-shot flows.

By default, only the old defended support set is few-shot. `max_classes=0`, `max_samples=0`, and `fresh_eval_per_class=0` mean all selected run classes, full clean train/val for the base DF, and the full selected test split for fresh defended evaluation. Use `--fresh_eval_per_class`, `--base_max_train_traces`, `--base_max_val_traces`, or `--fresh_max_test_traces` only for explicit debug probes.

Key fields:

- `clean_base_accuracy`: clean DF quality before adaptation.
- `base_train_full_selected_split`, `base_val_full_selected_split`, and `fresh_eval_full_selected_test`: should be true/1 for the default formal protocol.
- `before_finetune_old_support_accuracy` and `after_finetune_old_support_accuracy`: whether the attacker memorized or adapted to the old support set.
- `before_finetune_fresh_defended_accuracy` and `after_finetune_fresh_defended_accuracy`: the main robustness signal.
- `unique_clean_trace_class_counts`: verifies unique clean trace counts for support/eval.
- `selected_clean_trace_overlap`: must be `0` for old support versus fresh eval.

Outputs are written to `<run_dir>\attack_eval\few_shot_adaptive_df*`.
