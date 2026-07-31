# DMMPv3 运行手册

## 2026-07-14 入口更新

- `scripts\run_defense.py` / `scripts\train_defense.py --stage` 现在生效：`all` 依次执行 Stage 1/2/3，`1` 只生成 Stage 1 artifacts，`2` 复用 `--output_dir\--run_name` 中已有 Stage 1 artifacts，`3` 复用已有 Stage 1/2 artifacts。
- `--stage 2` 和 `--stage 3` 必须提供 `--run_name`，并会在开始前检查前置 artifacts；恢复阶段会读取该 run 的 `run_config.json`，只保留本次 CLI 的 device/progress/stage 设置。
- Stage 3 resume 仍可用 `scripts\resume_stage3.py`；该入口默认保留 hard quality gate。只有显式传入 `--allow_diagnostic_fallback` 时才会保存未过 gate 的诊断 fallback。
- Stage 3 hard quality gate 现在使用 `stage3_max_label_free_attack_pressure`，不是真实标签 accuracy。`selection_attack_pressure` 来自 prefix-only policy pressure，只使用已观察 prefix、policy logits、candidate mask 和攻击模型输出分布；完整 trace 下的 `selection_attack_accuracy` 与 rendered pressure 只作为离线诊断保存。
- `scripts\run_attack_eval.py` 默认复用当前 run 下 `attack_eval\<protocol>\<kind>\*_metrics.json` 和 `*_checkpoint.pt` 缓存；缓存会校验 attack 配置签名和 run artifact 签名，不一致会自动重训。defended dataset 缓存文件名也绑定 run artifact 签名，重跑 Stage 3 或替换 checkpoint 后不会误用旧 defended traces。需要无条件重新训练时使用 `--force_retrain`。真正重新训练后的 fixed/mixed 权重应写入当前指定的 run/output 目录，不应回写历史 checkpoint。
- mixed 入口只负责训练/评估 mixed DF/RF：默认 `--attackers mixed_df,mixed_rf --adaptive_protocol same_user`，会基于当前 run 的 clean + defended traces 重新训练；fixed checkpoint 只能作为初始化或缓存评估器，不能标记为 mixed 最终 checkpoint。

本文档记录当前 DMMPv3 Harness 的标准运行方式。所有命令默认在 `D:\learning\TOR\defence\DMMPv3` 下执行。

如果通过 `conda run` 执行长实验，建议加 `--no-capture-output`，否则部分 conda 版本会缓存 stdout/stderr，导致训练中看不到实时阶段日志。

## 运行前检查

```powershell
cd D:\learning\TOR\defence\DMMPv3
python scripts\verify_project.py
```

该检查会确认：

- 必要代码、文档和目录存在；
- `DefenseConfig.version` 默认是 `v3`；
- `DefenseConfig.output_dir` 默认是 `D:\learning\TOR\defence\DMMPv3\results`。

## 查看参数

```powershell
conda run --no-capture-output -n llm python scripts\train_defense.py --help
conda run --no-capture-output -n llm python scripts\run_attack_eval.py --help
```

## 正式防御训练

示例：

```powershell
conda run --no-capture-output -n llm python scripts\train_defense.py --run_name dmmpv3_cw_seed0_b30_formal
```

默认行为：

- `--version v3`
- `--data_root D:\learning\TOR\datasets\CW`
- `--output_dir D:\learning\TOR\defence\DMMPv3\results`
- `--budgets 0.30`
- `--candidate_mode executable`
- `--probe_attacker both`
- `--guidance_attackers both`
- `--guidance_label_mode pseudo`

正式 run 会创建：

```text
D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

如果目标 run directory 已存在且非空，DMMPv3 会拒绝覆盖。

## fixed DF/RF 评测

示例：

```powershell
conda run --no-capture-output -n llm python scripts\evaluate_fixed.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

该入口会委托 `scripts\run_attack_eval.py`，默认：

- `--attackers fixed_df,fixed_rf`
- `--adaptive_protocol fixed`

fixed DF/RF checkpoint 可以复用已验证候选，但它们的路径和兼容性前提必须写入结果摘要。

## mixed DF/RF 训练与评测

示例：

```powershell
conda run --no-capture-output -n llm python scripts\train_mixed_attackers.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
conda run --no-capture-output -n llm python scripts\evaluate_mixed.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

这两个入口都会委托 `scripts\run_attack_eval.py`，默认：

- `--attackers mixed_df,mixed_rf`
- `--adaptive_protocol same_user`

mixed attacker 必须随当前 DMMPv3 defended traces 重新训练。不能把 fixed checkpoint 直接当成 mixed 最终模型。

## Stage 3 恢复

如果已有 run 完成 Stage 1/2，可以只恢复 Stage 3：

```powershell
conda run --no-capture-output -n llm python scripts\resume_stage3.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

该命令会复用已有 Stage 1/2 checkpoint 和 profile 文件，重新执行 Stage 3 refinement/selection。

## 方向修正与只下行消融

DMMPv3 主线默认不做固定方向修正，让 DF/RF guidance、candidate utility 和部署约束决定 dummy 分布：

```text
--direction_target none
--direction_correction_strength 0
--min_incoming_dummy_share 0
```

偏好与用户画像条件默认降级：`--no-condition_profile_mask --no-condition_selected_mask --no-condition_preference_map --no-condition_preference_weights`。随机偏好仍可通过低权重 prior 和 `--preference_attack_gate` 提供多样性，但只有在不提高 surrogate attack risk 时才作为 soft loss 参与训练。

如果需要做“只插入下行虚拟包”的对照实验，可以新建一个 run：

```powershell
conda run --no-capture-output -n llm python scripts\train_defense.py --run_name dmmpv3_cw_seed0_b30_incoming_only --direction_target incoming --direction_correction_strength 1.0 --min_incoming_dummy_share 1.0
```

该设置用于消融分析，不建议直接替代正式主线默认值。它仍然受 candidate mask、allowed mask 和 bandwidth budget 约束。

## Label-dependent guidance 消融

默认 `--guidance_label_mode pseudo` 保持 Stage 2/3 的 label-free surrogate pseudo-label guidance。若要评估真实网站标签能给防御指导带来多少上界提升，可以新建一个独立 run：

```powershell
conda run --no-capture-output -n llm python scripts\train_defense.py --run_name dmmpv3_cw_seed0_b30_true_label_guidance --guidance_label_mode true
```

该开关只改变 Stage 2 diffusion guidance 和 Stage 3 guided DDIM/refinement 的 target，不改变 Stage 1、encoder、candidate scorer 或 profile/preference 条件逻辑。

成对全量训练和 fixed DF/RF 评测可以直接运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_label_guidance_full.ps1 -Seed 0 -Budgets 0.30 -Mode both
```

默认会生成同 seed、同 budget 的两个 run：`pseudo_label_free` 主线和 `true_label_oracle` 上界消融，并只运行 fixed DF/RF 评测，不运行 mixed 训练/评测。该脚本不设置 `max_samples`、`max_classes`、轻量 epoch 或 smoke-only 参数。

如果脚本因窗口重连、中断或机器重启后需要继续执行，直接重跑同一命令即可。脚本会检查目标 run 目录：Stage 3 已完成时跳过 defense 并补 fixed DF/RF 评测；只有 Stage 1 或 Stage 2 完成时分别用 `--stage 2` / `--stage 3` 接续；不可恢复的半成品目录会直接报错而不是覆盖。每次非 dry-run 执行都会写入 `logs\<run_prefix>_*_label_guidance_*.log` transcript 和 invocation manifest。

为避免 `conda run` 阻塞，脚本默认直接调用 `llm` 环境的 `python.exe`。如环境不在默认位置，可显式传入：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_label_guidance_full.ps1 -Seed 0 -Budgets 0.30 -Mode both -PythonExe D:\Miniconda3\envs\llm\python.exe
```

## Guidance sweep

在已有 run 上做 guidance sweep：

```powershell
conda run --no-capture-output -n llm python scripts\sweep_guidance.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

输出写入该 run 的：

```text
stage3_inference_margin_sweep
```

## Stage A 关键点发现

Stage A 是独立的可审查闭环：先在冻结攻击器上对 clean TAM 样本做 deletion-style DynaMask，再对 mask 做 PCA+KMeans 聚类。所有输出仍写入 `results\<run_name>`。

原生 RF/TAM quick audit 示例：

```powershell
conda run --no-capture-output -n llm python scripts\stage_a_run_dyn_mask.py --attacker rf --run_name stage_a_rf_native_w1800_n96_s60_seed0 --max_samples 96 --steps 60 --width 1800 --batch_size 4 --plot_limit 16 --device auto
conda run --no-capture-output -n llm python scripts\stage_a_cluster_masks.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --k_values "4,5,6,7,8,9,10,12" --pca_components 16 --representatives_per_cluster 4 --seed 0
conda run --no-capture-output -n llm python scripts\stage_a_summarize_results.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --cluster_result results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_clustering\cluster_result.npz
```

如需人工可解释性优先的 K=4 prototype，可另存审核版本：

```powershell
conda run --no-capture-output -n llm python scripts\stage_a_cluster_masks.py --archive results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_masks_rf\all_masks.npz --output_dir results\stage_a_rf_native_w1800_n96_s60_seed0\stage_a_clustering_k4_review --k_values "4" --pca_components 16 --representatives_per_cluster 4 --seed 0
```

`W=200` 可用于快速 smoke，但若使用现有 fixed RF checkpoint，正式解释应优先使用 `W=1800`，因为该 checkpoint 原生输入就是 `[2, 1800]` TAM。

## 训练期 loss 超参轻量搜索

如果需要筛选 `preference_weight`、`defense_soft_objective_scale`、`defense_soft_utility_weight`、`prefix_hidden_align_weight` 等训练期 loss 权重，不建议直接跑全量数据。先用轻量网格搜索生成短 run：

```powershell
conda run --no-capture-output -n llm python scripts\sweep_loss_hparams.py --sweep_name loss_probe_seed0 --device auto
```

默认轻量设置会限制类别数、样本数、surrogate/encoder epoch、diffusion steps、Pareto samples 和 fixed-probe 样本，并关闭 Stage 3 hard gate 中断。由于 strong surrogate 仍是短训轻量 probe，轻量搜索默认把 Stage 1 surrogate validation gate 从正式训练的 `0.85` 放宽到 `0.65`；该设置只用于筛选候选超参，正式全量实验仍应保留严格门槛。每个 trial 是一个独立 DMMPv3 run，结果汇总写入：

如果同名 sweep 的某个 trial 之前失败并留下了非空目录，脚本默认不会覆盖旧结果，而是自动使用 `<trial>_retryNNN` 新 run 名继续执行。

```text
results\hparam_sweeps\<sweep_name>\results_ranked.csv
results\hparam_sweeps\<sweep_name>\results_ranked.json
```

排序优先级是 label-free attack pressure、rendered RF diagnostic、diagnostic conservative accuracy 和 visible bandwidth。推荐只把 ranked 前 3-5 个配置放大到正式数据集重跑；轻量搜索结果不能直接作为最终防御结论。

## 轻量验证

验证 DF/RF surrogate 的梯度连通性：

```powershell
conda run --no-capture-output -n llm python scripts\validate_strong_surrogates.py
```

验证已有 run 的阶段性正确性：

```powershell
conda run --no-capture-output -n llm python scripts\validate_v4.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

## 禁止事项

- 不要把 DMMPv3 新结果写入 `D:\learning\TOR\results`。
- 不要覆盖 `defence\DMMP` 或 `defence\DMMP2` 的历史结果。
- 不要把旧工程 defended datasets、encoder、diffusion、stage outputs 当成 DMMPv3 正式结果。
- 不要把 fixed checkpoint 标记为 mixed 最终 checkpoint。
