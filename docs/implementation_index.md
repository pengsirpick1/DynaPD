# DMMPv3 实现索引

## 2026-07-14 实现更新

- `dmmp\diffusion\profile_pipeline.py::run_v4_pipeline` 现在按 `DefenseConfig.stage` 执行 `1`、`2`、`3`、`all`；Stage 2/3 通过 `_require_artifacts` 检查前置产物，Stage 3 复用 `stage2_user_diffusion` 下的 encoder/diffusion/profile artifacts。
- `dmmp\guidance\strong_surrogates.py::resolve_guidance_positions` 为 Stage 2/3 defense guidance 选择 target：默认 `guidance_label_mode=pseudo` 时从 frozen DF/RF observed-prefix prediction 得到 label-free pseudo label position；显式 `true` 时使用真实标签映射到 frozen surrogate class position，作为 oracle / upper-bound 消融。
- `dmmp\diffusion\profile_pipeline.py::train_v4_diffusion` 和 `generate_v4_ragged_dataset` 保存真实 `y` 用于 metadata/评测；默认主线传入 guided DDIM、continuous refinement 和 risk guard 的 target 来自 observed-prefix pseudo labels，`guidance_label_mode=true` 仅用于 label-dependent 消融。
- `dmmp\evaluation\profile_attacks.py::run_v4_attack_evaluation` 默认复用当前 run 的 attack eval 缓存，`AttackConfig.force_retrain` / `scripts\run_attack_eval.py --force_retrain` 可强制重训。

本文档按当前 `D:\learning\TOR\defence\DMMPv3` 中已经迁移完成的代码生成，用来回答“代码放在哪里、入口调用哪一层、结果写到哪里”。

## 当前结论

- DMMP2 最新强 DF/RF guidance 流程已经迁移到 DMMPv3 工程内。
- DMMPv3 的 Python package 位于 `dmmp\`。
- 可执行脚本位于 `scripts\`。
- `dmmp\` 顶层只保留 package 入口，业务代码已经按类别放入子目录。
- 新结果默认写入 `D:\learning\TOR\defence\DMMPv3\results`。
- 本次迁移没有启动正式训练，也没有移动或覆盖历史 checkpoint。

## 可执行入口

| 脚本 | 当前作用 | 主要调用模块 |
|---|---|---|
| `scripts\train_defense.py` | DMMPv3 防御训练主入口。 | `dmmp.diffusion.profile_pipeline.run_v4_pipeline` |
| `scripts\run_defense.py` | 与 `train_defense.py` 同形状的防御运行入口。 | `dmmp.diffusion.profile_pipeline.run_v4_pipeline` |
| `scripts\run_attack_eval.py` | fixed/mixed attack evaluation 主入口。 | `dmmp.evaluation.profile_attacks.run_v4_attack_evaluation` |
| `scripts\evaluate_fixed.py` | fixed DF/RF 评测包装入口。 | `scripts\run_attack_eval.py`，默认 `fixed_df,fixed_rf` |
| `scripts\train_mixed_attackers.py` | mixed DF/RF 训练与评测包装入口。 | `scripts\run_attack_eval.py`，默认 `mixed_df,mixed_rf` |
| `scripts\evaluate_mixed.py` | mixed DF/RF 评测包装入口。 | `scripts\run_attack_eval.py`，默认 `mixed_df,mixed_rf` |
| `scripts\resume_stage3.py` | 从已有 Stage 1/2 run 恢复 Stage 3。 | `dmmp.diffusion.profile_pipeline.run_v4_stage3` |
| `scripts\sweep_guidance.py` | 在已有 checkpoint 上做 guidance sweep。 | `dmmp.diffusion.profile_pipeline.generate_v4_ragged_dataset` |
| `scripts\validate_strong_surrogates.py` | 快速验证 DF/RF surrogate 梯度连通性。 | `dmmp.guidance.strong_surrogates` |
| `scripts\validate_v4.py` | 验证迁移流程的阶段性正确性。 | `dmmp.diffusion.profile_pipeline` |
| `scripts\verify_project.py` | 检查 Harness 结构和默认结果目录。 | `dmmp.utils.config.DefenseConfig` |
| `scripts\stage_a_run_dyn_mask.py` | Stage A：在冻结 DF/RF 上为 TAM 样本生成 deletion-style DynaMask 关键点图。 | `dmmp.stage_a.dyn_mask`、`dmmp.stage_a.modeling` |
| `scripts\stage_a_cluster_masks.py` | Stage A：对关键点图做 PCA+KMeans 聚类并输出 prototype 与审核图。 | `dmmp.stage_a.clustering`、`dmmp.stage_a.viz` |
| `scripts\stage_a_summarize_results.py` | Stage A：汇总 mask、扰动效果和簇级解释指标。 | `dmmp.stage_a.clustering` |
| `scripts\stage_a_train_mask_predictor.py` | Stage A 可选扩展：用 teacher masks 训练轻量 TAM mask predictor。 | `dmmp.stage_a.student` |

## Package 分层

| 子目录 | 文件 | 职责 |
|---|---|---|
| `dmmp\data` | `cw.py` | CW 数据加载、memmap `.npz` 读取、stratified split。 |
| `dmmp\encoders` | `prefix.py` | prefix condition、early causal region、patch count 特征。 |
| `dmmp\encoders` | `leakage.py` | 多视角 prefix leakage profile。 |
| `dmmp\encoders` | `condition_encoders.py` | leakage-aware condition encoder 与监督损失。 |
| `dmmp\diffusion` | `models.py` | TopKLeakageEncoder、policy diffusion 模型。 |
| `dmmp\diffusion` | `policy.py` | policy prior、模板概率和辅助策略。 |
| `dmmp\diffusion` | `profile_pipeline.py` | 当前 DMMPv3 主流程：candidate、encoder、guided diffusion、Stage 3。 |
| `dmmp\diffusion` | `pipeline.py` | 较轻量的 random-preference pipeline，保留作兼容与参考。 |
| `dmmp\guidance` | `candidate_scorer.py` | executable candidate cells、allowed masks、utility probe。 |
| `dmmp\guidance` | `diffusion_guidance.py` | guided DDIM、continuous refinement、risk guard。 |
| `dmmp\guidance` | `strong_surrogates.py` | ProjectDF/RF surrogate ensemble 和防御效用。 |
| `dmmp\projection` | `padding.py` | budget projection、rounding、template、fixed/variable renderer。 |
| `dmmp\renderer` | `__init__.py` | renderer-facing 导出，当前复用 `projection.padding` 的实现。 |
| `dmmp\constraints` | `preferences.py` | preference pool、random mixer、canonical preference。 |
| `dmmp\constraints` | `combination_catalogue.py` | pair/triple preference catalogue。 |
| `dmmp\constraints` | `user_profiles.py` | private user profile、visit selection、profile overlap。 |
| `dmmp\evaluation` | `attack_models.py` | ProjectDF、ProjectRF、DF/RF 输入适配。 |
| `dmmp\evaluation` | `attacks.py` | 非 profile-aware attack evaluation 兼容逻辑。 |
| `dmmp\evaluation` | `profile_attacks.py` | 当前 DMMPv3 fixed/mixed profile-aware attack evaluation。 |
| `dmmp\utils` | `config.py` | `DefenseConfig`、`AttackConfig` 和 CSV 参数解析。 |
| `dmmp\utils` | `common.py` | logging、seed、device、JSON/CSV/NPZ 写入工具。 |
| `dmmp\losses` | `__init__.py` | 预留给后续 loss 抽取。 |
| `dmmp\stage_a` | `dyn_mask.py`、`tam.py`、`modeling.py`、`clustering.py`、`viz.py`、`student.py` | Counterfactual TAM keypoint discovery：TAM 载入、冻结攻击器适配、DynaMask 优化、聚类、可视化和 teacher-student 预留。 |

## 阶段到代码的对应关系

| 阶段 | 内容 | 主要文件 |
|---|---|---|
| 数据划分 | 读取 CW、生成 seed 0 split。 | `dmmp\data\cw.py` |
| Stage 1 | prefix leakage、candidate cells、allowed masks。 | `dmmp\guidance\candidate_scorer.py` |
| Stage 2 encoder | condition encoder 训练。 | `dmmp\encoders\condition_encoders.py` |
| Stage 2 diffusion | warm-start 与 guided diffusion 训练。 | `dmmp\diffusion\profile_pipeline.py`、`dmmp\guidance\diffusion_guidance.py` |
| Stage 3 | guided DDIM sampling、projection、refinement、renderer。 | `dmmp\diffusion\profile_pipeline.py`、`dmmp\projection\padding.py` |
| fixed evaluation | fixed DF/RF 评测 fresh defended test。 | `dmmp\evaluation\profile_attacks.py` |
| mixed evaluation | clean + defended 训练 mixed DF/RF，评测 fresh deployment test。 | `dmmp\evaluation\profile_attacks.py` |
| Stage A keypoint discovery | 对 clean TAM 样本提取关键点图、聚类成关键点模式簇、输出 prototype 和人工审核图。 | `dmmp\stage_a\*`、`scripts\stage_a_*.py` |

## 结果归属

DMMPv3 新结果只能写入：

```text
D:\learning\TOR\defence\DMMPv3\results
```

`dmmp\utils\config.py` 中 `DefenseConfig.output_dir` 的默认值已经指向上述目录。`scripts\verify_project.py` 会检查这个默认值，防止结果误写到旧根目录 `D:\learning\TOR\results`。

## 历史资产边界

DMMP2 的 fixed DF/RF checkpoint 仍然保留在历史结果目录，只作为 DMMPv3 内部迭代时的固定评测器引用。旧工程中的 defense checkpoint、encoder、diffusion、defended datasets、stage outputs 和 mixed attacker checkpoint 都不能直接作为 DMMPv3 正式结果。
