# DMMPv3 方法说明

## 2026-07-16 更新说明：用户固定二元偏好子池

- DMMPv3 主线将用户偏好从“每次 visit 随机选择 pair/triple 组合并重新采样 Dirichlet 权重”改为“每个用户建档时固定一个二元偏好子池组合，并固定两项随机权重”。五个偏好 primitive 仍然保留，但它们只作为 Stage 1 已选 executable candidate cells 上的偏好引导源，而不是硬性防御机制。
- 每个用户 profile 只激活一个 pair combination。pair 在 5 个 primitive 的两两组合中随机选择；两个 raw weight 在固定区间 `[0, 1]` 内随机采样，并归一化为实际融合配比。这样跨用户仍然有随机性和个性化，但同一用户的训练和部署访问保持稳定偏好。
- 该设计替代旧的“多 pair/triple 子池 + visit-level Dirichlet 随机融合”。原因是旧设计会让 profile/preference 条件熵过高，容易把 diffusion 训练推向满足偏好扰动而不是优先压制 DF/RF attack pressure，尤其在此前实验中已经观察到用户个性化约束可能削弱防御严格性。
- 新主线假设是：二元偏好已经足够表达用户个性化方向，同时降低拟合难度和组合搜索噪声。抗对抗训练的随机性来自两个层面：不同用户选择的 pair 不同；不同用户 pair 内的权重配比不同。同一用户内部不再通过 visit-level 重新抽组合来制造随机性。
- 实现默认配置为 `profile_combination_mode=fixed_pair`、`combination_sizes=2`、`active_pair_count=1`、`active_triple_count=0`、`pair_probability=1.0`、`profile_pair_weight_min=0.0`、`profile_pair_weight_max=1.0`。旧的 multi-combo 行为保留为 `profile_combination_mode=legacy_pool`，只建议作为消融对照。
- 训练期 loss 权重不再建议只凭经验固定。新增 `scripts/sweep_loss_hparams.py` 做轻量 grid/random search，先在小类数、小样本和短 diffusion steps 下筛选 `preference_weight`、`defense_soft_objective_scale`、`defense_soft_utility_weight`、`prefix_hidden_align_weight` 等敏感项，再只把排名靠前的少数配置放大全量验证。轻量搜索默认将短训 strong surrogate 的 validation gate 放宽为 `0.65`，只用于避免 probe 被正式训练的 `0.85` 门槛提前中止；最终正式实验仍应使用严格 surrogate gate 和完整 fixed/mixed attack evaluation。
- 2026-07-16 seed0 轻量细分结果显示：反向去噪项没有独立的 `denoise_weight`，而是在 guidance 激活后通过 `defense_soft_objective_scale * soft_loss` 间接控制；当 `preference_weight=0`、`prefix_hidden_align_weight=0` 做隔离搜索时，较优区间集中在 `defense_soft_objective_scale=0.015~0.0175`、`defense_soft_utility_weight=0.030~0.0325`。这是 provisional candidate，只用于下一轮全量验证，不直接替代正式默认。
- 2026-07-16 seed0 轻量 prior-preference sweep 显示：用户个性化偏好作为 prior guidance 时不应回到与 leakage 同量级的强权重。固定 `preference_weight=0`、`prefix_hidden_align_weight=0` 并扫 `prior_preference_weight in {0,0.025,0.05,0.10,0.15,0.225,0.30,0.45}` 后，单点 best 为 `0`，但均值更支持 `0.10~0.15` 作为保守候选区间；`0.30` 以上整体变差。该 prior 只改变候选位置偏置，不直接控制带宽，带宽仍由 `budgets/pareto_budgets`、projection/rounding 和 renderer 约束。

## 2026-07-14 更新说明

- Stage 2/3 的防御 guidance target 已改为 label-free surrogate pseudo label：只对已观察 prefix 用 frozen ProjectDF/ProjectRF ensemble 做预测，按 ensemble 权重融合后取预测类别作为 guidance target。真实 labels 只用于训练/验证监督、数据集元数据和离线诊断指标，不再作为防御 guidance label，也不参与 Stage 3 候选排序和 hard gate。
- 主链路中的 top cells 指可执行的 direction-patch candidate cells，来源是 prefix condition、allowed mask、candidate scorer 和用户 profile/preference 的组合约束；它们不是用真实网站 label 直接挑选的 cells。
- Stage 3 顺序以当前实现为准：加载 Stage 1/2 artifacts -> 对 validation split 枚举 budget/keep ratio -> label-free guided DDIM 生成 -> continuous refinement -> projection/rounding/rendering -> surrogate/probe 评估候选 -> label-free pressure gate/selection -> 保存 selected policy。
- `direction_target=balanced` 表示将 dummy 方向比例拉向 50% incoming / 50% outgoing；`direction_target=clean` 才会在 prefix incoming ratio 基础上应用 `min_incoming_dummy_share` 下限。

本文件说明 DMMPv3 当前已迁移实现的具体构思。标有 `[需要人工审核]` 的段落是本次根据当前代码补充的实现解释，待人工审核确认后再移除标记。

## 方法定位

DMMPv3 将网站指纹防御建模为条件防御策略生成问题。模型生成的是连续 padding policy，不直接生成完整流量。迁移自 DMMP2 的最新强 DF/RF guidance 流程已经纳入本工程；在 DMMPv3 中，这套方案正式称为 DMMPv3，不再使用旧版本编号命名。

DMMPv3 的输出不是“直接改写包序列”，而是先在方向-时间片 cell 上生成一个策略图：

```text
policy logits: [2, patch_num]
```

其中 `2` 表示 outgoing/incoming 两个方向，`patch_num` 当前默认是 `200`。后续再经过 mask、预算投影、整数化和 renderer，变成真实 dummy packet。

## 总体执行链路

```text
observed prefix
-> prefix condition
-> leakage-aware condition encoder
-> conditional policy diffusion
-> DF/RF guided DDIM sampling
-> continuous refinement
-> allowed-mask and budget projection / integer rounding
-> dummy packet renderer
-> defended trace
```

## 基本约束

- Prefix-only：防御条件只能来自已观察到的 prefix，当前默认 `prefix_n = 500`。
- Label-free：网站标签不得作为防御条件。
- Strategy-only：diffusion 输出 direction-patch cell 上的策略，不输出真实包时间戳。
- Causal execution：投影后的 dummy counts 必须落在 executable allowed cells 内。
- Budgeted execution：当前正式预算只设计 `0.30` 带宽开销。
- Renderer 必须只插入 dummy packet，不删除真实包，不改变真实包相对顺序。
- raw variable-length defended traces 与 attacker-specific crop/pad 解耦。

任何改变这些约束的操作，都必须同步更新 `docs/decisions.md` 与 `docs/experiment_protocol.md`。

## 已迁移核心组件

| DMMPv3 文件 | 作用 |
|---|---|
| `dmmp\data\cw.py` | CW 数据加载、memmap-aware `.npz` 读取、stratified split。 |
| `dmmp\evaluation\attack_models.py` | ProjectDF、ProjectRF、DF input 和 RF TAM adapters。 |
| `dmmp\encoders\prefix.py` | prefix condition 提取和 early causal region。 |
| `dmmp\encoders\leakage.py` | 多视角 prefix leakage profile。 |
| `dmmp\guidance\candidate_scorer.py` | executable candidate cells、allowed masks 和 utility probes。 |
| `dmmp\encoders\condition_encoders.py` | leakage-aware condition encoder 与监督目标。 |
| `dmmp\diffusion\models.py` | TopKLeakageEncoder 与 policy diffusion 模型。 |
| `dmmp\guidance\diffusion_guidance.py` | DF/RF guidance、guided DDIM sampling、continuous refinement。 |
| `dmmp\projection\padding.py` | budget projection、rounding、fixed/variable renderer。 |
| `dmmp\guidance\strong_surrogates.py` | 冻结 ProjectDF/RF surrogate ensemble 与防御效用。 |
| `dmmp\diffusion\profile_pipeline.py` | 当前 DMMPv3 主训练与生成流程的迁移载体。 |
| `dmmp\evaluation\profile_attacks.py` | 当前 DMMPv3 fixed/mixed attack evaluation 的迁移载体。 |

完整实现索引见 `docs\implementation_index.md`。运行命令见 `docs\runbook.md`。

## 核心设计：攻击器引导不是最终评测

DMMPv3 在训练和生成阶段使用 ProjectDF/ProjectRF surrogate ensemble 作为“攻击视角反馈”。这些 surrogate 是完整训练过的攻击分类网络，但在这里的角色不是最终验收裁判，而是给防御策略提供可微分方向。

 ProjectDF/ProjectRF 首先作为普通攻击器用 clean traces 训练，训练分类器时使用交叉熵：

```text
loss_attacker = cross_entropy(attacker_logits, true_site_label)
```

当它们被用于防御 guidance 时，目标不再是让攻击器分类更准，而是让当前 padding policy 让攻击器更难维持原有判别。Stage 1 的监督/诊断目标可以使用真实标签；Stage 2/3 的 defense guidance 使用 frozen surrogate 伪标签位置作为 target，不直接使用真实网站标签。当前实现使用两个主要量：

```text
target risk = softplus(target_logit - max_other_logit + margin)
defense utility = (1 - target_confidence) + normalized_entropy + margin_related_term
```

diffusion/refinement 中的核心防御引导目标近似为：

```text
loss_defense = mean(target risk) - soft_utility_weight * mean(defense utility)
```

因此，防御训练不是简单最大化交叉熵，而是显式压低 target 类别优势、提高预测不确定性，并通过 robust term 关注 DF/RF ensemble 中更难被扰动的攻击器。

## Stage 0：数据与固定设置

DMMPv3 默认读取 CW 数据，生成 seed 0 的 stratified train/val/test split，并把 split 写入 run 目录。当前 fixed DF/RF checkpoint 复用资格也依赖这个 split、95 类标签映射、DF 输入长度和 RF TAM 表示保持不变。

主要固定形状：

- `prefix_n = 500`
- `patch_num = 200`
- `max_trace_length = 5000`
- `surrogate_rf_num_slots = 1800`
- `surrogate_rf_max_load_time = 80.0`
- `budgets = 0.30`

## Stage 1：Prefix Leakage 与 Executable Candidate Cells

### 0. 与 RF 论文互信息泄露设置的关系

[需要人工审核] DMMPv3 当前的 leakage 分析与 RF 论文中的信息泄露思想是同一类目标：都希望衡量某种流量表征 `F` 中包含多少关于网站类别 `C` 的信息。RF 论文使用的核心定义是：

```text
I(F; C) = H(C) - H(C | F)
```

也就是表征 `F` 越能降低网站类别的不确定性，泄露越强。DMMPv3 当前实现中的 `mutual_information_discrete` / `score_feature_matrix` 也是在估计“特征与类别标签之间的互信息”，因此方向上是相似的。

[需要人工审核] 但是当前实现还不是 RF 论文中完整意义上的 representation-level information leakage。主要差异有四点：

1. RF 论文比较的是完整 traffic representation 的信息泄露；DMMPv3 当前主要做 prefix-only leakage，目的是给防御策略提供早期可执行候选位置。
2. RF 论文中的 `F` 更接近一个完整表征集合；当前 `score_feature_matrix` 会先对每一维特征做分箱，再计算单列互信息，最后取 top feature 的平均值，因此它是 column-wise MI proxy，不是严格的联合表征互信息。
3. RF 论文把统计特征和 per-packet feature sequence 作为两类表征进行比较；DMMPv3 当前 view 包含 `V_raw`、`V_count`、`V_interval`、`V_burst`、`V_rate`、`V_cumul` 和 `V_patch`，其中 `V_patch` 更接近 direction-patch 粒度的序列表征，但仍然是 prefix 片段上的工程化近似。
4. `profile_views_v4` 的最终 view 分数不是纯互信息，而是 `MI proxy + centroid cross-validation accuracy + log-loss gain` 的综合分数。也就是说它更像“防御候选选择实用分数”，不等同于论文里单独报告的 `I(F; C)`。

[需要人工审核] 当前方向处理需要特别标注。代码中并非完全没有区分上下行：`prefix_patch_counts` 返回 `[2, patch_num]`，其中正方向包进入 outgoing/upstream 轴，负方向包进入 incoming/downstream/download 轴；`V_count` 也包含 `out_count`、`in_count`、`out_ratio`、`in_ratio`；`V_patch` 的 cell-level MI 会保留两个方向，并在 top cells 中写出 `outgoing` 或 `incoming`。因此 candidate cell、policy logits、mask 和 renderer 的基本形状已经是方向敏感的。

[需要人工审核] 但当前 view-level leakage 还没有把上行和下行拆成两个独立泄露视角来报告。`V_interval`、`V_rate`、`V_burst` 等视角大多仍是聚合统计；`V_patch` 虽然内部有 `[2, patch_num]`，但 view 打分时会 flatten 后综合计算，容易掩盖“下行/download 包更多、下行泄露更强”的数据事实。对于 CW 数据集，如果负方向代表服务端到客户端的 download/downstream 包，那么更合理的核对项是单独输出：

- `incoming/downstream_mi` 与 `outgoing/upstream_mi`；
- incoming top cells 与 outgoing top cells；
- top-k candidate 中 incoming cell 的占比；
- direction-specific `V_rate_in` / `V_rate_out`、`V_interval_in` / `V_interval_out`、`V_burst_in` / `V_burst_out`；
- 可选的 TAM-like direction-time 表征泄露分数，用于更贴近 RF 对 direction/timing 表征的讨论。

[需要人工审核] 因此，当前文档应把 Stage 1 的互信息称为“RF-style MI proxy”或“prefix leakage proxy”，而不是声称已经完全复现 RF 论文的信息泄露评估。若后续要严格对齐论文，应新增 direction-specific leakage report，并明确互信息单位使用 bits 还是 nats；当前 `np.log` 形式默认更接近 nats，若要报告 bits 需要除以 `log(2)`。

### 1. Prefix condition

每条 trace 先通过 `extract_prefix_condition` 提取 prefix-only 条件。该条件只使用已经观察到的前缀，不允许使用网站标签或未来包。它会生成：

- prefix 统计向量；
- direction-patch 形状的 saliency/structure 信息；
- causal allowed mask；
- burst、rate、interval 等辅助特征。

### 2. 多视角泄露分析

`profile_views_v4` 会构造多个 view，例如 `V_raw`、`V_count`、`V_interval`、`V_burst`、`V_rate`、`V_cumul` 和 `V_patch`。每个 view 用互信息 proxy、centroid cross-validation accuracy、log-loss gain 综合打分，然后选出当前最有用的 prefix leakage views。

### 3. 强 surrogate 训练或加载

[需要人工审核] Stage 1 会训练或加载 strong ProjectDF/ProjectRF surrogate ensemble。训练攻击器本身时使用交叉熵，之后检查 validation accuracy；若低于 `surrogate_min_val_accuracy`，防御训练应中止，避免弱攻击器误导防御模型。

### 4. Candidate utility map

[需要人工审核] 对每个候选 direction-patch cell，DMMPv3 需要判断“往这里加 dummy 是否有防御价值”。当前实现会用 DF/RF ensemble 产生 utility map：

- 近似版本：对 soft allocation 求 defense utility 关于 allocation 的梯度；
- 校验版本：使用 finite-difference insertion 做 exact probe；
- 输出限制在 allowed mask 内，避免不可执行位置参与候选。

### 5. CandidateScorer

[需要人工审核] `CandidateScorer` 学习从 prefix/candidate features 预测 utility map。训练目标不是交叉熵，而是：

```text
loss_candidate = MSE(predicted_utility, target_utility)
                 + 0.10 * cosine_ranking_loss
```

[需要人工审核] 训练完成后，通过 `soft_topk_mask` 或 hard top-k 选出 executable candidate cells。Stage 1 会保存 candidate scorer、utility probes、candidate metrics 和 summary。

## Stage 2A：Condition Encoder

[需要人工审核] Encoder 的任务不是直接生成防御策略，而是把 prefix leakage、candidate utility、candidate mask 和全局 view 信息编码成 diffusion 可用的条件表示。

[需要人工审核] 当前 `V4LeakageEncoder` 输出三个核心量：

- `c_global`：全局条件向量；
- `c_leakage`：direction-patch 形状的泄露/utility 表示；
- `structure`：用于重构选中 views 的结构信息。

[需要人工审核] Encoder 的输入由 `candidate_features`、DF/RF utility map、candidate mask 和 view vector 拼接而成。训练目标来自 Stage 1 的 utility/candidate 数据，以及 strong surrogate 产生的 global targets。

[需要人工审核] Encoder loss 不是分类交叉熵，而是多目标监督：

```text
loss_encoder =
  exec_mse(predicted_leakage, utility_target)
  + rank_weight * cosine_rank_loss
  + struct_weight * structure_reconstruction
  + global_weight * global_target_matching
  + fusion_weight * mean_response_matching
  + smooth_weight * local_smoothness
```

[需要人工审核] 这个阶段的目的是让 encoder 学会“prefix 泄露在哪里、哪些 future cells 可执行且有防御价值”，为后续 diffusion 生成策略提供条件。

## Stage 2B：Profile/Preference 条件

[需要人工审核] DMMPv3 引入 user profile 和固定二元 preference，是为了避免所有用户都使用同一种 padding 习惯，同时避免 visit-level 随机组合给 diffusion 训练带来过高扰动。`user_profiles.py` 生成 train/validation/test profiles；每个 profile 在建档时固定一个 pair combination、两项 raw preference weights，以及归一化后的实际融合权重。visit selection 仍然产生 keyed diffusion/render seeds，但不再改变该用户的 pair 或权重。

[需要人工审核] 每次训练 diffusion 时，会为 trace 选择一个 profile 与 visit，然后按该用户固定 pair 构造 mixed preference map。这个 map 会结合：

- primitive preference；
- candidate utility；
- candidate mask；
- profile mask；
- selected primitive mask；
- primitive weights。当前主线中 `condition_profile_mask`、`condition_selected_mask`、`condition_preference_map` 和 `condition_preference_weights` 默认关闭，因此这些 profile/preference 信号主要通过 prior、低权重 preference auxiliary loss 和记录统计发挥作用，而不是作为硬条件压过 attack guidance。

[需要人工审核] `CompositionalConditionEncoder` 把 `c_global`、`c_leakage`、candidate mask、mixed preference、primitive weights、selected mask 和 profile mask 融合成最终 diffusion condition。

## Stage 2C：Guided Policy Diffusion

[需要人工审核] 整个 Stage 2 不能简单理解为“条件指导型前向加噪”。更准确地说，Stage 2 由三层组成：Stage 2A 先训练条件编码器，Stage 2B 加入 profile/preference 条件，Stage 2C 才是 conditional diffusion 的前向加噪与反向去噪训练。也就是说，前向加噪只发生在 Stage 2C 的 diffusion 训练内部；Stage 2A/2B 的作用是让 diffusion 获得足够准确的流量表征、泄露位置、可执行候选位置和用户偏好条件。

### 1. Warm-start / denoising

[需要人工审核] Diffusion 的基础训练仍然是 denoising：从 prior policy logits 加噪，denoiser 预测噪声，使用 MSE 学会还原 policy 分布。

```text
loss_denoise = MSE(predicted_noise, target_noise)
```

[需要人工审核] prior policy logits 由 candidate utility、mixed preference 和 candidate mask 生成，作用是让 diffusion 一开始学到“可执行且偏好合理”的策略，而不是完全从随机策略开始。

### 2. Prefix-hidden alignment

[需要人工审核] DMMPv3 还保留了“流量表征 encoder 与 diffusion 深层 hidden 对齐”的设计。训练时，denoiser 不只输出 `predicted_noise`，还会返回一个中间 hidden 表示 `denoiser_hidden`。系统用两个 projector 分别把 `c_global` 和 `denoiser_hidden` 投影到同一维度：

```text
prefix_z = normalize(prefix_projector(c_global))
hidden_z = normalize(hidden_projector(denoiser_hidden))
```

[需要人工审核] 对齐目标由两部分直觉组成：同一条 trace 的 prefix 表征和 denoiser hidden 应该更接近，不同 trace 的表征应该更可区分。当前 `_prefix_hidden_alignment_loss` 使用对称 cross-entropy / contrastive 形式：

```text
alignment_loss =
  0.5 * CE(prefix_z @ hidden_z.T / temperature, batch_identity)
  + 0.5 * CE(hidden_z @ prefix_z.T / temperature, batch_identity)
```

[需要人工审核] 这一步的目的不是让 diffusion 生成原始流量，而是让 diffusion 虽然输出的是防御模板 `policy logits: [2, patch_num]`，但它的深层表示仍然理解原始 prefix 流量的类别相关结构、方向结构和泄露结构。该项通过 `prefix_hidden_align_weight` 加入 soft objective，当前默认权重为 `0.03`。

### 3. Soft allocation

[需要人工审核] 在把 policy logits 送给 DF/RF surrogate 前，需要先转成可微分的 dummy allocation。`soft_allocation` 会：

1. flatten `[2, patch_num]` logits；
2. 用 candidate mask 把不可执行位置设为极小值；
3. softmax 得到 cell 概率；
4. 按目标 dummy count 缩放成连续 dummy allocation。

### 4. DF/RF 可微输入近似

[需要人工审核] 为了让 DF/RF 能对 allocation 反传梯度，当前实现使用 soft defended input：

- DF：把 clean direction signal 与 dummy direction signal 在 patch 级别混合，得到近似 `[N, 1, 5000]` 输入；
- RF：把 dummy allocation scatter 到 TAM slots 上，叠加到 clean TAM，得到近似 `[N, 2, 1800]` 输入。

[需要人工审核] 这一步是 surrogate guidance 的核心近似。它不等同于最终 renderer 生成的离散 trace，但用于训练阶段提供可微分方向。

### 5. Defense-first loss

[需要人工审核] diffusion 训练后半段开始加入 attack guidance。对于抽样出来的一小批 `surrogate_gradient_batch_size`，当前 policy 先转成 soft allocation，再送入冻结的 ProjectDF/RF。随后计算：

```text
loss_defense = mean(pseudo-label target risk) - soft_utility_weight * mean(defense utility)
```

[需要人工审核] 总 loss 分为 soft objective 和 hard defense objective：

```text
soft_loss = denoise
          + preference_weight * gated_preference_loss
          + diversity_weight * diversity_loss
          + constraint_weight * allowed_mask_loss
          + profile_weight * profile_loss
          + prefix_hidden_align_weight * alignment_loss

if guidance is active:
    loss = defense_hard_weight * defense_loss
           + defense_soft_objective_scale * soft_loss
else:
    loss = soft_loss
```

[需要人工审核] 当前主线默认开启 `preference_attack_gate`：偏好一致性在 guidance 未激活时不进入 `soft_loss`；guidance 激活后，只有当偏好原型的 surrogate pseudo-label target risk 不高于当前 defense policy 的 risk 加 `preference_attack_gate_margin` 时，`preference_loss` 才以低权重进入 soft objective。这样随机偏好只提供多样性/不可预测性来源，不再压过攻击压制目标。

[需要人工审核] 当前还使用 defense-first backward：先关注 hard defense gradient，再把 soft objective 作为辅助项，降低“偏好/平滑目标把防御目标抵消”的风险。

## Stage 2D：Full-sample Guidance 与 Diversity

[需要人工审核] 除了对 predicted x0 做直接 defense loss，训练中还会周期性运行 differentiable DDIM sampling，用完整采样结果再算一次 defense guidance。若 full-sample defense loss 更差，则取更保守的防御损失。

[需要人工审核] diversity loss 会比较同一条件下两次 diffusion 输出的 policy 分布，避免模型塌缩成单一 padding 模式。

## Stage 3：Guided DDIM、Refinement 与 Selection

### 1. Guided DDIM sampling

[需要人工审核] 部署策略生成时，DDIM 采样最后若干步会启动 attack guidance。每一步先得到当前 predicted policy，再计算 DF/RF surrogate-pseudo-label target risk 和 utility，对 predicted policy 做一次梯度下降。这里的 target 来自 frozen surrogate 对已观察 prefix 的伪标签预测，不是真实网站标签：

```text
candidate_policy = current_policy - guidance_weight * gradient(loss_defense)
```

[需要人工审核] 更新不是无条件接受。系统会重新计算 candidate policy 的 pseudo-label target risk，只有当 risk 没有变差超过 `risk_tolerance` 时才接受，否则保留原 policy。

### 2. Continuous refinement

[需要人工审核] DDIM 之后、整数 rounding 之前，`continuous_refine_logits` 会学习一个 gate，对 logits 做局部稀疏化和再优化。它同样使用：

```text
loss_refine = mean(pseudo-label target risk)
              - soft_utility_weight * mean(defense utility)
              + sparsity_penalty
```

[需要人工审核] refinement 的目标是保留有效 dummy 分配，减少无效或风险更高的 cell。refinement 后也会比较 risk before/after，只接受没有使攻击风险变差的结果。

### 3. Projection、rounding 和 renderer

[需要人工审核] 连续 policy 会通过 `project_policy_to_template` 转成整数 padding template。该步骤负责：

- 应用 allowed mask；
- 按 clean trace 长度和 bandwidth budget 计算目标 dummy count；
- 使用预算投影和 largest-remainder 风格 rounding；
- 记录 allowed-mask violation、clip、rounding 和 bandwidth 统计。

[需要人工审核] 上下行偏置当前放在 Stage 3 的 projection/rounding 层，而不是强行改 Stage 2 的 diffusion 结构。`_direction_target_incoming_share` 会根据 `direction_target` 计算目标下行 dummy 占比，再传入 `project_policy_to_template`。当前支持四种方向策略：

- `direction_target=none`：不做方向校正，完全使用 diffusion/refinement 给出的方向概率；
- `direction_target=balanced`：把目标下行 dummy 占比拉向 `0.5`，不应用 `min_incoming_dummy_share` 下限；
- `direction_target=clean`：根据已观察 prefix 中负方向包比例设置目标下行占比，并应用 `min_incoming_dummy_share` 下限；
- `direction_target=incoming`：把目标下行 dummy 占比设为 `1.0`，用于“只插入下行虚拟包”的消融实验。

[需要人工审核] 当前默认是保守下行偏置，而不是强制只下行：

```text
direction_target = clean
direction_correction_strength = 0.75
min_incoming_dummy_share = 0.65
```

含义是：先看已观察 prefix 本身的下行比例，再至少保证目标 dummy 中约 `65%` 倾向于 incoming/downstream/download 方向；`direction_correction_strength=0.75` 表示不是完全覆盖 diffusion 的方向分布，而是把当前分布向目标分布拉近。这样可以利用 CW 数据中下行包更多、下行泄露可能更强的事实，同时不完全破坏 diffusion、candidate utility 和 profile preference 已经学到的方向结构。

[需要人工审核] “只下行虚拟包”不会要求重写模型，因为 renderer 本来就支持 `[2, patch_num]` 两个方向的 counts；但它会更强地覆盖模型生成的方向比例，因此应作为对照实验而不是默认主线。建议命令行设置为：

```text
--direction_target incoming --direction_correction_strength 1.0 --min_incoming_dummy_share 1.0
```

该设置仍然受到 allowed/candidate mask 和 budget projection 约束；如果某条 trace 没有可执行的 incoming cell，代码会回退到可执行方向，不会凭空违反 mask。

[需要人工审核] Renderer 只插入 dummy packet，不删除真实包，不改变真实包相对顺序。raw variable-length defended traces 会保存真实长度；DF/RF 所需 crop/pad 或 TAM 转换由 attacker adapter 单独处理。

### 4. Pareto selection

[需要人工审核] Stage 3 会在 validation split 上遍历 `pareto_budgets` 和 `refine_keep_ratios`，生成多个候选 defended datasets。每个候选会记录：

- surrogate DF/RF defended accuracy；
- visible bandwidth；
- raw real-packet retention；
- dummy incoming/outgoing share；
- template entropy；
- allowed-mask violation；
- refinement/projection 统计。

[需要人工审核] Stage 3 候选排序和 hard gate 不再使用真实标签 accuracy，也不使用完整 rendered trace 的攻击准确率。当前主选择量是 `selection_attack_pressure`，它等于 `prefix_policy_label_free_attack_pressure`：只用已观察 prefix、当前 policy logits、candidate mask 和 frozen surrogate 输出分布计算。公式为 `max_confidence + 0.50 * margin - 0.50 * entropy`，越低表示攻击器越不确定。`stage3_max_label_free_attack_pressure` 是 hard gate 阈值；`selection_attack_accuracy`、`surrogate_label_free_attack_pressure` 和 fixed probe 指标仍会保存，但只作为完整 trace 下的离线诊断，不参与候选排序或 gate。若没有候选通过 label-free gate，则保存 best diagnostic fallback，并在 hard gate 开启时中止。

## Fixed 与 Mixed Evaluation

[需要人工审核] Stage 3 的 surrogate/probe selection 仍然不是最终结论。正式评测需要通过 `scripts\run_attack_eval.py` 或包装入口执行：

- fixed DF/RF：只在 clean train/val 上训练或复用已验证 checkpoint，然后评测 fresh defended test；
- mixed DF/RF：使用 clean + defended train 混合训练，用 clean + defended val 选 checkpoint，再在 fresh deployment defended test 上评测。

[需要人工审核] fixed checkpoint 可以作为 DMMPv3 内部迭代的稳定评测器复用，但 mixed attacker 必须跟随当前 DMMPv3 defended traces 重新训练。

## 训练流程摘要

DMMPv3 当前流程：

1. 读取 CW 数据并生成 seed 0 stratified train/val/test split。
2. 训练或加载强 DF/RF surrogate，用于防御监督。
3. 分析 prefix leakage 和 executable candidate cells。
4. 训练 prefix/condition encoder。
5. warm-start policy diffusion。
6. 使用 DF/RF ensemble defense loss 训练 guided diffusion。
7. 通过 guided DDIM 采样部署策略。
8. 执行 continuous refinement、projection、rounding 和 renderer，生成 defended traces。
9. 使用独立 fixed 与 mixed attackers 评测。

## 重跑与复用规则

跨项目迁移时，防御流程相关阶段必须全部在 DMMPv3 下重新运行并重新保存结果。原因是旧工程产物的配置、路径、命名、日志、代码版本和结果归属都不属于 DMMPv3 Harness，直接继承会破坏复现链路。

必须重跑或重新生成：

- seed split 校验与 DMMPv3 记录；
- strong surrogate 的训练或兼容性验证记录；
- prefix leakage profile；
- executable candidate cells 与 allowed masks；
- prefix/condition encoder；
- diffusion warm-start 与 guided diffusion；
- guided DDIM deployment sampling；
- continuous refinement、projection、rounding 和 renderer；
- defended traces；
- mixed adaptive attackers；
- fixed/mixed 评测摘要与 DMMPv3 result metadata。

不能直接作为 DMMPv3 正式结果继承：

- 旧工程 encoder checkpoint；
- 旧工程 diffusion checkpoint；
- 旧工程 defense checkpoint；
- 旧工程 defended datasets；
- 旧工程 stage outputs；
- 旧工程 mixed attacker checkpoint。

可以 warm-start 或复用的内容：

- 已验证兼容的 fixed DF/RF checkpoint 可作为稳定固定评测器复用。
- 在 DMMPv3 内部一步步改进防御策略时，只要 CW 数据集、类别数、标签映射、train/val/test split、DF 输入表示、RF TAM 表示和攻击器结构不变，就不需要重训 fixed DF/RF。
- 如果上述任一条件改变，fixed DF/RF 必须重新验证；无法验证时必须重新训练。

## 推理流程

```text
clean trace prefix
-> condition and candidate context
-> user/profile visit selection
-> conditioned reverse diffusion
-> attack-guided final sampler steps
-> continuous refinement
-> causal/budget projection and rounding
-> dummy renderer
-> defended trace
```

当前离线评测通过已有 trace 生成 defended datasets 来模拟部署。真正 rolling online execution 仍为 `[需要人工确认]`。

## 2026-07-15 隔离实验后的理论修订建议

本节记录三组隔离实验后的方法层修订，用于后续论文/报告和新一轮 DMMPv3 设计。结论来自：

- 30% 原版 DMMP/V1 fixed 参照：DF defended acc `14.30%`，RF defended acc `21.72%`。
- 当前 V3 TAM/RF-aware 复用同一 fixed checkpoint：DF defended acc `51.26%`，RF defended acc `70.86%`，说明差距不是 fixed 攻击器不一致造成的。
- reliable Stage 3 probe：`selection_policy_valid=0`，`selection_used_quality_fallback=1`，rendered RF acc `71.20%`，说明放大 fixed probe 后 Stage 3 仍找不到合格策略。
- minimal ablation：关闭 profile/preference/direction/noise 约束后，DF defended acc 降到 `27.06%`，RF defended acc 降到 `31.14%`，说明 DF/RF guidance 本身有效，负面效果主要来自新增约束层和选择机制。

### 应从主方法中移除或降级的设计

以下设计不建议继续作为 DMMPv3 主理论的核心约束；可以移到消融实验、可解释性模块或用户偏好扩展中。

| 设计 | 当前配置 | 建议 | 原因 |
|---|---:|---|---|
| 用户画像组合子池条件 `condition_profile_mask` | `True` | 从主方法移除，最多作为可选软正则 | 它不是 renderer 层的硬 mask，而是把某个 profile 激活的 20 维组合子池作为 diffusion 条件输入；这会诱导模型按 profile 子池生成策略，间接缩小有效搜索空间，可能挡住真正有效的攻击压制策略。 |
| 已选偏好掩码 `condition_selected_mask` | `True` | 从主方法移除 | 实验没有证明它能降低 fixed DF/RF，反而 minimal ablation 关闭后效果明显改善。 |
| 混合偏好图条件 `condition_preference_map` | `True` | 主方法默认关闭 | 偏好图可保留为 prior 的低权重扰动和多样性来源，但不应默认作为 denoiser condition 直接约束生成方向。 |
| 偏好权重条件 `condition_preference_weights` | `True` | 从主防御目标中移除 | 它让模型优先满足偏好组合，而不是优先降低攻击准确率。 |
| 偏好一致性损失 `preference_weight` | `0.05` | 从主优化目标降级为低权重、带攻击收益门控的辅助项 | 偏好一致性与攻击压制目标存在竞争；若偏好原型不能带来不劣于当前防御输出的攻击风险，就不允许它产生梯度。 |
| 用户画像一致性损失 `profile_weight` | `0.01` | 默认设为 `0`，仅作解释性分析 | 用户画像约束会缩小策略空间，当前没有带来 fixed 攻击收益。 |
| 朝 clean 方向修正 `direction_target=clean` | `clean` | 主方法改为 `none` | 朝 clean 分布靠拢不等价于让攻击器失效；RF/TAM 仍能利用保留下来的结构。 |
| 方向修正强度 `direction_correction_strength` | `0.75` | 主方法设为 `0` | 强方向修正会覆盖 diffusion/guidance 学到的有效扰动分布。 |
| 最小 incoming dummy 比例 `min_incoming_dummy_share` | `0.65` | 主方法设为 `0`，不要固定阈值 | TAM/RF-aware run 的 incoming share 约 `0.65` 但 RF acc 仍高；minimal ablation incoming share 约 `0.167` 反而更强。固定 incoming 比例是负面约束。 |
| 策略 logit 噪声 `policy_logit_noise_std` | `0.10` | 主方法设为 `0`，只作多样性消融 | 噪声能增加多样性，但不保证攻击压制，且会增加选择不稳定性。 |
| fallback 作为可接受策略 | fallback 被保存 | 禁止作为正式防御结论 | fallback 只能说明当前 gate 下没有合格策略；若把 fallback 当正式结果，会掩盖 Stage 3 失败。 |

### 可以保留的设计

以下设计仍应保留在 DMMPv3 主线中，但需要以“攻击压制优先”为原则重新排序。

| 设计 | 建议 | 原因 |
|---|---|---|
| DF/RF 联合 guidance | 保留为主目标 | minimal ablation 证明关闭负面约束后，DF/RF fixed acc 大幅下降，说明联合 guidance 是有效核心。 |
| TAM/RF-aware surrogate | 保留，但不能单独决定选择 | RF 是主要失败点，TAM/RF-aware 方向必要；但必须由 rendered/fixed probe 指标验证，不能只看 surrogate 分数。 |
| label-free attack pressure | 保留为辅助指标 | 它满足在线/无真实标签约束，但 pseudo confidence 约 `0.47`，信号偏弱，不能作为唯一主导目标。 |
| reliable fixed probe gate | 保留并加强 | reliable probe 能暴露 Stage 3 fallback 和 RF 失败，应作为诊断 gate 和正式报告依据。 |
| rendered RF/DF diagnostic gate | 保留 | 最终策略必须在渲染后的离散 trace 上通过攻击准确率验证，而不是只在 soft/surrogate 空间通过。 |
| 带宽预算和 allowed mask | 保留为硬约束 | 这些是可部署性和合法插入位置约束，不能为降低攻击准确率而破坏。 |
| causal/budget projection 与 renderer | 保留 | 它保证只插入 dummy packet，不删除真实包，不改变真实包相对顺序，是部署语义的基础。 |
| continuous refinement | 保留但以攻击压制为首要目标 | refinement 可以改善策略，但接受准则必须以 rendered/fixed 攻击表现为主。 |
| 用户画像建模 | 保留为解释性或后处理模块 | 它可以解释策略多样性和用户偏好，但不应作为主防御优化的硬条件。 |

### 修订后的理论主线

DMMPv3 主理论应从“用户画像/偏好约束驱动的个性化防御”改为：

```text
attack-pressure-first defense generation
with bandwidth/allowed-mask deployment constraints
and optional profile regularization
```

中文表述为：

> 以 DF/RF 联合攻击压力最小化为主目标，以带宽预算、allowed mask 和因果 dummy renderer 为硬约束；用户画像和偏好仅作为可解释性或次级软正则项，不参与主防御策略筛选。Stage 3 必须由 rendered RF/DF 指标和可靠 fixed probe 验证，fallback 不作为正式防御策略。

实现上，随机偏好组合不再承担“必须按用户画像满足某个组合子池”的硬约束角色，而只作为多样性和不可预测性的来源；攻击压制是主优化目标；偏好一致性保留为低权重辅助项，并由攻击风险收益门控控制是否生效。

推荐主线默认配置：

```text
condition_profile_mask = false
condition_selected_mask = false
condition_preference_map = false
condition_preference_weights = false
prior_leak_weight = 1.50
prior_preference_weight = 0.15
prior_noise_std = 0
preference_weight = 0.01
preference_attack_gate = true
preference_attack_gate_margin = 0.02
profile_weight = 0
direction_target = none
direction_correction_strength = 0
min_incoming_dummy_share = 0
policy_logit_noise_std = 0
```

这不是否定 V3 的攻击感知升级，而是把负面约束从主方法中剥离：保留 DF/RF guidance、TAM/RF-aware surrogate、可靠 fixed probe gate 和部署约束；移除会把策略推向错误区域的 profile/preference/direction 条件约束。这里的 `condition_profile_mask` 不是原始理论中的必备设计，而是 V3 实现阶段加入的 profile 条件化扩展；后续论文/报告中不应把它写成主方法核心。
