# DMMPv3 决策记录

## 2026-07-14 Stage / Guidance / Checkpoint 决策

- Stage 2/3 defense guidance 不再使用真实 labels 作为 guidance label；生成侧统一使用 frozen ProjectDF/ProjectRF 对 observed prefix 的 label-free surrogate pseudo labels。真实 labels 仅限训练/验证监督、metadata 和离线评测。
- `--stage` 是正式入口参数：`all` 跑完整流水线，`1` 只跑 Stage 1，`2`/`3` 复用指定 run 并检查前置 artifacts。
- attack eval 默认复用当前 run 缓存；`--force_retrain` 用于显式重训。新训练出的 checkpoint 必须落入当前指定目录，历史 fixed checkpoint 仅能作为经过兼容性验证的固定评估器引用。

## 2026-07-20 Label-dependent guidance 消融决策

- 新增 `guidance_label_mode`，默认值为 `pseudo`，保持 2026-07-14 的 label-free Stage 2/3 主线不变。
- 只有显式设置 `guidance_label_mode=true` 时，Stage 2 diffusion guidance 和 Stage 3 guided DDIM/continuous refinement 的 target 才使用真实网站标签映射到 frozen surrogate class position。
- `true` 模式必须作为 label-dependent oracle / upper-bound ablation 报告，不得写成可部署的 label-free 防御结果；Stage 1、encoder、candidate scorer 和 profile/preference 条件逻辑不因该开关改变。

## 已确认决策

### 2026-07-14：DMMPv3 作为并列新工程创建

DMMPv3 创建在 `D:\learning\TOR\defence\DMMPv3` 下，不覆盖 `defence\DMMP`、`defence\DMMP2`、根目录 `experiments`、根目录 `models` 或任何历史结果。

原因：当前项目中存在多代实验代码和结果。独立 Harness 可以避免误覆盖 checkpoint 或结果，并为后续任务提供稳定入口。

### 2026-07-14：DMMP2 最新强引导流程迁移为 DMMPv3

`defence\DMMP2` 中最新的强 DF/RF guidance 流程已经迁移到 DMMPv3 的 `dmmp/` 和 `scripts/` 下。迁移后的正式方案统一称为 DMMPv3，不再使用旧版本编号命名。

原因：用户确认当前目标是把 DMMP2 中最新内容重新提炼并转移到 V3 中，而不是继续维护旧版本编号下的方案。

### 2026-07-14：DMMPv3 新结果只写入本工程 results

所有 DMMPv3 新训练、生成、评测和失败记录默认写入：

```text
D:\learning\TOR\defence\DMMPv3\results
```

原因：根目录 `D:\learning\TOR\results` 已经包含多代历史结果。DMMPv3 必须在本工程内形成独立可追踪结果树。

### 2026-07-14：跨项目防御流程必须重跑，DMMPv3 内部 fixed DF/RF 可复用

从旧工程迁移到 DMMPv3 时，防御流程相关内容必须在 DMMPv3 下重新运行并重新产出结果。旧工程中的 encoder、diffusion、defense checkpoint、defended datasets、stage outputs 和 mixed attacker checkpoint 不能直接作为 DMMPv3 正式结果。

已验证兼容的 fixed DF/RF checkpoint 可以作为稳定固定评测器复用。只要 DMMPv3 内部迭代没有改变数据集、类别数、标签映射、train/val/test split、DF 输入表示、RF TAM 表示或攻击器结构，就不需要重训 fixed DF/RF。

原因：跨项目继承防御中间产物会混淆 DMMPv3 正式结果的归属和可复现性；而 fixed DF/RF 是固定评测器，保持不变有利于比较 DMMPv3 内部每一步防御改进。

### 2026-07-14：fixed checkpoint 先作为外部路径引用

已验证 fixed DF/RF checkpoint 仍保留在 `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30` 原路径，只在 `docs/model_registry.md` 建立索引。

原因：当前不移动或复制历史 checkpoint。后续如需复制进 `models/attackers/fixed`，必须先加入 checksum 和 provenance 文件。

### 2026-07-14：当前正式预算只采用 30%

DMMPv3 第一版正式实验只设计 `0.30` 带宽开销。历史 sweep 中出现的其他预算点只保留为参考。

原因：用户已确认实验只设计 30% bandwidth overhead。

### 2026-07-14：本次迁移不启动正式训练

本次任务完成工程迁移、脚本入口、文档和校验，不启动大规模训练，不重训 fixed attacker，不训练 mixed attacker。

原因：迁移和 Harness 收口应先完成，再由用户选择正式 run 名称和资源配置。

### 2026-07-14：dmmp 顶层代码按职责放入子包

`dmmp\` 顶层只保留 package 入口；业务代码已经放入 `data`、`encoders`、`diffusion`、`guidance`、`projection`、`renderer`、`constraints`、`evaluation`、`utils` 和 `losses` 子目录。

原因：用户已经为 DMMPv3 设计了目录类别。按类别放置代码可以降低后续维护成本，也能让文档、入口和模块职责更一致。

### 2026-07-15：V3 主线改为攻击压制优先，个性化硬约束降级

三组隔离实验显示：当前 V3 TAM/RF-aware 在复用同一 fixed checkpoint 时仍显著弱于原版 30% fixed 参照；reliable Stage 3 probe 仍只能得到 diagnostic fallback；minimal ablation 关闭 profile/preference/direction/noise 约束后，DF/RF defended acc 明显下降。因此，主理论不再把用户画像掩码、偏好权重、clean 方向修正、固定 incoming dummy 比例和 logit noise 作为核心约束。

保留 DF/RF 联合 guidance、TAM/RF-aware surrogate、带宽/allowed-mask/renderer 部署硬约束和 reliable fixed probe gate。用户画像与偏好仅作为可解释性、后处理或可选软正则，不参与主防御策略筛选。详细修订见 `docs/method_spec.md` 末尾“2026-07-15 隔离实验后的理论修订建议”。

### 2026-07-15：随机偏好组合改为多样性来源，偏好一致性由攻击收益门控

新的默认实现把随机偏好组合从硬条件改为多样性/不可预测性来源：`condition_profile_mask=false`、`condition_selected_mask=false`、`condition_preference_map=false`、`condition_preference_weights=false`，同时降低 prior 中偏好成分占比为 `prior_preference_weight=0.15`，以 `prior_leak_weight=1.50` 继续突出 prefix leakage。

偏好一致性不再作为主优化目标，而是 `preference_weight=0.01` 的低权重辅助项，并默认开启 `preference_attack_gate=true`。只有当偏好原型的 DF/RF 目标风险不高于当前防御输出风险加 `preference_attack_gate_margin=0.02` 时，该辅助项才产生梯度。否则偏好项被置零，避免把策略拉回攻击器容易识别的区域。

## 已否定或暂缓路线

### 不把 smoke 结果作为正式证据

`D:\learning\TOR\results\dmmp2_v4_phase1_smoke_seed0_bwo18` 不能作为正式 checkpoint 结论。它的 fixed DF checkpoint 可以结构加载，但历史 best validation accuracy 太低，不能用于正式复用。

### 不把旧单脚本实验作为 DMMPv3 主入口

根目录 `experiments\run_random_preference_diffusion_pipeline.py` 和 `experiments\train_current_method_attackers.py` 仅作为历史参考。

原因：DMMPv3 已经拥有 self-contained 的 package 和 CLI。

## 需要人工确认

- DMMPv3 pass/fail 阈值。
- ALERT 与 FRUGAL 是否纳入首版 DMMPv3 baseline 协议。
- fixed checkpoint 后续是继续外部引用，还是复制进 DMMPv3 `models/` 并记录 checksum。
- rolling online execution 是否作为首版正式目标。

### 2026-07-18: class-conditional x0* upper-bound probe

新增 `target_policy_direct_v1` 的类别条件化上界实验入口。默认仍保持 label-free / clean-pseudo-label 行为；只有显式传入
`--teacher_target_mode true_label` 或 `--candidate_class_condition_mode train_saliency` 时，才使用真实类别信息。

本设计不是默认在线部署假设，而是 oracle upper-bound probe：用于判断当前 x0* 构造器效果弱，是否主要来自“没有类别条件”。
轻量 n=200 结果显示，单独把 teacher target 从 clean pseudo label 改为 true label 不改变结果；原因是 fixed DF/RF clean accuracy
很高，pseudo target 与 true label 基本一致。将 train-split per-class saliency prior 注入候选生成后，`w=0.35` 可进一步降低
DF defended accuracy，但会降低 deployable coverage，并未同时改善 RF。因此后续若继续该路线，应把 class prior 作为待网格搜索的
候选空间偏置，而不是直接作为默认方法。
## 2026-07-28 Stage A 独立关键点发现闭环

- 新增 Stage A：`Counterfactual TAM Keypoint Discovery`，代码位于 `dmmp/stage_a`，入口位于 `scripts/stage_a_*.py`，配置样例位于 `configs/stage_a`，说明位于 `stage_a/README.md`。
- Stage A 不直接生成防御策略，而是为 clean TAM 样本生成样本级关键点图 `S_i`，再聚类为关键点模式簇 `c_i`，作为后续 Stage B/C 的位置先验和人工审核依据。
- 第一版以 fixed RF checkpoint 作为忠实 TAM 攻击器；现有 fixed DF checkpoint 原生输入是 `sign(trace)->[1,5000]`，因此 Stage A 中的 DF/TAM 路径仅作为近似 adapter，不能优先作为正式解释结论。
- 对现有 fixed RF checkpoint，`W=200` 只作为快速 smoke；正式 RF keypoint audit 应优先用原生 `W=1800`，避免 downsample/upsample 损伤攻击器输入。
- 2026-07-28 quick audit 已保存到 `results/stage_a_rf_native_w1800_n96_s60_seed0`。该 run 使用 96 条 test 样本、60 步 DynaMask 优化，适合作为闭环验证和初步模式观察，不作为全量正式统计结论。
