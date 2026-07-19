# DMMPv3 Handoff Memory - 2026-07-19

本文件是本轮超长对话的长期记忆和新对话交接摘要。新建对话后可以直接让 Codex 读取本文件继续。

## 当前研究主线

目标是改进 DMMPv3 的网站指纹防御方法，使其在完整 CW 数据集上同时压低三类主流攻击视角：

- DF: 基于方向序列的深度模型。
- AWF: 基于 burst/方向片段结构的攻击视角。
- RF: 基于 TAM/time-slot 统计的攻击视角。

当前认为防御不能只针对 RF/TAM，也不能只在 DF 方向序列上有效。最终目标是完整 diffusion 防御链路下，fixed 和 mixed/adaptive attacker 都要全量评测。

## 关键方法判断

1. Stage2 中 diffusion 训练包含正向加噪 `q_sample(x0, t)` 和反向去噪预测噪声。条件指导本质上进入反向去噪网络/condition encoder，而不是“只加在正向加噪过程”。

2. 反向去噪 loss 不适合独自承担“带宽和防御有效性平衡”。带宽应主要由 `budget`、projection/rounding 和 renderer 控制；攻击压制应由 defense guidance/refinement 控制。

3. `preference_loss = 1 - cosine(allocation, preference)` 这类用户偏好约束不能和攻击压制同等重要。此前发现偏好/个性化约束会削弱防御严格性，所以后续设计应以 attack-pressure-first 为主。

4. 用户个性化/随机偏好不应作为硬防御目标。更合理的定位是：
   - 作为 x0/preference 模板的多样性来源。
   - 作为用户不可预测性的来源。
   - 不应压过 DF/RF/AWF 攻击压制目标。

5. 当前应该重点验证两个问题：
   - 旧五池模板本身是否比新五池构造器有效。
   - renderer 是否能把模板扰动真正映射成 DF/RF/AWF 可见的包序列扰动。

## 已做的组件诊断实验

所有以下结果均为 direct/template renderer 组件诊断，不是完整 diffusion 训练结果。

使用旧五池 direct template，不训练 diffusion，只改变 renderer，在完整 CW test split 上用已有 DF/RF checkpoint 评估：

- `rf_tam`: DF defended acc 约 0.9013，RF defended acc 约 0.9408。基本失败。
- `tam_obfuscation`: DF 约 0.8155，RF 约 0.8462。比 `rf_tam` 好，但仍较弱。
- `trace_index`: DF 约 0.5266，RF 约 0.8893。DF 明显改善，RF 仍弱。
- `multi_view split`: 四组 share 均约 DF 0.696-0.708，RF 0.904-0.905。由于预算被切成三份，效果有限。
- `multi_view fused`: 最好组为 DF/AWF/RF=0.50/0.25/0.25，结果：
  - DF defended acc 0.775085
  - RF defended acc 0.888679
  - worst defended acc 0.888679
  - raw bandwidth 0.299998
  - real packet retention 1.0

解释：

- `multi_view fused` 比旧 `rf_tam` 好，但没有继承 `trace_index` 对 DF 的强扰动能力，也没有继承 `tam_obfuscation` 对 RF 的最好效果。
- 当前 share 超参对结果影响很小，说明主要瓶颈不在 0.50/0.25/0.25 附近继续细分，而在 renderer 与模板/候选选择机制。

## renderer 改动

文件：

- `dmmp/projection/padding.py`
- `dmmp/utils/config.py`
- `scripts/train_defense.py`
- `scripts/run_defense.py`
- `scripts/evaluate_legacy_pool_direct.py`
- `scripts/evaluate_target_policy_direct_v1.py`
- `scripts/run_multi_view_defense_grid.ps1`
- `tests/target_policy_v1/test_core.py`

新增 renderer 选项：

```text
render_coordinate = multi_view
multi_view_mode = fused or split
multi_view_df_share = 0.40 default, best direct probe currently 0.50
multi_view_awf_share = 0.30 default, best direct probe currently 0.25
multi_view_rf_share = 0.30 default, best direct probe currently 0.25
```

`multi_view_mode=fused` 的含义：

- 不再把 dummy budget 切成 DF/AWF/RF 三份。
- 每个 dummy 同时根据 DF 方向邻域、AWF burst 边界、RF TAM slot 三个视角进行 slot scoring。
- 真实包时间戳、方向和相对顺序不改变。
- dummy 时间戳被限制在对应插入 gap 内，避免越过真实包。

已通过测试：

```powershell
conda run -n llm python -m py_compile dmmp\projection\padding.py dmmp\utils\config.py scripts\train_defense.py scripts\run_defense.py scripts\evaluate_target_policy_direct_v1.py scripts\evaluate_legacy_pool_direct.py
$env:PYTHONPATH='.'; conda run -n llm python tests\target_policy_v1\test_core.py
```

## 旧五池 mode pool 接入完整 diffusion

为了让完整 diffusion 的 x0/preference 模板和 direct 诊断里的旧五池口径一致，新增配置：

```text
--v1_mode_pool early
--v1_mode_pool legacy_direct
```

`legacy_direct` 对应五个旧防御 mode：

```text
gap-adaptive-padding
burst-obfuscation
direction-regularization
rate-smoothing
public-prototype-shaping
```

相关文件：

- `dmmp/diffusion/profile_pipeline.py`
- `dmmp/utils/config.py`
- `scripts/train_defense.py`
- `scripts/run_defense.py`

注意：

- 默认仍是 `v1_mode_pool=early`，避免破坏旧实验复现。
- 新完整实验脚本显式使用 `--v1_mode_pool legacy_direct`。

## 完整 diffusion 实验脚本

新增总控脚本：

```text
scripts/run_full_legacy_fused_diffusion.ps1
```

新增汇总脚本：

```text
scripts/summarize_full_legacy_fused_diffusion.py
```

完整实验设计：

- 全量 CW 数据集。
- 8:1:1 train/val/test split，由 DMMPv3 run 内部保存。
- Stage1 全量训练/验证 strong DF/RF surrogate 和 candidate scorer。
- Stage2 全量 train split 训练 encoder + guided diffusion。
- Stage3 全量 validation split 做 Pareto selection。
- Stage3 renderer 使用 `multi_view/fused`。
- 固定攻击评测使用全量 train/val/test。
- mixed/adaptive 攻击评测使用全量 clean + defended train/val/test。

默认实验参数中的关键项：

```text
budget = 0.30
profile_combination_mode = legacy_pool
active_pair_count = 10
active_triple_count = 5
v1_mode_pool = legacy_direct
v1_mode_prior_weight = 0.65
render_coordinate = multi_view
multi_view_mode = fused
multi_view_df_share = 0.50
multi_view_awf_share = 0.25
multi_view_rf_share = 0.25
surrogate_train_samples = 0
surrogate_val_samples = 0
encoder_train_samples = 1000000
max_samples = 0
max_classes = 0
max_generation_traces = 0
attack max_train/val/test = 0
stage3_require_quality_gate = false
attack_require_quality_gate = false
```

`stage3_require_quality_gate=false` 和 `attack_require_quality_gate=false` 是为了完整产出诊断结果，不让不达标阈值中断实验链路。

运行命令：

```powershell
cd D:\learning\TOR\defence\DMMPv3
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1
```

只查看命令不运行：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1 -DryRun
```

如果只想跑 fixed attack 后再决定 adaptive：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1 -SkipAdaptiveAttack
```

如果 defense 已经跑完，只补攻击评测：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1 -RunName <existing_run_name> -SkipDefense
```

## 当前代码验证状态

最近一次验证通过：

```powershell
conda run -n llm python -m py_compile dmmp\diffusion\profile_pipeline.py dmmp\utils\config.py scripts\train_defense.py scripts\run_defense.py scripts\summarize_full_legacy_fused_diffusion.py
$env:PYTHONPATH='.'; conda run -n llm python tests\target_policy_v1\test_core.py
```

dry-run 展开命令也通过：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_full_legacy_fused_diffusion.ps1 -DryRun -RunName dryrun_legacy_fused_fullcw -AdaptiveProtocols same_user -DiffusionTrainSteps 3 -EncoderEpochs 1 -SurrogateEpochs 1 -AttackEpochs 1 -AdaptiveEpochs 1
```

## 日志与可观测性要求

2026-07-19 正式全量实验运行时发现：`scripts/run_full_legacy_fused_diffusion.ps1` 使用普通 `conda run` 启动子进程，命令行长时间没有实时输出。实际进度是 Stage1 已完成、Stage2 diffusion checkpoint 已保存、Stage2 reference generation 已写出，但由于输出被 `conda run` 捕获/延迟，用户无法判断实验是在正常运行还是卡住。

后续写任何长实验脚本必须遵守：

1. 不要再写“长时间完全没有 log 打印”的训练/评测代码。
2. PowerShell 调用 `conda run` 时优先使用 `conda run --no-capture-output -n <env> ...`，避免 stdout/stderr 被缓存到子进程结束才显示。
3. Python 长循环必须按阶段和进度打印 `flush=True` 日志，至少包括：stage start/done、总样本数、当前 batch/step、输出文件路径、关键指标摘要。
4. 长时间无落盘的阶段，例如 defended trace generation、Stage3 Pareto candidate generation、mixed/adaptive defended pool generation，必须增加中间进度日志或 checkpoint/heartbeat 文件。
5. 总控脚本应同时把控制台日志写入 run 目录，例如 `run_dir/logs/full_experiment_*.log`，方便断点诊断；不要只依赖屏幕输出。
6. 如果已有 checkpoint 可复用，脚本应明确打印“正在复用/恢复哪个 checkpoint”，避免用户误以为从头重跑或卡死。

## 重要提醒

1. 当前完整 diffusion legacy-fused 实验脚本已经写好，但尚未在本轮对话中真正启动全量正式实验。

2. direct renderer 结果不能等价为完整 diffusion 结果。它只用于组件定位。

3. 完整实验会很重，尤其 `full_catalogue` adaptive mixed attack 会生成全量 defended train/val/test 并训练 DF/RF。

4. 后续新对话优先继续做：
   - 运行 `run_full_legacy_fused_diffusion.ps1`。
   - 监控日志。
   - 分析 `full_legacy_fused_experiment_summary.md/json`。
   - 若结果仍差，下一步应考虑 renderer candidate selector，而不是继续细分 `multi_view_df/awf/rf` share。

## 下一步候选方向

如果完整 diffusion 结果仍不理想，优先考虑：

1. Renderer candidate selector:
   - 同一条 trace、同一个 template 生成多个候选渲染版本。
   - 候选包括 `trace_index`、`tam_obfuscation`、`multi_view_fused_df_heavy` 等。
   - 用 frozen DF/RF checkpoint 的 pseudo-label confidence/margin drop 选择最差识别版本。

2. 不继续盲目细分 `multi_view_df/awf/rf` share:
   - direct fused 四组差异很小。
   - 当前主要瓶颈不是 share，而是渲染自由度和候选选择。

3. 如果论文方法需要更严谨，可把当前主线表述为：

```text
attack-pressure-first diffusion policy generation
with legacy defense-template mode priors,
multi-view TAM/sequence-aware dummy rendering,
and full fixed/mixed adaptive evaluation.
```

## RUDOLF / FRUGAL 参考结论
2026-07-19 补充：当前 DMMPv3 legacy-fused 版本在 fixed attack 下已经把 DF 从 0.974 压到 0.184，但 RF 仍为 0.448，未过 0.40 gate。核心判断是：当前渲染器主要加 outgoing dummy、保留真实时间戳，对 RF/TAM 的 incoming row、slot occupancy、burst timing 破坏不足；同时 Stage3 surrogate RF 低估了真实 fixed RF，因此后续选择器必须直接加入 fixed RF probe。

RUDOLF 可参考点：没有找到可用公开源码；公开摘要显示其关键是 burst-level real-time SAC，每个真实 burst 后同步决定扰动，不依赖完整 trace，并用 RL 在防御效果和带宽之间权衡。启发不是复现其数值，而是把 DMMPv3 的 action 从“整条 trace 的静态模板渲染”推进到“burst/TAM slot 级别的局部决策”。

FRUGAL 可参考点：公开论文和 GitHub 代码可用。论文主张以 traffic-label mutual information reduction 为目标，通过迭代选择最能降低累计 MI 的 dummy insertion positions；公开代码中 SAC actor 在 5-packet block 特征上做 top-k 位置选择，每次插入 +1 dummy 并截断尾部，训练 reward 主要等价于降低原类别置信度。该思想适合作为 DMMPv3 下一版 selector 的灵感：不要平均撒 dummy，而要 sparse/iterative 地挑高泄漏位置。

FRUGAL 结果需谨慎使用：用户指出其 Palette 对比结果与 Palette 原文观感不一致，这个怀疑应保留。另一个实现层面风险是公开 quick-start 虽写 attack_model 支持 RF，但 dqn_train_sac.py 中实际 choices/model map 主要列 DF/VarCNN/TF/AWF/NetCLR，RF 路径在 config.py 中为注释状态。因此 FRUGAL 更适合作为设计启发，不应把其论文表格数值当作 DMMPv3 的直接 benchmark。

下一版优先方向：
1. 增加 `dmmpv3_rfprobe_mi_sparse_v1` 风格 selector：Stage3 每个 trace/template 生成多个 sparse candidates，用 frozen DF/RF 的 true-label confidence、margin、entropy 或 CLUB-like proxy 选最难识别版本。
2. `stage3_fixed_probe_samples` 不再为 0；先以 fixed RF <= 0.40 作为进入 mixed/adaptive 的硬门槛。
3. Renderer 要显式 TAM-aware：action 空间从 raw packet index 扩展到 `(direction row, TAM slot/block, count, local timing spread)`，特别提高 incoming dummy share 和 incoming TAM shift。
4. 参考 FRUGAL 的 early-segment 规律，继续强化 prefix/early burst 位置，但不要只加 outgoing；建议候选 gate 包含 `dummy_incoming_share >= 0.20`、`tam_incoming_distribution_l1_shift >= 0.12~0.15`、visible overhead <= 0.30。
5. 对外表述保持为：attack-pressure-first diffusion policy generation + sparse MI/RF-probed candidate selection + full fixed/mixed adaptive evaluation。

## RF-probed MI sparse v1 实验入口
2026-07-19 已实现 RF 专项修正版入口：`scripts/run_rfprobe_mi_sparse_v1.ps1`。该版本当前只做 fixed RF 测试，不跑 fixed DF，也不跑 mixed/adaptive。

核心修正：
1. 实验入口强制实时输出：所有 `conda run` 自动展开为 `conda run --no-capture-output ... python -u ...`，并设置 `PYTHONUNBUFFERED=1`、`PYTHONIOENCODING=utf-8`。新脚本还会把 transcript 写到 `logs/<run_name>_rfprobe_mi_sparse_v1_*.log`。
2. 旧的 `scripts/run_full_legacy_fused_diffusion.ps1` 的执行 helper 也已修正为 `--no-capture-output + python -u`，避免后续再次出现长时间无控制台输出。
3. 新版本 defense 使用 RF-only guidance：`--guidance_attackers rf`、`--surrogate_df_weight 0.0`、`--surrogate_rf_weight 1.0`、`--probe_attacker rf`。
4. Renderer 改为 RF/TAM 更直接的 `--render_coordinate tam_obfuscation`，并启用 incoming 方向修正：`--direction_target incoming`、默认 `--direction_correction_strength 0.50`、`--min_incoming_dummy_share 0.20`。
5. Stage3 quick fixed probe 改为 RF-only：`--stage3_fixed_probe_attackers rf`，默认 probe 训练/验证/评估样本为 8000/2500/1024，epochs=5。
6. Stage3 增加 RF/TAM 约束字段：`stage3_min_dummy_incoming_share`、`stage3_min_tam_incoming_l1_shift`、`stage3_incoming_metric_weight`。这些默认不影响旧实验；新脚本默认 gate 为 incoming share >= 0.20、TAM incoming L1 shift >= 0.12。
7. 新脚本默认 `--no-stage3_require_quality_gate` 和 `--no-attack_require_quality_gate`，保证即使 RF gate 失败也会继续保存诊断结果和 fixed RF 指标。

运行正式 RF 专项实验：
```powershell
cd D:\learning\TOR\defence\DMMPv3
powershell -ExecutionPolicy Bypass -File scripts\run_rfprobe_mi_sparse_v1.ps1
```

只查看命令展开：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rfprobe_mi_sparse_v1.ps1 -DryRun -RunName dryrun_rfprobe_mi_sparse_v1
```

如果前面 defense 已跑完，只补 fixed RF：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rfprobe_mi_sparse_v1.ps1 -RunName <existing_run_name> -SkipDefense
```

### RF-probed MI sparse v1 首轮结果
Run: `results/dmmpv3_rfprobe_mi_sparse_v1_fullcw_seed0_20260719_171729`

结论：失败，且 fixed RF 比上一轮 legacy-fused 更差。完整 fixed RF 评估结果为 clean_acc=0.979364，fresh_defended_acc=0.657358，visible_bandwidth=0.288922，raw_retention=1.0，RF gate<=0.40 未通过。上一轮 legacy-fused fixed RF 是 0.448252，因此单纯把 outgoing/incoming share 拉平到约 52% incoming 并不能压制真实 RF。

Stage3 选中策略为 fallback，不是 valid policy：selection_policy_valid=0，selection_used_quality_fallback=1。三个 keep ratio 候选全部未过 gate：
- keep=1.00: surrogate_rf=0.5481, fixed_probe_rf=0.2646, selected, real fixed RF=0.6574
- keep=0.95: surrogate_rf=0.5669, fixed_probe_rf=0.2725
- keep=0.90: surrogate_rf=0.5840, fixed_probe_rf=0.2910

RF/TAM 方向指标虽然过了手工 gate：dummy_incoming_share≈0.521，tam_incoming_distribution_l1_shift≈0.167，但真实 RF 仍高。这说明问题不再只是 incoming share 太低，而是扰动结构对 RF 的 class-discriminative TAM pattern 破坏不足；quick fixed RF probe 也严重低估完整 RF（0.2646 vs 0.6574）。

下一步不要继续提高 incoming share。优先改：
1. Stage3 probe 必须更接近完整 RF：增大 probe train/val/sample 或直接复用 full fixed RF checkpoint/更大子集，否则 selector 会继续误判。
2. Renderer 不能只做 direction rebalance；需要 class/TAM slot-distribution flattening 或 per-class target TAM barycenter/mixup，让 added packets 破坏具体 slot pattern。
3. 加入真实 RF feedback loop：先小样本跑完整 fixed RF candidate eval，再选 candidate，而不是依赖 quick probe。

## RF TAM-shape natural-direction v2 设计
2026-07-19 已根据用户提醒修正设计：不要把 incoming dummy 当成越多越好的变量。WF/Tor trace 中 outgoing packet count 通常显著多于 incoming，因此 incoming 插得少本身是自然结构的一部分。上一轮 v1 把 dummy_incoming_share 拉到约 0.52，既不自然，也让真实 fixed RF 退化到 0.657。

新分支入口：`scripts/run_rf_tam_shape_v2.ps1`。

核心设计：
1. 方向比例跟随 clean trace：`--direction_target clean`，默认 `--direction_correction_strength 0.85`，不再强行 incoming。
2. 只设 incoming 上限，不设 incoming 下限：默认 `--stage3_max_dummy_incoming_share 0.20`、`--stage3_min_dummy_incoming_share 0.0`。
3. 新增 `tam_flatten_strength` / `tam_flatten_floor`：在每个方向内部，把 dummy placement 从已经很密的 TAM slot 推向低密度/空白 slot，目的是破坏 RF 的 slot occupancy shape，而不是改变方向比例。
4. 默认启用 `--tam_flatten_strength 0.45`、`--tam_flatten_floor 1.0`。
5. Stage3 RF probe 加强：默认 train/val/eval samples 为 30000/5000/4096，epochs=8，比 v1 的 8000/2500/1024 更接近完整 RF。
6. 仍然只做 fixed RF 测试，不跑 DF/mixed/adaptive；所有命令继续使用 `conda run --no-capture-output` + `python -u` 实时输出。

运行：
```powershell
cd D:\learning\TOR\defence\DMMPv3
powershell -ExecutionPolicy Bypass -File scripts\run_rf_tam_shape_v2.ps1
```

只看命令展开：
```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_rf_tam_shape_v2.ps1 -DryRun -RunName dryrun_rf_tam_shape_v2
```

### RF TAM-shape natural-direction v2 首轮结果
Run: `results/dmmpv3_rf_tam_shape_v2_fullcw_seed0_20260719_201036`

结论：失败，而且比 v1 更差。fixed RF clean_acc=0.980594，fresh_defended_acc=0.777010，visible_bandwidth=0.283507，raw_retention=1.0，RF gate<=0.40 未通过。

对比：
- legacy-fused fixed RF: 0.448252
- rfprobe_mi_sparse_v1 fixed RF: 0.657358
- rf_tam_shape_v2 fixed RF: 0.777010

Stage3 仍是 fallback：selection_policy_valid=0，selection_used_quality_fallback=1。选中 keep=1.0，surrogate_rf_accuracy=0.818848，quick fixed_probe_rf_accuracy=0.513916，real full fixed RF=0.777010。Stage3 四个 keep ratio 候选全部失败，keep 越小 RF 越高。

最关键诊断：v2 原本想让方向比例跟随 clean trace 并限制 incoming 上限为 0.20，但实际 dummy_incoming_share=0.710447，超过上限 gate，且 selected 只是 fallback。进一步检查数据发现：CW split 中负号方向占约 83.7%，正号方向约 16.3%；而当前代码 `build_rf_tam_input` 把 `raw > 0` 当 outgoing、`raw < 0` 当 incoming。结合用户提醒和文献共识“outgoing packet count 通常远大于 incoming”，这强烈暗示项目当前方向命名可能与数据符号约定相反：负号大概率才是 outgoing，大头被代码记成了 incoming。

下一步优先级：
1. 先确认数据符号约定，修正 DMMPv3 内部 outgoing/incoming 语义或至少在 renderer/metrics 中统一方向映射；不要继续基于当前 incoming/outgoing 命名调参。
2. 在方向语义修正前，不要继续跑 v2/v1 大实验；结果会被错误方向约束误导。
3. 方向修正后再做 TAM slot flatten/mixup，并让 Stage3 在真实 fixed RF 子集上直接评估候选。
