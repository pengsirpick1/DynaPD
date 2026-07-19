# DMMPv3 实验协议

## 2026-07-14 协议更新

- 训练/验证阶段允许使用真实 labels 监督 strong surrogate、candidate scorer/encoder 目标和 quick fixed probe；Stage 2/3 defense guidance 的 guidance label 必须使用 frozen ProjectDF/ProjectRF 对 observed prefix 的 label-free pseudo label。
- Stage 3 候选选择使用 `selection_attack_pressure` 和 `stage3_max_label_free_attack_pressure`；该指标是 prefix-only policy pressure，只依赖已观察 prefix、policy logits、candidate mask 和攻击模型输出分布，不使用真实 labels，也不使用完整 rendered trace 攻击结果。真实标签下的 surrogate/probe accuracy 只能作为离线诊断和最终评测报告，不得回流为生成 guidance、候选排序或 hard gate 条件。
- fixed attacker 评估优先复用当前 run 下已生成且 attack 配置签名、run artifact 签名一致的 fixed metrics/checkpoint 缓存；若签名不一致或使用 `--force_retrain`，新权重和指标必须落在当前指定 run/output 目录。历史 fixed checkpoint 只能在兼容性已验证且文档记录路径/前提时作为固定评估器引用。
- mixed attacker 必须基于当前 run 的 clean + fresh defended train/val 重新训练，并在 fresh deployment defended test 上评估；不得把 fixed checkpoint 直接登记为 mixed 最终权重。

## 数据集

当前正式协议默认使用 CW：

- CW：`D:\learning\TOR\datasets\CW\CW.npz`
- seed：`0`
- classes：`95`
- labels：连续 `0..94`
- split sizes：train `84602`，val `10564`，test `10564`

OW、TemporalDrift 和 VersionDrift 暂不纳入 DMMPv3 第一版正式协议，除非后续在 `docs/decisions.md` 中另行确认。

## Prefix、模型与策略形状

- prefix length：`500`
- patch count：`200`
- early fraction：`0.40`
- attacker adapter 最大长度：`5000`
- RF TAM slots：`1800`
- RF max load time：`80.0`
- guidance attackers：`both`
- policy generator：`diffusion`

如果改变这些值，fixed checkpoint 复用资格必须重新检查。

## 带宽预算

当前 DMMPv3 第一版只设计 `0.30` 带宽开销。历史 sweep 中出现的 `0.09`、`0.18` 只保留为参考，不作为当前正式实验网格。

正式结果必须记录：

- raw dummy overhead；
- visible dummy overhead；
- selected keep ratio；
- real-packet retention；
- dummy incoming/outgoing share；
- allowed-mask violation。

## 结果目录

所有 DMMPv3 新结果必须写入：

```text
D:\learning\TOR\defence\DMMPv3\results
```

不得把 DMMPv3 新结果写入根目录 `D:\learning\TOR\results`，也不得覆盖 `defence\DMMP` 或 `defence\DMMP2` 的历史结果。

默认防御入口已经设置：

```text
output_dir = D:\learning\TOR\defence\DMMPv3\results
```

每次正式运行必须创建唯一目录，例如：

```text
D:\learning\TOR\defence\DMMPv3\results\dmmpv3_strong_surrogate_seed0_20260714_153000
```

失败或中断的正式实验必须记录到：

```text
D:\learning\TOR\defence\DMMPv3\results\failed_runs
```

## 跨项目重跑原则

从 `defence\DMMP2`、`defence\DMMP`、根目录 `experiments` 或其他旧工程迁移到 DMMPv3 时，防御流程相关内容必须在 DMMPv3 下重新运行、重新记录配置、重新保存结果。

必须重跑或重新生成：

- prefix leakage profile；
- executable candidate cells 和 allowed masks；
- prefix/condition encoder；
- policy diffusion warm-start 和 guided diffusion；
- DDIM deployment sampling；
- continuous refinement、projection、rounding 和 renderer；
- defended traces；
- mixed adaptive attackers；
- DMMPv3 fixed/mixed 评测摘要。

可以复用：

- 已验证兼容的 fixed DF/RF checkpoint 作为固定评测器。

复用 fixed DF/RF 的原因是它们不是 DMMPv3 防御方法的中间产物，而是稳定评测器。保持 fixed 评测器不变，有利于比较 DMMPv3 内部逐步改进的防御策略。

## Fixed DF/RF 评测

fixed attackers 只在 clean train/val 上训练，然后评测 fresh defended test。DMMPv3 内部迭代可复用已验证 fixed DF/RF checkpoint，但每次正式结果必须记录 checkpoint 路径和兼容性前提。

复用前必须确认：

- 数据集路径与类别数一致；
- 标签映射一致；
- train/val/test 划分一致；
- DF 输入表示：`sign(trace)`，形状 `[N, 1, 5000]`；
- RF 输入表示：TAM，形状 `[N, 2, 1800]`；
- 模型结构与 checkpoint 参数兼容；
- checkpoint 可加载并可 forward；
- clean test 指标与历史记录一致。

当前可复用候选见 `docs/model_registry.md`。

## Mixed Adaptive DF/RF 评测

mixed attackers 必须随对应 DMMPv3 defended traces 重新训练。fixed checkpoint 可以作为初始化，但不能被标记为最终 mixed checkpoint。

必须记录：

- clean/defended 比例；
- defended train/val/test 来源；
- 防御 run directory；
- 初始化 checkpoint；
- 最终 mixed checkpoint；
- 训练配置；
- clean 与 fresh defended 指标。

不同防御 run 或不同 defended dataset 训练出的 mixed attacker 不得互相覆盖。

## 正式结果内容

每个正式结果目录至少包含：

- 完整实验配置；
- seed 和 split metadata；
- 数据集来源；
- 防御 checkpoint 路径；
- 攻击 checkpoint 路径；
- 日志；
- DF 和 RF 指标；
- 带宽开销；
- allowed-mask violation；
- clipping、refinement、projection 和 rounding 统计；
- smoke/formal 标记；
- success、failed 或 interrupted 状态；
- 中文结果摘要；
- 代码版本；若没有 Git 仓库，则记录 “无 Git 仓库”。

## 公平对比规则

- fixed 与 mixed 结论必须来自独立攻击训练和评测摘要。
- smoke run 不得作为正式结论。
- defended train/val 样本不得复用为最终 fresh deployment test。
- 若类别数、标签映射、prefix length 或 attacker input representation 不同，不得直接对比，除非差异被明确说明。
- 必须同时报告 DF 和 RF，不得只优化更容易被扰动的攻击器。

## 验收指标

至少记录：

- clean accuracy；
- defended accuracy；
- true-label confidence；
- prediction entropy；
- max confidence；
- raw 和 visible bandwidth overhead；
- real-packet retention；
- allowed-mask violation rate；
- clipping rate；
- dummy incoming/outgoing share；
- template entropy；
- projection 或 rounding error；
- mixed adaptive defended accuracy。

DMMPv3 pass/fail 阈值仍为 `[需要人工确认]`。
