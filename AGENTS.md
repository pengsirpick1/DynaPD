# DMMPv3 协作指南

DMMPv3 是当前网站指纹防御项目的新 Harness 工程，用于训练防御、评测攻击器、管理模型权重和保存实验结果。

## 不可违反的约束

- 不得删除、移动或覆盖旧工程代码、旧 checkpoint、数据集或历史结果。
- 当前方案统一称为 DMMPv3，不再使用旧版本编号命名。
- 所有 DMMPv3 新结果必须保存到 `D:\learning\TOR\defence\DMMPv3\results`。
- DMMPv3 生成的是防御策略，不是完整流量。
- 在线防御条件只能来自已观察到的 prefix，不得把网站标签或未来包作为防御条件。
- fixed DF/RF checkpoint 只有在数据集、标签映射、数据划分、输入表示、模型结构、加载能力和 clean 指标都通过检查后才能复用。
- mixed attacker 必须基于 clean 与 defended traces 重新训练，不得把 fixed checkpoint 标记为最终 mixed checkpoint。
- 无法从代码或已有文档中确认的内容，一律写成 `[需要人工确认]`。

## 任务阅读顺序

- 方法或架构任务：先读 `docs/method_spec.md`，再读 `docs/project_map.md`。
- 实验或协议任务：先读 `docs/experiment_protocol.md`，再读 `docs/model_registry.md`。
- 入口或模块路径任务：读 `docs/implementation_index.md` 和 `docs/runbook.md`。
- checkpoint 任务：修改或引用 `models/` 前，必须先读 `docs/model_registry.md`。
- 当前阶段状态：读 `tasks/current_task.md`。
- 设计变更：同步更新 `docs/decisions.md`。

## fixed 与 mixed 攻击器规则

- fixed attacker 只在 clean train/val 上训练，用于评测 fresh defended test。
- DMMPv3 内部逐步改进防御方法时，已验证 fixed DF/RF 可作为稳定评测器复用。
- mixed attacker 必须在 clean 与 defended 混合数据上训练，并用独立 fresh deployment defended test 评测。
- fixed checkpoint 可以作为 mixed 初始化，但不能作为 mixed 最终模型。
- 每个 mixed checkpoint 必须记录初始化 checkpoint、clean/defended 比例、defended traces 来源、防御 checkpoint、训练配置、最终 checkpoint 和指标。

## 结果与 checkpoint 保存规则

- 所有 DMMPv3 新结果必须保存到 `results/` 下的唯一运行目录，不得覆盖历史运行。
- 失败或中断的正式实验必须记录到 `results/failed_runs/`。
- 新防御模型权重保存到 `models/defense/` 或对应 run 目录。
- 新 fixed attacker 权重保存到 `models/attackers/fixed/{df,rf}/`，前提是验证通过；否则只在 `docs/model_registry.md` 中记录外部路径。
- 新 mixed attacker 权重保存到 `models/attackers/mixed/{df,rf}/` 或对应 run 的 attack evaluation 目录。

## smoke test 与正式实验

- smoke test 只用于验证导入、shape、投影约束、renderer 或 tiny data flow。
- smoke test 结果不能写成正式实验结论。
- 正式实验必须保存完整配置、seed、数据划分、防御 checkpoint、攻击 checkpoint、日志、指标、带宽开销、allowed-mask violation、projection/rounding 统计、运行状态、摘要 Markdown，以及代码版本信息。若当前项目没有 Git，则记录 “无 Git 仓库”。

## 完成任务前必须执行

在 `D:\learning\TOR\defence\DMMPv3` 下运行：

```powershell
python scripts\verify_project.py
```

同时确认：

- 没有覆盖旧 checkpoint 或历史结果。
- 若任务改变协议、路径、模型状态或设计决策，相关文档已更新。
