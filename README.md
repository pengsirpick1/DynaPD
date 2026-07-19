# DMMPv3

DMMPv3 是当前网站指纹防御项目的新 Harness 工程，位置为 `D:\learning\TOR\defence\DMMPv3`。它与旧工程 `defence\DMMP`、`defence\DMMP2`、根目录 `experiments`、`models`、`defenses` 和历史 `results` 分离。

最新 DMMP2 强 DF/RF guidance 实现已经迁移到本工程的 `dmmp/` 与 `scripts/` 下。迁移后的正式方案统一称为 DMMPv3，不再使用旧版本编号命名。DMMP2 中对应内容只作为迁移来源和历史证据。

## 当前状态

- 已迁移 CW 数据加载、seed 0 stratified split、prefix/candidate 分析、condition encoder、guided diffusion、projection/refinement/renderer、fixed/mixed attack evaluation 入口。
- 防御训练默认入口为 `scripts\train_defense.py` 或 `scripts\run_defense.py`。
- fixed/mixed 评测入口为 `scripts\run_attack_eval.py`，并提供 `evaluate_fixed.py`、`evaluate_mixed.py`、`train_mixed_attackers.py` 包装入口。
- 所有 DMMPv3 新运行默认写入 `D:\learning\TOR\defence\DMMPv3\results`。
- 本次迁移没有启动正式大规模训练，也没有移动或覆盖历史 checkpoint。

## 已确认可复用资产

已验证兼容的 fixed DF/RF checkpoint 仍保留在历史结果目录，只在 `docs/model_registry.md` 中登记：

- DF：`D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt`
- RF：`D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt`

它们是 DMMPv3 内部迭代时的稳定 fixed 评测器。只要 CW 数据集、95 类标签映射、seed 0 split、DF/RF 输入表示和 ProjectDF/ProjectRF 结构不变，就不需要因为每次改进防御策略而重训 fixed DF/RF。

## 运行入口

先检查工程结构和默认输出目录：

```powershell
cd D:\learning\TOR\defence\DMMPv3
python scripts\verify_project.py
```

查看防御训练参数：

```powershell
conda run -n llm python scripts\train_defense.py --help
```

正式训练会默认写入 `D:\learning\TOR\defence\DMMPv3\results`：

```powershell
conda run -n llm python scripts\train_defense.py --run_name dmmpv3_cw_seed0_b30_formal
```

对某个 DMMPv3 run 进行 fixed 评测：

```powershell
conda run -n llm python scripts\evaluate_fixed.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

对同一 run 进行 mixed adaptive 训练与评测：

```powershell
conda run -n llm python scripts\train_mixed_attackers.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
conda run -n llm python scripts\evaluate_mixed.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

## 关键文档

- `docs/method_spec.md`：DMMPv3 方法和阶段说明。
- `docs/experiment_protocol.md`：正式实验、评测和结果目录协议。
- `docs/project_map.md`：迁移来源、当前文件结构和入口。
- `docs/implementation_index.md`：当前代码分层、入口和阶段到文件的对应关系。
- `docs/runbook.md`：标准运行命令、评测命令和禁止事项。
- `docs/model_registry.md`：checkpoint 兼容性和复用规则。
- `docs/decisions.md`：已确认决策与仍需人工确认事项。
