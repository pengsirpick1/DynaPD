# DMMPv3 项目地图

扫描日期：2026-07-14

扫描根目录：`D:\learning\TOR`

DMMPv3 根目录：`D:\learning\TOR\defence\DMMPv3`

## 顶层资产

| 路径 | 作用 |
|---|---|
| `datasets` | CW、OW、TemporalDrift、VersionDrift 数据。 |
| `defence\DMMPv3` | 当前 DMMPv3 Harness 和正式工作目录。 |
| `defence\DMMP2` | 最新强 DF/RF guidance 流程的迁移来源。 |
| `defence\DMMP` | 旧 DMMP 方法说明、历史实现和历史结果。 |
| `attack` | Website Fingerprinting attack 参考实现。 |
| `defence\ALERT_Code` | ALERT 与 concept drift 参考。 |
| `defence\reinforce\FRUGAL-ndss-main` | FRUGAL/RL 参考。 |
| `experiments` | 早期单脚本实验，当前只作历史参考。 |
| `results` | 旧工程历史结果；DMMPv3 新结果不得写入这里。 |
| `tmp` | smoke 和诊断输出。 |

## DMMPv3 当前目录

| 路径 | 职责 |
|---|---|
| `dmmp` | 已迁移的核心 Python package。 |
| `scripts` | 防御训练、fixed/mixed 评测、辅助 sweep 和校验入口。 |
| `docs` | 方法、协议、项目地图、决策和模型注册表。 |
| `models` | DMMPv3 自有 checkpoint 或后续复制进来的固定资产。 |
| `results` | DMMPv3 新训练、生成、评测和失败结果。 |
| `configs` | 后续正式 YAML/JSON 配置位置。 |
| `tests` | 后续单元测试和 smoke tests。 |
| `logs` | 后续独立日志。 |
| `tasks` | 当前任务状态。 |

## 已迁移代码

| DMMPv3 文件 | 来源 | 状态 |
|---|---|---|
| `dmmp\data\cw.py` | `defence\DMMP2\dmmp2\data.py` | 已迁移。 |
| `dmmp\evaluation\attack_models.py` | `defence\DMMP2\dmmp2\attack_models.py` | 已迁移。 |
| `dmmp\encoders\prefix.py` | `defence\DMMP2\dmmp2\prefix.py` | 已迁移。 |
| `dmmp\encoders\leakage.py` | `defence\DMMP2\dmmp2\leakage.py` | 已迁移。 |
| `dmmp\guidance\candidate_scorer.py` | `defence\DMMP2\dmmp2\candidate_scorer.py` | 已迁移。 |
| `dmmp\encoders\condition_encoders.py` | `defence\DMMP2\dmmp2\condition_encoders.py` | 已迁移。 |
| `dmmp\diffusion\models.py` | `defence\DMMP2\dmmp2\models.py` | 已迁移。 |
| `dmmp\guidance\diffusion_guidance.py` | `defence\DMMP2\dmmp2\diffusion_guidance.py` | 已迁移。 |
| `dmmp\projection\padding.py` | `defence\DMMP2\dmmp2\padding.py` | 已迁移。 |
| `dmmp\guidance\strong_surrogates.py` | `defence\DMMP2\dmmp2\strong_surrogates.py` | 已迁移。 |
| `dmmp\constraints\preferences.py` | `defence\DMMP2\dmmp2\preferences.py` | 已迁移。 |
| `dmmp\constraints\user_profiles.py` | `defence\DMMP2\dmmp2\user_profiles.py` | 已迁移。 |
| `dmmp\diffusion\profile_pipeline.py` | `defence\DMMP2\dmmp2\v4_pipeline.py` | 已迁移，当前承载 DMMPv3 主流程。 |
| `dmmp\evaluation\profile_attacks.py` | `defence\DMMP2\dmmp2\v4_attacks.py` | 已迁移，当前承载 DMMPv3 profile-aware attack evaluation。 |

说明：内部 `v4_` 文件名和函数名前缀暂不大规模重命名，以降低迁移风险；对外文档、默认配置、run 名称和结果目录统一使用 DMMPv3。

## 当前命令

进入工程：

```powershell
cd D:\learning\TOR\defence\DMMPv3
```

检查 Harness：

```powershell
python scripts\verify_project.py
```

查看防御训练入口：

```powershell
conda run -n llm python scripts\train_defense.py --help
```

运行正式防御训练时默认输出到本工程 `results`：

```powershell
conda run -n llm python scripts\train_defense.py --run_name dmmpv3_cw_seed0_b30_formal
```

评测入口：

```powershell
conda run -n llm python scripts\evaluate_fixed.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
conda run -n llm python scripts\train_mixed_attackers.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
conda run -n llm python scripts\evaluate_mixed.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

辅助入口：

```powershell
conda run -n llm python scripts\resume_stage3.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
conda run -n llm python scripts\sweep_guidance.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
conda run -n llm python scripts\validate_strong_surrogates.py
conda run -n llm python scripts\validate_v4.py --run_dir D:\learning\TOR\defence\DMMPv3\results\<run_name>
```

更完整的命令说明见 `docs\runbook.md`。当前代码分层、入口和阶段到文件的对应关系见 `docs\implementation_index.md`。

## 数据目录

| 数据集 | 路径 | 当前状态 |
|---|---|---|
| CW | `D:\learning\TOR\datasets\CW\CW.npz` | DMMPv3 第一版正式数据集。 |
| OW | `D:\learning\TOR\datasets\OW\OW.npz` | 暂不纳入第一版正式协议。 |
| TemporalDrift | `D:\learning\TOR\datasets\TemporalDrift` | 暂不纳入第一版正式协议。 |
| VersionDrift | `D:\learning\TOR\datasets\VersionDrift` | 暂不纳入第一版正式协议。 |

## 结果与 checkpoint

DMMPv3 新结果路径：

```text
D:\learning\TOR\defence\DMMPv3\results
```

DMMPv3 自有模型路径：

```text
D:\learning\TOR\defence\DMMPv3\models
```

历史 fixed DF/RF checkpoint 仍保留在旧路径，仅在 `docs\model_registry.md` 中索引。正式 DMMPv3 run 不得覆盖旧结果。

## 迁移后仍需注意

1. 任何跨项目防御中间产物都不能直接作为 DMMPv3 正式结果。
2. DMMPv3 内部改进防御策略时，fixed DF/RF 可作为稳定评测器复用。
3. mixed attacker 必须跟随新的 defended traces 重新训练。
4. 正式训练前先运行 `scripts\verify_project.py`。
5. 新增协议、阈值、baseline 或 checkpoint 复制策略时，必须同步更新 `docs\decisions.md`。
