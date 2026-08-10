**中文** | [English](README.md)

# DynaPD

**面向网站指纹攻击的动态流量扰动防御框架**

DynaPD 面向闭世界网站指纹识别（Website Fingerprinting, WF）防御，当前包含两条互补的研究与部署分支：

| 分支 | 目标 | 决策时可使用的信息 |
|---|---|---|
| **离线 DynaPD** | 追求高防御效果，并研究随机化与抗自适应训练 | 完整 trace 与多替代攻击模型反馈 |
| **DynaPD-RT** | 实现严格因果、无标签的实时流式防御 | 当前及历史已到达的数据包 |

本仓库公开源码、紧凑配置与可复核实验记录；不公开 CW 数据集、攻击模型权重、WFlib 源码、生成后的防御流量及日志。

## 方法概览

```text
离线 DynaPD
完整 trace -> 候选动作生成 -> RF / DF / AWF 收益评估
          -> 带约束的动作选择 -> dummy 插包 + 时延扰动

DynaPD-RT
流量逐包到达 -> 运行预算 + 下载 burst 状态 -> utility 查表
            -> burst 尾部插 dummy + 有界因果 delay
```

### 离线 DynaPD

离线控制器从完整 trace 中生成分层 Top-128 候选动作，并用三类互补的替代攻击模型评估归一化 margin 收益：RF（TAM/burst）、DF（方向序列）和 AWF（burst-family）。主线确定性配置为 `norm_weighted_r80_d10_v10`，RF/DF/AWF 权重为 `0.80/0.10/0.10`。

随机化动作选择与 Completion 带宽补全属于离线研究分支，用于探索自适应攻击下的鲁棒性；它们不是 DynaPD-RT 的在线推理链路。

### DynaPD-RT

DynaPD-RT 将离线经验压缩为一个按 `(phase, direction, dose)` 索引的紧凑 utility 表。在线运行仅维护：已观测数据包数、正方向/下载 burst 状态、已用 dummy 预算和 utility 表。当一个 burst 结束时，系统依据当前预算实施有界扰动；全过程不需要网站真实标签，也不在线调用 RF、DF、AWF、TF 或 VarCNN。

```python
from streaming_state_machine import defend_stream

defended_trace = defend_stream(clean_trace, seed=0, rho=0.25)
```

部署主结果采用 **`tail0`**：最后一个尚未通过真实 timeout 结束的 burst 不执行动作。`tail1` 是允许 trace 结束时额外动作的消融；batch controller 使用完整 trace 信息，仅作为上限对照，不能视为在线部署结果。

## 主要结果

以下均为闭世界攻击准确率，数值越低表示防御越强。`WC` 为被评估攻击模型中的最高准确率。

### 离线 DynaPD：CW 512-trace 子集

| 配置 | RF | DF | AWF | VarCNN held-out | TF held-out | WC |
|---|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 12.70% | 8.40% | 3.32% | 7.03% | 17.77% | 17.77% |

### DynaPD-RT：完整 CW 评估（105,730 条 trace）

| 版本 | RF | DF | TF | AWF | VarCNN | WC | 实测带宽开销 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Streaming `tail0` | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | 15.31% | 16.36% |
| Streaming `tail1`（消融） | 10.13% | 13.68% | 10.43% | 8.18% | 6.07% | 13.68% | 17.34% |
| Batch 全信息上限 | 5.85% | 5.57% | 5.90% | 6.96% | 3.53% | 6.96% | 17.20% |

`tail0` 的因果审计记录为 0 次 future-packet access。全量生成使用 20 个 CPU worker，平均摊销生成时间为 1.98 ms/trace；单条在线 state-machine 的独立测量约为 12 ms/trace。

完整带宽曲线、同口径随机基线、原始 manifest 与实验边界说明见 [docs/RESULTS.md](docs/RESULTS.md)。

## 目录结构

```text
dynapd/                         核心数据、渲染、目标函数与模型工具
scripts/                        离线 Teacher/search 与评估脚本
streaming_state_machine.py      DynaPD-RT 严格因果状态机
streaming_allcw_mp.py           全量 CW RT 评估
streaming_allcw_bw_sweep.py     RT 带宽扫描
random_streaming_baseline_bw_sweep.py
                                 同口径因果随机基线
configs/dynapd_rt_utility.json  公开的紧凑 RT utility 表
reproducibility/                已追踪的 manifest 与运行摘要
docs/RESULTS.md                 整理后的实验记录
docs/REPRODUCIBILITY.md         数据、切分与复现协议
```

## 安装

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
```

五模型评估还需要兼容版本的 WFlib、CW signed-timestamp trace 和干净攻击模型 checkpoint。可将 WFlib 放在 `wflib_copy/`，或通过 `PYTHONPATH` 提供；运行评估时传入本地的数据与模型路径。

## 常用入口

```bash
# 离线多替代模型控制器
python scripts/stage_b_run_ensemble_oracle_e2b_completion.py --help
python scripts/stage_b_run_ensemble_oracle_e2b_rand.py --help

# 流式 DynaPD-RT 评估
python streaming_allcw_mp.py --help
python streaming_allcw_bw_sweep.py --help
python random_streaming_baseline_bw_sweep.py --help
```

## 实验边界

- 当前公开的 DynaPD-RT 结果是**非自适应攻击者**评估结果。
- RT utility 表应称为**小规模校准（small calibration）**，而不是按类别学习的 few-shot defense；它本质上是全局的 12-cell phase/direction/dose 汇总表。
- delay 上限为 64 bins；在 80 秒、1800 bins 表征下，单个受影响包的上界约为 2.84 秒。页面完成时间开销必须使用未截断 trace 计算，见 `scripts/measure_page_completion.py`。
- 当前尚未指定开源许可证。

完整复现说明见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)。
