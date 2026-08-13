**中文** | [English](README.md)

# DynaPD

**面向网站指纹攻击的动态流量扰动防御**

DynaPD 是面向闭世界网站指纹（Website Fingerprinting, WF）攻击的研究型防御框架。
可部署控制器 **DynaPD-RT** 将离线多替代攻击模型获得的动作收益压缩为紧凑 utility
表，并以严格因果、timeout 驱动的流式状态机执行。在线运行时不需要网站真实标签、
完整 trace，也不查询在线攻击模型。

本仓库不公开 CW 原始流量、WFlib 源码、攻击模型权重、生成后的防御流量和日志。

## DynaPD-RT

```text
离线校准
完整 trace + RF / DF / AWF 替代攻击模型收益
  -> 可观测下载 burst 事件的 utility
  -> (流量阶段, 方向, 持续时间分桶, 包量分桶) -> 动作 profile

在线控制器
流量逐包到达 -> token 预算 + 下载 burst 状态
  -> 空闲 timeout -> utility 查表
  -> timeout 后 dummy 注入 + 仅作用于后续到达包的有界 delay
```

事件状态无需标签即可观察。每个动作 profile 包含 dummy 剂量系数、间隔、前向 delay
窗口和最大 delay。utility 表仅保存全局聚合经验，不包含原始流量、网站标签或模型权重。

默认控制器为 [`streaming_state_machine.py`](streaming_state_machine.py)：

```python
from streaming_state_machine import defend_stream, load_utility

load_utility("configs/dynapd_rt_event_utility.npy")
defended_trace = defend_stream(clean_trace, seed=0, rho=0.35)
```

对于每个下载方向 burst，控制器在 `burst_end + GAP_THRESH + 1` 设置 timer。
timer 到期后，控制器查 utility 表选择动作 profile；dummy 严格从决策时刻之后发出，
delay 只作用于激活后才到达且尚未发出的包。每条 trace 都记录动作时间语义审计。

## 全量 CW 结果

下表为闭世界攻击准确率，越低说明防御越强。`WC` 是 RF、DF、TF、AWF、VarCNN
五个攻击模型中防御后准确率的最大值。

| 控制器 | RF | DF | TF | AWF | VarCNN | WC | 实测带宽开销 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **DynaPD-RT** | 7.01% | 11.41% | 9.87% | 11.96% | 10.18% | **11.96%** | 19.11% |

该结果覆盖 `105,634` 条 CW trace，并与 `96` 条校准区间不重叠。配置为 `rho=0.35`
及仓库中的 utility 表。18 个 worker 下的并行生成吞吐量为 1.37 ms/trace。时间语义
审计中的 `dummy_before_decision`、`delay_before_activation`、
`delay_after_emission` 和 `future_packet_read` 均为零。

标准评测遵循项目中各攻击模型的原生固定前缀预处理，CW 表示长度为 5,000 包。
物理 defended stream 保留所有输入包和显式时间戳。对 `92,940` 条“前 5,000 个
观测包内没有已知真实包被挤出”的控制子集，WC 仍为 10.83%。因此结果并不主要依赖
真实包被截断；但该结论仅对应 fixed-prefix attacker，不能外推为任意长度自适应攻击者。

### 离线参考结果

离线版本用于发现动作经验、刻画完整 trace 上的上限控制器，并非在线部署结果。

| 配置 | RF | DF | TF | AWF | VarCNN | WC | 实测带宽开销 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

## 安装

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
```

五模型评测还需要兼容版本的 WFlib、带符号时间戳的 CW 流量，以及 clean
RF/DF/TF/AWF/VarCNN checkpoint。这些材料需要由使用者在本地提供。

## 复现入口

```bash
# 使用独立校准区间重建 profile utility 表。
python scripts/build_event_keypoint_utility.py --help

# 小规模严格流式导出。
python scripts/run_dynapd_rt_eval.py --data /path/to/CW.npz --output-dir results/rt_small

# 多进程导出完整 CW 区间。
python scripts/run_dynapd_rt_fullcw.py --data /path/to/CW.npz --output-dir results/rt_full --workers 18
```

审计要求见 [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)，结果记录见
[docs/RESULTS.md](docs/RESULTS.md)。

## 实验边界

- 当前结果针对非自适应攻击者。
- 公共 utility 表属于全局小规模校准（small calibration），不是类别级 few-shot 模型。
- 当前攻击评测是 fixed-prefix，不能外推至任意未截断或长度自适应攻击者。
- 当前尚未指定开源许可证。
