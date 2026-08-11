**中文** | [English](README.md)

# DynaPD

**面向网站指纹攻击的动态流量扰动防御**

DynaPD 面向闭世界网站指纹（Website Fingerprinting, WF）防御。其部署链路
**DynaPD-RT** 将离线阶段昂贵的多替代攻击模型搜索压缩为紧凑的 burst 事件
utility 表，并在流量逐包到达时严格因果地执行。在线部署不需要网站真实标签、
完整 trace，也不查询在线攻击模型。

仓库不公开 CW 原始流量、WFlib 源码、攻击模型权重、生成后的防御流量和运行日志。

## 方法概览

```text
离线经验发现
完整 trace + RF/DF/AWF 替代攻击模型收益
    -> 聚合可观测下载 burst 事件的动作效用
    -> (流量阶段, out, 持续时间分桶, 包量分桶) -> 分配系数

在线 DynaPD-RT
流量逐包到达 -> token 预算 + 下载 burst 状态
    -> 识别已结束的局部 burst 事件 -> utility 查表
    -> burst 尾部 dummy 注入 + 有界因果 delay
```

默认在线控制器是
[`streaming_state_machine.py`](streaming_state_machine.py)。每当一个服务器到
客户端（下载方向）的 burst 结束时，控制器只基于该 burst 已观测到的持续时间和
包量匹配事件类型，并选择离线校准的分配系数。

```python
from streaming_state_machine import defend_stream

defended_trace = defend_stream(clean_trace, seed=0, rho=0.213)
```

部署协议采用 `tail0`：最后一个尚未由真实网络 timeout 确认结束的 burst 不执行
动作。旧的 phase-only 实现保留在
[`streaming_state_machine_phase_baseline.py`](streaming_state_machine_phase_baseline.py)，
用于消融与对照。

## 实验结果

下表为闭世界攻击准确率，越低表示防御越强；`WC` 是 RF、DF、TF、AWF、VarCNN
中最高的防御后准确率。

### 离线 DynaPD 参考结果

离线控制器使用完整 trace，并以 `RF/DF/AWF = 0.80/0.10/0.10` 的归一化收益
选择动作。它用于离线经验发现与效果参考，不是在线部署结果。

| 配置 | RF | DF | TF | AWF | VarCNN | WC | 实测 BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| `norm_weighted_r80_d10_v10` | 13.36% | 11.41% | 17.55% | 6.47% | 25.60% | 25.60% | 7.19% |

### DynaPD-RT 严格因果流式评估

| 控制器 | RF | DF | TF | AWF | VarCNN | WC | 实测 BWO |
|---|---:|---:|---:|---:|---:|---:|---:|
| **事件关键点 RT（`tail0`，默认）** | 11.54% | 15.39% | 11.63% | 10.15% | 16.24% | 16.24% | 16.17% |
| Phase-only RT（`tail0`，基线） | 11.72% | 15.31% | 12.01% | 9.64% | 7.21% | **15.31%** | 16.36% |

事件关键点控制器在与其 96 条校准区间不重叠的 105,634 条 CW trace 上完成评测，
因果审计记录为零次 future-packet access。当前“持续时间 + 包量”的事件定义与
phase-only 基线性能接近，但尚未带来可声称的显著提升；它的价值是让在线动作能
明确对应离线发现的局部 burst 形状。完整记录见
[docs/EVENT_KEYPOINT_RT.md](docs/EVENT_KEYPOINT_RT.md)。

## 仓库结构

```text
dynapd/                                   核心数据、目标函数和模型工具
scripts/build_event_keypoint_utility.py   离线事件 utility 校准脚本
scripts/                                  离线 Teacher/search 与评估工具
streaming_state_machine.py                默认事件关键点 RT 控制器
streaming_state_machine_phase_baseline.py Phase-only RT 基线
configs/dynapd_rt_event_utility.npy       公开的紧凑事件 utility 表
reproducibility/event_keypoint_rt_fullcw/ 全量 CW manifest 和五模型结果
docs/RESULTS.md                           基线实验记录
docs/EVENT_KEYPOINT_RT.md                 事件关键点实验记录
docs/REPRODUCIBILITY.md                   数据、切分和复现协议
```

## 安装

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
python -m pip install -r requirements.txt
python -m pip install -e .
```

五模型评测还需要兼容版本的 WFlib、CW signed-timestamp trace 及干净攻击模型
checkpoint。请将 WFlib 放在 `wflib_copy/`，或通过 `PYTHONPATH` 提供；评测时
传入本地数据和模型路径。

## 复现入口

```bash
# 使用本地校准数据重新构建事件条件 utility 表。
python scripts/build_event_keypoint_utility.py --help

# 使用本地 CW 数据和模型权重运行严格因果 RT 带宽扫描。
python streaming_allcw_bw_sweep.py --help

# 离线多替代攻击模型参考控制器。
python scripts/stage_b_run_ensemble_oracle_e2b_completion.py --help
```

## 实验边界

- 当前 RT 结果为**非自适应攻击者**评测。
- 公开事件表属于全局的**小规模校准（small calibration）**，不是类别级 few-shot
  模型。
- delay 上限为 64 bins；页面完成时间开销必须在未截断 trace 上测量，参见
  `scripts/measure_page_completion.py`。
- 当前尚未指定开源许可证。
