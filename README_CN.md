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
    -> timeout 后 dummy 注入 + 仅作用于后续到达包的有界 delay
```

默认在线控制器是
[`streaming_state_machine.py`](streaming_state_machine.py)。当一个服务器到
客户端（下载方向）的 burst 停止后，控制器等待 5 个 bin 的真实 timeout；timer
触发时，只基于该 burst 已观测到的持续时间和包量匹配事件类型，并选择离线校准
的分配系数。dummy 严格在 timeout 之后发送；delay 仅作用于 timeout 激活后才到达
且尚未发出的包。

```python
from streaming_state_machine import defend_stream

defended_trace = defend_stream(clean_trace, seed=0, rho=0.35)
```

部署协议采用 `tail0`：最后一个尚未由真实网络 timeout 确认结束的 burst 不执行
动作。此前 phase-only 流式实现保留在 Git 历史中；其动作时间语义已被 timeout
协议取代，不能再作为严格因果基线。

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
| **Timeout 事件关键点 RT（默认）** | 11.03% | 21.71% | 17.88% | 20.31% | 19.08% | **21.71%** | 15.24% |

事件关键点控制器在与其 96 条校准区间不重叠的 105,634 条 CW trace 上完成评测。
`dummy_before_decision`、`delay_before_activation`、`delay_after_emission` 和
`future_packet_read` 四项审计均为零。该设计使在线动作明确对应离线发现的局部
burst 形状。此前存在回填时间戳问题的实现及修正说明见
[docs/EVENT_KEYPOINT_RT.md](docs/EVENT_KEYPOINT_RT.md)。

## 仓库结构

```text
dynapd/                                   核心数据、目标函数和模型工具
scripts/build_event_keypoint_utility.py   离线事件 utility 校准脚本
scripts/                                  离线 Teacher/search 与评估工具
streaming_state_machine.py                timeout 驱动的事件关键点 RT 控制器
causal_event_renderer.py                  显式时间戳渲染器
configs/dynapd_rt_event_utility_timeout.npy timeout 事件 utility 表
reproducibility/event_keypoint_timeout_fullcw/ 修正后的全量 CW 记录
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
