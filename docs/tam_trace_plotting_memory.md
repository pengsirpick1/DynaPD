# TAM Trace Plotting Memory

以后如果需要进行“流量表征绘图”“TAM 绘图”“clean/defended/purified 对比图”，默认使用本文件记录的绘图方式。

## Canonical Script

```text
D:\learning\TOR\defence\DMMPv3\scripts\plot_tam_clean_defended_purified.py
```

## Plotting Representation

默认采用 TAM 时间槽计数图，参考：

```text
A. Panchenko, F. Lanze, J. Pennekamp, T. Engel, A. Zinnen,
M. Henze, and K. Wehrle.
Website fingerprinting at internet scale.
NDSS 2016.
```

以及当前参考论文：

```text
C:\Users\Pengtor\Desktop\papers\2026-f1760-paper.pdf
page 5, Figure 1
```

绘图规则：

```text
num_slots = 1000
slot_ms = 80
x axis = Time Slots
y axis = Pkt. Number
outgoing packets = blue positive bars
incoming packets = red negative bars
```

当前 signed timestamp trace 中：

```text
value > 0  -> outgoing packet
value < 0  -> incoming packet
abs(value) -> packet timestamp in seconds
slot = floor(abs(value) / 0.08)
```

## Default Comparison Panels

基础三子图：

```text
Clean Trace
Defended Trace
Purified Trace
```

如果存在截断后数据，默认扩展为四子图：

```text
Clean Trace
Defended Trace
Actual Purified Trace
Direction-count Truncated Purified Trace
```

## Standard Commands

三子图，中文标签，共享纵轴：

```powershell
D:\Miniconda3\envs\llm\python.exe -u scripts\plot_tam_clean_defended_purified.py --purifier-run-dir results\purifier_runs\purifier_b010_model_policy_retrain_20260722 --row-index 0 --num-slots 1000 --slot-ms 80 --language zh --y-scale shared
```

三子图，中文标签，独立纵轴：

```powershell
D:\Miniconda3\envs\llm\python.exe -u scripts\plot_tam_clean_defended_purified.py --purifier-run-dir results\purifier_runs\purifier_b010_model_policy_retrain_20260722 --row-index 0 --num-slots 1000 --slot-ms 80 --language zh --y-scale independent
```

四子图，加入 direction-count truncated purified trace：

```powershell
D:\Miniconda3\envs\llm\python.exe -u scripts\plot_tam_clean_defended_purified.py --purifier-run-dir results\purifier_runs\purifier_b010_model_policy_retrain_20260722 --row-index 0 --truncated-manifest results\purifier_runs\purifier_b010_model_policy_retrain_20260722\manifests\diffusion_defense_truncated_manifest.csv --num-slots 1000 --slot-ms 80 --language zh --y-scale independent
```

## Notes

- `shared` 纵轴用于比较数量级，尤其适合展示 actual purified 是否出现 dense trace。
- `independent` 纵轴用于观察每条 trace 自身的局部形态。
- 论文复现风格可使用 `--language paper`，中文展示可使用 `--language zh`。
- 如果用户后续只说“按之前 TAM 方式画图”，默认就是本文件定义的方式。
