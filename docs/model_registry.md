# DMMPv3 模型注册表

## 2026-07-14 复用规则补充

- 当前实现的 attack eval 会先复用当前 run/output 目录中已有的 fixed/mixed metrics + checkpoint 缓存；这只是同一 run 内的结果缓存复用，不等同于把历史 checkpoint 登记为新结果。
- 使用 `--force_retrain` 重新训练 attack eval 时，新 checkpoint 应写入当前指定目录，例如 `results\<run_name>\attack_eval\<protocol>\<kind>\` 或显式 `--output_dir`，不得覆盖历史目录中的候选 checkpoint。
- 已验证历史 fixed DF/RF checkpoint 仍只能作为固定评估器候选引用；正式使用前仍需满足本表的兼容性条件。mixed attacker 不适用 fixed 复用规则，必须随当前 defended traces 重新训练并登记最终 mixed checkpoint。

本注册表只索引已有 checkpoint，不移动、不删除、不覆盖原文件。

兼容性检查时间：2026-07-14。

检查环境：`llm` conda 环境。检查时使用 DMMP2 原始 ProjectDF/ProjectRF 结构与迁移到 DMMPv3 后的同构 `dmmp\evaluation\attack_models.py` 结构进行对照。由于这些 checkpoint 是本地项目生成的字典文件，内部字段为 `model_state`，在 PyTorch 2.10 下检查时使用受信任本地加载 `torch.load(..., weights_only=False)`，再提取 `model_state` 进行 strict state load。

## 已验证 fixed checkpoint 候选

| 模型类型 | 攻击器 | 场景 | checkpoint 路径 | 数据集 | seed | 状态 | 对应结果 |
|---|---|---|---|---|---:|---|---|
| attacker | DF | fixed | `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt` | CW 95 类 | 0 | 可复用固定评测器：strict state load 通过；clean test acc `0.973684`；best val `0.972075` | `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\df\fixed_df_metrics.json` |
| attacker | RF | fixed | `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt` | CW 95 类 | 0 | 可复用固定评测器：strict state load 通过；clean test acc `0.976619`；best val `0.978512` | `D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_metrics.json` |
| attacker | DF | mixed | 尚未训练 | - | - | unavailable | - |
| attacker | RF | mixed | 尚未训练 | - | - | unavailable | - |

## 兼容性证据

共享 run metadata：

- run directory：`D:\learning\TOR\results\dmmp2_v5_fixed_oriented_seed0_bwo30`
- data source：`D:\learning\TOR\datasets\CW\CW.npz`
- seed：`0`
- prefix length：`500`
- patch count：`200`
- budget：`0.30`
- split sizes：train `84602`，val `10564`，test `10564`
- label mapping：连续 `0..94`
- splits：互不重叠，覆盖全部 `105730` 个样本

DF 检查：

- architecture：`ProjectDF`
- input representation：`sign(trace)` -> `[N, 1, 5000]`
- checkpoint keys：`best_val`、`classes`、`model_state`
- checkpoint classes：`95`
- missing/unexpected state keys：`0/0`
- zero batch forward output shape：`[2, 95]`

RF 检查：

- architecture：`ProjectRF`
- input representation：signed timestamp TAM -> `[N, 2, 1800]`
- max load time：`80.0`
- checkpoint keys：`best_val`、`classes`、`model_state`
- checkpoint classes：`95`
- missing/unexpected state keys：`0/0`
- zero batch forward output shape：`[2, 95]`

同一历史 run 中记录的 clean/defended 指标：

| 攻击器 | Clean Test Acc | Fresh Defended Acc | Visible Bandwidth | Allowed-mask Violation |
|---|---:|---:|---:|---:|
| DF | `0.973684` | `0.497507` | `0.299998` | `0.0` |
| RF | `0.976619` | `0.874385` | `0.299998` | `0.0` |

这些 defended 指标只作为历史证据，不作为 DMMPv3 正式结果。DMMPv3 正式 defended traces 与评测摘要必须在 `D:\learning\TOR\defence\DMMPv3\results` 下重新生成。

## 其他扫描到的 fixed checkpoint

| 攻击器 | 场景 | checkpoint 路径 | 状态 |
|---|---|---|---|
| DF | fixed | `D:\learning\TOR\results\dmmp2_cw_seed0_bwo30_fixed_protocol\dmmp2_attack_eval\fixed_df\fixed_df_checkpoint.pt` | 结构兼容；较旧协议，不作为主候选。 |
| RF | fixed | `D:\learning\TOR\results\dmmp2_cw_seed0_bwo30_fixed_protocol\dmmp2_attack_eval\fixed_rf\fixed_rf_checkpoint.pt` | 结构兼容；较旧协议，不作为主候选。 |
| DF | fixed | `D:\learning\TOR\results\dmmp2_v4_user_diffusion_seed0_bwo30\attack_eval\fixed\df\fixed_df_checkpoint.pt` | 结构兼容；历史协议，不作为主候选。 |
| RF | fixed | `D:\learning\TOR\results\dmmp2_v4_user_diffusion_seed0_bwo30\attack_eval\fixed\rf\fixed_rf_checkpoint.pt` | 结构兼容；历史协议，不作为主候选。 |
| DF | fixed smoke | `D:\learning\TOR\results\dmmp2_v4_phase1_smoke_seed0_bwo18\attack_eval\fixed\df\fixed_df_checkpoint.pt` | 结构兼容，但属于 smoke run，不能作为正式结论。 |
| DF | fixed legacy | `D:\learning\TOR\defence\DMMP\results\cw_final_guided_pool_neural_align_001_seed0\models\fixed_clean_DF_split_safe.pth` | `[需要人工确认]`，结构与 ProjectDF 大小不同。 |
| RF | fixed legacy | `D:\learning\TOR\defence\DMMP\results\cw_final_guided_pool_neural_align_001_seed0\models\fixed_clean_RF_split_safe.pth` | `[需要人工确认]`，旧 DMMP 协议。 |
| DF | fixed legacy | `D:\learning\TOR\defence\DMMP\results\dmmp_final_pool_old_handcrafted_only_20260705_100308\models\fixed_clean_DF_split_safe.pth` | `[需要人工确认]`，旧 DMMP 协议。 |
| RF | fixed legacy | `D:\learning\TOR\defence\DMMP\results\dmmp_final_pool_old_handcrafted_only_20260705_100308\models\fixed_clean_RF_split_safe.pth` | `[需要人工确认]`，旧 DMMP 协议。 |

## 复用规则

checkpoint 标记为 DMMPv3 可复用固定评测器前，必须确认：

1. 数据集路径和类别数一致。
2. 标签映射一致。
3. train/val/test 划分一致。
4. DF/RF 输入表示一致。
5. 模型结构与 state dict 兼容。
6. clean test 指标与历史摘要基本一致。
7. result directory 和 config 已记录。

当前只有上述 DF/RF fixed 候选满足这些检查。

## DMMPv3 内部迭代规则

已验证 fixed DF/RF checkpoint 是固定评测器，不是 DMMPv3 防御方法的中间产物。因此：

- DMMPv3 内部逐步改进防御方法时，可以继续复用同一组 fixed DF/RF 权重。
- 复用前提是 CW 数据集、95 类标签映射、seed 0 split、DF `[N, 1, 5000]` 输入、RF `[N, 2, 1800]` TAM 输入和 ProjectDF/ProjectRF 结构均保持不变。
- 如果上述任一条件改变，必须重新验证 checkpoint；若无法验证，则重新训练 fixed DF/RF。
- mixed attacker 不适用本规则，必须随对应 defended traces 重新训练。
