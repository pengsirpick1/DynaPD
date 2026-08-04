[English](README.md) | **中文**

# DynaPD

基于**多替代模型教师优化**和**随机化补全**的网站指纹动态填充防御框架。

## 概述

DynaPD 通过在 RF 识别的关键点迭代注入 dummy 包和重分布延迟来生成防御流量。候选动作经过一组替代攻击模型评估，通过加权效用评分选择最优动作。

### 核心创新

| 组件 | 描述 |
|------|-------------|
| **多替代模型教师** | RF（TAM/突发）、DF（方向 CNN）、AWF（浅层方向 CNN）提供三种不同攻击视角，实现跨架构鲁棒性。 |
| **归一化增益** | `gain_m = (当前margin - 候选margin) / abs(原始margin)` 使跨模型增益可比。 |
| **随机化动作选择** | ε-greedy 随机化（ε=0.5）确保同一干净流量产生不同的防御结果，抵抗对抗训练。 |
| **带宽感知补全** | 两阶段：先翻所有模型，再用剩余带宽填满（`preserve_flip_fill`）。 |
| **表征族覆盖** | 替代模型按表征族（DIR/突发/TAM）而非模型名称选择，增强泛化性。 |

### 设计流程

```
干净流量
    ↓
RF 关键点 → 分层 Top128 候选（5维交叉组合）
    ↓
三模型评估（RF / DF / AWF）
    ↓
加权分数 = 0.8×gain_rf + 0.1×gain_df + 0.1×gain_awf - λ×成本
    ↓
ε-greedy 选择动作 → 应用 → 循环直到全翻或预算耗尽
    ↓
Completion: 剩余带宽随机填满有效候选
```

## 当前最佳结果

Completion b30 ε=0.5，全量 CW Test（10564 条），T640g0 RTX 4090

| 模型 | 角色 | 干净准确率 | 防御后准确率 | 翻转率 |
|------|------|------|------|------|
| RF | Teacher | 97.66% | **13.37%** | 86.59% |
| DF | Teacher | 98.25% | 9.94% | 89.91% |
| AWF | Teacher | 94.86% | 4.95% | 95.05% |
| TF | Held-out | 96.16% | **10.20%** | 89.80% |
| **WC** | — | — | **13.37%** | — |

三替代模型全翻: 78.75% | 平均带宽: 24.50% | 平均动作数: 12.09

### 演进历史

| 版本 | 替代模型 | 预算 | 随机ε | Completion | WC | TF held-out |
|------|------|------|------|------|------|------|
| RF-only Student | RF | 10% | 0 | 无 | 37.96% | — |
| E2b (512) | RF+DF+AWF | 15% | 0 | 无 | 17.77% | 17.77% |
| E2b (全量) | RF+DF+AWF | 15% | 0 | 无 | 17.84% | 17.84% |
| **Completion (全量)** | **RF+DF+AWF** | **30%** | **0.5** | **有** | **13.37%** | **10.20%** |

WC: RF-only 基线 37.96% → Completion 13.37%（降低 24.59pp）

## 抗对抗训练性能（512 子集预览）

| 方法 | 带宽 | DF 自适应 | AWF 自适应 |
|------|------|------|------|
| E2b b20 ε=0.1, 无 completion | 7.60% | 65.04% | 81.64% |
| Completion b20 ε=0.5 | 17.51% | 17.19% | 18.36% |
| Completion b30 ε=0.5 | 25.13% | 11.72% | 14.84% |

无随机化 + 无补全时，2 epoch 对抗微调即可破防（恢复率 65-82%）。加上随机化 + 补全后，自适应攻击恢复率降至 12-18%。

## 为什么选这三个模型

| 模型 | 攻击视角 | 对应表征族 |
|------|---------|------|
| RF（sklearn） | TAM / burst 统计特征 | Burst 族 |
| DF（CNN） | 方向序列长程模式 | DIR 族 |
| AWF（浅 CNN） | 方向序列局部模式 | DIR 族（浅层变体） |

三种不同攻击视角需要同时被欺骗，防御才具跨架构泛化能力。

## 候选空间

动作由五维交叉组合生成，~3万原始组合 → Top128：

| 维度 | 来源 | 取值 |
|------|------|------|
| 窗口 | RF TAM 关键点峰值 | 最多 8 个 |
| 锚点 | 7 类结构位置 | 关键点偏移、速率峰值、方向切换点、间隙等 |
| 剂量 | 绝对 + 相对 | {1,2,4,8,16,32} + {10%-100% 窗口流量} |
| 宽度 | 注入范围 | 窗口长度、半长、8、16、32 |
| 模式 | 方向策略 | 方向平衡、因果掩码 |

## 安装

```bash
git clone https://github.com/pengsirpick1/DynaPD.git
cd DynaPD
conda create -n pyda python=3.10
conda activate pyda
pip install torch numpy tqdm
pip install -e .
```

实验环境: Python 3.10.20, PyTorch 2.13.0+cu130, RTX 4090 24GB

## 快速开始

### 构建关键点存档

```bash
python scripts/stage_b_prepare_fast_keypoint_archive.py \
  --data_root datasets/CW.npz \
  --attacker rf \
  --checkpoint models/attacks/fixed_rf_checkpoint.pt \
  --split_name all \
  --batch_size 512 \
  --device cuda \
  --run_name stage_b_fast_keypoint_full_cw_all_seed0
```

### 运行 E2b Oracle（512 子集）

```bash
python scripts/stage_b_run_ensemble_oracle_e2b.py \
  --output_dir results --run_name stage_b_e2b_512 \
  --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
  --data_root datasets/CW.npz \
  --rf_checkpoint models/attacks/fixed_rf_checkpoint.pt \
  --df_checkpoint wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth \
  --varcnn_checkpoint wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth \
  --methods norm_weighted_r80_d10_v10 \
  --dummy_budgets 0.30 \
  --max_samples 512 --sample_end 512 \
  --random_epsilon 0.5 \
  --completion_mode preserve_flip_fill \
  --completion_target_bandwidth 0.30 \
  --compact_candidate_generation \
  --device cuda --batch_size 128
```

### 运行 Completion 全量（64 分片并行）

```bash
samples=10564
for s in $(seq 0 63); do
  start=$((s * samples / 64))
  end=$(((s + 1) * samples / 64))
  python scripts/stage_b_run_ensemble_oracle_e2b_completion.py \
    --output_dir results/$RUN --run_name shard_$(printf '%03d' $s) \
    --archive results/stage_b_fast_keypoint_full_cw_all_seed0/fast_keypoint_archive.npz \
    --data_root datasets/CW.npz \
    --rf_checkpoint models/attacks/fixed_rf_checkpoint.pt \
    --df_checkpoint wflib_copy/checkpoints/CW/DF/dynapd_clean_seed0.pth \
    --varcnn_checkpoint wflib_copy/checkpoints/CW/AWF/dynapd_clean_seed0.pth \
    --methods norm_weighted_r80_d10_v10 --dummy_budgets 0.30 \
    --max_samples $samples --sample_start $start --sample_end $end \
    --random_epsilon 0.5 --completion_mode preserve_flip_fill \
    --completion_target_bandwidth 0.30 --completion_topk 16 \
    --compact_candidate_generation --candidate_batch_size 4096 \
    --candidate_device cuda --candidate_score_device cpu \
    --device cuda --batch_size 128 \
    --cost_lambda 0.05 --robust_min_positive_models 1 \
    --robust_epsilon 0.20 --max_dummy_steps 20 --seed 202 &
done
wait
```

## 文件布局

```
dynapd/
  data/               CW 数据加载
  stage_a/            攻击模型（RF/DF），关键点提取
  stage_b/            候选生成、目标函数、策略数据
  utils/              运行配置、设备管理

scripts/
  stage_b_prepare_fast_keypoint_archive.py    # RF 关键点存档
  stage_b_run_ensemble_oracle_e2b.py          # E2b 基础版
  stage_b_run_ensemble_oracle_e2b_completion.py  # E2b + Completion + 随机化
  stage_b_run_ensemble_oracle_e2b_rand.py     # E2b + 随机化
  stage_b_build_policy_dataset.py             # Student 策略数据
  stage_b_train_candidate_policy.py           # Student 训练
  stage_b_run_student_policy_controller.py    # Student 推理
```

## 实验文档

- [英文完整报告](docs/COMPLETION_B30_FULLCW_RESULTS.md)
- [中文完整报告](docs/COMPLETION_B30_全量CW实验报告.md)

## 说明

本项目为学术研究发布。数据集、模型权重、实验输出均不包含在内。
当前主线使用 RF + DF + AWF 作为 Teacher 替代模型组，配合补全模式和随机化动作选择。
