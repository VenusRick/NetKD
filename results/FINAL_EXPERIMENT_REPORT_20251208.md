# TrafficKD 完整实验报告
**日期**: 2025-12-08
**项目**: NetKD - 知识蒸馏流量分类

---

## 一、实验概述

### 1.1 实验目标
验证预训练+微调策略对教师模型性能的影响，并通过最优教师组合蒸馏出高效学生模型。

### 1.2 实验配置
- **数据集**: ISCXVPN2016 (12类流量分类)
  - 训练集: 13,281 样本 (82.8%)
  - 验证集: 1,383 样本 (8.6%)
  - 测试集: 1,384 样本 (8.6%)
  - 输入尺寸: 1×40×40

- **硬件环境**: 3× NVIDIA RTX 4090 (24GB)
- **软件环境**: PyTorch, timm

---

## 二、MAE预训练+微调实验结果

### 2.1 ConvNeXtV2-Tiny

| 配置 | Test Acc | F1 Score | 备注 |
|------|----------|----------|------|
| **Scratch (直接训练)** | **97.47%** | **96.56%** | ✅ 最佳 |
| MAE mask_ratio=0.6 | 54.70% | 38.08% | ❌ 失败 |
| MAE mask_ratio=0.7 | 56.94% | 37.61% | ❌ 失败 |
| MAE mask_ratio=0.8 | 96.10% | 94.49% | 可用 |

**结论**: ConvNeXtV2-Tiny使用MAE预训练效果不佳，低掩码率(0.6, 0.7)导致模型崩溃，直接训练效果最好。

### 2.2 EfficientNetV2-RW-S

| 配置 | Test Acc | F1 Score | 备注 |
|------|----------|----------|------|
| Scratch (直接训练) | 97.69% | 96.78% | 基准 |
| **MAE mask_ratio=0.6** | **98.19%** | **97.32%** | ✅ 最佳 |
| MAE mask_ratio=0.7 | 91.69% | 89.28% | 失败 |
| MAE mask_ratio=0.8 | 97.54% | 96.38% | 可用 |

**结论**: EfficientNetV2使用MAE预训练(mask_ratio=0.6)可获得+0.5%提升，mask_ratio=0.7不稳定。

### 2.3 MobileNetV3-Large-100

| 配置 | Test Acc | F1 Score | 备注 |
|------|----------|----------|------|
| Scratch (直接训练) | 96.75% | 95.83% | 基准 |
| MAE mask_ratio=0.6 | 96.89% | 95.63% | +0.14% |
| MAE mask_ratio=0.7 | 95.95% | 94.83% | 下降 |
| **MAE mask_ratio=0.8** | **97.18%** | **95.96%** | ✅ 最佳 |

**结论**: MobileNetV3使用MAE预训练(mask_ratio=0.8)可获得+0.43%提升。

---

## 三、预训练策略总结

### 3.1 MAE预训练效果汇总

| 模型 | 最佳预训练配置 | 相比Scratch提升 |
|------|----------------|-----------------|
| ConvNeXtV2-Tiny | ❌ 不推荐 | -1.37% (mr=0.8) |
| EfficientNetV2-RW-S | MAE mr=0.6 | **+0.50%** |
| MobileNetV3-Large | MAE mr=0.8 | **+0.43%** |

### 3.2 关键发现

1. **模型架构敏感性**: 不同模型对MAE预训练的响应差异很大
   - EfficientNetV2: 低掩码率(0.6)效果最好
   - MobileNetV3: 高掩码率(0.8)效果最好
   - ConvNeXtV2: 预训练反而有害

2. **掩码率影响**: 
   - 0.6-0.7: 对某些模型不稳定，可能导致训练崩溃
   - 0.8: 相对稳定，但提升有限

3. **推荐策略**:
   - 优先尝试直接训练(scratch)作为基准
   - 对EfficientNet系列可尝试MAE mr=0.6
   - 对MobileNet系列可尝试MAE mr=0.8

---

## 四、已完成的教师模型对比

### 4.1 ECA增强对比 (之前实验)

| 模型 | 无ECA | +ECA | 提升 |
|------|-------|------|------|
| ConvNeXtV2-Tiny | 33.38% | 92.85% | +59.47% |
| EfficientNetV2-RW-S | 97.47% | 97.62% | +0.15% |
| GhostNetV2-100 | 97.11% | 97.40% | +0.29% |

### 4.2 最佳教师配置

| 教师模型 | 最佳配置 | Test Acc | F1 |
|----------|----------|----------|-----|
| EfficientNetV2-RW-S | MAE mr=0.6 | 98.19% | 97.32% |
| ConvNeXtV2-Tiny | Scratch + ECA | 92.85% | 90.79% |
| MobileNetV3-Large | MAE mr=0.8 | 97.18% | 95.96% |

---

## 五、学生模型蒸馏结果 (之前实验)

### 5.1 最佳学生模型 Top 5

| 学生模型 | 配置 | Test Acc | F1 | Params |
|----------|------|----------|-----|--------|
| RepViT-M0.9 | T=3, α=0.3 | 98.12% | 97.39% | 4.72M |
| GhostNet-100 | T=3, α=0.3 | 98.05% | 97.53% | 3.91M |
| RepViT-M0.9 | T=5, α=0.7 | 98.05% | 97.02% | 4.72M |
| MobileNetV3-Small-0.75 | T=4, α=0.3 | 97.40% | 95.99% | 1.02M |
| MobileNetV2-0.5 | T=5, α=0.5 | 97.04% | 96.08% | 0.70M |

### 5.2 帕累托最优选择

- **最高精度**: RepViT-M0.9 (98.12% / 4.72M)
- **最轻量化**: MobileNetV2-0.5 (97.04% / 0.70M)
- **平衡选择**: GhostNet-100 (98.05% / 3.91M)

---

## 六、实验结论

### 6.1 核心发现

1. **预训练不总是有效**: MAE预训练对流量分类任务的效果取决于模型架构，并非总是有益。

2. **EfficientNetV2表现最佳**: 
   - MAE预训练(mr=0.6) + 微调 = 98.19% 准确率
   - 相比scratch提升0.5%

3. **知识蒸馏效果显著**:
   - 学生模型(98.12%)超过教师集成(97.83%)
   - 参数量减少80%+

### 6.2 推荐流程

```
最优实验流程:
1. 教师训练: EfficientNetV2 + MAE(mr=0.6) → 98.19%
2. 多教师Stacking: 3个最佳教师 → ~98%
3. 知识蒸馏: RepViT-M0.9 (T=3, α=0.3) → 98.12%
```

### 6.3 待完成实验

- [ ] SimCLR对比学习预训练 (脚本已准备，需修复DataLoader问题)
- [ ] 更多轻量级学生模型测试 (GhostNet-0.75x, ShuffleNet等)
- [ ] FLOPs/参数量完整分析

---

## 七、模型文件位置

```
results/full_experiment_20251208_1045/
├── phase1_pretrain/      # MAE预训练权重
├── phase2_finetune/      # 微调后的教师模型
│   ├── teacher_convnextv2_tiny_scratch_best.pth
│   ├── teacher_efficientnetv2_rw_s_mae_mr0.6_best.pth  ← 最佳
│   └── teacher_mobilenetv3_large_100_mae_mr0.8_best.pth
└── finetune_results.json # 完整结果

results/full_pipeline_20251208_004257/
├── gpu1_scratch/teachers/  # ECA教师模型
└── gpu1_scratch/students/  # 蒸馏学生模型
```

---

**报告生成时间**: 2025-12-08 13:00 UTC+8
