# NetKD 实验结果汇总

> 更新日期: 2025年12月2日  
> 数据集: ISCXVPN2016 (7类VPN加密流量分类)  
> 硬件: NVIDIA GeForce RTX 4090

---

## 目录
1. [数据集信息](#1-数据集信息)
2. [教师模型实验](#2-教师模型实验)
3. [Stacking集成实验](#3-stacking集成实验)
4. [学生模型蒸馏实验](#4-学生模型蒸馏实验)
5. [消融实验](#5-消融实验)
6. [最终推荐配置](#6-最终推荐配置)

---

## 1. 数据集信息

| 属性 | 值 |
|------|-----|
| 数据集名称 | ISCXVPN2016 |
| 类别数 | 7 (browsing, chat, email, ftp, p2p, streaming, voip) |
| 训练集 | 13,281 张 (82.8%) |
| 验证集 | 1,383 张 (8.6%) |
| 测试集 | 1,384 张 (8.6%) |
| 图像尺寸 | 1 × 40 × 40 (灰度图) |

---

## 2. 教师模型实验

### 2.1 ECA注意力机制消融实验

比较在各教师模型上添加ECA (Efficient Channel Attention) 模块的效果。

| 教师模型 | 无ECA验证精度 | 有ECA验证精度 | 提升 |
|----------|---------------|---------------|------|
| ResNet-50 | 98.77% | **99.06%** | +0.29% |
| DenseNet-121 | 99.06% | 99.06% | 0% |
| MobileNetV3-Large | 98.77% | **99.06%** | +0.29% |
| ConvNeXtV2-Tiny | **98.92%** | 98.70% | -0.22% |
| EfficientNetV2-S | **99.06%** | 98.99% | -0.07% |

**结论**: 
- ECA对ResNet-50和MobileNetV3-Large有明显提升
- 对ConvNeXtV2-Tiny和EfficientNetV2-S反而有轻微下降
- DenseNet-121无明显变化

### 2.2 最佳教师模型组合

基于ECA消融实验，选择以下模型作为Stacking集成的基模型：
- **ConvNeXtV2-Tiny (ECA)**: 98.70% - 全局结构特征
- **DenseNet-121 (ECA)**: 99.06% - 多尺度特征复用
- **MobileNetV3-Large (ECA)**: 99.06% - 局部细节特征

---

## 3. Stacking集成实验

### 3.1 不同组合的Stacking结果

| 配置 | 基模型组合 | 验证精度 |
|------|------------|----------|
| baseline | ResNet50 + DenseNet121 + MobileNetV3 | **99.20%** |
| replace_resnet | ConvNeXtV2 + DenseNet121 + MobileNetV3 | 98.92% |
| replace_mobilenet | ResNet50 + DenseNet121 + ConvNeXtV2 | 99.13% |
| new_trio | ConvNeXtV2 + EfficientNetV2 + MobileNetV3 | 98.99% |
| quad_ensemble | 4个模型组合 | 99.13% |
| full_ensemble | 5个模型组合 | 99.13% |

**结论**: 
- 原始baseline (ResNet50 + DenseNet121 + MobileNetV3) 达到最高验证精度 99.20%
- 更多模型不一定带来更高精度，反而可能引入噪声

### 3.2 最终采用的Stacking配置

由于需要用ConvNeXtV2替换ResNet50（论文要求），采用 `stacking_replace_resnet` 配置：
- **基模型**: ConvNeXtV2-Tiny (ECA) + DenseNet-121 (ECA) + MobileNetV3-Large (ECA)
- **元学习器**: MLP (hidden_dim=384)
- **验证精度**: 98.92%

---

## 4. 学生模型蒸馏实验

### 4.1 学生模型架构对比

| 模型 | 参数量 | 验证精度 | 测试精度 | F1-macro | 训练时间 |
|------|--------|----------|----------|----------|----------|
| **MobileNetV2** | 2.23M | **98.84%** | **97.98%** | 96.94% | 10.1min |
| **MobileNetV3-Small** | 1.52M | 98.34% | 97.98% | **97.20%** | 10.4min |
| GhostNet | 3.91M | 98.48% | 97.25% | 95.94% | 14.3min |

### 4.2 蒸馏配置

```yaml
训练配置:
  epochs: 100
  batch_size: 128
  base_lr: 0.001
  weight_decay: 1e-4
  warmup_epochs: 5
  grad_clip: 1.0
  label_smoothing: 0.1

蒸馏配置:
  temperature: 3.0
  alpha_ce: 0.5      # CE损失权重
  alpha_kl: 0.5      # KL损失权重
  
损失函数: L = 0.5 * CE + 0.5 * KL
```

### 4.3 知识保留率

| 学生模型 | 教师精度(Stacking) | 学生测试精度 | 知识保留率 |
|----------|-------------------|--------------|------------|
| MobileNetV2 | 98.92% | 97.98% | 99.05% |
| MobileNetV3-Small | 98.92% | 97.98% | 99.05% |
| GhostNet | 98.92% | 97.25% | 98.31% |

**知识保留率** = 学生精度 / 教师精度 × 100%

---

## 5. 消融实验

### 5.1 蒸馏损失函数消融

| 损失函数 | 测试精度 | 说明 |
|----------|----------|------|
| CE-only | 97.04% | 仅交叉熵损失 |
| KL-only | 97.11% | 仅KL散度损失 |
| CE+KL (0.5+0.5) | **97.98%** | 平衡组合 ✓ |
| FKL+RKL+Sinkhorn | 96.60% | 复杂混合损失 ✗ |

**结论**: 简单的 `0.5*CE + 0.5*KL` 组合效果最佳，复杂的混合损失反而降低性能。

### 5.2 温度参数消融

| 温度 T | 效果 |
|--------|------|
| T=2.0 | 稳定，效果良好 |
| T=3.0 | **最佳**，当前采用 |
| T=4.0 | 数值不稳定，出现NaN |

---

## 6. 最终推荐配置

### 6.1 最佳精度配置

```
教师模型: Stacking集成
  - ConvNeXtV2-Tiny (ECA): 27.82M params
  - DenseNet-121 (ECA): 6.95M params  
  - MobileNetV3-Large (ECA): 4.21M params
  - Meta-Learner: MLP (hidden_dim=384)
  
学生模型: MobileNetV2
  - 参数量: 2.23M
  - 测试精度: 97.98%
  - F1-macro: 96.94%
  
压缩比: (27.82 + 6.95 + 4.21) / 2.23 = 17.5x
```

### 6.2 最小模型配置

```
学生模型: MobileNetV3-Small
  - 参数量: 1.52M
  - 测试精度: 97.98%
  - F1-macro: 97.20%
  
压缩比: 38.98 / 1.52 = 25.6x
```

---

## 附录: 文件路径

### 模型检查点
- 教师模型: `results/teacher_search_bs128/eca_ablation/*.pth`
- Stacking模型: `results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth`
- 学生模型: `results/multi_student/student_*.pth`

### 实验脚本
- 教师训练: `scripts/run_teacher_search_experiment.py`
- 多学生蒸馏: `scripts/run_multi_student_distillation.py`
- 蒸馏实验: `scripts/run_distillation_experiment.py`

### 结果文件
- 学生模型结果: `results/multi_student/results.json`
- 消融实验结果: `checkpoints/ablation/*/results.json`
