# 🚀 TrafficKD 完整实验方案

## 📋 实验概述

本方案涵盖从**自监督预训练**到**知识蒸馏学生训练**的完整流程，旨在探索：
1. 自监督预训练（MAE/对比学习）是否能提升教师模型性能
2. 更好的教师模型是否能通过Stacking产生更好的集成效果
3. 更强的Stacking是否能蒸馏出更好的学生模型

---

## 🔬 Phase 1: 自监督预训练实验

### 1.1 MAE预训练（不同掩码率）

| 实验ID | Backbone | Mask Ratio | Epochs | Batch Size | GPU |
|--------|----------|------------|--------|------------|-----|
| MAE-01 | ConvNeXtV2-Tiny | 0.6 | 100 | 512 | 0 |
| MAE-02 | ConvNeXtV2-Tiny | 0.7 | 100 | 512 | 0 |
| MAE-03 | ConvNeXtV2-Tiny | 0.8 | 100 | 512 | 0 |
| MAE-04 | EfficientNetV2-S | 0.6 | 100 | 512 | 1 |
| MAE-05 | EfficientNetV2-S | 0.7 | 100 | 512 | 1 |
| MAE-06 | EfficientNetV2-S | 0.8 | 100 | 512 | 1 |
| MAE-07 | MobileNetV3-Large | 0.6 | 100 | 512 | 2 |
| MAE-08 | MobileNetV3-Large | 0.7 | 100 | 512 | 2 |
| MAE-09 | MobileNetV3-Large | 0.8 | 100 | 512 | 2 |

### 1.2 对比学习预训练（SimCLR）

| 实验ID | Backbone | Temperature | Epochs | Batch Size | GPU |
|--------|----------|-------------|--------|------------|-----|
| CL-01 | ConvNeXtV2-Tiny | 0.1 | 100 | 256 | 0 |
| CL-02 | ConvNeXtV2-Tiny | 0.5 | 100 | 256 | 0 |
| CL-03 | EfficientNetV2-S | 0.1 | 100 | 256 | 1 |
| CL-04 | EfficientNetV2-S | 0.5 | 100 | 256 | 1 |
| CL-05 | MobileNetV3-Large | 0.1 | 100 | 256 | 2 |
| CL-06 | MobileNetV3-Large | 0.5 | 100 | 256 | 2 |

---

## 🎓 Phase 2: 教师模型微调

对每个预训练检查点进行分类微调：
- Epochs: 50
- LR: 1e-4 (预训练模型) / 1e-3 (从头训练)
- Optimizer: AdamW, Weight Decay: 1e-4
- Batch Size: 128

### 对照组
- 从头训练（Baseline）
- 从头训练 + ECA注意力

---

## 🏗️ Phase 3: Stacking集成

使用Phase 2中最佳的3个教师模型构建Stacking：
- Epochs: 30
- LR: 1e-3
- Hidden Dim: 384

---

## 👨‍🎓 Phase 4: 知识蒸馏学生训练

### 学生模型候选
| 模型 | 参数量 | 类型 |
|------|--------|------|
| RepViT-M0.9 | 4.72M | 高性能 |
| MobileNetV3-Small-050 | 0.58M | 超轻量 |
| MobileNetV3-Small-075 | 1.02M | 轻量 |
| GhostNet-1.0x | 3.90M | 轻量 |

### 蒸馏超参数
- Temperature: [2, 3, 4, 5]
- Alpha CE: [0.3, 0.5, 0.7]
- Epochs: 100

---

## 📁 输出结构
```
results/full_experiment_YYYYMMDD/
├── phase1_pretrain/
│   ├── mae/
│   └── simclr/
├── phase2_finetune/
├── phase3_stacking/
├── phase4_distill/
└── FINAL_REPORT.md
```

