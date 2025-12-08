# NetKD 实验历史记录
**创建日期**: 2025-12-08
**维护者**: CodeAgent

---

## 📋 实验索引

| 日期 | 实验名称 | 状态 | 最佳结果 |
|------|----------|------|----------|
| 2025-12-07 | MAE预训练实验 | ✅完成 | 98.19% |
| 2025-12-07 | 教师模型对比 | ✅完成 | 98.77% |
| 2025-12-07 | 学生蒸馏实验 | ✅完成 | 98.12% |
| 2025-12-07 | 注意力消融实验 | ✅完成 | 98.55% |
| 2025-12-08 | SimCLR预训练实验 | 🔄进行中 | - |

---

## 📊 实验详情

### 实验 #1: MAE预训练 (2025-12-07 ~ 12-08)

**目标**: 测试 Masked Autoencoder 预训练对流量分类的影响

**配置**:
- 数据集: ISCXVPN2016
- Backbones: efficientnetv2_rw_s, convnextv2_tiny, mobilenetv3_large_100
- Mask Ratios: 0.6, 0.7, 0.8
- 预训练 Epochs: 100
- 微调 Epochs: 50

**结果**:
| Backbone | Pretrain | Val Acc | Test Acc | Test F1 |
|----------|----------|---------|----------|---------|
| EfficientNetV2-RW-S | MAE mr=0.6 | 98.34% | **98.19%** | 97.32% |
| EfficientNetV2-RW-S | MAE mr=0.8 | 98.12% | 97.54% | 96.38% |
| EfficientNetV2-RW-S | Scratch | 98.26% | 97.69% | 96.78% |
| ConvNeXtV2-Tiny | Scratch | 98.34% | 97.47% | 96.56% |
| ConvNeXtV2-Tiny | MAE mr=0.8 | 96.02% | 96.10% | 94.49% |
| MobileNetV3-Large | MAE mr=0.8 | 97.40% | 97.18% | 95.96% |

**结论**: 
- MAE预训练对 EfficientNetV2 有效（mr=0.6 最佳）
- ConvNeXtV2 从头训练效果更好
- MobileNetV3 MAE 预训练略有帮助

---

### 实验 #2: 教师模型搜索 (2025-12-07)

**目标**: 寻找最优教师模型组合

**测试模型**:
1. ResNet50-ECA
2. DenseNet121-ECA
3. MobileNetV3-Large-ECA
4. ConvNeXtV2-Tiny-ECA
5. EfficientNetV2-S-ECA

**结果**:
| 模型 | Params | Test Acc | F1 Macro |
|------|--------|----------|----------|
| DenseNet121-ECA | 8.0M | **98.77%** | - |
| ResNet50-ECA | 25.6M | 98.48% | - |
| MobileNetV3-Large-ECA | 5.4M | 98.19% | - |
| ConvNeXtV2-Tiny-ECA | 28.6M | 97.47% | 96.56% |
| EfficientNetV2-S-ECA | 22.2M | 97.69% | 96.78% |

**最佳教师组合**: DenseNet121 + EfficientNetV2 + MobileNetV3

---

### 实验 #3: 学生蒸馏实验 (2025-12-07)

**目标**: 知识蒸馏到轻量级学生模型

**学生候选**:
- MobileNetV2
- MobileNetV3-Small
- GhostNet
- RepViT-M0.9

**KD配置**:
- Temperature: 3.0, 4.0, 5.0
- Alpha (soft weight): 0.3, 0.5, 0.7
- 损失: CE + Forward KL

**结果**:
| 学生模型 | T | α | Test Acc | Params |
|----------|---|---|----------|--------|
| RepViT-M0.9 | 3 | 0.3 | **98.12%** | 4.72M |
| GhostNet-100 | 3 | 0.3 | 98.05% | 3.91M |
| MobileNetV2 | 3 | 0.3 | 97.98% | 2.23M |
| MobileNetV3-Small | 4 | 0.3 | 97.40% | 1.02M |

---

### 实验 #4: 注意力机制消融 (2025-12-07)

**目标**: 对比不同注意力机制

**测试方案**:
1. No Attention (baseline)
2. SE (Squeeze-and-Excitation)
3. CBAM
4. ECA
5. Agent Attention

**结果**:
| 注意力 | Test Acc | 参数增加 |
|--------|----------|----------|
| Agent Attention | **98.55%** | +0.12M |
| ECA | 98.48% | +0.01M |
| SE | 98.26% | +0.05M |
| CBAM | 98.12% | +0.08M |
| No Attention | 97.98% | - |

---

### 实验 #5: SimCLR预训练 (2025-12-08, 进行中)

**目标**: 对比 SimCLR 对比学习与 MAE 预训练

**配置**:
- Temperature: 0.5
- 预训练 Epochs: 100
- 微调 Epochs: 50
- Batch Size: 256

**当前进度** (14:50 UTC+8):
| GPU | Backbone | 阶段 | 进度 | 备注 |
|-----|----------|------|------|------|
| 0 | efficientnetv2_rw_s | SimCLR预训练 | E66/100 | 正常 |
| 1 | convnextv2_tiny | 微调 | E17/50 | val=0.30 (异常) |
| 2 | mobilenetv3_large_100 | 微调 | E31/50 | val=0.97 |

---

## 📁 结果目录

```
results/
├── full_experiment_20251208_1045/    # MAE实验
│   ├── phase1_pretrain/
│   ├── phase2_finetune/
│   └── finetune_results.json
├── simclr_experiment_*/              # SimCLR实验
├── ablation_20251207/                # 消融实验
├── lightweight_students_20251207/    # 学生实验
└── FINAL_EXPERIMENT_REPORT_20251208.md
```

---

## 📝 后续实验建议

1. **SimCLR调参**: GPU 1 的 convnextv2 表现异常，可能需要调整学习率
2. **更多学生模型**: 测试 EfficientNet-Lite, ShuffleNet v2
3. **跨数据集测试**: 在 ISCXTor2016, USTC-TFC2016 上验证

---

**更新日志**:
- 2025-12-08: 创建文档，添加 SimCLR 实验记录
