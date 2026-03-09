# 知识蒸馏实验报告

## 实验概述

**目标**: 使用最佳教师组合 (replace_resnet) 进行知识蒸馏，将教师知识迁移到轻量级学生模型

**实验日期**: 2024-12-01

---

## 1. 教师模型配置

### 教师组合: replace_resnet (bs=128)
| 模型 | ECA | 测试精度 | Checkpoint |
|------|-----|---------|------------|
| ConvNeXtV2-Tiny | ✓ | 98.70% | eca_ablation/convnextv2_tiny_eca_best.pth |
| DenseNet121 | ✓ | 98.63% | eca_ablation/densenet121_eca_best.pth |
| MobileNetV3 | ✓ | 97.83% | eca_ablation/mobilenetv3_eca_best.pth |

### Stacking 集成
- **Stacking 精度**: 99.06%
- **Checkpoint**: stacking/stacking_replace_resnet_best.pth

---

## 2. 蒸馏配置

### 训练参数
| 参数 | 值 |
|------|-----|
| Epochs | 100 |
| Batch Size | 128 |
| Learning Rate | 0.0003 |
| Warmup Epochs | 10 |
| Label Smoothing | 0.1 |
| Gradient Clip | 0.5 |

### 蒸馏损失
| 参数 | 值 |
|------|-----|
| Temperature (T) | 3.0 |
| CE Loss Weight (α_ce) | 0.5 |
| KL Loss Weight (α_kl) | 0.5 |

---

## 3. 实验结果

### 最终性能
| 指标 | 值 |
|------|-----|
| **测试集精度** | **96.60%** |
| **F1-macro** | **95.23%** |
| **最佳 Epoch** | 93 |
| **验证集最佳精度** | 97.76% |
| **训练时间** | 9.8 分钟 |

### 模型压缩
| 指标 | 值 |
|------|-----|
| 学生模型参数量 | 9.81M |
| 教师模型参数量 | 38.99M |
| **压缩比** | **4.0x** |

---

## 4. 性能对比

| 模型 | 参数量 | 测试精度 | F1-macro |
|------|--------|----------|----------|
| Stacking 教师 (3x) | 38.99M | 99.06% | - |
| 学生 (蒸馏后) | 9.81M | 96.60% | 95.23% |
| 精度差异 | -75% params | -2.46% | - |

### 关键发现
1. **蒸馏效果良好**: 学生模型保留了教师模型 97.5% 的性能
2. **4倍压缩**: 参数量从 39M 降低到 9.8M
3. **稳定训练**: 100 epoch 训练无 NaN，损失平稳下降
4. **快速收敛**: 约 30 epoch 后精度趋于稳定

---

## 5. 训练曲线特征

- **Warmup 阶段** (Epoch 1-10): 精度从 52.6% 快速上升到 84.0%
- **快速学习阶段** (Epoch 10-40): 精度从 84% 上升到 96%
- **精细调整阶段** (Epoch 40-100): 精度在 97-98% 波动

---

## 6. 结论

本次蒸馏实验成功将 Stacking 教师集成的知识迁移到轻量级学生模型：

✅ **成功点**:
- 使用简化的 CE + KL 损失函数，避免了数值不稳定
- 较低的学习率 (0.0003) 和较长的 warmup (10 epochs) 保证了稳定训练
- 实现了 4x 模型压缩，仅损失 2.46% 精度

⚠️ **待改进**:
- 学生模型与教师集成仍有 2.46% 精度差距
- 可尝试更复杂的蒸馏损失（如 feature distillation）
- 可考虑更长的训练或更大的学生模型

---

## 附录: 文件清单

```
results/distillation_bs128_full/
├── student_distilled_best.pth   # 最佳学生模型 checkpoint
├── distillation_result.json     # 实验结果 JSON
└── experiment_report.md         # 本报告
```
