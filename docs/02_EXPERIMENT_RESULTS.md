# NetKD 实验结果记录
**最后更新**: 2025-12-10 16:00 UTC+8

---

## 📊 实验总览

| # | 日期 | 数据集 | 教师最佳 | 学生最佳 | 状态 |
|---|------|--------|----------|----------|------|
| 1 | 12-07 | ISCXVPN2016 | 98.77% (DenseNet121-ECA) | 98.12% (RepViT) | ✅ |
| 2 | 12-10 | USTC-TFC2016 | 进行中 | - | ⏳ |
| 3 | 12-10 | CICIoT2022 | 进行中 | - | ⏳ |
| 4 | 12-10 | ISCXVPN2016 | 进行中 | - | ⏳ |
| 5 | - | ISCXTor2016 | - | - | 待启动 |
| 6 | - | CrossPlatform-Android | - | - | 待启动 |
| 7 | - | CrossPlatform-iOS | - | - | 待启动 |

---

## 🏆 最佳实验结果 (ISCXVPN2016 - 基准)

### 教师模型
| 模型 | Test Acc | F1 | 参数量 |
|------|----------|-----|--------|
| **DenseNet121-ECA** | **98.77%** | 0.9831 | 8.0M |
| EfficientNetV2-RW-S | 98.19% | 0.9772 | 22.2M |
| MobileNetV3-Large-ECA | 98.19% | 0.9772 | 5.4M |
| ResNet50 | ~97% | - | 25.6M |

### 学生模型 (KD后)
| 模型 | Test Acc | F1 | 参数量 | KD配置 |
|------|----------|-----|--------|--------|
| **RepViT-M0.9** | **98.12%** | 0.9766 | 4.72M | T=3, α=0.3 |
| GhostNet-100 | 98.05% | 0.9758 | 3.91M | T=3, α=0.3 |
| MobileNetV3-Small | 97.40% | 0.9693 | 1.02M | T=4, α=0.3 |

---

## 📈 当前实验进度 (2025-12-10)

### USTC-TFC2016 (GPU 0)
- 数据集: 37K/7K/7K (19类)
- ResNet50: E6/30, val_acc=97.83%
- DenseNet121: 待训练
- MobileNetV3: 待训练

### CICIoT2022 (GPU 1)
- 数据集: 19K/2K/2K
- ResNet50: E12/30, val_acc=96.72%
- DenseNet121: 待训练
- MobileNetV3: 待训练

### ISCXVPN2016 (GPU 2)
- 数据集: 13K/1.4K/1.4K (7类)
- ResNet50: E16/30, val_acc=97.90%
- DenseNet121: 待训练
- MobileNetV3: 待训练

---

## 🔬 消融实验结果

### MAE预训练对比
| Backbone | MAE+FT | Scratch | 提升 |
|----------|--------|---------|------|
| EfficientNetV2 | 98.19% | 97.69% | **+0.50%** |
| ConvNeXtV2 | 97.05% | 97.40% | -0.35% |
| MobileNetV3 | 97.98% | 97.76% | +0.22% |

**结论**: MAE对EfficientNetV2有效

### 注意力机制消融
| 注意力 | Test Acc | 参数增加 |
|--------|----------|----------|
| **Agent Attention** | **98.55%** | +0.12M |
| ECA | 98.48% | +0.01M |
| SE | 98.26% | +0.05M |
| CBAM | 98.12% | +0.08M |
| None | 97.98% | - |

**结论**: ECA性价比最高

### KD超参数消融
| T | α | Test Acc |
|---|---|----------|
| 2 | 0.3 | 97.85% |
| **3** | **0.3** | **98.12%** |
| 4 | 0.3 | 98.05% |
| 3 | 0.5 | 97.92% |

**结论**: T=3, α=0.3 最优

---

## 📁 结果文件位置

```
runs/
├── USTC-TFC2016/          # 当前训练中
├── CICIoT2022/            # 当前训练中
├── ISCXVPN2016/           # 当前训练中
│   ├── resnet50_teacher.pth
│   ├── densenet121_teacher.pth
│   ├── mobilenetv3_teacher.pth
│   ├── stacking_model.pth
│   └── student_sd_mkd.pth

results/
├── full_experiment_20251208_1045/  # MAE实验
└── ablation_20251207/              # 消融实验
```
