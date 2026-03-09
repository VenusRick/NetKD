# NetKD 模型架构文档
**最后更新**: 2025-12-10 16:00 UTC+8

---

## 🎯 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                     NetKD Pipeline                       │
├─────────────────────────────────────────────────────────┤
│  Stage 1: Teacher Training                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                │
│  │ ResNet50 │ │DenseNet  │ │MobileNet │                │
│  │  +ECA    │ │ 121+ECA  │ │ V3+ECA   │                │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘                │
│       │            │            │                        │
│       └────────────┼────────────┘                        │
│                    ▼                                     │
│  Stage 2: Stacking Ensemble                              │
│  ┌─────────────────────────────────────┐                │
│  │  Stacking Model (MLP)               │                │
│  │  Input: concat(logits1,2,3)         │                │
│  │  Output: ensemble prediction        │                │
│  └─────────────────┬───────────────────┘                │
│                    │                                     │
│                    ▼                                     │
│  Stage 3: Knowledge Distillation                         │
│  ┌─────────────────────────────────────┐                │
│  │  Student (RepViT/GhostNet/MBV3)     │                │
│  │  Loss: CE + α*KL(student||teacher)  │                │
│  └─────────────────────────────────────┘                │
└─────────────────────────────────────────────────────────┘
```

---

## 🏫 教师模型配置

### ResNet50 (with ECA)
```python
models.ResNet50Teacher(num_classes, use_eca=True, pretrained=False)
# 参数量: 25.6M
# 特征维度: 2048
```

### DenseNet121 (with ECA)
```python
models.DenseNet121Teacher(num_classes, use_eca=True, pretrained=False)
# 参数量: 8.0M
# 特征维度: 1024
# 最佳教师模型 ★
```

### MobileNetV3-Large (with ECA)
```python
models.MobileNetV3LargeTeacher(num_classes, use_eca=True, pretrained=False)
# 参数量: 5.4M
# 特征维度: 1280
```

---

## 📚 Stacking模型

```python
class StackingModel(nn.Module):
    def __init__(self, num_classes, hidden_dim=384):
        # Input: 3 * num_classes (三个教师的logits拼接)
        # Hidden: hidden_dim
        # Output: num_classes
        self.mlp = nn.Sequential(
            nn.Linear(3 * num_classes, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )
```

---

## 🎓 学生模型配置

### RepViT-M0.9 (推荐)
```python
# 最佳学生模型
# 参数量: 4.72M
# KD后精度: 98.12%
timm.create_model('repvit_m0_9', num_classes=num_classes)
```

### GhostNet-100
```python
# 参数量: 3.91M
# KD后精度: 98.05%
timm.create_model('ghostnet_100', num_classes=num_classes)
```

### MobileNetV3-Small
```python
# 极致轻量
# 参数量: 1.02M
# KD后精度: 97.40%
timm.create_model('mobilenetv3_small_100', num_classes=num_classes)
```

---

## 🔧 灰度图适配

所有模型第一层卷积需要修改输入通道从3→1:

```python
def adapt_to_grayscale(model):
    first_conv = model.conv1  # 或 model.features[0]
    new_conv = nn.Conv2d(
        1, first_conv.out_channels,
        first_conv.kernel_size, first_conv.stride,
        first_conv.padding, bias=first_conv.bias is not None
    )
    # 平均RGB权重初始化
    new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = new_conv
    return model
```

---

## 📉 损失函数

### KD Loss
```python
def distillation_loss(student_logits, teacher_logits, labels, T=3, alpha=0.3):
    # Hard Loss (CE)
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # Soft Loss (KL Divergence)
    soft_student = F.log_softmax(student_logits / T, dim=1)
    soft_teacher = F.softmax(teacher_logits / T, dim=1)
    kl_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)
    
    # Combined Loss
    return (1 - alpha) * ce_loss + alpha * kl_loss
```

### 最佳超参数
- **Temperature (T)**: 3
- **Alpha (α)**: 0.3
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Scheduler**: CosineAnnealingLR

---

## 🏆 最佳组合推荐

| 场景 | 教师 | 学生 | 预期精度 |
|------|------|------|----------|
| 最高精度 | DenseNet121-ECA | RepViT-M0.9 | ~98% |
| 平衡 | DenseNet121-ECA | GhostNet-100 | ~98% |
| 极致轻量 | DenseNet121-ECA | MobileNetV3-Small | ~97% |
| 快速训练 | ResNet50 | GhostNet-100 | ~97% |
