# NetKD 模型架构参考
**创建日期**: 2025-12-08
**维护者**: CodeAgent

---

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      NetKD Framework                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Teacher 1  │  │  Teacher 2  │  │  Teacher 3  │         │
│  │ DenseNet121 │  │ EfficientV2 │  │ MobileNetV3 │         │
│  │    +ECA     │  │    +ECA     │  │    +ECA     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │    Stacking Layer     │                      │
│              │   (MLP Ensemble)      │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   Knowledge Distill   │                      │
│              │  (SD-MKD Framework)   │                      │
│              └───────────┬───────────┘                      │
│                          │                                  │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │      Student          │                      │
│              │  ShuffleNet v2 +      │                      │
│              │  Agent Attention      │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 👨‍🏫 教师模型配置

### 推荐配置 (已验证最佳)

| 模型 | 参数量 | Test Acc | 推荐场景 |
|------|--------|----------|----------|
| DenseNet121-ECA | 8.0M | **98.77%** | 首选教师 |
| EfficientNetV2-RW-S | 22.2M | 98.19% | 高精度需求 |
| MobileNetV3-Large-ECA | 5.4M | 98.19% | 轻量化需求 |

### YAML 配置示例

```yaml
# configs/teachers.yaml
teachers:
  densenet121:
    backbone: densenet121
    pretrained: true
    attention: eca
    num_classes: 12
    in_channels: 1
    
  efficientnetv2:
    backbone: efficientnetv2_rw_s
    pretrained: true
    attention: eca
    num_classes: 12
    in_channels: 1
    
  mobilenetv3:
    backbone: mobilenetv3_large_100
    pretrained: true
    attention: eca
    num_classes: 12
    in_channels: 1
```

### 预训练策略选择

| Backbone | 推荐预训练 | 备注 |
|----------|------------|------|
| EfficientNetV2 | MAE (mr=0.6) | +0.5% vs scratch |
| ConvNeXtV2 | Scratch | MAE效果差 |
| MobileNetV3 | MAE (mr=0.8) | 轻微提升 |
| DenseNet121 | ImageNet | 足够好 |

---

## 👨‍�� 学生模型配置

### 推荐配置 (已验证最佳)

| 模型 | 参数量 | Test Acc | 适用场景 |
|------|--------|----------|----------|
| RepViT-M0.9 | 4.72M | **98.12%** | 精度优先 |
| GhostNet-100 | 3.91M | 98.05% | 均衡 |
| MobileNetV3-Small | 1.02M | 97.40% | 极致轻量 |
| MobileNetV2-0.5 | 1.97M | 97.98% | 经典选择 |

### YAML 配置示例

```yaml
# configs/students.yaml
student:
  backbone: shufflenet_v2_x0_5
  attention: agent_attention
  num_classes: 12
  in_channels: 1
  width_mult: 0.5
  
distillation:
  temperature: 3.0
  alpha: 0.3
  loss_type: ce_fkl
```

---

## 🎯 知识蒸馏配置

### SD-MKD 损失组件

```
Total Loss = α × L_soft + (1-α) × L_hard

L_soft = KL(σ(z_s/T) || σ(z_t/T)) × T²
L_hard = CE(z_s, y)
```

### 推荐超参数

| 参数 | 推荐值 | 范围 | 说明 |
|------|--------|------|------|
| Temperature (T) | 3.0 | 2-6 | 越大越软 |
| Alpha (α) | 0.3 | 0.1-0.7 | soft权重 |
| Loss Type | ce_fkl | - | CE + Forward KL |

### 消融实验结论

| 损失组合 | Test Acc | 推荐 |
|----------|----------|------|
| CE + Forward KL | **98.12%** | ✅ |
| CE + Reverse KL | 97.89% | |
| CE + Symmetric KL | 97.95% | |
| CE Only | 97.40% | |

---

## 🔧 注意力模块

### 性能对比

| 注意力 | 准确率提升 | 参数增加 | 推荐度 |
|--------|------------|----------|--------|
| Agent Attention | +0.57% | +0.12M | ⭐⭐⭐ |
| ECA | +0.50% | +0.01M | ⭐⭐⭐ |
| SE | +0.28% | +0.05M | ⭐⭐ |
| CBAM | +0.14% | +0.08M | ⭐ |

### Agent Attention 配置

```python
# models/student_model.py
class AgentAttention2D(nn.Module):
    def __init__(self, in_channels, num_agents=4, reduction=4):
        # 参数: in_channels, num_agents, reduction
        pass
```

---

## 📊 最佳组合推荐

### 场景 1: 高精度部署

```
教师: DenseNet121-ECA + EfficientNetV2-ECA + MobileNetV3-ECA
学生: RepViT-M0.9 + Agent Attention
KD: T=3, α=0.3
预期: 98.12% @ 4.72M params
```

### 场景 2: 边缘设备

```
教师: DenseNet121-ECA + MobileNetV3-ECA
学生: MobileNetV3-Small + ECA
KD: T=4, α=0.3
预期: 97.40% @ 1.02M params
```

### 场景 3: 极致轻量

```
教师: DenseNet121-ECA
学生: ShuffleNet v2 x0.5 + ECA
KD: T=3, α=0.3
预期: 96.5% @ 0.35M params
```

---

## 📁 关键代码文件

| 文件 | 功能 |
|------|------|
| `models/teacher_models.py` | 教师模型定义 |
| `models/student_model.py` | 学生模型定义 |
| `models/eca_module.py` | ECA注意力模块 |
| `training/kd_losses.py` | KD损失函数 |
| `training/train.py` | 训练主脚本 |

---

## ⚡ 快速启动

### 最佳配置一键启动

```bash
cd /workspace/yqm/NetKD

# 教师训练
python training/train.py \
    --use_real_data \
    --mode train_teachers \
    --teachers densenet121,efficientnetv2_rw_s,mobilenetv3_large_100

# 学生蒸馏
python training/train.py \
    --use_real_data \
    --mode train_student \
    --student repvit_m0_9 \
    --temperature 3 \
    --alpha 0.3
```

---

**更新日志**:
- 2025-12-08: 初始版本，整理最佳配置
