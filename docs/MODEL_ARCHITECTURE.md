# NetKD 模型架构参考
**创建日期**: 2025-12-08
**最后更新**: 2025-12-08 15:43 UTC+8

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                      NetKD Framework                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  Teacher 1  │  │  Teacher 2  │  │  Teacher 3  │         │
│  │ DenseNet121 │  │  ResNet50   │  │ MobileNetV3 │         │
│  │    +ECA     │  │             │  │    +ECA     │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │   Knowledge Distill   │                      │
│              │  T=3, α=0.3, CE+FKL   │                      │
│              └───────────┬───────────┘                      │
│                          ▼                                  │
│              ┌───────────────────────┐                      │
│              │      Student          │                      │
│              │  GhostNet/MobileNet   │                      │
│              └───────────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 教师模型推荐

| 模型 | 参数量 | Test Acc | 推荐场景 | 路径 |
|------|--------|----------|----------|------|
| DenseNet121-ECA | 8.0M | **98.77%** | 首选教师 | checkpoints/densenet121_teacher.pth |
| ResNet50 | 23.5M | 98.19% | 高容量 | checkpoints/resnet50_teacher.pth |
| MobileNetV3-Large | 5.4M | 98.19% | 轻量教师 | checkpoints/mbv3_teacher.pth |

---

## 学生模型推荐

### 性能排行 (无KD基准)

| 排名 | 模型 | 参数量 | Test Acc | 场景 |
|------|------|--------|----------|------|
| 1 | ghostnet_100 | 3.91M | **97.40%** | 精度优先 |
| 2 | efficientnet_lite0 | 3.38M | 97.11% | 均衡推荐 |
| 3 | mobilenetv3_small_050 | 0.58M | 94.08% | 极致轻量 |

### timm 可用模型名

```python
# 轻量级 (<1M)
'mobilenetv3_small_050'   # 0.58M
'mobilenetv3_small_075'   # 1.0M

# 中等 (1-5M)
'mobilenetv3_small_100'   # 1.5M
'efficientnet_lite0'      # 3.38M
'ghostnet_100'            # 3.91M

# 较大 (5-10M)
'mobilenetv3_large_100'   # 5.4M
'efficientnet_lite1'      # 5.4M
```

---

## 知识蒸馏配置

### 损失函数

```
Total Loss = α × L_soft + (1-α) × L_hard

L_soft = KL(σ(z_s/T) || σ(z_t/T)) × T²
L_hard = CE(z_s, y)
```

### 推荐超参数

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| Temperature (T) | 3.0 | 软化程度 |
| Alpha (α) | 0.3 | soft loss 权重 |
| Loss Type | CE + Forward KL | 最优组合 |

---

## 注意力模块

| 注意力 | 准确率提升 | 参数增加 | 推荐度 |
|--------|------------|----------|--------|
| Agent Attention | +0.57% | +0.12M | ⭐⭐⭐ |
| ECA | +0.50% | +0.01M | ⭐⭐⭐ |
| SE | +0.28% | +0.05M | ⭐⭐ |
| CBAM | +0.14% | +0.08M | ⭐ |

---

## 训练配置推荐

### 学生训练 (纯CE)
```python
epochs = 30        # 20 epochs 可用于快速实验
lr = 1e-3
batch_size = 64
optimizer = AdamW(weight_decay=0.01)
```

### 学生蒸馏 (KD)
```python
epochs = 30
lr = 1e-4          # 蒸馏用较小学习率
temperature = 3.0
alpha = 0.3
```

---

## 最佳配置组合

### 场景1: 高精度部署 (>97%)
```
教师: DenseNet121-ECA
学生: ghostnet_100
KD: T=3, α=0.3
预期: ~98% @ 3.91M params
```

### 场景2: 边缘设备 (1-4M)
```
教师: DenseNet121-ECA
学生: efficientnet_lite0
KD: T=3, α=0.3
预期: ~97.5% @ 3.38M params
```

### 场景3: 极致轻量 (<1M)
```
教师: DenseNet121-ECA
学生: mobilenetv3_small_050
KD: T=4, α=0.3
预期: ~95-96% @ 0.58M params
```

---

## 快速启动命令

```bash
cd /workspace/yqm/NetKD

# 单个学生训练
python -u scripts/simple_student_kd.py --gpu 0 --student ghostnet_100

# 3 GPU 并行
nohup python -u scripts/simple_student_kd.py --gpu 0 --student mobilenetv3_small_050 > logs/gpu0.log 2>&1 &
nohup python -u scripts/simple_student_kd.py --gpu 1 --student ghostnet_100 > logs/gpu1.log 2>&1 &
nohup python -u scripts/simple_student_kd.py --gpu 2 --student efficientnet_lite0 > logs/gpu2.log 2>&1 &
```

---

## 关键代码文件

| 文件 | 功能 |
|------|------|
| `data_preprocessing/image_loader.py` | 数据加载 |
| `models/teacher_models.py` | 教师模型 |
| `models/student_model.py` | 学生模型 |
| `scripts/simple_student_kd.py` | 简单训练脚本 |
