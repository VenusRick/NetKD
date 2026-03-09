# 🏗️ NetKD 模型架构指南

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    NetKD 知识蒸馏框架                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Teacher 1   │    │ Teacher 2   │    │ Teacher 3   │     │
│  │ EfficientV2 │    │ ConvNeXtV2  │    │ MobileNetV3 │     │
│  │   21.5M     │    │   28.6M     │    │    5.5M     │     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘     │
│         │                  │                  │             │
│         └──────────────────┼──────────────────┘             │
│                            │                                 │
│                    ┌───────▼───────┐                        │
│                    │   Stacking    │                        │
│                    │   Ensemble    │                        │
│                    │   (学习权重)   │                        │
│                    └───────┬───────┘                        │
│                            │                                 │
│              ┌─────────────▼─────────────┐                  │
│              │    Knowledge Distillation │                  │
│              │    CE + KL (T=3)          │                  │
│              └─────────────┬─────────────┘                  │
│                            │                                 │
│  ┌─────────────────────────▼─────────────────────────────┐  │
│  │                    Student Models                      │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │  │
│  │  │EdgeNeXt  │ │MobileV3  │ │MobileOne │ │ RepViT   │  │  │
│  │  │  1.33M   │ │  1.52M   │ │  2.08M   │ │  4.72M   │  │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## 2. 教师模型详情

| 模型 | 参数量 | 特点 | 贡献度 |
|------|--------|------|--------|
| efficientnetv2_rw_s | 21.5M | SE注意力 + 复合缩放 | +1.5% F1 |
| convnextv2_tiny | 28.6M | 现代化设计 + LayerNorm | +1.0% F1 |
| mobilenetv3_large_100 | 5.5M | 轻量级 + 硬件友好 | +0.5% F1 |

**总计**: 55.6M 参数 → 集成后 F1: 98%

## 3. 学生模型推荐

### Pareto 最优模型

| 场景 | 推荐模型 | 参数量 | F1 Score | 压缩比 |
|------|----------|--------|----------|--------|
| IoT/嵌入式 | edgenext_xx_small | 1.33M | 94.0% | 42x |
| 移动端 | mobilenetv3_small | 1.52M | 96.2% | 37x |
| 边缘服务器 | mobileone_s0 | 2.08M | 95.5% | 27x |
| 云端/高性能 | repvit_m0_9 | 4.72M | 97.4% | 12x |

### 选择指南

```python
def recommend_student(latency_budget_ms, memory_budget_mb):
    """根据部署约束推荐学生模型"""
    if latency_budget_ms < 5 and memory_budget_mb < 10:
        return "edgenext_xx_small"  # 1.33M, 极致轻量
    elif latency_budget_ms < 10 and memory_budget_mb < 20:
        return "mobilenetv3_small"  # 1.52M, 最佳性价比
    elif latency_budget_ms < 20 and memory_budget_mb < 30:
        return "mobileone_s0"  # 2.08M, 平衡选择
    else:
        return "repvit_m0_9"  # 4.72M, 最高性能
```

## 4. 知识蒸馏配置

### 最佳配置 (CE + KL)

```python
KD_CONFIG = {
    "temperature": 3.0,      # 软化温度
    "alpha_ce": 0.5,         # CE损失权重
    "alpha_kl": 0.5,         # KL损失权重
}
```

### 损失函数

```python
def kd_loss(student_logits, teacher_logits, labels, T=3.0, alpha=0.5):
    # 硬标签损失
    ce_loss = F.cross_entropy(student_logits, labels)
    
    # 软标签损失
    student_soft = F.log_softmax(student_logits / T, dim=1)
    teacher_soft = F.softmax(teacher_logits / T, dim=1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
    
    return alpha * ce_loss + (1 - alpha) * kl_loss
```

## 5. 训练配置

### 推荐超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| batch_size | 128 | RTX 4090 单卡最优 |
| learning_rate | 0.001 | AdamW 初始学习率 |
| weight_decay | 1e-4 | L2正则化 |
| epochs | 100 | 约40-60 epoch收敛 |
| warmup_epochs | 5 | 学习率预热 |
| scheduler | cosine | 余弦退火 |

### 数据增强

```python
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
```

## 6. 快速开始

### 训练学生模型

```bash
# CE-only baseline
python scripts/run_kd_simple.py \
    --student mobilenetv3_small \
    --train_fraction 1.0 \
    --gpu 0

# CE+KL 知识蒸馏
python scripts/run_kd_simple.py \
    --student mobilenetv3_small \
    --use_kd \
    --train_fraction 1.0 \
    --gpu 0
```

### 批量实验

```bash
# 运行所有配置
bash scripts/run_kd_parallel.sh
```

## 7. 性能基准

### ISCXVPN2016 数据集

| 模型 | 参数量 | 100%数据 | 50%数据 | 20%数据 |
|------|--------|----------|---------|---------|
| repvit_m0_9 (CE) | 4.72M | 97.4% | 91.9% | 83.6% |
| mobilenetv3_small (CE) | 1.52M | 96.2% | 86.4% | 77.4% |

### 数据效率

- repvit_m0_9: 50%数据仅下降 5.5% F1
- mobilenetv3_small: 50%数据下降 9.8% F1
- **结论**: 较大模型对数据量更鲁棒

## 8. 文件结构

```
NetKD/
├── models/
│   ├── teacher_models.py      # 教师模型定义
│   ├── student_models_v2.py   # 学生模型定义
│   └── stacking.py            # Stacking集成
├── scripts/
│   ├── run_kd_simple.py       # 简化版KD训练
│   ├── run_kd_parallel.sh     # 并行实验脚本
│   └── run_simclr_bs128.py    # SimCLR预训练
├── analysis/
│   ├── teacher_analysis.py    # STEP 1 教师分析
│   ├── pareto_analysis.py     # STEP 2 Pareto分析
│   └── analyze_step3_step4.py # STEP 3&4 分析
└── results/
    ├── kd_ablation_*/         # 消融实验结果
    └── kd_simple_*/           # KD实验结果
```

---

**最后更新**: 2025-12-08
**作者**: CodeAgent
