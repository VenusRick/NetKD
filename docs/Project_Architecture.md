# NetKD 项目架构说明 (最新版)

## 项目概述

NetKD是一个基于多教师知识蒸馏的网络流量分类框架,采用ECA注意力机制增强的教师模型,通过Stacking集成和多分布蒸馏将知识传递给轻量级学生模型。

**更新日期**: 2025-11-22

---

## 目录结构

```
NetKD/
├── data_preprocessing/          # 数据预处理模块
│   ├── dataset.py              # 数据集定义
│   ├── quick_load_dataset.py   # 快速数据加载
│   ├── augmentation.py         # 数据增强
│   └── ...
├── models/                      # 模型定义
│   ├── teacher_models.py       # 教师模型(ResNet50/DenseNet121/MobileNetV3)
│   ├── eca_module.py          # ECA注意力模块
│   └── student_model.py       # 学生模型
├── training/                    # 训练相关
│   ├── train.py               # 主训练脚本 ⭐核心
│   ├── engine.py              # 训练引擎
│   ├── loss_functions.py      # 损失函数(KL/Sinkhorn等)
│   └── evaluation.py          # 评估工具
├── checkpoints/                 # 模型检查点
│   ├── eca_pipeline/          # ECA教师模型和学生模型
│   ├── mbv3_final/            # MobileNetV3不同batch size实验
│   └── ablation/              # 消融实验结果(待生成)
├── docs/                        # 📚 项目文档
│   ├── Project_Architecture.md      # 本文档
│   ├── Paper_Materials.md           # 论文素材(实验结果/表格/段落)
│   ├── Ablation_Experiment_Plan.md  # 消融实验方案
│   └── Experiment_Results.md        # 实验结果汇总(待生成)
├── logs/                        # 训练日志
└── Dataset/                     # 数据集根目录
    ├── ISCXVPN2016/
    ├── CICIoT2022/
    └── ...
```

---

## 核心模块说明

### 1. 数据预处理 (`data_preprocessing/`)

**quick_load_dataset.py** - 快速数据加载接口
```python
def quick_load_dataset(
    dataset_name: str,           # 数据集名称
    dataset_root: str,           # 数据集根目录
    batch_size: int = 256,
    num_workers: int = 4,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[DataLoader, DataLoader, DataLoader, int]
```

**特点**:
- 自动检测预分割数据集(train/val/test文件夹)
- 支持运行时动态分割
- 返回训练/验证/测试DataLoader和类别数

---

### 2. 模型架构 (`models/`)

#### 2.1 教师模型 (`teacher_models.py`)

三个ECA增强的教师模型:

```python
# ResNet50-ECA (25.6M参数)
def resnet50_eca_teacher(num_classes, use_eca=True, pretrained=False)

# DenseNet121-ECA (8.0M参数)  
def densenet121_eca_teacher(num_classes, use_eca=True, pretrained=False)

# MobileNetV3-ECA (5.5M参数)
def mobilenetv3_eca_teacher(num_classes, use_eca=True, pretrained=False)
```

**关键发现**:
- ResNet50/DenseNet121对batch size不敏感(bs=512正常)
- MobileNetV3对batch size极度敏感:
  * bs=128: 98.19% ✓
  * bs=512: 50.54% ✗ (训练崩溃)

#### 2.2 ECA模块 (`eca_module.py`)

Efficient Channel Attention - 轻量级通道注意力机制

```python
class ECA(nn.Module):
    def __init__(self, k_size=3):
        # 1D卷积捕获通道间依赖
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2)
```

**优势**:
- 参数量极少(~100参数)
- 计算开销低
- 对ResNet/DenseNet提升明显

#### 2.3 学生模型 (`student_model.py`)

轻量级CNN模型(0.78M参数)

```python
class StudentModel(nn.Module):
    # 简化的ResNet-like架构
    # 通过知识蒸馏学习教师知识
```

---

### 3. 训练流程 (`training/train.py`) ⭐

#### 3.1 三阶段训练架构

```python
# Stage 1: 训练教师模型
train_teachers(
    dataset_name="ISCXVPN2016",
    batch_size=512,              # ResNet50/DenseNet121
    # batch_size=128,            # MobileNetV3(需要小batch size)
    epochs=25,
    resnet_use_eca=True,
    mbv3_use_eca=True
)

# Stage 2: 训练Stacking集成
train_stacking(
    # 使用已训练的教师模型
    batch_size=256,
    epochs=5
)

# Stage 3: 训练学生模型
train_student(
    # 使用Stacking作为teacher
    batch_size=128,              # 最优配置
    epochs=100,
    distill_mode='full'          # ce/kl/kl2/full
)
```

#### 3.2 蒸馏损失函数 (`loss_functions.py`)

四种蒸馏模式:

**1. S-CE (仅硬标签)**
```python
loss = CE(student_logits, ground_truth)
```

**2. S-KL (标准KD)**
```python
loss = (1-α)·CE + α·KL(teacher||student)
```

**3. S-KL2 (双向KL)**
```python
loss = (1-α)·CE + α·FKL(teacher||student) + β·RKL(student||teacher)
```

**4. S-Full (多分布蒸馏)**
```python
loss = (1-α)·CE + α·FKL + β·RKL + γ·Sinkhorn(teacher, student)
```

**默认参数值**:
- α (lamb_f) = 0.5  # Forward KL权重
- β (lamb_r) = 0.5  # Reverse KL权重
- γ (lamb_s) = 0.1  # Sinkhorn权重

---

## 实验配置

### 硬件环境
- GPU: 单张NVIDIA GPU
- CPU: 8核(num_workers=8)
- 内存: 充足(数据集较小,1×40×40图像)

### 超参数设置

```python
# 教师模型训练
TEACHER_CONFIG = {
    "epochs": 25,
    "batch_size": 512,           # ResNet50/DenseNet121
    # "batch_size": 128,         # MobileNetV3
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "CosineAnnealingLR",
    "amp": True                  # 混合精度训练
}

# Stacking训练
STACKING_CONFIG = {
    "epochs": 5,
    "batch_size": 256,
    "optimizer": "AdamW",
    "lr": 1e-3
}

# 学生模型训练
STUDENT_CONFIG = {
    "epochs": 100,
    "batch_size": 128,           # 最优精度
    # "batch_size": 512,         # 最快速度(4min)
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "scheduler": "ReduceLROnPlateau",
    "distill_mode": "full"       # ce/kl/kl2/full
}
```

---

## 实验结果

### 教师模型性能

| 模型 | 参数量 | Batch Size | 验证准确率 | 训练时间 |
|------|--------|-----------|-----------|---------|
| ResNet50-ECA | 25.6M | 512 | 98.48% | 8.6 min |
| DenseNet121-ECA | 8.0M | 512 | 98.77% | 8.8 min |
| MobileNetV3-ECA | 5.5M | 128 | 98.19% | 9.0 min |
| Stacking | 0.047M | 256 | 98.70% | 2.2 min |

### 学生模型性能(不同Batch Size)

| Batch Size | 验证准确率 | 知识保留率 | 训练时间 |
|-----------|-----------|-----------|---------|
| 128 | 98.34% | 99.64% | 11 min |
| 256 | 97.61% | 98.90% | 6 min |
| 512 | 97.76% | 99.05% | 4 min |

**知识保留率** = (学生准确率 / Stacking准确率) × 100%

---

## 命令行使用示例

### 1. 训练教师模型
```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --use_real_data \
  --mode train_teachers \
  --dataset ISCXVPN2016 \
  --dataset_root /path/to/Dataset \
  --batch_size 512 \
  --num_workers 8 \
  --epochs_teacher 25 \
  --resnet_use_eca \
  --mbv3_use_eca \
  --output_dir checkpoints/teachers
```

### 2. 训练Stacking
```bash
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --use_real_data \
  --mode train_stacking \
  --dataset ISCXVPN2016 \
  --batch_size 256 \
  --epochs_stacking 5 \
  --output_dir checkpoints/pipeline
```

### 3. 训练学生模型(消融实验)
```bash
# Baseline (CE only)
CUDA_VISIBLE_DEVICES=0 python training/train.py \
  --use_real_data \
  --mode train_student \
  --dataset ISCXVPN2016 \
  --batch_size 128 \
  --epochs_student 100 \
  --distill_mode ce \
  --output_dir checkpoints/ablation/s_ce

# Standard KD
python training/train.py ... --distill_mode kl

# Bidirectional KL
python training/train.py ... --distill_mode kl2

# Full (当前最优)
python training/train.py ... --distill_mode full
```

---

## 关键发现与洞察

### 1. Batch Size敏感性

**轻量级模型(MobileNetV3)**:
- 极度敏感,大batch size导致训练崩溃
- 原因: Batch Normalization统计不准确
- 建议: bs ≤ 256

**重量级模型(ResNet50/DenseNet121)**:
- 对batch size不敏感
- 可使用bs=512提高训练效率

### 2. ECA模块有效性

- ✓ ResNet50: 基线 → +ECA 提升明显
- ✓ DenseNet121: 达到最高准确率98.77%
- ✓ MobileNetV3: 在合适batch size下表现良好(98.19%)

### 3. 知识蒸馏效果

- Stacking集成准确率: 98.70%
- 最佳学生模型: 98.34%
- 知识保留率: 99.64%
- 参数压缩比: 33× (25.6M → 0.78M)

### 4. 训练效率

- 完整pipeline: ~40分钟
- 学生模型: 4-11分钟(取决于batch size)
- 非常适合快速实验迭代

---

## 消融实验设计

### 目标
证明多分布蒸馏(FKL+RKL+Sinkhorn)优于标准KD

### 四组对比实验

| 模式 | 损失函数 | 预期准确率 | 训练时间 |
|------|---------|-----------|---------|
| S-CE | GT only | ~95-96% | 11 min |
| S-KL | GT + FKL | ~97-97.5% | 11 min |
| S-KL2 | GT + FKL + RKL | ~97.8-98.1% | 11 min |
| S-Full | GT + FKL + RKL + Sinkhorn | 98.34% | 11 min |

### 预期趋势
```
Acc(S-CE) < Acc(S-KL) < Acc(S-KL2) ≤ Acc(S-Full)
```

---

## 未来工作

### 短期计划
- [x] 修正MobileNetV3训练配置
- [x] 生成完整论文材料
- [ ] 完成消融实验
- [ ] 跨数据集泛化实验

### 中期计划
- [ ] 单教师vs多教师对比
- [ ] 更多数据集验证(CICIoT2022, USTC-TFC2016)
- [ ] 学生模型架构优化

### 长期计划
- [ ] 在线蒸馏支持
- [ ] 模型量化和剪枝
- [ ] 部署优化

---

## 参考资料

### 论文材料
- `docs/Paper_Materials.md` - 完整实验结果和LaTeX表格
- `docs/Ablation_Experiment_Plan.md` - 消融实验设计

### 代码入口
- `training/train.py` - 主训练脚本(支持三阶段训练)
- `models/teacher_models.py` - 教师模型定义
- `training/loss_functions.py` - 蒸馏损失函数

### 实验记录
- `checkpoints/eca_pipeline/` - 已训练模型
- `logs/` - 训练日志
- `docs/Experiment_Results.md` - 实验结果汇总(待生成)

---

**维护者**: AI Assistant  
**最后更新**: 2025-11-22  
**版本**: v2.0 (消融实验版)
