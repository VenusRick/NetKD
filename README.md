# NetKD - 网络流量知识蒸馏框架

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

基于多教师知识蒸馏的加密网络流量分类框架，支持自动化实验、模型集成和高效部署。

---

## 🎯 核心特性

### 教师模型
- **5种教师架构**: DenseNet121, MobileNetV3, ConvNeXtV2-Tiny, ResNet50, EfficientNetV2-S
- **ECA注意力增强**: 显著提升模型性能
- **动态Stacking集成**: 智能组合多个教师模型

### 学生模型
- **轻量级架构**: RepViT-M0.9 (4.72M参数)
- **知识蒸馏**: CE + Forward/Reverse KL + Sinkhorn损失
- **高性能压缩**: 保持>98%准确率，减少80%参数

### 自动化工具
- **完整实验流程**: 数据比例实验 (100%/75%/50%/25%)
- **实时监控**: 自动进度跟踪和报告生成
- **分析工具**: 教师多样性分析、贡献度评估

---

## 📊 最新实验结果 (2025-12-11)

### 🏆 最佳性能

| 指标 | 值 | 模型/数据集 |
|------|-----|-----------|
| **学生准确率** | **99.25%** ± 0.30% | RepViT-M0.9 @ ISCXTor2016 |
| **Stacking准确率** | **99.82%** ± 0.14% | 三教师集成 @ ISCXTor2016 |
| **参数压缩** | **80%+** | RepViT-M0.9 (4.72M) |

### 📈 数据效率

| 数据集 | 100%数据 | 50%数据 | 性能下降 |
|--------|---------|---------|---------|
| ISCXTor2016 | 99.25% | 89.83% | -9.42% |
| ISCXVPN2016 | 97.54% | 92.85% | -4.70% |

**详细结果**: 查看 `results/complete_experiment/FINAL_REPORT.md`

---

## 🚀 快速开始

### 1. 环境配置
```bash
# 创建环境
conda create -n netkd python=3.12 -y
conda activate netkd

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备
```
Dataset/
├── ISCXVPN2016/
│   ├── train/class_name/*.png
│   ├── valid/class_name/*.png
│   └── test/class_name/*.png
└── ISCXTor2016/...
```

### 3. 运行完整实验
```bash
# 完整流程: 教师训练 → Stacking → 学生蒸馏
python scripts/run_complete_experiment.py \
  --datasets ISCXVPN2016 ISCXTor2016 \
  --ratios 1.0 0.75 0.5 0.25 \
  --runs 3 \
  --gpu 0
```

### 4. 监控实验进度
```bash
# 实时查看进度
python scripts/monitor_and_summarize.py

# 启动自动监控 (每5分钟更新)
bash scripts/auto_monitor_loop.sh
```

### 5. 生成实验报告
```bash
python scripts/generate_final_report.py
# 输出: results/complete_experiment/FINAL_REPORT.md
```

---

## 🛠️ 核心工具

### 监控与报告
| 脚本 | 功能 |
|------|------|
| `monitor_and_summarize.py` | 实时进度监控 |
| `auto_monitor_loop.sh` | 自动循环监控 |
| `generate_final_report.py` | 完整实验报告生成 |

### 分析工具
| 脚本 | 功能 |
|------|------|
| `compute_disagreement.py` | 教师多样性分析 |
| `leave_one_out_stacking.py` | 教师贡献度评估 |
| `check_teacher2.0_progress.py` | 任务进度检查 |

---

## 📁 项目结构

```
NetKD/
├── models/                    # 模型定义
│   ├── teacher_models.py     # 教师模型
│   ├── student_models_v2.py  # 学生模型
│   └── teacher_registry.py   # 模型注册
├── training/                  # 训练模块
│   ├── train.py              # 主训练脚本
│   └── loss_functions.py     # 损失函数 (含Sinkhorn)
├── scripts/                   # 实验脚本
│   ├── run_complete_experiment.py
│   ├── monitor_and_summarize.py
│   └── generate_final_report.py
├── analysis/                  # 分析工具
│   └── compute_disagreement.py
├── configs/                   # 配置文件
│   ├── teachers.yaml
│   └── students.yaml
├── results/                   # 实验结果
│   └── complete_experiment/
│       └── FINAL_REPORT.md   # 📊 最新报告
└── docs/                      # 文档
    ├── INDEX.md              # 文档索引
    ├── 01_AGENT_HANDOVER_GUIDE.md
    ├── 02_EXPERIMENT_RESULTS.md
    └── 03_MODEL_ARCHITECTURE.md
```

---

## 📖 文档

### 新手入门
- 📘 **[文档索引](docs/INDEX.md)** - 所有文档导航
- 🚀 **[Agent交接指南](docs/01_AGENT_HANDOVER_GUIDE.md)** - 快速上手
- 📊 **[实验结果](docs/02_EXPERIMENT_RESULTS.md)** - 当前进度

### 高级主题
- 🏗️ **[模型架构](docs/03_MODEL_ARCHITECTURE.md)** - 详细设计
- 📋 **[TODO列表](docs/TODO_Teacher2.0_Student2.0.md)** - 开发计划
- 📈 **[完整实验计划](docs/COMPLETE_EXPERIMENT_PLAN.md)** - Phase 1-4

---

## 🔬 实验配置

### 教师模型配置
| 模型 | 参数量 | 注意力 | 预训练 |
|------|--------|--------|--------|
| DenseNet121-ECA | 8.0M | ECA | ✓ |
| MobileNetV3-Large-ECA | 5.4M | ECA | ✓ |
| ConvNeXtV2-Tiny-ECA | 28.6M | ECA | ✓ |

配置文件: `configs/teachers.yaml`

### 学生模型配置
| 模型 | 参数量 | 适用场景 |
|------|--------|---------|
| RepViT-M0.9 | 4.72M | 高性能 |
| GhostNet-1.0x | 3.90M | 轻量级 |
| MobileNetV3-Small | 1.02M | 极致轻量 |

配置文件: `configs/students.yaml`

### 知识蒸馏配置
- **Temperature**: 3.0
- **Alpha (CE权重)**: 0.3-0.5
- **损失函数**: CE + Forward KL + Reverse KL + Sinkhorn
- **训练轮数**: 25-30 epochs

---

## 📈 数据集支持

| 数据集 | 类别数 | 样本数 | 图像尺寸 | 状态 |
|--------|--------|--------|---------|------|
| ISCXVPN2016 | 7 | 13K+ | 40×40 灰度 | ✅ |
| ISCXTor2016 | 8 | 11K+ | 40×40 灰度 | ✅ |
| USTC-TFC2016 | 10 | 15K+ | 40×40 灰度 | ⏳ |
| CICIoT2022 | 6 | 18K+ | 40×40 灰度 | ⏳ |
| CrossPlatform-Android | 124 | 40K+ | 40×40 灰度 | 🔧 |
| CrossPlatform-iOS | 124 | 48K+ | 40×40 灰度 | 🔧 |

数据集路径: `/workspace/yqm/Dataset/`

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 许可证

MIT License

---

## 📮 联系方式

- **GitHub**: [VenusRick/NetKD](https://github.com/VenusRick/NetKD)
- **分支**: Ubuntu

---

## 🔖 版本历史

### v0.3.0 (2025-12-11)
- ✅ 完整实验系统 (6数据集×4比例×3轮)
- ✅ 自动监控和报告生成
- ✅ 教师多样性分析工具
- ✅ Leave-One-Out贡献度分析
- ✅ 文档整理和索引

### v0.2.0 (2025-12-08)
- ✅ MAE预训练实验
- ✅ 新增ConvNeXtV2和EfficientNetV2教师
- ✅ RepViT学生模型

### v0.1.0 (2025-12-07)
- ✅ 基础框架
- ✅ DenseNet121-ECA教师
- ✅ SD-MKD知识蒸馏

---

*最后更新: 2025-12-11*
