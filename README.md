# NetKD - 网络流量知识蒸馏框架

基于ECA增强的多教师知识蒸馏框架，用于加密网络流量分类。

---

## 🎯 核心特性

- **ECA增强教师集成**: 支持5种教师架构
  - ResNet50-ECA, DenseNet121-ECA, MobileNetV3-ECA (现有)
  - ConvNeXtV2-Tiny-ECA, EfficientNetV2-S-ECA (新增)
- **灵活教师组合**: YAML配置驱动的教师搜索实验
- **动态Stacking集成**: 支持任意数量教师的MLP融合
- **自适应知识蒸馏**: SD-MKD (Self-Distillation Multi-Knowledge Distillation)
- **Agent Attention机制**: 高性能注意力模块用于学生网络

---

## 📊 模型性能总览

### 教师模型

| 模型 | 验证准确率 | 参数量 | 模型大小 |
|------|----------|--------|---------|
| **DenseNet121-ECA** | **98.77%** ⭐ | 8M | 28MB |
| ResNet50-ECA | 98.48% | 25M | 91MB |
| MobileNetV3-ECA | 98.19% | 4.2M | 17MB |
| ConvNeXtV2-Tiny-ECA | *待验证* | 28.6M | - |
| EfficientNetV2-S-ECA | *待验证* | 21.5M | - |

### 学生模型

| 蒸馏模式 | 测试准确率 | 知识保留率 |
|---------|----------|----------|
| **S-KL** | **97.11%** ⭐ | 98.39% |
| S-CE | 97.04% | 98.32% |

### 注意力机制对比

| 注意力类型 | 测试准确率 | 参数量 |
|-----------|----------|--------|
| **Agent Attention** | **98.55%** ⭐ | 9.8M |
| CBAM | 98.05% | 480K |
| Baseline (无) | 97.83% | 349K |

---

## 🚀 快速开始

### 环境配置
```bash
conda create -n netkd python=3.12 -y
conda activate netkd
pip install -r requirements.txt
```

### 数据准备
```
Dataset/ISCXVPN2016/{train,valid,test}/class_name/*.png
```

### 训练流程

```bash
# 1. 训练教师模型
python training/train.py --use_real_data --mode train_teachers \
  --dataset ISCXVPN2016 --dataset_root /walnut_data/yqm/Dataset \
  --batch_size 256 --epochs_teacher 25 --resnet_use_eca --mbv3_use_eca

# 2. 训练Stacking集成
python training/train.py --use_real_data --mode train_stacking \
  --dataset ISCXVPN2016 --dataset_root /walnut_data/yqm/Dataset \
  --batch_size 256 --epochs_stacking 5

# 3. 训练学生模型
python training/train.py --use_real_data --mode train_student \
  --dataset ISCXVPN2016 --dataset_root /walnut_data/yqm/Dataset \
  --batch_size 128 --epochs_student 100
```

---

## 🔬 教师搜索实验 (Teacher Search)

### 教师搜索流程

使用配置驱动的教师搜索实验，寻找最优教师组合：

```bash
# 1. 训练单个教师
python experiments/teacher_search/run_train_teacher.py \
  --teacher convnextv2_tiny_eca \
  --dataset ISCXVPN2016 \
  --epochs 50

# 2. 训练教师组合的Stacking
python experiments/teacher_search/run_train_stacking.py \
  --teacher_set new_trio \
  --dataset ISCXVPN2016 \
  --epochs 30

# 3. 汇总实验结果
python analysis/summarize_teacher_search.py \
  --results_dir results/teacher_search \
  --output report.md
```

### 配置文件

教师配置位于 `configs/teachers.yaml`：

```yaml
teacher_candidates:
  resnet50_eca:
    class: ResNet50Teacher
    pretrained: true
    use_eca: true

  convnextv2_tiny_eca:
    class: ConvNeXtV2TinyTeacher
    pretrained: true
    use_eca: true

teacher_sets:
  baseline:
    teachers: [resnet50_eca, densenet121_eca, mobilenetv3_large_eca]
  
  new_trio:
    teachers: [convnextv2_tiny_eca, densenet121_eca, efficientnetv2_s_eca]
```

### 支持的教师组合

| 组合名称 | 教师模型 | 描述 |
|---------|---------|------|
| `baseline` | ResNet50 + DenseNet121 + MobileNetV3 | 原始三教师 |
| `replace_resnet` | ConvNeXtV2 + DenseNet121 + MobileNetV3 | 替换ResNet |
| `replace_mobilenet` | ResNet50 + DenseNet121 + EfficientNetV2 | 替换MobileNet |
| `new_trio` | ConvNeXtV2 + DenseNet121 + EfficientNetV2 | 全新组合 |
| `quad_ensemble` | ResNet50 + DenseNet121 + ConvNeXtV2 + EfficientNetV2 | 四教师 |
| `full_ensemble` | 全部5个教师 | 完整组合 |

---

## 📁 项目结构

```
NetKD/
├── configs/                    # 配置文件
│   └── teachers.yaml          # 教师模型配置
├── training/                   # 训练核心
│   ├── train.py               # 三阶段训练主入口
│   ├── engine.py              # 训练/验证引擎
│   └── loss_functions.py      # 知识蒸馏损失函数
├── models/                    # 模型定义
│   ├── teacher_models.py      # 教师模型 (5种架构)
│   ├── teacher_registry.py    # 教师注册表 (工厂模式)
│   ├── student_model.py       # 学生模型
│   └── eca_module.py          # ECA注意力
├── experiments/               # 实验框架
│   └── teacher_search/        # 教师搜索实验
│       ├── run_train_teacher.py    # 单教师训练
│       ├── run_train_stacking.py   # Stacking训练
│       ├── metrics.py              # 评估指标
│       └── result_schema.py        # 结果数据结构
├── analysis/                  # 结果分析
│   └── summarize_teacher_search.py  # 结果汇总
├── data_preprocessing/        # 数据处理
│   └── image_loader.py        # 图像加载器
├── scripts/                   # 辅助脚本
│   ├── run_full_training.sh   # 完整训练流程
│   ├── run_ablation_experiments.sh  # 蒸馏消融
│   ├── plot_*.py              # 结果可视化
│   └── balance_dataset.py     # 数据平衡
├── checkpoints/               # 模型权重
│   ├── *_teacher.pth          # 教师模型
│   ├── stacking_model.pth     # Stacking模型
│   ├── student_sd_mkd.pth     # 学生模型
│   ├── teacher_search/        # 教师搜索实验结果
│   ├── ablation/              # 蒸馏消融结果
│   └── attention_ablation/    # 注意力消融结果
├── results/                   # 实验结果
│   └── teacher_search/        # 教师搜索结果JSON
├── docs/                      # 文档与图表
├── logs/                      # 训练日志
├── runs/                      # TensorBoard
└── trash/                     # 归档文件
```

---

## �� 消融实验

### 蒸馏模式消融
```bash
bash scripts/run_ablation_experiments.sh
# 输出: checkpoints/ablation/s_{ce,kl,kl2}/
```

### 教师ECA消融
通过 `configs/teachers.yaml` 中的 `ablation_experiments` 配置进行。

### 结果可视化
```bash
python scripts/plot_distillation_results.py
python scripts/plot_attention_performance_final.py
python analysis/summarize_teacher_search.py --latex  # 生成LaTeX表格
```

---

## 📈 性能基准

- **GPU**: NVIDIA RTX 4090 (单卡)
- **总训练时间**: ~1小时 (单教师)
- **最佳知识保留率**: 98.39%
- **参数压缩率**: >97%

---

## 📧 联系方式

- **Email**: yuqiming24@nudt.edu.cn

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**最后更新**: 2025-12-01
