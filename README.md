# NetKD - 网络流量知识蒸馏框架

基于ECA增强的多教师知识蒸馏框架，用于加密网络流量分类。

---

## 🎯 核心特性

- **ECA增强教师集成**: ResNet50-ECA + DenseNet121-ECA + MobileNetV3-ECA
- **Stacking集成学习**: 线性MLP融合多教师输出
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

## 📁 项目结构

```
NetKD/
├── training/                    # 训练核心
│   ├── train.py                # 三阶段训练主入口
│   ├── engine.py               # 训练/验证引擎
│   └── loss_functions.py       # 知识蒸馏损失函数
├── models/                     # 模型定义
│   ├── teacher_models.py       # 教师模型
│   ├── student_model.py        # 学生模型
│   └── eca_module.py           # ECA注意力
├── data_preprocessing/         # 数据处理
│   └── image_loader.py         # 图像加载器
├── scripts/                    # 辅助脚本
│   ├── run_full_training.sh    # 完整训练流程
│   ├── run_ablation_experiments.sh  # 蒸馏消融
│   ├── plot_*.py               # 结果可视化
│   └── balance_dataset.py      # 数据平衡
├── checkpoints/                # 模型权重
│   ├── *_teacher.pth           # 教师模型
│   ├── stacking_model.pth      # Stacking模型
│   ├── student_sd_mkd.pth      # 学生模型
│   ├── ablation/               # 蒸馏消融结果
│   └── attention_ablation/     # 注意力消融结果
├── docs/                       # 文档与图表
├── logs/                       # 训练日志
├── runs/                       # TensorBoard
└── trash/                      # 归档文件
```

---

## 🔬 消融实验

### 蒸馏模式消融
```bash
bash scripts/run_ablation_experiments.sh
# 输出: checkpoints/ablation/s_{ce,kl,kl2}/
```

### 结果可视化
```bash
python scripts/plot_distillation_results.py
python scripts/plot_attention_performance_final.py
```

---

## 📈 性能基准

- **GPU**: NVIDIA RTX 4090 (单卡)
- **总训练时间**: ~1小时
- **最佳知识保留率**: 98.39%
- **参数压缩率**: >97%

---

## 📧 联系方式

- **Email**: yuqiming24@nudt.edu.cn

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

---

**最后更新**: 2025-11-30
