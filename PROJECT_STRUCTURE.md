# NetKD 项目结构说明

## 📁 整体架构

```
NetKD/
├── 📂 models/                    # 神经网络模型定义
├── 📂 training/                  # 训练逻辑和工具
├── 📂 experiments/               # 实验脚本
├── 📂 data_preprocessing/        # 数据预处理管道
├── 📂 util/                      # 通用工具函数
├── 📄 main.py                    # 主程序入口
└── 📄 README.md                  # 项目说明
```

---

## 🎯 核心模块详解

### 1️⃣ **models/** - 模型定义模块

#### 📄 `student_model.py` - 学生模型
**功能**: 轻量级学生网络实现
- **ShuffleNetBackbone**: ShuffleNetV2 0.5x骨干网络（适配单通道输入）
- **AgentAttention2D**: 高效注意力机制（O(N×M)复杂度，使用agent tokens）
- **StudentNet**: 完整学生模型（ShuffleNetV2 + AgentAttention + 分类头）

**特点**:
- 参数量小，适合部署
- 使用Agent Attention降低计算复杂度
- 支持单通道流量图像输入

#### 📄 `teacher_models.py` - 教师模型集合
**功能**: 三个重量级教师网络及集成方法
- **ResNet50Teacher**: 基于ResNet50的教师模型
- **MobileNetV3LargeTeacher**: 基于MobileNetV3-Large的教师模型
- **DenseNet121Teacher**: 基于DenseNet121的教师模型
- **TeacherEnsemble**: 教师集成类（管理三个教师模型）
- **StackingModel**: MLP融合层（将三个教师的输出融合）
- **train_single_teacher()**: 单个教师训练函数
- **train_stacking_model()**: Stacking模型训练函数

**特点**:
- 所有教师都适配单通道输入
- 支持ImageNet预训练权重
- 包含完整的训练逻辑

#### 📂 `teacher_models_package/` - 教师模型扩展包
高级教师模型功能（模块化设计，备用）

---

### 2️⃣ **training/** - 训练逻辑模块

#### 📄 `loss_functions.py` - 损失函数
**功能**: SD-MKD的复合蒸馏损失
- **ce_loss()**: 交叉熵损失（学生对真实标签）
- **forward_kl()**: 前向KL散度 KL(P_teacher || P_student)
- **reverse_kl()**: 反向KL散度 KL(P_student || P_teacher)
- **sinkhorn_distance()**: Sinkhorn最优传输距离（考虑类别间结构）
- **distillation_loss()**: 复合蒸馏损失（CE + FKL + RKL + Sinkhorn）
- **class_cost_matrix()**: 类别代价矩阵生成

**核心公式**:
```
Total Loss = λ_ce·CE + λ_f·FKL + λ_r·RKL + λ_s·Sinkhorn
```

#### 📄 `train.py` - 三阶段训练流程
**功能**: SD-MKD的完整训练管道

**三个训练阶段**:
1. **train_teachers()**: 教师预训练（数据集A）
   - 独立训练ResNet50、MobileNetV3、DenseNet121
   - 使用交叉熵损失
   
2. **train_stacking_model_stage()**: Stacking集成训练（数据集B）
   - 冻结教师模型
   - 训练MLP融合层
   
3. **train_student_stage()**: 学生蒸馏（数据集B）
   - 使用复合蒸馏损失
   - 从教师集成和Stacking模型学习知识

**Demo功能**:
- **run_demo_pipeline()**: 快速演示完整流程（使用虚拟数据）

#### 📄 `evaluation.py` - 评估工具
**功能**: 模型性能评估
- **evaluate_model()**: 计算准确率、F1分数、混淆矩阵
- **summarize_metrics()**: 生成可读的评估报告

#### 📄 `engine.py` - 通用训练引擎
**功能**: 底层训练和评估函数（支持分布式、混合精度）
- **train_one_epoch()**: 单轮训练
- **evaluate()**: 完整评估
- **pretrain_one_epoch()**: 预训练轮次（支持MAE等自监督方法）

---

### 3️⃣ **experiments/** - 实验脚本模块

#### 📄 `sd_mkd.py` - SD-MKD完整实现
**功能**: 独立的SD-MKD训练脚本
- 完整的三阶段训练流程
- 可作为参考实现或独立运行

#### 📄 `ablation.py` - 消融实验
**功能**: 测试不同损失组合的效果
- **AblationConfig**: 消融实验配置（控制各损失权重）
- **default_ablation_configs()**: 预定义的消融配置
  - baseline: 仅CE
  - +FKL: CE + 前向KL
  - +FKL+RKL: CE + 前向KL + 反向KL
  - full: 完整损失（CE + FKL + RKL + Sinkhorn）
- **run_ablation_study()**: 执行消融实验

---

### 4️⃣ **data_preprocessing/** - 数据预处理模块

**功能**: 加密流量数据处理管道

#### 核心组件:
- **FlowExtractor**: 从PCAP文件提取流量
- **ByteExtractor**: 字节级特征提取
- **FlowDataset**: PyTorch数据集封装
- **LabelEncoder**: 标签编码和类别权重计算
- **DataPreprocessor**: 完整预处理流程
- **CacheManager**: 缓存管理（加速重复实验）
- **DataValidator**: 数据验证
- **DatasetStatistics**: 数据集统计分析

#### 数据集适配器:
- **ISCXVPN2016Adapter**: VPN流量数据集
- **ISCXTor2016Adapter**: Tor流量数据集
- **USTCTFC2016Adapter**: TFC流量数据集

**特点**:
- 支持多种公开数据集
- 自动缓存加速处理
- 灵活的字节聚合策略

---

### 5️⃣ **util/** - 通用工具模块

**功能**: 训练辅助工具

- **misc.py**: 杂项工具（分布式训练、检查点保存、日志等）
- **lr_sched.py**: 学习率调度器
- **lr_decay.py**: 学习率衰减策略
- **lars.py**: LARS优化器
- **pos_embed.py**: 位置编码工具
- **crop.py**: 数据增强

---

### 6️⃣ **main.py** - 主程序入口

**功能**: 命令行接口

**运行模式**:
```bash
# 训练教师模型（阶段I）
python main.py --mode train_teachers

# 训练Stacking模型（阶段II）
python main.py --mode train_stacking

# 训练学生模型（阶段III）
python main.py --mode train_student

# 评估学生模型
python main.py --mode eval --student_ckpt student_sd_mkd.pth
```

---

## 🔄 数据流程

```
原始PCAP文件
    ↓
[data_preprocessing] 流量提取和转换
    ↓
单通道流量图像 [B, 1, H, W]
    ↓
[models/teacher_models] 教师预训练（阶段I）
    ↓
[models/teacher_models] Stacking集成（阶段II）
    ↓
[models/student_model] 学生蒸馏（阶段III）
    ↓
[training/evaluation] 性能评估
```

---

## 📊 训练流程

### 阶段I: 教师预训练（数据集A）
- 训练3个教师: ResNet50, MobileNetV3, DenseNet121
- 损失: 交叉熵
- 保存检查点: `resnet50_teacher.pth`, `mbv3_teacher.pth`, `densenet121_teacher.pth`

### 阶段II: Stacking集成（数据集B）
- 冻结教师模型
- 训练MLP融合层
- 保存检查点: `stacking_model.pth`

### 阶段III: 学生蒸馏（数据集B）
- 学生: ShuffleNetV2 + AgentAttention
- 教师: 冻结的教师集成 + Stacking
- 损失: CE + FKL + RKL + Sinkhorn
- 保存检查点: `student_sd_mkd.pth`

---

## 🎓 关键技术

1. **知识蒸馏**: 从大模型向小模型传递知识
2. **Stacking集成**: 元学习融合多个教师
3. **多级知识**: 结合logits和softmax分布
4. **Sinkhorn距离**: 考虑类别间结构的OT距离
5. **Agent Attention**: 降低自注意力复杂度

---

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate netkd

# 2. 运行完整训练流程（使用虚拟数据）
python main.py --mode train_student

# 3. 运行消融实验
python -m experiments.ablation

# 4. 使用真实数据（需先预处理）
python -m data_preprocessing.preprocess_data \
    --dataset ISCXVPN2016 \
    --data_path /path/to/raw/data
```

---

## 📦 依赖环境

**核心依赖**:
- PyTorch >= 2.0
- torchvision
- scikit-learn
- numpy
- scipy
- timm
- tqdm

**可选依赖**:
- matplotlib (数据可视化)
- tensorboard (训练监控)

---

## 📝 论文方法

**SD-MKD**: Stacking-based Distillation with Multi-level Knowledge and Sinkhorn Distances

**核心创新**:
1. 使用Stacking集成提升教师质量
2. 多级知识蒸馏（logits + 分布）
3. Sinkhorn距离捕捉类别间关系
4. Agent Attention降低学生复杂度

**应用场景**: 加密流量分类（VPN、Tor等网络流量识别）
