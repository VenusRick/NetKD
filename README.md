# NetKD - 高效加密流量分类知识蒸馏框架

基于Stacking集成与多级知识蒸馏的轻量级网络流量分类系统

---

## 📁 项目结构

```
NetKD/
├── 📂 models/              # 神经网络模型定义
│   ├── student_model.py    # ShuffleNetV2 + AgentAttention学生模型
│   ├── teacher_models.py   # ResNet50/MobileNetV3/DenseNet121教师模型
│   └── teacher_models_package/  # 教师模型扩展包
│
├── 📂 training/            # 训练逻辑与工具
│   ├── train.py            # 三阶段训练流程（教师→Stacking→学生）
│   ├── loss_functions.py   # 复合蒸馏损失（CE+FKL+RKL+Sinkhorn）
│   ├── evaluation.py       # 模型评估工具
│   └── engine.py           # 通用训练引擎
│
├── 📂 experiments/         # 实验脚本
│   ├── sd_mkd.py           # SD-MKD完整实现
│   └── ablation.py         # 消融实验
│
├── 📂 data_preprocessing/  # 数据预处理管道
│   ├── flow_extractor.py   # PCAP流量提取
│   ├── dataset.py          # PyTorch数据集
│   ├── preprocessor.py     # 预处理流程
│   └── adapters.py         # 数据集适配器（VPN/Tor/TFC）
│
├── 📂 util/                # 通用工具
│   ├── misc.py             # 分布式训练、日志等
│   └── lr_sched.py         # 学习率调度
│
└── 📄 main.py              # 主程序入口
```

详细说明请查看 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

---

## 🎯 核心功能

### 1. **SD-MKD知识蒸馏**
三阶段训练架构：
- **阶段I**: 教师预训练（ResNet50, MobileNetV3, DenseNet121）
- **阶段II**: Stacking集成训练（MLP融合层）
- **阶段III**: 学生蒸馏（ShuffleNetV2 + AgentAttention）

### 2. **复合蒸馏损失**
```
Total Loss = λ_ce·CE + λ_f·FKL + λ_r·RKL + λ_s·Sinkhorn
```
- **CE**: 交叉熵（硬标签）
- **FKL**: 前向KL散度
- **RKL**: 反向KL散度
- **Sinkhorn**: 最优传输距离（考虑类别结构）

### 3. **高效学生模型**
- ShuffleNetV2 0.5x骨干网络（轻量级）
- Agent Attention机制（O(N×M)复杂度）
- 单通道流量图像输入

---

## 🚀 快速开始

### 安装依赖
```bash
# 创建conda环境
conda create -n netkd python=3.12
conda activate netkd

# 安装依赖
pip install torch torchvision scikit-learn numpy scipy timm tqdm tensorboard matplotlib
```

### 运行训练
```bash
# 方式1: 使用虚拟数据快速演示
python main.py --mode train_student

# 方式2: 分阶段训练
python main.py --mode train_teachers    # 阶段I
python main.py --mode train_stacking    # 阶段II
python main.py --mode train_student     # 阶段III

# 评估模型
python main.py --mode eval --student_ckpt student_sd_mkd.pth
```

### 使用真实数据
```bash
# 1. 预处理数据
python -m data_preprocessing.preprocess_data \
    --dataset ISCXVPN2016 \
    --data_path /path/to/raw/pcap \
    --image_height 32 \
    --image_width 32

# 2. 修改训练脚本使用处理后的数据
# 详见 training/train.py
```

---

## 📊 实验功能

### 消融实验

测试不同损失组合的效果：
```python
from experiments import run_ablation_suite, default_ablation_configs

# 运行预定义的消融配置
configs = default_ablation_configs()  # baseline, +FKL, +FKL+RKL, full
results = run_ablation_suite(train_loader, val_loader, num_classes=5, device=device)
```

### 自定义配置
```python
from experiments import AblationConfig

config = AblationConfig(
    name="custom",
    lamb_ce=1.0,   # 交叉熵权重
    lamb_f=0.5,    # 前向KL权重
    lamb_r=0.5,    # 反向KL权重
    lamb_s=0.1,    # Sinkhorn权重
    temperature=4.0
)
```

---

## 📦 模型导入示例

```python
# 导入学生模型
from models import StudentNet
student = StudentNet(num_classes=10)

# 导入教师模型
from models import ResNet50Teacher, TeacherEnsemble
teacher = ResNet50Teacher(num_classes=10)
ensemble = TeacherEnsemble(num_classes=10)

# 导入训练函数
from training import train_teachers, train_student_stage
from training import distillation_loss

# 导入数据处理
from data_preprocessing import FlowDataset, DataPreprocessor
```

---

## 🎓 方法论

**SD-MKD** (Stacking-based Distillation with Multi-level Knowledge and Sinkhorn Distances)

### 核心创新：
1. **Stacking集成**: 使用元学习融合多个教师，提升知识质量
2. **多级知识**: 结合logits和softmax分布的知识
3. **Sinkhorn距离**: 利用最优传输理论捕捉类别间结构关系
4. **Agent Attention**: 降低学生模型的自注意力计算复杂度

### 应用场景：
- 加密流量分类（VPN、Tor等）
- 网络入侵检测
- 恶意流量识别
- 应用层协议识别

---

## 📈 性能特点

- ✅ **高准确率**: 通过教师集成和多级蒸馏提升分类性能
- ✅ **轻量化**: 学生模型参数量小，适合边缘部署
- ✅ **高效推理**: Agent Attention降低计算复杂度
- ✅ **灵活扩展**: 模块化设计，易于添加新教师或损失函数

---

## 📝 引用

如果使用本代码，请引用：

```bibtex
@article{netkd2024,
  title={Efficient Encrypted Traffic Classification with Multiple Knowledge Distillation},
  author={Your Name},
  journal={电子学报},
  year={2024}
}
```

---

## 📄 许可证

详见 [LICENSE](LICENSE) 文件

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📧 联系方式

如有问题，请联系：[your.email@example.com]
