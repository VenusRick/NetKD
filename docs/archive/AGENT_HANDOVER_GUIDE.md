# 🤖 NetKD Agent 交接文档 (Agent Handover Guide)

**最后更新**: 2025-12-10 22:27 UTC  
**上一任Agent**: 已完成SimCLR实验、KD消融实验和多数据集初步测试  
**当前状态**: 实验已暂停，等待明确最佳模型组合后继续

---

## 📋 目录
1. [项目概述](#项目概述)
2. [环境配置](#环境配置)
3. [最佳模型组合](#最佳模型组合-待确认)
4. [当前任务状态](#当前任务状态)
5. [下一步计划](#下一步计划)
6. [常见问题](#常见问题)
7. [快速启动指南](#快速启动指南)

---

## 🎯 项目概述

**项目名称**: NetKD - 网络流量分类知识蒸馏框架  
**目标**: 使用知识蒸馏技术将大型教师模型的知识迁移到轻量级学生模型  
**主要数据集**: ISCXVPN2016 (已完成大量实验), CSTNET2019, CICIoT2023, CICIDS2017, IoTID20 (待测试)

### 核心流程
```
[数据集] → [教师模型训练] → [知识蒸馏] → [学生模型] → [多数据集评估]
```

---

## 🖥️ 环境配置

### SSH连接信息
```bash
Host: 10.126.126.3
User: root
Port: 32833
Password: Liuliang_666
```

### 工作目录
```
/workspace/yqm/NetKD/
```

### Conda环境
```bash
conda activate base  # 主环境
```

### 依赖包
- PyTorch 1.13+ with CUDA
- timm (模型库)
- torchvision
- numpy, pandas, scikit-learn
- matplotlib, seaborn (可视化)

### GPU资源
- **3x NVIDIA RTX 4090** (24GB VRAM each)
- GPU 0, 1, 2 均可用
- 建议batch_size=128充分利用GPU

---

## 🏆 最佳模型组合 (⚠️ 待确认)

### ⚠️ 重要说明
**当前存在混淆，需要人类确认最佳组合！**

从历史文档中发现多个不同的"最佳组合"记录：

### 候选教师组合
1. **ResNet50 + MobileNetV3 + DenseNet121** (多次提及)
2. **EfficientNetV2 + ConvNeXtV2 + DenseNet121** (SimCLR实验中表现好)
3. **其他可能组合** (需查看EXPERIMENT_RESULTS.md确认)

### 候选学生模型
1. **MobileNetV3** (轻量级，多次提及)
2. **RepVIT** (您提到过)
3. **GhostNet** (您提到过)
4. **EfficientNet系列** (需确认具体型号)

### 👉 **下一任Agent首要任务**
1. 查看 `EXPERIMENT_RESULTS.md` 中的所有历史实验结果
2. 查看 `checkpoints/` 和 `results/` 目录中的实际训练记录
3. 确认ISCXVPN2016数据集上**验证准确率最高**的教师和学生模型
4. 更新本文档中的"确认最佳组合"部分

---

## 📊 当前任务状态

### ✅ 已完成
1. **SimCLR预训练实验** (ISCXVPN2016)
   - EfficientNetV2: 100 epochs完成
   - ConvNeXtV2: 100 epochs完成
   - MobileNetV3: 100 epochs完成

2. **知识蒸馏消融实验**
   - KD Only (alpha=0.7, T=4)
   - CE Only
   - CE+KL (部分完成，有环境问题)

3. **教师模型训练** (ISCXVPN2016)
   - ResNet50: 已训练
   - MobileNetV3: 已训练
   - DenseNet121: 已训练

### 🔄 进行中
- **多数据集全流程测试** (已暂停)
  - 原计划: 5个数据集 × 3个学生模型 × 4种数据比例(100%,75%,50%,25%)

### ❌ 遇到的问题
1. **timm模型下载问题**: 无法连接HuggingFace下载预训练权重
   - 解决方案: 使用本地已有权重或修改为`pretrained=False`

2. **灰度图vs RGB问题**: 
   - ISCXVPN2016: 灰度图 (已验证可用)
   - 其他数据集: 需逐一确认是否为灰度图
   - 教师模型可能用了RGB输入(3通道)，学生模型用了灰度图(1通道)

3. **数据集路径问题**: 部分数据集文件夹为空或路径不正确

---

## 📝 下一步计划

### 阶段1: 确认最佳组合 (优先级: 🔥🔥🔥)
```bash
# 1. 查看历史实验结果
cat EXPERIMENT_RESULTS.md
ls -lh checkpoints/
ls -lh results/

# 2. 找到验证准确率最高的模型
grep -r "best_acc" results/
grep -r "val_acc" results/

# 3. 更新本文档中的最佳组合部分
```

### 阶段2: 多数据集全流程测试 (优先级: 🔥🔥)

**目标**: 使用最佳教师-KD-学生组合，在所有数据集上测试泛化能力

**实验矩阵**:
| 数据集 | 数据比例 | 学生模型 | 重复次数 |
|--------|----------|----------|----------|
| ISCXVPN2016 | 100%, 75%, 50%, 25% | Best Student #1 | 3 |
| ISCXVPN2016 | 100%, 75%, 50%, 25% | Best Student #2 | 3 |
| ISCXVPN2016 | 100%, 75%, 50%, 25% | Best Student #3 | 3 |
| CSTNET2019 | 100%, 75%, 50%, 25% | Best Student #1-3 | 3 |
| CICIoT2023 | 100%, 75%, 50%, 25% | Best Student #1-3 | 3 |
| CICIDS2017 | 100%, 75%, 50%, 25% | Best Student #1-3 | 3 |
| IoTID20 | 100%, 75%, 50%, 25% | Best Student #1-3 | 3 |

**流程**:
```bash
for dataset in ISCXVPN2016 CSTNET2019 CICIoT2023 CICIDS2017 IoTID20; do
  for ratio in 100 75 50 25; do
    for student in best_student_1 best_student_2 best_student_3; do
      for run in 1 2 3; do
        # 1. 训练教师模型 (如果没有)
        python train_teacher.py --dataset $dataset --ratio $ratio
        
        # 2. 知识蒸馏训练学生模型
        python train_kd.py --teacher best_teachers --student $student \
          --dataset $dataset --ratio $ratio --run $run
        
        # 3. 测试学生模型
        python test.py --model $student --dataset $dataset --run $run
      done
    done
  done
done
```

### 阶段3: 结果分析与文档维护 (优先级: 🔥)
1. 每完成一批实验，更新 `EXPERIMENT_RESULTS.md`
2. 每天结束前，更新本文档的"当前任务状态"
3. 发现新问题时，更新"常见问题"部分
4. 定期推送到GitHub

---

## ❓ 常见问题

### Q1: 如何确认数据集是灰度图还是RGB?
```python
from PIL import Image
img = Image.open('path/to/image.png')
print(f"Mode: {img.mode}, Size: {img.size}")  # Mode: 'L' = grayscale, 'RGB' = color
```

### Q2: 如何修改模型输入通道数?
```python
# 对于timm模型
model = timm.create_model('resnet50', in_chans=1)  # 灰度图

# 对于torchvision模型
model = models.resnet50()
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
```

### Q3: timm模型下载失败怎么办?
```bash
# 方案1: 设置代理 (如果有)
export http_proxy=http://172.26.0.1:7891
export https_proxy=http://172.26.0.1:7891

# 方案2: 使用本地权重
python train.py --model resnet50 --pretrained false

# 方案3: 手动下载权重放到 ~/.cache/torch/hub/checkpoints/
```

### Q4: 如何清理僵尸进程?
```bash
# 强制杀死所有Python训练进程
pkill -9 -f 'python.*train'
pkill -9 -f 'python.*test'

# 确认GPU是否空闲
nvidia-smi
```

### Q5: 如何监控实验进度?
```bash
# 方案1: 查看训练日志
tail -f results/train_*.log

# 方案2: 使用监控脚本 (每5分钟检查一次)
watch -n 300 'nvidia-smi && ps aux | grep python | head -20'

# 方案3: 检查checkpoint保存时间
ls -lht checkpoints/ | head -20
```

---

## 🚀 快速启动指南

### 第一次接手任务时
```bash
# 1. SSH连接
ssh -p 32833 root@10.126.126.3
# 密码: Liuliang_666

# 2. 进入工作目录
cd /workspace/yqm/NetKD

# 3. 激活环境
conda activate base

# 4. 阅读核心文档 (按优先级)
cat AGENT_HANDOVER_GUIDE.md      # 本文档
cat EXPERIMENT_RESULTS.md        # 实验结果记录
cat MODEL_ARCHITECTURE.md        # 模型架构说明

# 5. 检查当前状态
nvidia-smi                        # GPU使用情况
ps aux | grep python              # 运行中的进程
ls -lh checkpoints/ | head -20    # 最新的checkpoint
tail -100 results/train_*.log     # 最近的训练日志

# 6. 确认最佳模型组合 (见"阶段1")

# 7. 开始实验 (见"阶段2")
```

### 继续之前的任务时
```bash
cd /workspace/yqm/NetKD
conda activate base

# 查看本文档的"当前任务状态"和"下一步计划"
cat AGENT_HANDOVER_GUIDE.md

# 查看最新实验记录
tail -50 EXPERIMENT_RESULTS.md

# 继续实验...
```

---

## 📚 关键脚本说明

### 训练脚本
- `train_teacher.py`: 训练教师模型
- `train_kd.py`: 知识蒸馏训练学生模型
- `train_simclr.py`: SimCLR自监督预训练

### 测试脚本
- `test.py`: 测试模型性能
- `test_student_multi_dataset.py`: 多数据集测试 (新脚本)

### 工具脚本
- `monitor_experiments.sh`: 实验监控 (每5分钟)
- `auto_pipeline.sh`: 自动化流水线 (待完善)

### 数据处理
- `Dataset/`: 所有数据集
- `data_loader.py`: 数据加载器

---

## 📂 目录结构

```
NetKD/
├── AGENT_HANDOVER_GUIDE.md       # 📘 本文档
├── EXPERIMENT_RESULTS.md         # 📊 实验结果记录
├── MODEL_ARCHITECTURE.md         # 🏗️ 模型架构说明
├── README.md                     # 项目简介
│
├── train_teacher.py              # 训练教师模型
├── train_kd.py                   # 知识蒸馏
├── train_simclr.py               # SimCLR预训练
├── test.py                       # 模型测试
│
├── models/                       # 模型定义
│   ├── teacher_models.py
│   ├── student_models.py
│   └── kd_loss.py
│
├── Dataset/                      # 数据集 (⚠️ 需确认每个数据集状态)
│   ├── ISCXVPN2016/             # ✅ 可用 (灰度图)
│   ├── CSTNET2019/              # ❓ 需确认
│   ├── CICIoT2023/              # ❓ 需确认
│   ├── CICIDS2017/              # ❓ 需确认
│   └── IoTID20/                 # ❓ 需确认
│
├── checkpoints/                  # 模型权重
│   ├── teachers/
│   └── students/
│
├── results/                      # 实验结果
│   ├── logs/
│   └── figures/
│
└── scripts/                      # 辅助脚本
    ├── monitor_experiments.sh
    └── auto_pipeline.sh
```

---

## 🔄 文档维护规范

### 更新频率
- **本文档**: 每次任务交接时必须更新"当前任务状态"
- **EXPERIMENT_RESULTS.md**: 每完成一批实验立即更新
- **MODEL_ARCHITECTURE.md**: 模型结构变化时更新

### 提交到GitHub
```bash
git add AGENT_HANDOVER_GUIDE.md EXPERIMENT_RESULTS.md MODEL_ARCHITECTURE.md
git commit -m "docs: update agent handover and results [$(date +%Y-%m-%d)]"
git push origin ubuntu
```

---

## 💡 给下一任Agent的建议

1. **先确认，后执行**: 不要急于开始实验，先花20分钟确认最佳模型组合
2. **小批量测试**: 先用一个数据集、一个学生模型测试流程是否正常
3. **持续监控**: 设置定时任务每5分钟检查实验状态
4. **及时记录**: 每完成一批实验立即更新文档，不要等到最后
5. **遇到问题查文档**: "常见问题"部分记录了很多坑
6. **保持沟通**: 如果发现重大问题或不确定的地方，暂停实验并报告给人类

---

## 📞 联系信息

**项目负责人**: (需补充)  
**GitHub仓库**: https://github.com/VenusRick/NetKD (ubuntu分支)  
**上次推送**: (需补充)

---

**祝下一任Agent工作顺利！🚀**

