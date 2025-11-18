# NetKD 训练指南

## 环境配置

已成功配置 conda netkd 环境，安装的关键库包括:
- PyTorch 2.9.1 + torchvision 0.24.1
- scikit-learn 1.7.2
- timm 1.0.22 (PyTorch Image Models)
- matplotlib 3.10.7 + seaborn 0.13.2
- tensorboard 2.20.0

## 数据集准备

### 数据位置
数据集位于: `G:\数据集\Dataset`

### 支持的数据集
1. **ISCXVPN2016** - 7类VPN流量分类
   - 类别: browsing, chat, email, ftp, p2p, streaming, voip
   - 样本数: 13,832 图像

2. **ISCXTor2016** - 8类Tor流量分类
   - 类别: AUDIO, BROWSING, CHAT, FILE-TRANSFER, MAIL, P2P, VIDEO, VOIP

3. **USTC-TFC2016** - 18类流量分类
   - 类别: BitTorrent, Cridex, Facetime, FTP, Geodo, Gmail, 等

4. **CrossPlatform-Android** - Android应用流量分类
5. **CrossPlatform-iOS** - iOS应用流量分类
6. **CICIoT2022** - IoT设备流量分类

### 数据格式
- 每个数据集包含 `images_sampled_new/` 目录
- 目录下按类别分文件夹
- 每个样本为 40×40 灰度 PNG 图像
- 配套 JSON 文件包含元数据

## 训练方式

### 1. 快速测试训练流程

验证完整的 SD-MKD 三阶段训练是否正常工作:

```bash
python quick_test_training.py
```

这将执行:
- 阶段 I: 训练3个教师模型 (ResNet50, MobileNetV3, DenseNet121) - 2 epochs
- 阶段 II: 训练Stacking集成模型 - 2 epochs  
- 阶段 III: 训练学生模型(知识蒸馏) - 3 epochs
- 评估学生模型性能

**预期输出**: 训练流程正常完成，准确率约30-40% (由于epoch较少)

### 2. 完整训练

使用更多epoch进行完整训练:

#### 训练学生模型 (默认模式)

```bash
python train_with_real_data.py --dataset ISCXVPN2016 \
                                --epochs_teacher 30 \
                                --epochs_stacking 20 \
                                --epochs_student 50 \
                                --batch_size 64 \
                                --lr 1e-3
```

#### 仅训练教师模型

```bash
python train_with_real_data.py --mode train_teachers \
                                --dataset ISCXVPN2016 \
                                --epochs_teacher 30
```

#### 完整三阶段训练

```bash
python train_with_real_data.py --mode full_pipeline \
                                --dataset ISCXVPN2016 \
                                --epochs_teacher 30 \
                                --epochs_stacking 20 \
                                --epochs_student 50
```

#### 评估已训练模型

```bash
python train_with_real_data.py --mode eval \
                                --dataset ISCXVPN2016 \
                                --student_ckpt checkpoints/student_sd_mkd.pth
```

### 3. 主要参数说明

#### 数据集参数
- `--dataset`: 数据集名称 (ISCXVPN2016, ISCXTor2016, USTC-TFC2016, 等)
- `--data_root`: 数据集根目录 (默认: `G:\数据集\Dataset`)
- `--batch_size`: 批次大小 (默认: 64)
- `--num_workers`: 数据加载线程数 (Windows建议0)

#### 训练参数
- `--mode`: 训练模式
  - `train_teachers`: 仅训练教师模型
  - `train_stacking`: 训练Stacking模型
  - `train_student`: 训练学生模型 (默认)
  - `full_pipeline`: 完整三阶段训练
  - `eval`: 仅评估

- `--epochs_teacher`: 教师模型训练轮数 (默认: 10)
- `--epochs_stacking`: Stacking模型训练轮数 (默认: 10)
- `--epochs_student`: 学生模型训练轮数 (默认: 20)
- `--lr`: 学习率 (默认: 1e-3)
- `--weight_decay`: 权重衰减 (默认: 1e-4)

#### 蒸馏损失权重
- `--lamb_ce`: 交叉熵损失权重 (默认: 1.0)
- `--lamb_f`: 前向KL损失权重 (默认: 0.5)
- `--lamb_r`: 反向KL损失权重 (默认: 0.5)
- `--lamb_s`: Sinkhorn损失权重 (默认: 0.1)
- `--temperature`: 蒸馏温度 (默认: 4.0)

#### 其他参数
- `--output_dir`: 检查点保存目录 (默认: `./checkpoints`)
- `--student_ckpt`: 学生模型检查点文件名 (默认: `student_sd_mkd.pth`)

## SD-MKD 训练流程详解

### 阶段 I: 教师模型预训练
- 在数据集A上独立训练3个教师模型
- 教师架构:
  - ResNet50 (深度网络)
  - MobileNetV3-Large (轻量网络)
  - DenseNet121 (密集连接)
- 损失函数: 交叉熵 (CE)
- 输出: 3个教师模型检查点

### 阶段 II: Stacking集成训练
- 冻结教师模型参数
- 训练Stacking元学习器
- 输入: 3个教师的logits
- 输出: 集成预测
- 损失函数: 交叉熵 (CE)
- 输出: Stacking模型检查点

### 阶段 III: 学生模型蒸馏
- 学生架构: ShuffleNetV2 + AgentAttention (轻量级)
- 教师: 冻结的Stacking集成模型
- 损失函数: CE + FKL + RKL + Sinkhorn
  - CE: 交叉熵 (真实标签)
  - FKL: 前向KL散度 (教师→学生)
  - RKL: 反向KL散度 (学生→教师)
  - Sinkhorn: Sinkhorn距离 (特征对齐)
- 输出: 学生模型检查点

## 训练监控

### TensorBoard
可以使用TensorBoard监控训练过程:

```bash
tensorboard --logdir=./runs
```

### 检查点管理
所有检查点保存在 `--output_dir` 指定的目录下:
- `resnet50_teacher.pth`: ResNet50教师
- `mbv3_teacher.pth`: MobileNetV3教师
- `densenet121_teacher.pth`: DenseNet121教师
- `stacking_model.pth`: Stacking集成模型
- `student_sd_mkd.pth`: 学生模型

### 结果保存
评估结果保存为JSON格式:
- `results_<dataset_name>.json`: 包含准确率、F1分数、混淆矩阵等

## 测试结果 (快速测试)

使用 ISCXVPN2016 数据集，2-3 epochs训练:

```
阶段 I - 教师模型 (2 epochs):
  ResNet50:     val_acc=0.8814
  MobileNetV3:  val_acc=0.2467
  DenseNet121:  val_acc=0.8646

阶段 II - Stacking模型 (2 epochs):
  val_acc=0.9176

阶段 III - 学生模型 (3 epochs):
  val_acc=0.3181
  test_acc=0.3104
  test_f1=0.2251
```

**注意**: 这只是快速测试结果。完整训练(30-50 epochs)应该能达到更高的准确率。

## 数据加载示例

在自己的代码中使用数据加载:

```python
from data_preprocessing import quick_load_dataset

# 加载数据集
train_loader, val_loader, test_loader, metadata = quick_load_dataset(
    dataset_name="ISCXVPN2016",
    dataset_root=r"G:\数据集\Dataset",
    batch_size=64,
    num_workers=0,  # Windows建议0
)

# 获取元数据
num_classes = metadata['num_classes']
class_names = metadata['class_names']
print(f"数据集: {metadata['dataset_name']}")
print(f"类别数: {num_classes}")
print(f"训练集: {metadata['train_size']} 样本")

# 训练循环
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # images: [batch_size, 1, 40, 40]
        # labels: [batch_size]
        # 训练代码...
```

## 常见问题

### Q1: 为什么MobileNetV3准确率很低?
A: MobileNetV3在灰度图像上表现可能不如ResNet和DenseNet。这是正常的，Stacking模型会自动调整各教师的权重。

### Q2: 可以使用GPU吗?
A: 可以！如果有CUDA GPU，代码会自动使用。检测到GPU时会自动启用 `pin_memory`。

### Q3: 如何处理类别不平衡?
A: 数据加载时使用了stratified splitting保持类别比例。可以在训练中添加类别权重或使用focal loss。

### Q4: 可以训练其他数据集吗?
A: 可以！只需修改 `--dataset` 参数为其他数据集名称即可。

### Q5: Windows上 num_workers > 0 报错?
A: Windows多进程有限制，建议设置 `--num_workers 0`。

## 下一步计划

1. ✅ 环境配置
2. ✅ 数据加载实现
3. ✅ 训练流程验证
4. **待完成**: 完整训练 (30-50 epochs)
5. **待完成**: 超参数调优
6. **待完成**: 消融实验 (测试不同损失函数组合)
7. **待完成**: 多数据集评估

## 参考文献

- SD-MKD: Stacked Distillation with Multi-stage Knowledge Distillation
- ShuffleNetV2: Practical Guidelines for Efficient CNN Architecture Design
- AgentAttention: On the Integration of Softmax and Linear Attention
