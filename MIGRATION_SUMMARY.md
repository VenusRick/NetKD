# NetKD Linux 迁移总结

## 完成日期
2025年11月19日

## 迁移概述
成功将NetKD项目从Windows系统迁移到Linux系统,解决了所有环境和兼容性问题。

## 主要修改

### 1. 环境配置
- **Conda环境**: netkd (Python 3.10.19)
- **主要依赖**:
  - PyTorch 2.2.2 + CUDA 12.1
  - torchvision 0.17.2
  - timm 0.9.16
  - scikit-learn 1.4.2
  - scipy 1.11.4
  - matplotlib 3.8.4
  - seaborn 0.13.2
  - tensorboard 2.16.2
  - dpkt 1.9.8
  - scapy 2.5.0

### 2. 路径修复
- 将所有硬编码的Windows路径 `G:\数据集\Dataset` 替换为Linux路径 `/walnut_data/yqm/Dataset`
- 修改的文件:
  - `data_preprocessing/image_loader.py`
  - `quick_test_training.py`
  - `train_with_real_data.py`
  - `test_data_loading.py`

### 3. 数值稳定性改进
- **问题**: 学生模型训练时出现NaN损失
- **解决方案**:
  1. 改进损失函数的数值稳定性 (`training/loss_functions.py`):
     - 在KL散度计算中添加epsilon裁剪
     - 在Sinkhorn算法中添加数值范围裁剪
     - 添加NaN检测和安全返回机制
  2. 添加梯度裁剪 (`experiments/sd_mkd.py`):
     - 最大梯度范数: 1.0
     - 防止梯度爆炸

### 4. 多进程配置
- Windows下 `num_workers=0` (单进程)
- Linux下可以使用 `num_workers=4` (多进程加速)

## 测试结果

### 快速测试 (ISCXVPN2016数据集)
```bash
conda run -n netkd python quick_test_training.py \
    --dataset_name ISCXVPN2016 \
    --num_workers 4 \
    --batch_size 64 \
    --epochs 2
```

**结果**:
- 教师模型训练正常
- Stacking集成模型训练正常
- 学生模型训练正常,准确率 55.23%

### 完整训练测试
```bash
conda run -n netkd python train_with_real_data.py \
    --dataset ISCXVPN2016 \
    --num_workers 4 \
    --batch_size 128 \
    --epochs_teacher 1 \
    --epochs_stacking 1 \
    --epochs_student 2 \
    --mode full_pipeline
```

**结果**:
- 测试准确率: 78.46%
- F1分数: 78.06%
- 所有阶段运行正常

## 使用方法

### 1. 激活环境
```bash
conda activate netkd
```

### 2. 数据加载测试
```bash
python test_data_loading.py ISCXVPN2016
```

### 3. 快速训练测试
```bash
python quick_test_training.py \
    --dataset_name ISCXVPN2016 \
    --num_workers 4 \
    --batch_size 64
```

### 4. 完整训练
```bash
python train_with_real_data.py \
    --dataset ISCXVPN2016 \
    --num_workers 4 \
    --batch_size 128 \
    --epochs_teacher 10 \
    --epochs_stacking 10 \
    --epochs_student 20 \
    --mode full_pipeline
```

## 可用数据集
- ISCXVPN2016 ✅ (已测试)
- ISCXTor2016
- USTC-TFC2016
- CrossPlatform-Android
- CrossPlatform-iOS
- CICIoT2022

## 已知问题和解决方案

### 问题1: 学生模型NaN损失
**解决**: 
- 改进损失函数数值稳定性
- 添加梯度裁剪
- 添加epsilon裁剪和NaN检测

### 问题2: Windows路径不兼容
**解决**:
- 使用自动路径检测
- 将默认路径改为Linux格式

### 问题3: 多进程数据加载
**解决**:
- Linux下可以使用 `num_workers > 0`
- 建议值: 4-8 (根据CPU核心数)

## 性能建议

1. **批次大小**: 根据GPU内存调整
   - 16GB GPU: batch_size=128-256
   - 8GB GPU: batch_size=64-128

2. **数据加载线程**: 
   - Linux: `num_workers=4-8`
   - 应小于CPU核心数

3. **训练轮数**:
   - 教师模型: 10-20 epochs
   - Stacking模型: 10-15 epochs
   - 学生模型: 20-30 epochs

## 文件备份
- `training/loss_functions_backup.py` - 原始损失函数
- `training/loss_functions.py` - 改进后的损失函数

## 下一步
1. 在其他数据集上测试 (ISCXTor2016, USTC-TFC2016等)
2. 调优超参数以获得更好性能
3. 尝试不同的蒸馏损失权重组合
4. 评估模型在不同批次大小下的性能

## 联系方式
如有问题,请参考:
- README.md - 项目说明
- TRAINING_GUIDE.md - 训练指南
- PROJECT_STRUCTURE.md - 项目结构

---
迁移完成! 🎉
