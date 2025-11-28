# 数据增强快速指南 - 提升Class 5精度

## 问题
- **Class 5 (streaming)**: 当前测试精度 **92.86%** (52/56)
- **原因**: 训练样本仅640个,远低于平均值1897个 (仅33.73%)

## 解决方案

### 方案1: 仅增强Class 5 (推荐,快速) ⚡

```bash
cd /walnut_data/yqm/NetKD

# 增强Class 5: 640 → 2560 样本 (4倍)
python scripts/balance_dataset.py --classes streaming --multiplier 3
```

**预期效果**: Class 5精度 92.86% → **95-97%** ✓

---

### 方案2: 全面平衡 (更优,耗时较长)

```bash
# 同时增强所有样本不足的类别
python scripts/balance_dataset.py --classes "email,p2p,streaming" --multiplier 4
```

**预期效果**: 
- Class 5: 92.86% → **96-98%**
- 整体: 98.41% → **98.7-99.0%** ✓

---

## 重新训练 Stacking 模型

增强数据后,需要重新训练:

```bash
cd /walnut_data/yqm/NetKD

python -c "
import sys
sys.path.insert(0, '/walnut_data/yqm/NetKD')

from training.stacking import train_stacking_model
import torch

device = torch.device('cuda:0')
model, metrics = train_stacking_model(
    teacher_checkpoints=[
        'checkpoints/resnet50_eca_bs256_teacher.pth',
        'checkpoints/mbv3_eca_bs256_teacher.pth', 
        'checkpoints/densenet121_bs256_teacher.pth'
    ],
    data_path='/walnut_data/yqm/Dataset/ISCXVPN2016',
    num_classes=7,
    batch_size=128,
    epochs=30,
    device=device,
    save_path='checkpoints/stacking/stacking_augmented.pth'
)

print(f'\n=== 增强效果对比 ===')
print(f'增强前测试精度: 98.41%')
print(f'增强后测试精度: {metrics[\"test_accuracy\"]*100:.2f}%')
"
```

---

## 技术细节

### 增强策略
- ✅ 小角度旋转 (±5°): 模拟时间偏移
- ✅ 亮度调整 (0.9-1.1x): 模拟流量强度
- ✅ 对比度调整: 模拟网络条件
- ✅ 高斯噪声 (2%): 模拟网络抖动
- ✅ 轻微模糊: 模拟采样误差

### 为什么有效?
1. **缓解类别不平衡**: 将Class 5样本数提升到与其他类相当
2. **增加样本多样性**: 模拟不同网络条件下的流量变化
3. **保留核心特征**: 增强策略专门设计,不破坏流量模式

---

## 查看完整文档

```bash
cat /walnut_data/yqm/NetKD/docs/DATA_AUGMENTATION_GUIDE.md
```

包含:
- 详细方案对比
- 参数调优建议  
- Mixup/采样平衡等高级策略
- 常见问题解答
