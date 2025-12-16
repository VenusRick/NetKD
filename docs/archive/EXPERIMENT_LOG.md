# NetKD 实验记录

## 最后更新: 2025-12-09 23:55

## 数据集训练状态

| 数据集 | Teachers | Stacking | Student | 最终准确率 | 状态 |
|--------|----------|----------|---------|------------|------|
| ISCXTor2016 | ✅ 30 epochs | ✅ 20 epochs | ✅ 50 epochs | 82.14% | 完成 |
| CICIoT2022 | 🔄 进行中 | - | - | - | 训练中 |
| USTC-TFC2016 | 🔄 进行中 | - | - | - | 训练中 |
| ISCXVPN2016 | ✅ 30 epochs | ❌ 失败 | - | - | 需重新运行 |

## 各数据集详情

### 1. ISCXTor2016 (完成)
- Teachers: ResNet50(99.59%), MobileNetV3(98.28%), DenseNet121(99.93%)
- Stacking: 99.86%
- **Student: 82.14%** (待优化)
- 训练时间: ~15分钟

### 2. CICIoT2022 (进行中)
- 当前阶段: teacher_mbv3 E14/30
- 当前准确率: 97.95%
- GPU: 0

### 3. USTC-TFC2016 (进行中)
- 当前阶段: teacher_mbv3 E14/30
- 当前准确率: 98.51%
- GPU: 1

### 4. ISCXVPN2016 (需重新运行)
- 问题: checkpoint冲突导致stacking失败
- 解决方案: 等待其他任务完成后串行运行

## 已知问题
1. **Checkpoint冲突**: 并行训练时模型保存到同一目录会互相覆盖
   - 临时解决: 串行运行不同数据集
   - 长期解决: 修改train.py保存逻辑

## 模型文件位置
- ISCXTor2016: results/full_ISCXTor/
- CICIoT2022: results/full_CICIoT2022/ (训练中)
- USTC: results/full_USTC/ (训练中)
- ISCXVPN: results/full_ISCXVPN/ (需重新训练)
