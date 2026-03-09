# NetKD 实验执行计划

## 🎯 目标
收集学生模型在所有数据集上的测试结果 (100%, 75%, 50%, 25% 数据比例)

## 📊 当前状态 (2025-12-10 22:05)

### 已完成结果
| 数据集 | 比例 | 教师(DenseNet121) | Stacking | 学生(RepViT-M0.9) |
|--------|------|-------------------|----------|-------------------|
| ISCXVPN2016 | 100% | 98.19% / F1=97.13% | 98.27% | **97.83% / F1=97.02%** |
| CICIoT2022 | 100% | 99.18% / F1=98.63% | 99.54% | **99.64% / F1=99.56%** |

### 待完成任务
- [ ] ISCXVPN2016: 75%, 50%, 25% (各3轮)
- [ ] ISCXTor2016: 100%, 75%, 50%, 25% (各3轮)
- [ ] USTC-TFC2016: 100%, 75%, 50%, 25% (各3轮)
- [ ] CICIoT2022: 75%, 50%, 25% (各3轮)
- [ ] CrossPlatform-Android: 100%, 75%, 50%, 25% (各3轮)
- [ ] CrossPlatform-iOS: 100%, 75%, 50%, 25% (各3轮)

## 🔧 最佳配置

### 教师组合 (Stacking Ensemble)
- DenseNet121 (最佳单教师)
- MobileNetV3-Large
- ConvNeXtV2-Tiny

### 学生模型
- **RepViT-M0.9** (最佳, 4.72M参数)
  - 知识蒸馏配置: α=0.5, T=3.0

### 训练配置
| 阶段 | Epochs | LR | Early Stop |
|------|--------|-----|------------|
| 教师 | 25 | 0.001 | 5 |
| Stacking | 12 | 0.0005 | - |
| 学生KD | 25 | 0.0003 | - |

## 📂 数据集信息
| 数据集 | 类别数 | 训练样本 | 图像尺寸 |
|--------|--------|----------|----------|
| ISCXVPN2016 | 7 | 13,281 | 40x40 灰度 |
| ISCXTor2016 | 8 | ~10,000 | 40x40 灰度 |
| USTC-TFC2016 | 10 | ~15,000 | 40x40 灰度 |
| CICIoT2022 | 6 | 18,734 | 40x40 灰度 |
| CrossPlatform-Android | 124 | ~40,000 | 40x40 灰度 |
| CrossPlatform-iOS | 124 | ~48,000 | 40x40 灰度 |

## 🚀 运行命令
```bash
# 完整实验 (已启动)
nohup python scripts/run_complete_experiment.py \
  --datasets ISCXVPN2016 ISCXTor2016 USTC-TFC2016 CICIoT2022 CrossPlatform-Android CrossPlatform-iOS \
  --ratios 1.0 0.75 0.5 0.25 \
  --runs 3 \
  --gpu 0 \
  > logs/complete_exp_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 监控进度
tail -f logs/complete_exp_*.log | grep -E "Dataset|Ratio|Run|test_acc"
```

## 📈 预计时间
- 每个数据集×比例×运行: ~10-15分钟
- 6数据集 × 4比例 × 3运行 = 72次实验
- 总计: ~12-18小时

---
更新时间: $(date '+%Y-%m-%d %H:%M:%S')
