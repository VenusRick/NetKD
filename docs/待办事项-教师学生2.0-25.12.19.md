# TrafficKD Teacher 2.0 & Student 2.0 实施计划

> 创建时间: 2025-12-07 17:50
> 执行者: GitHub Copilot CLI
> 项目路径: /workspace/yqm/NetKD

---

## 📋 任务总览

基于 `12.7-下一步模型改进计划.md` 文档，实施 Teacher 2.0 和 Student 2.0 实验框架。

---

## 🎯 Teacher 2.0 任务

### 已完成 ✅
- [x] ResNet50-ECA 教师模型 (98.48% → 99.06% with ECA)
- [x] DenseNet121-ECA 教师模型 (99.06%)
- [x] MobileNetV3-Large-ECA 教师模型 (99.06%)
- [x] ConvNeXtV2-Tiny-ECA 教师模型 (98.70%)
- [x] EfficientNetV2-S-ECA 教师模型 (98.99%)
- [x] Stacking 集成框架
- [x] 教师注册表 (teacher_registry.py)
- [x] 教师配置文件 (configs/teachers.yaml)

### 待实施 📝
- [ ] 添加 GhostNetV3 教师模型
- [ ] 添加 RepViT 教师模型
- [ ] 实现教师多样性计算 (compute_disagreement.py)
- [ ] 实现 Leave-One-Out 贡献分析 (leave_one_out_stacking.py)
- [ ] 新增教师组合实验:
  - [ ] dense_convnexts_ghostv3: DenseNet121 + ConvNeXtV2 + GhostNetV3
  - [ ] convnexts_repvit_ghostv3: ConvNeXtV2 + RepViT + GhostNetV3

---

## 🎯 Student 2.0 任务

### 已完成 ✅
- [x] MobileNetV2 学生模型 (97.98%)
- [x] MobileNetV3-Small 学生模型 (97.98%)
- [x] GhostNet 学生模型 (97.25%)
- [x] 基础蒸馏框架 (CE + KL)

### 待实施 📝
- [ ] 创建 configs/students.yaml 配置文件
- [ ] 创建 models/student_registry.py 学生注册表
- [ ] 添加新学生模型:
  - [ ] GhostNetV3 0.75x
  - [ ] RepViT-M0.9 (Tiny)
- [ ] 实现完整 TrafficKD 损失函数 (CE + FKL + RKL + Sinkhorn)
- [ ] 创建 experiments/student_kd/ 目录结构:
  - [ ] train_student_baseline.py (CE only)
  - [ ] train_student_kd.py (完整 TrafficKD)
  - [ ] train_student_kd_subsampled.py (数据效率实验)

---

## 🎯 分析与汇总任务

### 待实施 📝
- [ ] analysis/summarize_teachers.py - 教师模型汇总
- [ ] analysis/summarize_teacher_sets.py - 教师组合汇总
- [ ] analysis/summarize_students.py - 学生模型汇总
- [ ] analysis/pareto.py - Pareto 前沿计算
- [ ] analysis/summarize_all_experiments.py - 总汇总脚本
- [ ] 自动生成 Markdown 表格和 CSV 文件

---

## 📊 实验配置

### 教师模型候选 (Teacher 2.0)
| 模型 | 骨干网络 | 注意力 | 预期参数量 |
|------|---------|--------|-----------|
| convnextv2_tiny | ConvNeXtV2-Tiny | ECA | ~28M |
| convnextv2_small | ConvNeXtV2-Small | ECA | ~50M |
| efficientnetv2_s | EfficientNetV2-S | ECA | ~21M |
| ghostnetv3_1_1x | GhostNetV3 | - | ~6M |
| repvit_m1_0 | RepViT-M1.0 | - | ~7M |

### 学生模型候选 (Student 2.0)
| 模型 | 骨干网络 | 预期参数量 |
|------|---------|-----------|
| mobilenetv2 | MobileNetV2 | ~2.2M |
| mobilenetv3_small | MobileNetV3-Small | ~1.5M |
| ghostnet_v1_1_0x | GhostNet-V1 | ~3.9M |
| ghostnetv3_0_75x | GhostNetV3 0.75x | ~3M |
| repvit_tiny_m0_9 | RepViT-M0.9 | ~5M |

### KD 配置
| 配置ID | CE | FKL | RKL | Sinkhorn | 温度 |
|--------|----|----|-----|----------|------|
| ce_only | 1.0 | 0 | 0 | 0 | 1.0 |
| ce_kl | 1.0 | 1.0 | 0 | 0 | 3.0 |
| traffic_kd_full | 1.0 | 1.0 | 1.0 | 0.1 | 3.0 |

---

## 🔄 执行顺序

1. **Phase 1: 配置与注册表** (当前)
   - 创建 configs/students.yaml
   - 创建 models/student_registry.py
   - 扩展 teacher_models.py (添加 GhostNetV3, RepViT)

2. **Phase 2: 实验脚本**
   - 创建 experiments/student_kd/ 目录
   - 实现训练脚本

3. **Phase 3: 分析工具**
   - 完善 analysis/ 汇总脚本
   - 实现 Pareto 分析

4. **Phase 4: 运行实验**
   - 训练新教师模型
   - 训练学生模型 (各种 KD 配置)
   - 数据效率实验

5. **Phase 5: 结果汇总**
   - 生成所有 Markdown 表格
   - Pareto 前沿分析

---

## 📝 执行日志

### 2025-12-07 17:50 - 初始化
- 连接到远程服务器 10.126.126.3
- 确认项目路径: /workspace/yqm/NetKD
- 创建 TODO 文档


### 2025-12-07 17:56 - Phase 1 完成
- ✅ 创建 configs/students.yaml - 学生模型和KD配置
- ✅ 创建 models/student_registry.py - 学生模型注册表
- ✅ 扩展 models/student_models_v2.py - 添加 GhostNetV3Student 和 RepViTStudent
- ✅ 创建 experiments/student_kd/__init__.py
- ✅ 创建 experiments/student_kd/train_student_kd.py - 完整TrafficKD训练脚本
- ✅ 创建 analysis/pareto.py - Pareto前沿分析工具
- ✅ 创建 analysis/summarize_all_experiments.py - 实验汇总脚本

**下一步**: Phase 2 - 测试脚本可用性，运行示例实验


### 2025-12-07 18:17 - Phase 2 实验启动
- ✅ 修复数据路径问题 (/walnut_data → /workspace)
- ✅ 在3块RTX 4090上启动并行实验:
  - GPU 0: MobileNetV2 蒸馏训练 (100 epochs)
  - GPU 1: MobileNetV3-Small 蒸馏训练 (100 epochs)
  - GPU 2: GhostNet 蒸馏训练 (100 epochs)
- 配置: teacher_set=replace_resnet (ConvNeXtV2+DenseNet121+MobileNetV3)
- KD: temperature=3.0, alpha_ce=0.5, alpha_kl=0.5

**实验正在运行中...**

### 2025-12-07 18:28 - Phase 2 实验完成 ✅

#### 🎉 实验结果汇总

| 学生模型 | 参数量 | 验证精度 | 测试精度 | Macro-F1 | 训练时间 |
|---------|--------|---------|---------|---------|---------|
| **MobileNetV2** | 2.23M | 98.48% | 97.83% | 97.27% | 9.4min |
| **MobileNetV3-Small** | 1.52M | 98.34% | 97.83% | 96.79% | 8.7min |
| **GhostNet** | 3.91M | 98.63% | 97.83% | 96.93% | 12.7min |

#### 实验配置
- **教师组合**: replace_resnet (ConvNeXtV2-Tiny + DenseNet121 + MobileNetV3-Large)
- **蒸馏温度**: T = 3.0
- **损失权重**: α_ce = 0.5, α_kl = 0.5
- **训练**: 100 epochs, AdamW, cosine schedule with warmup

#### 检查点位置
- `results/student_kd_20251207/mobilenetv2/student_mobilenetv2_best.pth`
- `results/student_kd_20251207/mobilenetv3_small/student_mobilenetv3_small_best.pth`
- `results/student_kd_20251207/ghostnet/student_ghostnet_best.pth`

**下一步**: Phase 3 - 消融实验 (KD配置对比)


---

### 2025-12-07 20:32 - Phase 4 扩展学生模型实验完成 ✅

#### 实验结果

| 学生模型 | 参数量 | 验证精度 | 测试精度 | Macro-F1 | 训练时间 |
|---------|--------|---------|---------|---------|---------|
| **EfficientNet-Lite** | 3.38M | **98.63%** | **97.90%** | **96.86%** | 9.1min |
| **GhostNet** (重复) | 3.91M | 98.34% | 97.47% | 96.22% | 11.3min |
| **ShuffleNet** ⚠️ | 9.81M | 73.17% | 73.19% | 68.62% | 9.4min |

#### 发现与分析
- **EfficientNet-Lite** 表现出色，是最佳学生模型之一
- **ShuffleNet** 训练出现 NaN loss，最终只有~73%准确率（训练不稳定，需要调整学习率）
- **GhostNet** 结果与Phase 2一致，验证了实验可重复性

#### 检查点
- `results/student_extended_20251207/efficientnet_lite/`
- `results/student_extended_20251207/ghostnet_v2/`
- `results/student_extended_20251207/shufflenet/`


---

## 📊 实验总结报告 (2025-12-07)

### 完成的实验

#### Phase 2: 核心学生模型蒸馏实验 ✅
| 学生模型 | 参数量 | 验证精度 | 测试精度 | Macro-F1 | 训练时间 |
|---------|--------|---------|---------|---------|---------|
| MobileNetV2 | 2.23M | 98.48% | 97.83% | 97.27% | 9.4min |
| MobileNetV3-Small | 1.52M | 98.34% | 97.83% | 96.79% | 8.7min |
| GhostNet | 3.91M | 98.63% | 97.83% | 96.93% | 12.7min |

#### Phase 4: 扩展学生模型实验 ✅
| 学生模型 | 参数量 | 验证精度 | 测试精度 | Macro-F1 | 训练时间 |
|---------|--------|---------|---------|---------|---------|
| EfficientNet-Lite | 3.38M | **98.63%** | **97.90%** | **96.86%** | 9.1min |
| GhostNet (重复) | 3.91M | 98.34% | 97.47% | 96.22% | 11.3min |
| ShuffleNet ⚠️ | 9.81M | 73.17% | 73.19% | 68.62% | 9.4min |

### 🏆 最佳学生模型排名

1. **EfficientNet-Lite** - 测试精度 97.90%, F1 96.86%, 参数 3.38M
2. **GhostNet** - 测试精度 97.83%, F1 96.93%, 参数 3.91M  
3. **MobileNetV2** - 测试精度 97.83%, F1 97.27%, 参数 2.23M
4. **MobileNetV3-Small** - 测试精度 97.83%, F1 96.79%, 参数 **1.52M** (最轻量)

### 实验配置
- **教师模型组合**: ConvNeXtV2-Tiny + DenseNet121 + MobileNetV3-Large (带ECA注意力)
- **Stacking**: 动态权重堆叠元学习器
- **蒸馏温度**: T = 3.0
- **损失函数**: CE + KL Divergence (α_ce=0.5, α_kl=0.5)
- **训练**: 100 epochs, AdamW, cosine schedule with warmup
- **数据集**: ISCXVPN2016

### 待完成任务
- [ ] Phase 3: KD配置消融实验 (需要修改代码支持distill_mode)
- [ ] Phase 5: 新教师模型组合实验 (GhostNetV3/RepViT作为教师)
- [ ] ShuffleNet训练不稳定问题修复

---

---

## 📊 Phase 5: 新轻量学生模型实验完成 (2025-12-07 21:39)

### 🎯 新增轻量学生模型结果

| 学生模型 | 参数量 | 验证精度 | 测试精度 | Macro-F1 | 训练时间 |
|---------|--------|---------|---------|---------|---------|
| **RepViT-M0.9** | 4.72M | **98.70%** | **98.12%** | **97.25%** | 13.2min |
| **MobileNetV3-Small-050** | **0.58M** | 97.69% | 97.47% | 96.48% | 8.6min |
| **GhostNetV3-050** | 2.12M | 97.83% | 96.60% | 95.13% | 23.4min |
| **GhostNet-050** | 1.32M | 97.76% | 96.82% | 95.88% | 12.1min |
| **MobileNetV2-050** | **0.70M** | 97.69% | 97.11% | 95.88% | 9.4min |

### 🏆 总体最佳学生模型排名 (所有实验汇总)

| 排名 | 学生模型 | 参数量 | 测试精度 | Macro-F1 | 特点 |
|-----|---------|--------|---------|---------|------|
| 1 | **RepViT-M0.9** | 4.72M | 98.12% | 97.25% | 最高精度 |
| 2 | **EfficientNet-Lite** | 3.38M | 97.90% | 96.86% | 高精度/低参数比 |
| 3 | **MobileNetV2** | 2.23M | 97.83% | 97.27% | 高F1分数 |
| 4 | **GhostNet** | 3.91M | 97.83% | 96.93% | |
| 5 | **MobileNetV3-Small** | 1.52M | 97.83% | 96.79% | |
| 6 | **MobileNetV3-Small-050** | **0.58M** | 97.47% | 96.48% | **最轻量高精度** |
| 7 | **MobileNetV2-050** | 0.70M | 97.11% | 95.88% | 超轻量 |
| 8 | **GhostNet-050** | 1.32M | 96.82% | 95.88% | 轻量 |
| 9 | **GhostNetV3-050** | 2.12M | 96.60% | 95.13% | |

### 💡 关键发现

1. **RepViT-M0.9** 表现最佳 (98.12% 测试精度)，是轻量ViT架构的优势体现
2. **MobileNetV3-Small-050** 仅 **0.58M 参数** 却达到 **97.47%** 测试精度，是极致轻量化的最佳选择
3. 所有轻量模型 (0.58M-4.72M) 均达到 **95%+ 测试精度**
4. 知识蒸馏成功将教师集成的知识迁移到超轻量学生模型

### 检查点位置
- `results/lightweight_students_20251207/group1/` (GhostNet-050, GhostNetV3-050)
- `results/lightweight_students_20251207/group2/` (RepViT-M0.9, MobileNetV2-050)
- `results/lightweight_students_20251207/group3/` (MobileNetV3-Small-050)

---

## ✅ 实验完成状态

| Phase | 任务 | 状态 |
|-------|-----|------|
| Phase 1 | 教师模型预训练 (已有) | ✅ 完成 |
| Phase 2 | 核心学生模型蒸馏 | ✅ 完成 |
| Phase 3 | KD消融实验 | ⏸️ 需代码修改 |
| Phase 4 | 扩展学生模型 | ✅ 完成 |
| Phase 5 | 新轻量学生模型 | ✅ 完成 |

---

---

## 📊 MAE自监督预训练实验 (2025-12-07 23:30)

### 研究问题
教师模型如果经过自监督预训练（MAE风格），是否能获得更好的效果？

### 实验设置
- **预训练**: Batch=512, Epochs=100, LR=0.01(线性缩放), Mask=0.9
- **微调**: Epochs=50, LR=1e-4
- **对照**: 从头训练50 epochs

### 实验结果

| 模型 | 方法 | Val Acc | Test Acc | F1 |
|-----|------|---------|----------|-----|
| ConvNeXtV2-Tiny | MAE预训练+微调 | 64.71% | 64.09% | 62.29% |
| ConvNeXtV2-Tiny | **从头训练** | **97.47%** | **97.47%** | **96.53%** |
| EfficientNetV2-S | MAE预训练+微调 | 73.32% | 72.98% | 66.78% |
| EfficientNetV2-S | **从头训练** | **98.12%** | **98.19%** | **97.42%** |
| MobileNetV3-Large | MAE预训练+微调 | 96.24% | 94.51% | 93.90% |
| MobileNetV3-Large | **从头训练** | **98.05%** | **96.82%** | **96.21%** |

### 关键发现

🔴 **假设不成立**: MAE自监督预训练无法提升教师模型性能

**原因分析**:
1. 数据集规模小(~16k样本)，MAE需要大规模数据
2. 掩码率0.9过高，重建任务过难
3. 40x40图像分割patches后信息密度低
4. 网络流量图像与自然图像域差异大

### 结论与建议

✅ **对于ISCXVPN2016数据集，推荐直接使用监督学习从头训练**

如需预训练，建议:
- 降低掩码率到0.5-0.6
- 尝试对比学习(SimCLR, MoCo)
- 使用领域相关的数据增强

---

## 📈 完整实验汇总

### 最佳配置推荐

#### 教师模型
- **推荐**: 从头训练 + ECA注意力机制
- **最佳组合**: ConvNeXtV2-Tiny + DenseNet121 + MobileNetV3-Large

#### 学生模型 (知识蒸馏后)
| 排名 | 模型 | 参数量 | Test Acc | F1 |
|-----|------|--------|----------|-----|
| 1 | RepViT-M0.9 | 4.72M | 98.12% | 97.25% |
| 2 | MobileNetV3-Small-050 | **0.58M** | 97.47% | 96.48% |
| 3 | MobileNetV2-050 | 0.70M | 97.11% | 95.88% |

### 实验结论

1. **MAE预训练对小规模专业数据集无效**
2. **从头训练+知识蒸馏是最佳方案**
3. **最轻量高效模型**: MobileNetV3-Small-050 (0.58M参数, 97.47%精度)

