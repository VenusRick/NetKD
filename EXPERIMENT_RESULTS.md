
---

## 📊 STEP 3 & 4: KD 消融和数据效率实验 (2025-12-08)

### 实验配置

**学生模型**:
- repvit_m0_9: 4.72M 参数
- mobilenetv3_small: 1.52M 参数

**KD 配置**:
- `ce_only`: 仅交叉熵损失（无知识蒸馏）
- `ce_kl`: CE + KL 散度 (T=3, alpha_ce=0.5, alpha_kl=0.5)

**数据比例**: 100%, 50%, 20%

### 实验进度

| # | 学生模型 | KD 配置 | 数据比例 | GPU | 状态 | Test Acc | Macro F1 |
|---|----------|---------|----------|-----|------|----------|----------|
| 1 | repvit_m0_9 | ce_only | 100% | 1 | 运行中 | - | - |
| 2 | repvit_m0_9 | ce_only | 50% | 2 | 等待中 | - | - |
| 3 | repvit_m0_9 | ce_only | 20% | 0 | 等待中 | - | - |
| 4 | repvit_m0_9 | ce_kl | 100% | 1 | 等待中 | - | - |
| 5 | repvit_m0_9 | ce_kl | 50% | 2 | 等待中 | - | - |
| 6 | repvit_m0_9 | ce_kl | 20% | 0 | 等待中 | - | - |
| 7 | mobilenetv3_small | ce_only | 100% | 1 | 等待中 | - | - |
| 8 | mobilenetv3_small | ce_only | 50% | 2 | 等待中 | - | - |
| 9 | mobilenetv3_small | ce_only | 20% | 0 | 等待中 | - | - |
| 10 | mobilenetv3_small | ce_kl | 100% | 1 | 等待中 | - | - |
| 11 | mobilenetv3_small | ce_kl | 50% | 2 | 等待中 | - | - |
| 12 | mobilenetv3_small | ce_kl | 20% | 0 | 等待中 | - | - |

**注**: 结果将在所有实验完成后更新。

### 预期分析
- **STEP 3 分析**: CE+KL 应该在两个学生模型上都优于 CE-only
- **STEP 4 分析**: 当训练数据减少到 50% 和 20% 时，CE+KL 的优势应该更明显


---

## 📊 CE+KL 实验结果 (2025-12-08 补充)

**注**: CE+KL 实验使用简化版脚本 (run_kd_simple.py)，采用timm预训练教师模型。

### 实验配置
- 教师: resnet50 + densenet121 + mobilenetv3_large_100 (平均集成)
- 温度 T=3.0, alpha_ce=0.5, alpha_kl=0.5
- 训练: 100 epochs, batch=128, lr=0.001

### 待更新结果
实验运行中，结果将在完成后自动更新到此表格。

---

## 📈 STEP 1 & 2 分析完成

### STEP 1: 教师分析
- 详见: `analysis/STEP1_teacher_analysis.md`
- EfficientNetV2 贡献最大 (+1.5% F1)
- 三教师集成达到最优

### STEP 2: Pareto 前沿
- 详见: `analysis/STEP2_pareto_analysis.md`
- Pareto 最优模型: edgenext_xx_small, mobilenetv3_small, efficientnet_lite0, repvit_m0_9
- 1.33M参数即可达到94% F1

