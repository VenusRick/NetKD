
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

