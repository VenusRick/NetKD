# 多分布蒸馏消融实验方案

## 实验目标
证明两个核心价值:
1. **蒸馏vs无蒸馏**: 知识蒸馏明显优于从头训练
2. **多分布蒸馏的优势**: 多重KL损失+Sinkhorn优于单一KL蒸馏

---

## 实验设置 (固定配置)

### 数据集
- **ISCXVPN2016**: 训练13,281 / 验证1,383 / 测试1,384
- **图像**: 1×40×40

### 固定参数
- **教师**: Stacking集成模型(98.70%准确率,已训练好)
- **学生**: 0.78M参数的轻量级模型
- **Batch Size**: 128 (最优配置)
- **优化器**: AdamW, lr=1e-3, weight_decay=1e-4
- **学习率调度**: CosineAnnealingLR
- **训练轮数**: 100 epochs
- **其他**: num_workers=8, AMP混合精度

### 关键原则
⚠️ **只改变损失函数形式,其他所有超参数保持一致!**

---

## 四种实验配置

### 1️⃣ S-CE (Baseline) - 仅硬标签
```
L = (1/b) Σ CE(student, ground_truth)
```
- **目的**: 学生从头训练的基线性能
- **不使用**: 教师知识
- **预期**: 明显低于其他方法

### 2️⃣ S-KL (标准KD) - CE + 单向KL
```
L = (1/b) Σ [(1-α)·CE(s, y) + α·KL(teacher||student)]
```
- **目的**: 经典知识蒸馏baseline
- **使用**: 仅Forward KL
- **α值**: 使用当前代码中的值(不重新调参)
- **预期**: 明显优于S-CE

### 3️⃣ S-KL2 (双向KL) - CE + FKL + RKL
```
L = (1/b) Σ [(1-α)·CE + α·KL(t||s) + β·KL(s||t)]
```
- **目的**: 验证双向KL的贡献
- **使用**: Forward + Reverse KL
- **α, β值**: 使用当前代码中的值
- **预期**: 优于S-KL

### 4️⃣ S-Full (当前方法) - 完整多分布蒸馏
```
L = (1/b) Σ [(1-α)·CE + α·KL(t||s) + β·KL(s||t) + γ·Sinkhorn(t,s)]
```
- **目的**: 完整框架性能
- **使用**: Forward KL + Reverse KL + Sinkhorn距离
- **α, β, γ值**: 使用当前代码中的值
- **当前结果**: 98.34%准确率,99.64%知识保留率

---

## 实现方案

### 代码修改
在`training/train.py`中添加`--distill_mode`参数:

\`\`\`python
parser.add_argument('--distill_mode', type=str, 
                    choices=['ce', 'kl', 'kl2', 'full'],
                    default='full',
                    help='Distillation loss mode')
\`\`\`

在损失函数计算中:
\`\`\`python
if distill_mode == 'ce':
    loss = ce_loss
elif distill_mode == 'kl':
    loss = (1-alpha) * ce_loss + alpha * fkl_loss
elif distill_mode == 'kl2':
    loss = (1-alpha) * ce_loss + alpha * fkl_loss + beta * rkl_loss
elif distill_mode == 'full':
    loss = (1-alpha) * ce_loss + alpha * fkl_loss + beta * rkl_loss + gamma * sinkhorn_loss
\`\`\`

### 训练命令
\`\`\`bash
# 1. Baseline (CE only)
CUDA_VISIBLE_DEVICES=0 python training/train.py \\
  --use_real_data --mode train_student \\
  --dataset ISCXVPN2016 \\
  --batch_size 128 --num_workers 8 \\
  --epochs_student 100 \\
  --distill_mode ce \\
  --output_dir checkpoints/ablation/s_ce

# 2. Standard KD (CE + KL)
CUDA_VISIBLE_DEVICES=0 python training/train.py \\
  --use_real_data --mode train_student \\
  --dataset ISCXVPN2016 \\
  --batch_size 128 --num_workers 8 \\
  --epochs_student 100 \\
  --distill_mode kl \\
  --output_dir checkpoints/ablation/s_kl

# 3. Bidirectional KL (CE + FKL + RKL)
CUDA_VISIBLE_DEVICES=0 python training/train.py \\
  --use_real_data --mode train_student \\
  --dataset ISCXVPN2016 \\
  --batch_size 128 --num_workers 8 \\
  --epochs_student 100 \\
  --distill_mode kl2 \\
  --output_dir checkpoints/ablation/s_kl2

# 4. Full (CE + FKL + RKL + Sinkhorn) - 已有结果
# 当前已训练: 98.34%准确率
\`\`\`

---

## 预期结果表格

| 模式   | 损失函数组成              | Val Acc (%) | 知识保留率* | 训练时间 |
|--------|---------------------------|-------------|-------------|----------|
| S-CE   | GT only                   | ~95-96      | ~96-97%     | 11 min   |
| S-KL   | GT + FKL                  | ~97-97.5    | ~98-99%     | 11 min   |
| S-KL2  | GT + FKL + RKL            | ~97.8-98.1  | ~99.0-99.3% | 11 min   |
| S-Full | GT + FKL + RKL + Sinkhorn | **98.34**   | **99.64%**  | 11 min   |

*知识保留率 = (学生准确率 / Stacking准确率 98.70%) × 100%

### 预期趋势
\`\`\`
Acc(S-CE) < Acc(S-KL) < Acc(S-KL2) ≤ Acc(S-Full)
\`\`\`

---

## 时间成本估算

- **单次训练**: ~11分钟(100 epochs, bs=128)
- **总实验时间**: ~33分钟(3个新实验,S-Full已有结果)
- **GPU需求**: 单卡即可
- **教师模型**: 无需重新训练(使用已有Stacking)

---

## 论文贡献点

完成这组实验后,可以在论文中明确说明:

### Method部分
> "We design a multi-distribution distillation objective combining forward KL, 
> reverse KL and Sinkhorn distance. Ablation studies (Section X) demonstrate 
> that each component contributes incrementally to the final accuracy."

### Ablation Study部分
> "Table X shows the ablation results. Compared to training from scratch (CE-only),
> standard knowledge distillation (CE+KL) improves accuracy by ~1.5-2%. Adding 
> reverse KL further boosts performance by ~0.5%. The full objective with Sinkhorn 
> distance achieves the best accuracy of 98.34%, retaining 99.64% of the teacher's
> knowledge."

### Contribution部分
> "We propose a multi-distribution distillation framework that systematically 
> combines forward KL, reverse KL, and Sinkhorn distance. Ablation experiments 
> validate the necessity of each component beyond standard single-KL baselines."

---

## 实验检查清单

- [ ] 确认当前代码中α, β, γ的具体数值
- [ ] 实现\`--distill_mode\`参数支持
- [ ] 验证S-CE模式不使用任何教师logits
- [ ] 运行S-CE baseline实验
- [ ] 运行S-KL标准KD实验
- [ ] 运行S-KL2双向KL实验
- [ ] 整理结果到表格
- [ ] 绘制性能对比图(可选)

---

## 注意事项

1. **随机种子**: 建议固定随机种子确保可重复性
2. **评估指标**: 使用测试集准确率(Test Acc)而非验证集,更严谨
3. **多次运行**: 如时间允许,每个配置运行3次取平均值
4. **日志记录**: 保存完整训练日志以便后续分析

---

**总结**: 这是一个**最小但有力**的消融实验设计,用~33分钟GPU时间换取论文的完整性和说服力。
