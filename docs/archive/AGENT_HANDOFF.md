# NetKD 项目交接文档 (Agent Handoff)

**最后更新**: $(date '+%Y-%m-%d %H:%M')
**更新者**: CodeAgent

---

## 📍 快速定位

### 项目路径
```
SSH: root@10.126.126.3 -p 32833
密码: Liuliang_666
项目: /workspace/yqm/NetKD
```

### 关键文件
| 文件 | 说明 |
|------|------|
| `README.md` | 项目概述和性能基准 |
| `docs/AGENT_HANDOFF.md` | 本文档 - Agent交接指南 |
| `configs/teachers.yaml` | 教师模型配置 |
| `configs/students.yaml` | 学生模型和KD配置 |
| `results/FINAL_EXPERIMENT_REPORT_*.md` | 实验结果报告 |

---

## 🔄 当前运行状态

### 正在运行的实验 (检查方法)
```bash
# 查看GPU状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 查看运行中的Python进程
ps aux | grep python | grep -v grep

# 查看实验日志
tail -f logs/simclr_gpu*.log
```

### 待完成任务
- [ ] SimCLR对比学习预训练实验 (正在运行)
- [ ] 更多轻量级学生模型测试
- [ ] FLOPs/参数量完整分析

---

## 📊 已完成实验结果摘要

### 教师模型 (最佳配置)
| 模型 | 配置 | Test Acc | F1 |
|------|------|----------|-----|
| EfficientNetV2-RW-S | MAE mr=0.6 | **98.19%** | 97.32% |
| ConvNeXtV2-Tiny | Scratch | 97.47% | 96.56% |
| MobileNetV3-Large | MAE mr=0.8 | 97.18% | 95.96% |
| DenseNet121-ECA | Baseline | 98.77% | - |

### 学生模型 (蒸馏结果)
| 模型 | KD配置 | Test Acc | Params |
|------|--------|----------|--------|
| RepViT-M0.9 | T=3, α=0.3 | **98.12%** | 4.72M |
| GhostNet-100 | T=3, α=0.3 | 98.05% | 3.91M |

---

## 🚀 常用命令

### 启动实验
```bash
cd /workspace/yqm/NetKD

# SimCLR预训练 (单GPU)
python scripts/run_simclr_experiment.py --gpu 0 --backbone efficientnetv2_rw_s

# 教师训练
python training/train.py --use_real_data --mode train_teachers

# 学生蒸馏
python training/train.py --use_real_data --mode train_student
```

### 监控实验
```bash
# GPU监控
watch -n 5 nvidia-smi

# 日志监控
tail -f logs/*.log
```

---

## 📁 结果目录结构

```
results/
├── full_experiment_20251208_1045/     # MAE预训练实验
│   ├── phase1_pretrain/               # MAE预训练权重
│   ├── phase2_finetune/               # 微调教师模型
│   └── finetune_results.json          # 结果汇总
├── simclr_experiment_*/               # SimCLR实验结果
└── FINAL_EXPERIMENT_REPORT_*.md       # 报告文档

checkpoints/
├── *_teacher.pth                      # 教师模型权重
├── stacking_model.pth                 # Stacking模型
├── student_sd_mkd.pth                 # 学生模型
└── attention_ablation/                # 注意力消融
```

---

## ⚠️ 注意事项

1. **僵尸进程**: 服务器上可能有僵尸进程，无法直接kill
2. **SSH连接**: 长时间无操作可能断开，建议用 `nohup` 或 `tmux`
3. **数据集路径**: `/workspace/yqm/Dataset/ISCXVPN2016`

---

## 📝 变更日志

### 2025-12-08
- 修复 SimCLR 脚本 DataLoader 问题
- 修复 torch.amp.autocast 参数问题
- 启动 3-GPU 并行 SimCLR 实验
- 创建 Agent 交接文档

