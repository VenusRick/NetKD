# NetKD Agent 交接指南
**最后更新**: 2025-12-08 14:48 UTC+8
**维护者**: CodeAgent

---

## 🔑 快速连接

```bash
# SSH 连接
ssh -o StrictHostKeyChecking=no root@10.126.126.3 -p 32833
# 密码: Liuliang_666
# 项目目录: /workspace/yqm/NetKD
```

---

## 📁 关键文件索引

| 文件路径 | 说明 | 优先级 |
|----------|------|--------|
| `docs/AGENT_HANDOFF_GUIDE.md` | 本文档 - Agent交接指南 | ⭐⭐⭐ |
| `docs/EXPERIMENT_HISTORY.md` | 所有实验记录 | ⭐⭐⭐ |
| `docs/MODEL_ARCHITECTURE.md` | 模型架构参考 | ⭐⭐⭐ |
| `configs/teachers.yaml` | 教师模型配置 | ⭐⭐ |
| `configs/students.yaml` | 学生模型配置 | ⭐⭐ |
| `results/FINAL_EXPERIMENT_REPORT_*.md` | 实验报告 | ⭐⭐ |

---

## 🔄 当前状态检查命令

```bash
# 1. GPU状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv

# 2. 运行中的实验
ps aux | grep python | grep -v grep

# 3. 最新日志
tail -f logs/*.log

# 4. 僵尸进程数 (Docker环境下无法清除)
ps aux | grep -c '<defunct>'
```

---

## 🚀 常用实验命令

### SimCLR 预训练
```bash
cd /workspace/yqm/NetKD
# 单GPU
python scripts/run_simclr_experiment.py --gpu 0 --backbone efficientnetv2_rw_s

# 多GPU并行
nohup python scripts/run_simclr_experiment.py --gpu 0 --backbone efficientnetv2_rw_s > logs/simclr_gpu0.log 2>&1 &
nohup python scripts/run_simclr_experiment.py --gpu 1 --backbone convnextv2_tiny > logs/simclr_gpu1.log 2>&1 &
nohup python scripts/run_simclr_experiment.py --gpu 2 --backbone mobilenetv3_large_100 > logs/simclr_gpu2.log 2>&1 &
```

### MAE 预训练
```bash
python scripts/mae_pretrain_finetune.py --gpu 0 --backbone efficientnetv2_rw_s --mask_ratio 0.6
```

### 学生蒸馏
```bash
python training/train.py --use_real_data --mode train_student
```

---

## ⚠️ 已知问题

1. **僵尸进程**: Docker容器环境下僵尸进程由init(PID 1)领养，无法kill。不影响运行，重启容器可清除。

2. **torch.amp API**: PyTorch 2.x 更新了 autocast API
   - 旧: `autocast('cuda')` 
   - 新: `autocast(device_type='cuda')`

3. **DataLoader vs Dataset**: `quick_load_dataset()` 返回 DataLoader，不是 Dataset
   - 获取 Dataset: `train_loader.dataset`

---

## 📊 最新实验结果摘要

### 教师模型 Top 3
| 模型 | 配置 | Test Acc | 推荐度 |
|------|------|----------|--------|
| EfficientNetV2-RW-S | MAE mr=0.6 | **98.19%** | ⭐⭐⭐ |
| DenseNet121-ECA | Baseline | 98.77% | ⭐⭐⭐ |
| ConvNeXtV2-Tiny | Scratch | 97.47% | ⭐⭐ |

### 学生模型 Top 3
| 模型 | 配置 | Test Acc | Params |
|------|------|----------|--------|
| RepViT-M0.9 | T=3, α=0.3 | **98.12%** | 4.72M |
| GhostNet-100 | T=3, α=0.3 | 98.05% | 3.91M |
| MobileNetV3-Small | T=4, α=0.3 | 97.40% | 1.02M |

---

## 📝 任务清单

### 待完成
- [ ] SimCLR预训练实验 (正在运行)
- [ ] 更多轻量级学生模型测试
- [ ] FLOPs/参数量完整分析
- [ ] 跨数据集泛化测试

### 已完成
- [x] MAE预训练实验
- [x] 教师模型对比
- [x] 学生蒸馏实验
- [x] 注意力机制消融

---

## 🔧 Git 配置

```bash
git config user.email "agent@netkd.local"
git config user.name "CodeAgent"
```

---

**下一位 Agent 接手时，请先阅读本文档和 `EXPERIMENT_HISTORY.md`**
