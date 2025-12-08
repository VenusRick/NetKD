# NetKD Agent 交接指南
**最后更新**: 2025-12-08 15:43 UTC+8
**维护者**: CodeAgent

---

## 🚀 快速开始

### SSH 连接
```bash
ssh root@10.126.126.3 -p 32833
# 密码: Liuliang_666
cd /workspace/yqm/NetKD
```

### 环境检查
```bash
nvidia-smi                              # GPU状态
conda activate base                     # Python环境
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📁 项目结构

```
/workspace/yqm/NetKD/
├── data_preprocessing/
│   └── image_loader.py          # 数据加载 (quick_load_dataset)
├── models/
│   ├── teacher_models.py        # 教师模型
│   ├── student_model.py         # 学生模型
│   └── eca_module.py            # ECA注意力
├── training/
│   ├── train.py                 # 主训练脚本
│   └── kd_losses.py             # KD损失函数
├── scripts/
│   ├── run_simclr_experiment.py # SimCLR实验
│   └── simple_student_kd.py     # 简单学生训练
├── checkpoints/                  # 预训练教师模型
├── results/                      # 实验结果
├── logs/                         # 训练日志
└── docs/                         # 文档
    ├── AGENT_HANDOFF_GUIDE.md   # 本文档
    ├── EXPERIMENT_HISTORY.md    # 实验历史
    └── MODEL_ARCHITECTURE.md    # 模型架构
```

---

## 🔧 常用命令

### 启动实验 (3 GPU并行)
```bash
# 学生模型训练
nohup python -u scripts/simple_student_kd.py --gpu 0 --student mobilenetv3_small_050 > logs/gpu0.log 2>&1 &
nohup python -u scripts/simple_student_kd.py --gpu 1 --student ghostnet_100 > logs/gpu1.log 2>&1 &
nohup python -u scripts/simple_student_kd.py --gpu 2 --student efficientnet_lite0 > logs/gpu2.log 2>&1 &
```

### 监控实验
```bash
tail -f logs/gpu0.log                   # 实时查看
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
ps aux | grep python | grep -v grep     # 查看进程
```

### Git 提交
```bash
git add -A && git commit -m "描述"
git --no-pager log --oneline -5
```

---

## 📊 当前最佳结果

### 教师模型 (已训练好)
| 模型 | 参数量 | Test Acc | 路径 |
|------|--------|----------|------|
| DenseNet121-ECA | 8.0M | **98.77%** | checkpoints/densenet121_teacher.pth |
| MobileNetV3-Large | 5.4M | 98.19% | checkpoints/mbv3_teacher.pth |
| ResNet50 | 23.5M | 98.19% | checkpoints/resnet50_teacher.pth |

### 学生模型基准 (2025-12-08)
| 模型 | 参数量 | Test Acc | 适用场景 |
|------|--------|----------|----------|
| mobilenetv3_small_050 | 0.58M | 94.08% | 极致轻量 |
| efficientnet_lite0 | 3.38M | 97.11% | 性价比最高 |
| ghostnet_100 | 3.91M | 97.40% | 精度优先 |

---

## ⚠️ 已知问题

1. **僵尸进程**: Docker容器限制，无法清除，不影响运行
2. **timm预训练下载**: 网络问题，使用 `pretrained=False`
3. **教师模型加载**: 需判断是 state_dict 还是完整模型

---

## 📋 待办事项

- [ ] 知识蒸馏实验 (教师→学生)
- [ ] 跨数据集验证
- [ ] 模型量化部署测试

---

## 📝 实验建议

1. **训练 Epochs**: 30 足够，快速实验用 20
2. **学习率**: 学生模型用 1e-3，微调用 1e-4
3. **批大小**: 64 (显存够用 128)
4. **收敛判断**: E20 后 val_acc 变化 <0.5% 即收敛
