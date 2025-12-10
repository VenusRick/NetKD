# NetKD Agent 交接指南
**最后更新**: 2025-12-10 16:00 UTC+8
**维护者**: CodeAgent

---

## �� 快速开始

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
ps aux | grep python | grep -v grep     # 查看进程
```

---

## 📂 项目结构

```
/workspace/yqm/NetKD/
├── data_preprocessing/
│   └── image_loader.py          # 数据加载 (quick_load_dataset)
├── models/
│   ├── teacher_models.py        # 教师模型 (DenseNet121, ResNet50, MobileNetV3)
│   ├── student_model.py         # 学生模型 (StudentNet)
│   └── eca_module.py            # ECA注意力
├── training/
│   ├── train.py                 # 主训练脚本 ★
│   └── loss_functions.py        # KD损失函数
├── scripts/                     # 辅助脚本
├── runs/                        # 模型权重输出
├── results/                     # 实验结果
├── logs/                        # 训练日志
└── docs/                        # 文档
    ├── 01_AGENT_HANDOVER_GUIDE.md  # 本文档
    ├── 02_EXPERIMENT_RESULTS.md    # 实验结果记录
    └── 03_MODEL_ARCHITECTURE.md    # 模型架构
```

---

## 🎯 常用命令

### 启动全流程训练
```bash
# 单GPU训练
python training/train.py --mode train_student --use_real_data \
    --dataset ISCXVPN2016 --dataset_root /workspace/yqm/Dataset \
    --batch_size 128 --epochs_teacher 30 --epochs_stacking 10 \
    --epochs_student 30 --output_dir ./runs/ISCXVPN2016 \
    --no_teacher_pretrained

# 多GPU并行 (后台运行)
CUDA_VISIBLE_DEVICES=0 nohup python -u training/train.py ... > logs/gpu0.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python -u training/train.py ... > logs/gpu1.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u training/train.py ... > logs/gpu2.log 2>&1 &
```

### 监控实验
```bash
tail -f logs/train/USTC.log              # 实时查看
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
ps aux | grep python | grep -v grep      # 查看进程
```

---

## 📊 数据集列表

| 数据集 | 类别数 | 训练/验证/测试 | 图像尺寸 |
|--------|--------|----------------|----------|
| USTC-TFC2016 | 19 | 37K/7K/7K | 40×40 灰度 |
| ISCXTor2016 | 8 | 12K/1.5K/1.5K | 40×40 灰度 |
| CICIoT2022 | ? | 19K/2K/2K | 40×40 灰度 |
| CrossPlatform-Android | ? | - | 40×40 灰度 |
| CrossPlatform-iOS | ? | - | 40×40 灰度 |
| ISCXVPN2016 | 7 | 13K/1.4K/1.4K | 40×40 灰度 |

**数据集路径**: `/workspace/yqm/Dataset/`

---

## 🏆 最佳配置

### 教师模型
- **DenseNet121-ECA**: 98.77% (最佳)
- **ResNet50**: ~97%
- **MobileNetV3-Large**: ~97%

### 学生模型  
- **RepViT-M0.9**: 98.12% (T=3, α=0.3)
- **GhostNet-100**: 98.05%
- **MobileNetV3-Small**: 97.40%

### KD配置
- Temperature: T=3
- Alpha: α=0.3
- Loss: CE + Forward KL

---

## ⏳ 当前任务状态 (2025-12-10)

**正在运行的训练:**
- GPU 0: USTC-TFC2016 (ResNet50 教师训练中)
- GPU 1: CICIoT2022 (ResNet50 教师训练中)
- GPU 2: ISCXVPN2016 (ResNet50 教师训练中)

**待完成:**
- [ ] 完成第一批3个数据集的全流程
- [ ] 启动第二批 (ISCXTor2016, CrossPlatform-Android, CrossPlatform-iOS)
- [ ] 汇总所有结果到文档

---

## ⚠️ 已知问题及解决方案

1. **僵尸进程**: 使用 `pkill -9 -f python` 清理
2. **预训练权重下载失败**: 使用 `--no_teacher_pretrained`
3. **数据集路径**: 必须使用 `/workspace/yqm/Dataset`

---

## 📝 Git 操作
```bash
git add -A && git commit -m "描述"
git push origin Ubuntu
```

**GitHub**: https://github.com/VenusRick/NetKD.git (Ubuntu分支)
