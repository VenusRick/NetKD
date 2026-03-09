# NetKD Agent 交接文档
> 最后更新: 2024-12-10 18:45 (UTC+8)
> 当前状态: **第一批训练进行中**

## 🎯 项目核心目标
使用知识蒸馏(KD)方法，将多教师模型的知识迁移到轻量级学生模型，实现高效的网络流量分类。

## 📁 项目结构
```
/workspace/yqm/NetKD/
├── training/          # 训练代码
│   └── train.py       # 主训练脚本
├── models/            # 模型定义
├── data_preprocessing/ # 数据预处理
├── runs/              # 模型权重输出
├── logs/              # 训练日志
├── docs/              # 文档
└── configs/           # 配置文件
```

## 🔧 环境配置
```bash
# SSH连接
ssh -p 32833 root@10.126.126.3
# 密码: Liuliang_666

# Python环境
/root/miniconda3/bin/python  # base环境

# 数据集位置
/workspace/yqm/Dataset/
```

## 📊 数据集列表
| 数据集 | 类别数 | 训练/验证/测试 | 图像尺寸 |
|--------|--------|----------------|----------|
| USTC-TFC2016 | 20 | 37K/7K/7K | 40×40 灰度 |
| ISCXTor2016 | 8 | 12K/1.5K/1.5K | 40×40 灰度 |
| CICIoT2022 | ? | 19K/2K/2K | 40×40 灰度 |
| CrossPlatform-Android | ? | - | 40×40 灰度 |
| CrossPlatform-iOS | ? | - | 40×40 灰度 |
| ISCXVPN2016 | 7 | - | 40×40 灰度 |

## 🚀 训练命令
```bash
# 单数据集训练
CUDA_VISIBLE_DEVICES=0 python training/train.py \
    --mode train_student \
    --use_real_data \
    --dataset USTC-TFC2016 \
    --dataset_root /workspace/yqm/Dataset \
    --batch_size 128 \
    --epochs_teacher 15 \
    --epochs_stacking 3 \
    --epochs_student 20 \
    --output_dir ./runs/USTC-TFC2016 \
    --no_teacher_pretrained
```

## ⏳ 当前任务状态
**第一批训练 (2024-12-10 启动):**
- GPU 0: USTC-TFC2016 (PID 562439) ✅ 运行中
- GPU 1: ISCXTor2016 (PID 562664) ✅ 运行中
- GPU 2: CICIoT2022 (PID 562873) ✅ 运行中

**下一步计划:**
1. 第一批完成后启动第二批 (CrossPlatform-Android, CrossPlatform-iOS, ISCXVPN2016)
2. 汇总所有数据集的实验结果
3. 更新实验结果文档

## 📝 监控命令
```bash
# 查看GPU状态
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 查看训练日志
tail -50 logs/parallel/USTC.log
tail -50 logs/parallel/Tor.log
tail -50 logs/parallel/CICIOT.log

# 查看后台进程
jobs -l
```

## ⚠️ 已知问题及解决方案
1. **数据集路径**: 使用 `/workspace/yqm/Dataset` (不是 `/walnut_data/yqm/Dataset`)
2. **Python路径**: 使用 `python` (base环境已激活)
3. **预训练权重**: 使用 `--no_teacher_pretrained` 避免网络下载问题
