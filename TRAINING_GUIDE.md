# NetKD 训练指南（当前可用入口）

## 环境与数据
- 环境：`conda create -n netkd python=3.12`；`pip install -r requirements.txt`
- 数据根目录：`/walnut_data/yqm/Dataset`
- 预处理：若需从 PCAP 生成图像，运行  
  ```bash
  python -m data_preprocessing.preprocess_data \
    --dataset ISCXVPN2016 \
    --data_path /path/to/raw/pcap \
    --image_height 32 --image_width 32 \
    --val_ratio 0.15 --test_ratio 0.15
  ```
  生成的图像放置在 `Dataset/<name>/images_sampled_new/<class>/xxx.png` 或同级子目录。

## 训练路径

### 1) 全流程三阶段（教师 → Stacking → 学生）
入口：`training/train.py`
```bash
python training/train.py \
  --use_real_data \
  --mode train_student \             # train_teachers / train_stacking / train_student
  --dataset ISCXVPN2016 \
  --dataset_root /walnut_data/yqm/Dataset \
  --batch_size 256 \
  --epochs_teacher 20 \
  --epochs_stacking 5 \
  --epochs_student 100 \
  --resnet_use_eca --mbv3_use_eca \
  --output_dir checkpoints/full_run
```
- 会自动复用 `output_dir` 下同名教师/Stacking 权重，存在则跳过重训。
- 主要可调参数：`--distill_mode {ce,kl,kl2,full}`、`--temperature`、`--lamb_*`、`--allow_data_parallel`。

### 2) 仅训练学生（复用高精度教师）
入口：`train_student_direct.py`（依赖 `checkpoints/` 下已训好的教师与 Stacking）
```bash
python train_student_direct.py
```
- 默认：`batch_size=256`，`epochs=100`，数据 `ISCXVPN2016`。

### 3) 蒸馏消融
- 单实验：`python run_ablation_student.py full` （ce/kl/kl2/full）
- 批量：`bash scripts/run_ablation_experiments.sh`  
  输出：`checkpoints/ablation/`，日志：`logs/ablation/`

### 4) 注意力机制消融
- 标准：`python run_attention_ablation.py agent`（支持 agent/cbam/eca/simam/none）
- Agent 修复版（更稳）：`python run_attention_ablation_fixed.py agent`
  输出：`checkpoints/attention_ablation(_fixed)/`，日志：`logs/`

## 模型与损失
- **教师**：ResNet50 / MobileNetV3-L / DenseNet121（单通道输入，ECA 可选）
- **Stacking**：三路教师 logits → MLP 融合
- **学生**：ShuffleNetV2 0.5x + AgentAttention2D + 全局池化 + 线性分类头
- **损失**：CE + 前向 KL + 反向 KL + Sinkhorn (`training/loss_functions.py`)

## 日志与监控
- 文本日志：`logs/`
- TensorBoard：`tensorboard --logdir runs`
- 常用监控脚本：`scripts/monitor_gpu.py`（GPU 采样）、`scripts/check_ablation_progress.sh` 等

## 清理提示
- 本次整理将失效脚本/备份/重复教师权重移至 `垃圾/`，核心入口见上。
