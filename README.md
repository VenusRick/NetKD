# NetKD - 轻量化加密流量蒸馏框架

基于多教师 (ResNet50 / MobileNetV3-Large / DenseNet121) + Stacking 集成的知识蒸馏管线，将高精度教师知识迁移到带 AgentAttention 的 ShuffleNet 学生模型。

## 目录结构 (精简后)
- `training/`：三阶段训练入口 `train.py`，蒸馏损失 `loss_functions.py`，训练监控 `monitor.py`
- `data_preprocessing/`：图像流量加载与预处理 (`image_loader.py`, `preprocess_data.py`)
- `models/`：教师/学生/Stacking 模型与注意力模块
- `scripts/`：批量训练、消融与可视化辅助脚本
- `checkpoints/`：教师、Stacking、学生模型权重（当前高精度模型位于此）
- `logs/`, `runs/`：训练日志与 TensorBoard 运行目录
- `archive/`：历史归档
- `垃圾/`：本次整理后归档的失效脚本、旧权重、备份文件

## 快速开始
```bash
conda create -n netkd python=3.12 -y
conda activate netkd
pip install -r requirements.txt
```

### 1) 全流程三阶段训练（真实数据）
```bash
python training/train.py \
  --use_real_data \
  --mode train_student \        # train_teachers / train_stacking / train_student
  --dataset ISCXVPN2016 \
  --dataset_root /walnut_data/yqm/Dataset \
  --batch_size 256 \
  --epochs_teacher 20 --epochs_stacking 5 --epochs_student 100 \
  --resnet_use_eca --mbv3_use_eca \
  --output_dir checkpoints/full_run
```
- 支持自动复用已存在的教师/Stacking 权重（同名文件即跳过重训）。

### 2) 仅学生训练（复用已训教师）
```bash
python train_student_direct.py
```
- 依赖 `checkpoints/` 下的高精度教师与 Stacking 权重。

### 3) 蒸馏消融
- 单次：`python run_ablation_student.py full`（ce/kl/kl2/full）
- 批量：`bash scripts/run_ablation_experiments.sh`

### 4) 注意力消融
- 标准版：`python run_attention_ablation.py agent`
- 修复版（Agent 特殊 lr / bs）：`python run_attention_ablation_fixed.py agent`

## 数据预处理
预处理原始 PCAP：  
```bash
python -m data_preprocessing.preprocess_data \
  --dataset ISCXVPN2016 \
  --data_path /path/to/raw/pcap \
  --image_height 32 --image_width 32 \
  --val_ratio 0.15 --test_ratio 0.15
```
生成的图像数据放入 `Dataset/<dataset_name>/images_sampled_new/` 或同级类目录，`training/train.py` / `train_student_direct.py` 直接读取。

## 模型概览
- **教师**：ResNet50 / MobileNetV3-Large / DenseNet121，可选 ECA。单通道输入头已适配。
- **Stacking**：三路教师 logits -> MLP 融合，作为学生软目标。
- **学生**：ShuffleNetV2 0.5x + AgentAttention2D + 全局池化，全流程蒸馏损失 CE+FKL+RKL+Sinkhorn。

## 检查点与日志
- 教师/Stacking/学生权重：`checkpoints/`
- 实验日志：`logs/`，TensorBoard: `tensorboard --logdir runs`

## 清理说明
- 新增 `垃圾/`：存放失效脚本（如依赖缺失的 `run_student_training.sh`）、旧备份与重复教师权重，避免干扰当前管线。
