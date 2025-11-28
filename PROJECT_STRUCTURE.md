# NetKD 项目结构（当前可用版本）

## 顶层布局
- `training/`：三阶段训练管线与监控工具
- `data_preprocessing/`：流量转图像的数据加载与预处理
- `models/`：教师模型、学生模型、注意力与 Stack 集成
- `scripts/`：批量训练/监控/可视化辅助脚本
- `checkpoints/`：教师、Stacking、学生权重与实验结果
- `logs/`、`runs/`：训练日志与 TensorBoard
- `archive/`：历史归档（旧实验、重复模型）
- `垃圾/`：此次整理收纳的失效脚本、备份、重复权重

## 核心模块

### data_preprocessing/
- `image_loader.py`：`quick_load_dataset` 直接读取预处理好的 PNG 数据集（自动检测 train/val/test 或运行时划分）。
- `preprocess_data.py` + `preprocessor.py`：从原始 PCAP 生成流量图像，支持缓存与分割。
- 其他: `adapters.py`（多数据集适配）、`augmentation.py`、`label_encoder.py`、`statistics.py`。

### models/
- `student_model.py`：ShuffleNetV2 0.5x 骨干 + `AgentAttention2D` + 全局池化 + 线性分类头。
- `teacher_models.py`：ResNet50 / MobileNetV3-Large / DenseNet121（单通道输入，ECA 可选）、`TeacherEnsemble`、`StackingModel`，以及单教师/Stacking 训练 helper。
- `student_model_flexible.py`：可插拔注意力版本（用于注意力消融）。
- `attention_modules.py`、`eca_module.py`：注意力实现与 ECA 注入工具。

### training/
- `train.py`：主入口。支持 `train_teachers` / `train_stacking` / `train_student`，可切换演示数据或真实数据。
- `loss_functions.py`：CE + 前向/反向 KL + Sinkhorn 复合蒸馏损失。
- `monitor.py` + `status_tracker.py`：训练过程记录与进度监控。
- `engine.py`、`evaluation.py`、`stacking.py`：训练循环与评估辅助。

### scripts/
- `run_full_training.sh`：批量调度多 batch 的教师/Stacking/学生训练。
- `run_ablation_experiments.sh` + `run_ablation_student.py`：蒸馏损失消融。
- `run_attention_ablation.py` / `run_attention_ablation_fixed.py`：注意力类型对比（修复版含 Agent 专用 lr/bs）。
- `plot_*` 与 `monitor_*`：实验结果绘制与监控。

### 其他目录
- `checkpoints/`：已训练权重（教师/Stacking/学生）与实验输出。
- `logs/`、`runs/`：训练日志与 TensorBoard。
- `垃圾/`：最新清理收纳的无效文件（备份脚本、旧教师权重等）。

## 数据与训练流程
```
Raw PCAP → data_preprocessing.preprocess_data → 40×40/32×32 灰度图
           ↓
   quick_load_dataset() 生成 train/val/test DataLoader
           ↓
教师训练 (ResNet50 / MobileNetV3-L / DenseNet121, 可选 ECA)
           ↓
Stacking MLP 融合教师 logits
           ↓
学生蒸馏 (ShuffleNetV2 + AgentAttention, 损失=CE+FKL+RKL+Sinkhorn)
           ↓
模型/日志输出到 checkpoints/ 与 logs/
```
