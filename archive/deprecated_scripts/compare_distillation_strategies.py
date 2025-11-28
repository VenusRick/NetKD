"""
对比固定权重蒸馏 vs 自适应权重蒸馏
使用checkpoints/下已训练好的高精度教师模型 (98%+)
"""
import torch
from training.train import train_student_with_distillation
from pathlib import Path

# 配置
dataset = "ISCXVPN2016"
data_root = "/walnut_data/yqm/Dataset"
device = torch.device("cuda")

print("="*80)
print("使用已有高精度教师模型进行学生模型蒸馏对比实验")
print("教师模型路径: checkpoints/{resnet50,mbv3,densenet121}_teacher.pth")
print("Stacking路径: checkpoints/stacking_model.pth") 
print("="*80)

# 1. 训练学生 - 固定权重 (Baseline)
print("\n阶段1: 训练学生模型 - 固定权重蒸馏")
print("-"*80)
from experiments.sd_mkd import ExperimentConfig, run_experiment

config_fixed = ExperimentConfig(
    dataset=dataset,
    data_root=data_root,
    batch_size=256,
    num_workers=0,
    epochs_teacher=0,  # 跳过教师训练
    epochs_stacking=0,  # 跳过Stacking训练  
    epochs_student=100,
    lr=1e-3,
    weight_decay=1e-4,
    temperature=4.0,
    output_dir="./checkpoints/comparison",
    mode="student_only",
    student_ckpt="student_fixed_weights.pth",
    disable_monitor=False
)

print("开始训练 (固定权重)...")
results_fixed = run_experiment(config_fixed)
print(f"固定权重学生精度: {results_fixed.get('test_accuracy', 0)*100:.2f}%")

# 2. 训练学生 - 自适应权重
print("\n阶段2: 训练学生模型 - 自适应权重蒸馏") 
print("-"*80)
config_adaptive = ExperimentConfig(
    dataset=dataset,
    data_root=data_root,
    batch_size=256,
    num_workers=0,
    epochs_teacher=0,
    epochs_stacking=0,
    epochs_student=100,
    lr=1e-3,
    weight_decay=1e-4,
    temperature=4.0,
    output_dir="./checkpoints/comparison",
    mode="student_only_adaptive",  # 新模式: 使用自适应权重
    student_ckpt="student_adaptive_weights.pth",
    disable_monitor=False
)

print("开始训练 (自适应权重)...")
results_adaptive = run_experiment(config_adaptive)
print(f"自适应权重学生精度: {results_adaptive.get('test_accuracy', 0)*100:.2f}%")

# 3. 对比结果
print("\n" + "="*80)
print("最终对比结果")
print("="*80)
fixed_acc = results_fixed.get('test_accuracy', 0) * 100
adaptive_acc = results_adaptive.get('test_accuracy', 0) * 100
print(f"学生(固定权重):   {fixed_acc:.2f}%")
print(f"学生(自适应权重): {adaptive_acc:.2f}%")
print(f"改进幅度:         {adaptive_acc - fixed_acc:+.2f}%")
print(f"\n目标达成: {'✅' if adaptive_acc >= 98.5 else '❌'} (目标≥98.5%)")
