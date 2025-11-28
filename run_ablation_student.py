"""
消融实验专用学生训练脚本
支持不同的蒸馏模式 (ce/kl/kl2/full)
"""
import torch
import sys
import shutil
from pathlib import Path
from data_preprocessing.image_loader import quick_load_dataset
from training.train import train_student_stage
from training.monitor import LiveTrainingMonitor

if len(sys.argv) != 2:
    print("Usage: python run_ablation_student.py <distill_mode>")
    print("distill_mode: ce (GT only), kl (GT+FKL), kl2 (GT+FKL+RKL), full (all)")
    sys.exit(1)

distill_mode = sys.argv[1]
if distill_mode not in ['ce', 'kl', 'kl2', 'full']:
    print(f"Error: Invalid distill_mode '{distill_mode}'")
    print("Valid modes: ce, kl, kl2, full")
    sys.exit(1)

# 配置
DATASET = "ISCXVPN2016"
DATA_ROOT = "/walnut_data/yqm/Dataset"
BATCH_SIZE = 128  # 使用128以保证稳定性
NUM_WORKERS = 8
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 4.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 教师模型路径 - 使用高精度版本
TEACHER_CKPTS = {
    "resnet50": "checkpoints/resnet50_teacher.pth",
    "mbv3": "checkpoints/mbv3_teacher.pth",  
    "densenet121": "checkpoints/densenet121_teacher.pth"
}
STACKING_CKPT = "checkpoints/stacking_model.pth"  # 使用bs128版本
OUTPUT_DIR = f"checkpoints/ablation/s_{distill_mode}"

print("="*80)
print(f"消融实验: 蒸馏模式 = {distill_mode}")
print(f"数据集: {DATASET}")
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"输出目录: {OUTPUT_DIR}")
print("="*80)

# 验证教师模型文件存在
for name, path in TEACHER_CKPTS.items():
    if not Path(path).exists():
        raise FileNotFoundError(f"教师模型不存在: {path}")
if not Path(STACKING_CKPT).exists():
    raise FileNotFoundError(f"Stacking模型不存在: {STACKING_CKPT}")
print("✓ 所有教师模型文件验证通过\n")

# 加载数据
print("加载数据集...")
train_loader, val_loader, test_loader, meta = quick_load_dataset(
    dataset_name=DATASET,
    dataset_root=DATA_ROOT,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    val_ratio=0.15,
    test_ratio=0.15
)
num_classes = meta["num_classes"]
print(f"✓ 数据集加载完成: 类别数={num_classes}\n")

# 创建监控器
monitor = LiveTrainingMonitor()

# 创建输出目录
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# 训练学生模型
print("="*80)
print(f"开始训练学生模型 (模式: {distill_mode})...")
print("="*80)

student_ckpt = train_student_stage(
    train_loader_B=train_loader,
    val_loader_B=test_loader,
    num_classes=num_classes,
    device=DEVICE,
    T=TEMPERATURE,
    lamb_ce=1.0,
    lamb_f=0.5,
    lamb_r=0.5,
    lamb_s=0.1,
    num_epochs_student=EPOCHS,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
    teacher_ckpts=TEACHER_CKPTS,
    stacking_ckpt=STACKING_CKPT,
    monitor=monitor,
    distill_mode=distill_mode
)

print(f"\n训练完成! 学生模型: {student_ckpt}")

# 移动模型到输出目录
final_model_path = f"{OUTPUT_DIR}/student_model.pth"
shutil.move(student_ckpt, final_model_path)
print(f"模型已移动到: {final_model_path}")

# 评估最终精度
print("\n评估学生模型...")
from models.student_model import StudentNet
student = StudentNet(num_classes=num_classes).to(DEVICE)
student.load_state_dict(torch.load(final_model_path, map_location="cpu"))
student.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = student(images)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

final_acc = correct / total
print(f"测试集准确率: {final_acc*100:.2f}%")
print(f"知识保留率: {(final_acc/0.987)*100:.2f}% (相对Stacking的98.70%)")

# 保存结果
import json
results = {
    "distill_mode": distill_mode,
    "test_accuracy": final_acc,
    "knowledge_retention": final_acc / 0.987,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "model_path": final_model_path
}
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n结果已保存到: {OUTPUT_DIR}/results.json")
print("="*80)
