"""
直接使用training.train模块训练学生模型
使用已有的高精度教师模型 (98%+)
"""
import torch
from pathlib import Path
from data_preprocessing.image_loader import quick_load_dataset
from training.train import train_student_stage
from training.monitor import LiveTrainingMonitor

# 配置
DATASET = "ISCXVPN2016"
DATA_ROOT = "/walnut_data/yqm/Dataset"
BATCH_SIZE = 256
NUM_WORKERS = 0
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
TEMPERATURE = 4.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 教师模型路径
TEACHER_CKPTS = {
    "resnet50": "checkpoints/resnet50_teacher.pth",
    "mbv3": "checkpoints/mbv3_teacher.pth",
    "densenet121": "checkpoints/densenet121_teacher.pth"
}
STACKING_CKPT = "checkpoints/stacking_model.pth"

print("="*80)
print("使用高精度教师模型(98%+)训练学生模型")
print(f"数据集: {DATASET}")
print(f"Batch Size: {BATCH_SIZE}, Epochs: {EPOCHS}")
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

# 训练学生模型
print("="*80)
print("开始训练学生模型...")
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
    monitor=monitor
)

print(f"\n训练完成! 学生模型: {student_ckpt}")

# 评估最终精度
print("\n评估学生模型...")
from models.student_model import StudentNet
student = StudentNet(num_classes=num_classes).to(DEVICE)
student.load_state_dict(torch.load(student_ckpt, map_location="cpu"))
student.eval()

correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = student(images)
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

test_acc = 100 * correct / total
print(f"\n最终测试精度: {test_acc:.2f}%")
print(f"目标达成: {'✅' if test_acc >= 98.5 else '❌'} (目标≥98.5%)")
