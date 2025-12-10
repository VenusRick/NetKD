#!/usr/bin/env python3
"""检查Teacher 2.0任务进度"""
import os

print("=" * 80)
print("Teacher 2.0 & Student 2.0 任务检查")
print("=" * 80)

tasks = {
    "Teacher 2.0": [
        ("configs/teachers.yaml", "教师模型配置文件"),
        ("models/teacher_registry.py", "教师模型注册表"),
        ("models/teacher_models.py (GhostNetV3)", "GhostNetV3教师模型"),
        ("models/teacher_models.py (RepViT)", "RepViT教师模型"),
        ("analysis/compute_disagreement.py", "教师多样性计算"),
        ("scripts/leave_one_out_stacking.py", "Leave-One-Out分析"),
    ],
    "Student 2.0": [
        ("configs/students.yaml", "学生模型配置文件"),
        ("models/student_registry.py", "学生模型注册表"),
        ("models/student_models_v2.py (GhostNetV3)", "GhostNetV3-0.75x学生"),
        ("training/loss_functions.py (Sinkhorn)", "Sinkhorn损失函数"),
        ("experiments/student_kd/train_student_kd.py", "完整TrafficKD训练"),
    ],
}

for category, items in tasks.items():
    print(f"\n{category}:")
    print("-" * 80)
    for path, desc in items:
        if "(" in path:
            # 需要检查文件内容
            file_path = path.split("(")[0].strip()
            keyword = path.split("(")[1].split(")")[0]
            exists = os.path.exists(file_path)
            status = "⚠️  文件存在，需检查内容" if exists else "❌"
        else:
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
        
        print(f"  {status} {desc:<40} ({path})")

print("\n" + "=" * 80)
