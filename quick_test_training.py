"""
快速测试训练流程 - 使用小规模数据验证pipeline正确性

该脚本用于快速验证整个训练流程是否正常工作，使用少量epoch和小批次。
"""

import torch
from data_preprocessing import quick_load_dataset
from training import train_teachers, train_stacking_model_stage, train_student_stage
from models import StudentNet
from training.evaluation import evaluate_model


def quick_test():
    """快速测试完整训练pipeline"""
    
    print("\n" + "="*70)
    print("NetKD快速测试 - 验证训练流程")
    print("="*70 + "\n")
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")
    
    # 加载数据集（使用较小的批次）
    print("正在加载数据集...")
    train_loader, val_loader, test_loader, metadata = quick_load_dataset(
        dataset_name="ISCXVPN2016",
        dataset_root=r"G:\数据集\Dataset",
        batch_size=32,  # 小批次
        num_workers=0,
    )
    
    num_classes = metadata['num_classes']
    print(f"数据集: {metadata['dataset_name']}")
    print(f"类别数: {num_classes}")
    print(f"训练集: {metadata['train_size']} 样本")
    print(f"验证集: {metadata['val_size']} 样本")
    print(f"测试集: {metadata['test_size']} 样本\n")
    
    # 阶段 I: 训练教师模型（仅2个epoch用于快速测试）
    print("="*70)
    print("阶段 I: 训练教师模型（2 epochs）")
    print("="*70 + "\n")
    
    teacher_ckpts = train_teachers(
        train_loader_A=train_loader,
        val_loader_A=val_loader,
        num_classes=num_classes,
        device=device,
        num_epochs_teacher=2,  # 快速测试：仅2轮
        lr=1e-3,
    )
    
    print("\n✅ 教师模型训练完成！")
    for name, ckpt in teacher_ckpts.items():
        print(f"  {name}: {ckpt}")
    
    # 阶段 II: 训练Stacking模型（仅2个epoch）
    print("\n" + "="*70)
    print("阶段 II: 训练Stacking集成模型（2 epochs）")
    print("="*70 + "\n")
    
    stacking_ckpt = train_stacking_model_stage(
        train_loader_B=train_loader,
        val_loader_B=val_loader,
        num_classes=num_classes,
        device=device,
        teacher_ckpts=teacher_ckpts,
        num_epochs_stacking=2,  # 快速测试：仅2轮
        lr=1e-3,
    )
    
    print(f"\n✅ Stacking模型训练完成！检查点: {stacking_ckpt}")
    
    # 阶段 III: 训练学生模型（仅3个epoch）
    print("\n" + "="*70)
    print("阶段 III: 训练学生模型 - 知识蒸馏（3 epochs）")
    print("="*70 + "\n")
    
    student_ckpt = train_student_stage(
        train_loader_B=train_loader,
        val_loader_B=val_loader,
        num_classes=num_classes,
        device=device,
        teacher_ckpts=teacher_ckpts,
        stacking_ckpt=stacking_ckpt,
        num_epochs_student=3,  # 快速测试：仅3轮
        lr=1e-3,
        T=4.0,
        lamb_ce=1.0,
        lamb_f=0.5,
        lamb_r=0.5,
        lamb_s=0.1,
    )
    
    print(f"\n✅ 学生模型训练完成！检查点: {student_ckpt}")
    
    # 评估
    print("\n" + "="*70)
    print("评估学生模型")
    print("="*70 + "\n")
    
    student = StudentNet(num_classes=num_classes)
    student.load_state_dict(torch.load(student_ckpt, map_location="cpu"))
    student.to(device)
    
    test_acc, test_f1, test_cm = evaluate_model(student, test_loader, device)
    
    print(f"\n测试集结果:")
    print(f"  准确率: {test_acc:.4f}")
    print(f"  F1分数 (macro): {test_f1:.4f}")
    
    print("\n" + "="*70)
    print("✅ 快速测试完成！训练流程正常工作。")
    print("="*70 + "\n")
    
    return {
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    results = quick_test()
    
    print("\n最终结果:")
    print(f"  测试准确率: {results['test_accuracy']:.4f}")
    print(f"  测试F1: {results['test_f1']:.4f}")
    print(f"  类别数: {results['num_classes']}")
