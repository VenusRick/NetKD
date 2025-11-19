"""
使用真实数据集训练NetKD模型

该脚本使用G:\数据集\Dataset目录下的预处理图像数据进行训练。
"""

import argparse
import torch
from pathlib import Path

from data_preprocessing import quick_load_dataset
from training import train_teachers, train_stacking_model_stage, train_student_stage
from models import StudentNet
from training.evaluation import evaluate_model


def parse_args():
    parser = argparse.ArgumentParser(description="NetKD训练 - 使用真实数据集")
    
    # 数据集参数
    parser.add_argument(
        "--dataset",
        type=str,
        default="ISCXVPN2016",
        choices=["ISCXVPN2016", "ISCXTor2016", "USTC-TFC2016", 
                 "CrossPlatform-Android", "CrossPlatform-iOS", "CICIoT2022"],
        help="数据集名称"
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/walnut_data/yqm/Dataset",
        help="数据集根目录"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="批次大小"
    )
    
    # 训练参数
    parser.add_argument(
        "--mode",
        type=str,
        default="train_student",
        choices=["train_teachers", "train_stacking", "train_student", "full_pipeline", "eval"],
        help="训练模式"
    )
    parser.add_argument(
        "--epochs_teacher",
        type=int,
        default=10,
        help="教师模型训练轮数"
    )
    parser.add_argument(
        "--epochs_stacking",
        type=int,
        default=10,
        help="Stacking模型训练轮数"
    )
    parser.add_argument(
        "--epochs_student",
        type=int,
        default=20,
        help="学生模型训练轮数"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="学习率"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="权重衰减"
    )
    
    # 蒸馏损失权重
    parser.add_argument("--lamb_ce", type=float, default=1.0, help="交叉熵损失权重")
    parser.add_argument("--lamb_f", type=float, default=0.5, help="前向KL损失权重")
    parser.add_argument("--lamb_r", type=float, default=0.5, help="反向KL损失权重")
    parser.add_argument("--lamb_s", type=float, default=0.1, help="Sinkhorn损失权重")
    parser.add_argument("--temperature", type=float, default=4.0, help="蒸馏温度")
    
    # 其他参数
    parser.add_argument("--num_workers", type=int, default=0, help="数据加载线程数（Windows建议0）")
    parser.add_argument("--student_ckpt", type=str, default="student_sd_mkd.pth", help="学生模型检查点")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="检查点保存目录")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"NetKD训练 - 使用真实数据集")
    print(f"{'='*70}")
    print(f"数据集: {args.dataset}")
    print(f"数据路径: {args.data_root}")
    print(f"训练模式: {args.mode}")
    print(f"设备: {device}")
    print(f"{'='*70}\n")
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据集
    print("正在加载数据集...")
    train_loader, val_loader, test_loader, metadata = quick_load_dataset(
        dataset_name=args.dataset,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    
    num_classes = metadata['num_classes']
    print(f"\n数据集加载完成！")
    print(f"  类别数: {num_classes}")
    print(f"  训练集: {metadata['train_size']} 样本")
    print(f"  验证集: {metadata['val_size']} 样本")
    print(f"  测试集: {metadata['test_size']} 样本")
    print(f"  图像尺寸: {metadata['image_channels']}x{metadata['image_height']}x{metadata['image_width']}")
    print(f"  类别名称: {metadata['class_names']}\n")
    
    # 定义检查点路径
    teacher_ckpts = {
        "resnet50": str(output_dir / "resnet50_teacher.pth"),
        "mbv3": str(output_dir / "mbv3_teacher.pth"),
        "densenet121": str(output_dir / "densenet121_teacher.pth"),
    }
    stacking_ckpt = str(output_dir / "stacking_model.pth")
    student_ckpt = str(output_dir / args.student_ckpt)
    
    # 执行训练
    if args.mode == "train_teachers" or args.mode == "full_pipeline":
        print(f"\n{'='*70}")
        print("阶段 I: 训练教师模型")
        print(f"{'='*70}\n")
        
        teacher_ckpts = train_teachers(
            train_loader_A=train_loader,
            val_loader_A=val_loader,
            num_classes=num_classes,
            device=device,
            num_epochs_teacher=args.epochs_teacher,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        # 保存到输出目录
        for name, ckpt in teacher_ckpts.items():
            import shutil
            target = output_dir / Path(ckpt).name
            shutil.move(ckpt, target)
            teacher_ckpts[name] = str(target)
        
        print(f"\n✅ 教师模型训练完成！检查点已保存到: {output_dir}")
    
    if args.mode == "train_stacking" or args.mode == "full_pipeline":
        print(f"\n{'='*70}")
        print("阶段 II: 训练Stacking集成模型")
        print(f"{'='*70}\n")
        
        stacking_ckpt = train_stacking_model_stage(
            train_loader_B=train_loader,
            val_loader_B=val_loader,
            num_classes=num_classes,
            device=device,
            teacher_ckpts=teacher_ckpts,
            num_epochs_stacking=args.epochs_stacking,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        
        # 移动到输出目录
        import shutil
        target = output_dir / Path(stacking_ckpt).name
        shutil.move(stacking_ckpt, target)
        stacking_ckpt = str(target)
        
        print(f"\n✅ Stacking模型训练完成！检查点: {stacking_ckpt}")
    
    if args.mode == "train_student" or args.mode == "full_pipeline":
        print(f"\n{'='*70}")
        print("阶段 III: 训练学生模型（知识蒸馏）")
        print(f"{'='*70}\n")
        print(f"蒸馏损失配置:")
        print(f"  CE权重: {args.lamb_ce}")
        print(f"  FKL权重: {args.lamb_f}")
        print(f"  RKL权重: {args.lamb_r}")
        print(f"  Sinkhorn权重: {args.lamb_s}")
        print(f"  温度: {args.temperature}\n")
        
        student_ckpt = train_student_stage(
            train_loader_B=train_loader,
            val_loader_B=val_loader,
            num_classes=num_classes,
            device=device,
            teacher_ckpts=teacher_ckpts,
            stacking_ckpt=stacking_ckpt,
            num_epochs_student=args.epochs_student,
            lr=args.lr,
            weight_decay=args.weight_decay,
            T=args.temperature,
            lamb_ce=args.lamb_ce,
            lamb_f=args.lamb_f,
            lamb_r=args.lamb_r,
            lamb_s=args.lamb_s,
        )
        
        # 移动到输出目录
        import shutil
        target = output_dir / Path(student_ckpt).name
        shutil.move(student_ckpt, target)
        student_ckpt = str(target)
        
        print(f"\n✅ 学生模型训练完成！检查点: {student_ckpt}")
    
    # 评估模型
    if args.mode == "eval" or args.mode == "full_pipeline":
        print(f"\n{'='*70}")
        print("评估学生模型")
        print(f"{'='*70}\n")
        
        # 加载学生模型
        student = StudentNet(num_classes=num_classes)
        student.load_state_dict(torch.load(student_ckpt, map_location="cpu"))
        student.to(device)
        
        # 在测试集上评估
        test_acc, test_f1, test_cm = evaluate_model(student, test_loader, device)
        
        print(f"\n测试集结果:")
        print(f"  准确率: {test_acc:.4f}")
        print(f"  F1分数: {test_f1:.4f}")
        print(f"\n混淆矩阵:")
        print(test_cm)
        
        # 保存结果
        results = {
            "dataset": args.dataset,
            "num_classes": num_classes,
            "test_accuracy": float(test_acc),
            "test_f1": float(test_f1),
            "confusion_matrix": test_cm.tolist() if hasattr(test_cm, 'tolist') else test_cm,
            "class_names": metadata['class_names'],
        }
        
        import json
        results_file = output_dir / f"results_{args.dataset}.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n结果已保存到: {results_file}")
    
    print(f"\n{'='*70}")
    print("训练完成！")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
