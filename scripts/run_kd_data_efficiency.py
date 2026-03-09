#!/usr/bin/env python3
"""
STEP 3 & 4: KD Ablation + Data Efficiency Experiments
针对论文新计划的实验：
- STEP 3: KD损失配置消融 (CE-only vs CE+KL)
- STEP 4: 数据效率实验 (不同训练数据比例)
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Subset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset
from models.teacher_models import (DynamicStackingModel, ConvNeXtV2TinyTeacher,
                                    DenseNet121Teacher, MobileNetV3LargeTeacher)
from models.student_models_v2 import create_student_model_extended as create_student_model

# KD 配置定义
KD_CONFIGS = {
    "ce_only": {
        "name": "CE Only",
        "alpha_ce": 1.0,
        "alpha_kl": 0.0,
        "description": "仅使用交叉熵损失，无知识蒸馏"
    },
    "ce_kl": {
        "name": "CE + KL (T=3)",
        "alpha_ce": 0.5,
        "alpha_kl": 0.5,
        "temperature": 3.0,
        "description": "交叉熵 + KL散度蒸馏"
    },
}

# 全局配置
CONFIG = {
    "dataset": "ISCXVPN2016",
    "data_path": "/workspace/yqm/Dataset/ISCXVPN2016",
    "batch_size": 128,
    "num_workers": 8,
    "teachers": {
        "convnextv2_tiny": {
            "class": ConvNeXtV2TinyTeacher,
            "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/convnextv2_tiny_eca_best.pth"
        },
        "densenet121": {
            "class": DenseNet121Teacher,
            "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/densenet121_eca_best.pth"
        },
        "mobilenetv3": {
            "class": MobileNetV3LargeTeacher,
            "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/mobilenetv3_eca_best.pth"
        },
    },
    "stacking": {
        "checkpoint": "results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth",
        "hidden_dim": 384
    },
    "training": {
        "epochs": 100,
        "base_lr": 0.001,
        "weight_decay": 1e-4,
        "warmup_epochs": 5,
        "grad_clip": 1.0
    },
}

def subsample_dataset(dataset, train_fraction, seed=42):
    """子采样训练数据集"""
    if train_fraction >= 1.0:
        return dataset

    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    subset_size = int(len(dataset) * train_fraction)
    subset_indices = indices[:subset_size]

    return Subset(dataset, subset_indices)

def load_teachers(config, num_classes, device):
    """加载教师模型"""
    teachers = []
    for name, cfg in config["teachers"].items():
        model = cfg["class"](num_classes=num_classes, use_eca=cfg["use_eca"])
        ckpt_path = PROJECT_ROOT / cfg["checkpoint"]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        for p in model.parameters():
            p.requires_grad = False
        teachers.append(model)
    return teachers

def load_stacking(config, num_teachers, num_classes, device):
    """加载Stacking集成模型"""
    stacking = DynamicStackingModel(num_classes, num_teachers, config["stacking"]["hidden_dim"])
    ckpt_path = PROJECT_ROOT / config["stacking"]["checkpoint"]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        stacking.load_state_dict(ckpt["model_state_dict"])
    stacking = stacking.to(device).eval()
    for p in stacking.parameters():
        p.requires_grad = False
    return stacking

@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """详细评估：准确率、F1、精确率、召回率"""
    model.eval()
    preds, labels = [], []

    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        preds.extend(outputs.argmax(1).cpu().tolist())
        labels.extend(y.tolist())

    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    macro_f1 = f1_score(labels, preds, average='macro')
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": precision,
        "recall": recall
    }

def train_student(student, teachers, stacking, train_loader, val_loader, test_loader,
                 kd_config_id, config, device, output_dir):
    """训练学生模型"""
    kd_cfg = KD_CONFIGS[kd_config_id]
    train_cfg = config["training"]

    # 优化器和调度器
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=train_cfg["base_lr"],
                                  weight_decay=train_cfg["weight_decay"])

    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()

    # 训练循环
    best_val_f1 = 0
    best_epoch = 0
    history = []

    use_kd = (kd_config_id != "ce_only")
    T = kd_cfg.get("temperature", 3.0)

    for epoch in range(1, train_cfg["epochs"] + 1):
        student.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"E{epoch}/{train_cfg['epochs']}", leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast('cuda'):
                student_logits = student(images)

                # CE Loss
                ce_loss = F.cross_entropy(student_logits, labels)

                if use_kd and teachers is not None and stacking is not None:
                    # 获取教师预测
                    with torch.no_grad():
                        teacher_logits_list = [t(images) for t in teachers]
                        # Stack: (batch, num_teachers, num_classes)
                        teacher_logits_stacked = torch.stack(teacher_logits_list, dim=1)
                        teacher_logits = stacking(teacher_logits_stacked)

                    # KL Divergence Loss
                    student_soft = F.log_softmax(student_logits / T, dim=1)
                    teacher_soft = F.softmax(teacher_logits / T, dim=1)
                    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)

                    # 总损失
                    loss = kd_cfg["alpha_ce"] * ce_loss + kd_cfg["alpha_kl"] * kl_loss
                else:
                    loss = ce_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        # 每5个epoch评估一次
        if epoch % 5 == 0 or epoch == train_cfg["epochs"]:
            val_metrics = evaluate_detailed(student, val_loader, device)
            val_f1 = val_metrics["macro_f1"]

            history.append({
                "epoch": epoch,
                "train_loss": avg_loss,
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_f1
            })

            print(f"  E{epoch}: loss={avg_loss:.4f}, val_acc={val_metrics['accuracy']:.4f}, "
                  f"val_f1={val_f1:.4f}")

            # 保存最佳模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save({
                    "model_state_dict": student.state_dict(),
                    "epoch": epoch,
                    "val_f1": val_f1
                }, output_dir / "best_model.pth")

    # 加载最佳模型并测试
    best_ckpt = torch.load(output_dir / "best_model.pth", weights_only=False)
    student.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate_detailed(student, test_loader, device)

    print(f"\n🎉 训练完成! 最佳验证 F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"测试结果: acc={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}")

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "history": history
    }

def main():
    parser = argparse.ArgumentParser(description="STEP 3 & 4: KD Ablation + Data Efficiency")
    parser.add_argument("--student", type=str, required=True,
                       help="Student model name (e.g., repvit_m0_9, mobilenetv3_small_050)")
    parser.add_argument("--kd_config", type=str, required=True, choices=list(KD_CONFIGS.keys()),
                       help="KD configuration: ce_only or ce_kl")
    parser.add_argument("--train_fraction", type=float, default=1.0,
                       help="Fraction of training data to use (1.0, 0.5, 0.2)")
    parser.add_argument("--gpu", type=int, default=0,
                       help="GPU ID to use")
    parser.add_argument("--output_dir", type=str, default="results/kd_ablation",
                       help="Output directory for results")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = PROJECT_ROOT / args.output_dir / f"{args.student}_{args.kd_config}_frac{args.train_fraction}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据集
    print(f"\n加载数据集 (训练数据比例: {args.train_fraction})...")
    train_loader, val_loader, test_loader, dataset_info = quick_load_dataset(
        CONFIG["data_path"],
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )

    # 子采样训练数据
    if args.train_fraction < 1.0:
        train_dataset = train_loader.dataset
        subsampled_dataset = subsample_dataset(train_dataset, args.train_fraction)
        train_loader = torch.utils.data.DataLoader(
            subsampled_dataset,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=CONFIG["num_workers"],
            pin_memory=True
        )
        print(f"训练集大小: {len(train_dataset)} -> {len(subsampled_dataset)}")

    num_classes = dataset_info["num_classes"]

    # 加载教师模型（仅在需要KD时）
    teachers, stacking = None, None
    if args.kd_config != "ce_only":
        print("\n加载教师模型...")
        teachers = load_teachers(CONFIG, num_classes, device)
        stacking = load_stacking(CONFIG, len(teachers), num_classes, device)
        print(f"加载了 {len(teachers)} 个教师模型")

    # 创建学生模型
    print(f"\n创建学生模型: {args.student}")
    student = create_student_model(args.student, num_classes=num_classes, in_chans=1)
    student = student.to(device)

    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"学生模型参数量: {params:.2f}M")

    # 训练
    print(f"\n开始训练 (KD配置: {args.kd_config})...")
    results = train_student(
        student, teachers, stacking,
        train_loader, val_loader, test_loader,
        args.kd_config,
        CONFIG,
        device,
        output_dir
    )

    # 保存结果
    output_json = {
        "experiment_type": "kd_ablation_data_efficiency",
        "student_name": args.student,
        "kd_config_id": args.kd_config,
        "kd_config_name": KD_CONFIGS[args.kd_config]["name"],
        "train_fraction": args.train_fraction,
        "params_m": params,
        "test_acc": results["test_metrics"]["accuracy"],
        "macro_f1": results["test_metrics"]["macro_f1"],
        "precision": results["test_metrics"]["precision"],
        "recall": results["test_metrics"]["recall"],
        "best_val_f1": results["best_val_f1"],
        "best_epoch": results["best_epoch"],
        "config_summary": {
            "epochs": CONFIG["training"]["epochs"],
            "batch_size": CONFIG["batch_size"],
            "learning_rate": CONFIG["training"]["base_lr"]
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(output_json, f, indent=2)

    print(f"\n✅ 结果已保存到: {output_dir}/metrics.json")
    print(json.dumps(output_json, indent=2))

if __name__ == "__main__":
    main()
