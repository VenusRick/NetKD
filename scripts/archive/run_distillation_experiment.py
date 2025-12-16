#!/usr/bin/env python3
"""
知识蒸馏实验脚本 - 稳定版

使用最佳教师组合 (replace_resnet) 进行知识蒸馏:
- 教师: ConvNeXtV2-Tiny (ECA) + DenseNet121 (ECA) + MobileNetV3 (ECA)
- Stacking 模型: stacking_replace_resnet
- 学生: StudentNet (轻量级CNN)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset
from models.teacher_models import (DynamicStackingModel,
    ConvNeXtV2TinyTeacher,
    DenseNet121Teacher,
    MobileNetV3LargeTeacher,
)
from models.student_model import StudentNet



EXPERIMENT_CONFIG = {
    "dataset": "ISCXVPN2016",
    "batch_size": 128,
    "num_workers": 8,
    
    "teachers": {
        "convnextv2_tiny": {
            "class": "ConvNeXtV2TinyTeacher",
            "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/convnextv2_tiny_eca_best.pth",
        },
        "densenet121": {
            "class": "DenseNet121Teacher",
            "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/densenet121_eca_best.pth",
        },
        "mobilenetv3": {
            "class": "MobileNetV3LargeTeacher",
            "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/mobilenetv3_eca_best.pth",
        },
    },
    
    "stacking": {
        "checkpoint": "results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth",
        "hidden_dim": 384,
    },
    
    "student": {
        "epochs": 100,
        "base_lr": 0.0003,
        "weight_decay": 1e-4,
        "warmup_epochs": 10,
        "label_smoothing": 0.1,
        "grad_clip": 0.5,
    },
    
    "distillation": {
        "temperature": 3.0,
        "alpha_ce": 0.5,
        "alpha_kl": 0.5,
    },
}


def create_dataloaders(dataset_name, batch_size, num_workers=8):
    train_loader, val_loader, test_loader, info = quick_load_dataset(
        dataset_name, batch_size=batch_size, num_workers=num_workers)
    num_classes = info["num_classes"]
    print(f"Dataset: {dataset_name}")
    print(f"  Train: {info['train_size']}, Val: {info['val_size']}, Test: {info['test_size']}")
    print(f"  Classes: {num_classes}, Batch size: {batch_size}")
    return train_loader, val_loader, test_loader, num_classes


def load_teachers(config, num_classes, device):
    teacher_classes = {
        "ConvNeXtV2TinyTeacher": ConvNeXtV2TinyTeacher,
        "DenseNet121Teacher": DenseNet121Teacher,
        "MobileNetV3LargeTeacher": MobileNetV3LargeTeacher,
    }
    teachers = []
    for name, cfg in config["teachers"].items():
        model_class = teacher_classes[cfg["class"]]
        model = model_class(num_classes=num_classes, use_eca=cfg["use_eca"])
        ckpt_path = PROJECT_ROOT / cfg["checkpoint"]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"✓ Loaded {name}")
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        teachers.append(model)
    return teachers


def load_stacking(config, num_teachers, num_classes, device):
    stacking = DynamicStackingModel(num_classes, num_teachers, config["stacking"]["hidden_dim"])
    ckpt_path = PROJECT_ROOT / config["stacking"]["checkpoint"]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        stacking.load_state_dict(ckpt["model_state_dict"])
        print(f"✓ Loaded stacking")
    stacking = stacking.to(device)
    stacking.eval()
    for param in stacking.parameters():
        param.requires_grad = False
    return stacking


def compute_distillation_loss(student_logits, teacher_logits, labels, T, alpha_ce, alpha_kl, ls):
    ce_loss = F.cross_entropy(student_logits, labels, label_smoothing=ls)
    student_soft = F.log_softmax(student_logits / T, dim=-1)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)
    kl_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (T ** 2)
    total_loss = alpha_ce * ce_loss + alpha_kl * kl_loss
    return total_loss, {"ce": ce_loss.item(), "kl": kl_loss.item(), "total": total_loss.item()}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        _, preds = logits.max(1)
        correct += preds.eq(y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def compute_f1(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        _, preds = logits.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.numpy())
    return f1_score(all_labels, all_preds, average='macro')


def train_student(student, teachers, stacking, train_loader, val_loader, test_loader, config, device, output_dir):
    student_cfg = config["student"]
    dist_cfg = config["distillation"]
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=student_cfg["base_lr"], weight_decay=student_cfg["weight_decay"])
    
    total_steps = student_cfg["epochs"] * len(train_loader)
    warmup_steps = student_cfg["warmup_epochs"] * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    result = {"train_losses": [], "val_losses": [], "val_accs": [], "best_epoch": 0, "best_val_acc": 0.0}
    best_val_acc = 0.0
    checkpoint_path = output_dir / "student_distilled_best.pth"
    
    print(f"\n{'='*60}")
    print(f"Starting Distillation: Epochs={student_cfg['epochs']}, LR={student_cfg['base_lr']}, T={dist_cfg['temperature']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    step = 0
    
    for epoch in range(1, student_cfg["epochs"] + 1):
        student.train()
        total_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{student_cfg['epochs']}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                with torch.no_grad():
                    t_logits = [t(x) for t in teachers]
                    teacher_out = stacking(*t_logits)
                student_out = student(x)
                loss, _ = compute_distillation_loss(
                    student_out, teacher_out, y,
                    dist_cfg["temperature"], dist_cfg["alpha_ce"], dist_cfg["alpha_kl"],
                    student_cfg["label_smoothing"]
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), student_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            
            total_loss += loss.item() * x.size(0)
            _, preds = student_out.max(1)
            correct += preds.eq(y).sum().item()
            total += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
        
        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(student, val_loader, device)
        
        result["train_losses"].append(train_loss)
        result["val_losses"].append(val_loss)
        result["val_accs"].append(val_acc)
        
        print(f"Epoch {epoch} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            result["best_epoch"] = epoch
            result["best_val_acc"] = best_val_acc
            torch.save({"epoch": epoch, "model_state_dict": student.state_dict(), "val_acc": val_acc}, checkpoint_path)
            print(f"✓ Saved best model (val_acc={val_acc:.4f})")
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    
    test_loss, test_acc = evaluate(student, test_loader, device)
    test_f1 = compute_f1(student, test_loader, device)
    
    result["test_accuracy"] = test_acc
    result["test_f1"] = test_f1
    result["training_time"] = time.time() - start_time
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Epoch: {result['best_epoch']}")
    print(f"Test Acc: {test_acc:.4f}, F1-macro: {test_f1:.4f}")
    print(f"Time: {result['training_time']/60:.1f} min")
    print(f"{'='*60}")
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/distillation_bs128")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    config = {k: v.copy() if isinstance(v, dict) else v for k, v in EXPERIMENT_CONFIG.items()}
    if args.epochs:
        config["student"]["epochs"] = args.epochs
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        config["dataset"], config["batch_size"], config["num_workers"])
    
    print("\nLoading Teachers...")
    teachers = load_teachers(config, num_classes, device)
    stacking = load_stacking(config, len(teachers), num_classes, device)
    
    print("\nCreating Student...")
    student = StudentNet(num_classes=num_classes).to(device)
    
    student_params = sum(p.numel() for p in student.parameters()) / 1e6
    teacher_params = sum(sum(p.numel() for p in t.parameters()) for t in teachers) / 1e6
    print(f"Student: {student_params:.2f}M, Teachers: {teacher_params:.2f}M, Compression: {teacher_params/student_params:.1f}x")
    
    result = train_student(student, teachers, stacking, train_loader, val_loader, test_loader, config, device, output_dir)
    
    result_path = output_dir / "distillation_result.json"
    with open(result_path, "w") as f:
        json.dump({
            "test_accuracy": result["test_accuracy"],
            "test_f1": result["test_f1"],
            "best_epoch": result["best_epoch"],
            "best_val_acc": result["best_val_acc"],
            "training_time": result["training_time"],
        }, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
