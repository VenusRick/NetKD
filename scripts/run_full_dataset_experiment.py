#!/usr/bin/env python3
"""
全量数据集实验脚本 - 三轮实验
===========================
使用最佳教师组合 (ConvNeXtV2-Tiny + DenseNet121 + MobileNetV3) 进行知识蒸馏
在所有数据集上运行三轮完整流程

用法:
    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/run_full_dataset_experiment.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    "datasets": [
        "USTC-TFC2016",
        "ISCXTor2016", 
        "CICIoT2022",
        "CrossPlatform-Android",
        "CrossPlatform-iOS",
        "ISCXVPN2016",
    ],
    "dataset_root": "/workspace/yqm/Dataset",
    "output_root": "/workspace/yqm/NetKD/results/full_dataset_experiment",
    "num_runs": 3,  # 三轮实验
    "batch_size": 128,
    "num_workers": 8,
    # 教师训练配置
    "teacher": {
        "epochs": 30,  # 收敛后不需要太多
        "lr": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 3,
        "early_stop_patience": 5,
    },
    # Stacking配置
    "stacking": {
        "epochs": 15,
        "lr": 0.0005,
        "hidden_dim": 384,
    },
    # 学生蒸馏配置
    "student": {
        "epochs": 30,  
        "lr": 0.0003,
        "temperature": 3.0,
        "alpha_ce": 0.5,
        "alpha_kl": 0.5,
    },
    # 最佳教师组合: replace_resnet
    "teacher_combo": ["convnextv2_tiny", "densenet121", "mobilenetv3"],
}

# ============================================================================
# 工具函数
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def create_teacher_model(name: str, num_classes: int, use_eca: bool = True):
    """创建教师模型"""
    from models.teacher_models import (
        DenseNet121Teacher, MobileNetV3LargeTeacher, ConvNeXtV2TinyTeacher
    )
    
    teacher_classes = {
        "densenet121": DenseNet121Teacher,
        "mobilenetv3": MobileNetV3LargeTeacher,
        "convnextv2_tiny": ConvNeXtV2TinyTeacher,
    }
    
    if name not in teacher_classes:
        raise ValueError(f"Unknown teacher: {name}")
    
    # 不使用ImageNet预训练，使用随机初始化
    return teacher_classes[name](num_classes=num_classes, pretrained=False, use_eca=use_eca)

def create_student_model(num_classes: int):
    """创建学生模型"""
    from models.student_models import StudentNet
    return StudentNet(num_classes=num_classes)

def load_dataset(dataset_name: str, batch_size: int, num_workers: int):
    """加载数据集"""
    from data_preprocessing import quick_load_dataset
    
    train_loader, val_loader, test_loader, info = quick_load_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, info

def train_teacher(model, train_loader, val_loader, config, device):
    """训练单个教师模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        # 训练
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        scheduler.step()
        
        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config.get("early_stop_patience", 5):
            print(f"  Early stop at epoch {epoch+1}")
            break
            
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f} (best={best_acc:.4f})")
    
    model.load_state_dict(best_state)
    return model, best_acc

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    acc = (all_preds == all_labels).mean()
    f1 = f1_score(all_labels, all_preds, average='macro')
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1),
        "precision": float(precision),
        "recall": float(recall),
    }

def train_stacking(teachers, train_loader, val_loader, num_classes, config, device):
    """训练Stacking模型"""
    from models.teacher_models import StackingModel
    
    # 收集教师特征
    for t in teachers:
        t.eval()
    
    hidden_dim = config.get("hidden_dim", 384)
    num_teachers = len(teachers)
    stacking = StackingModel(num_classes=num_classes, hidden_dim=hidden_dim).to(device)
    
    optimizer = torch.optim.AdamW(stacking.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(config["epochs"]):
        stacking.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # 获取教师logits
            teacher_logits = []
            with torch.no_grad():
                for t in teachers:
                    teacher_logits.append(t(x))
            teacher_logits = torch.stack(teacher_logits, dim=1)  # [B, num_teachers, num_classes]
            
            optimizer.zero_grad()
            logits = stacking(teacher_logits)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # 验证
        stacking.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                teacher_logits = []
                for t in teachers:
                    teacher_logits.append(t(x))
                teacher_logits = torch.stack(teacher_logits, dim=1)
                logits = stacking(teacher_logits)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in stacking.state_dict().items()}
    
    stacking.load_state_dict(best_state)
    return stacking, best_acc

def train_student_kd(student, teachers, stacking, train_loader, val_loader, config, device):
    """使用知识蒸馏训练学生模型"""
    student = student.to(device)
    for t in teachers:
        t.eval()
    stacking.eval()
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    T = config.get("temperature", 3.0)
    alpha_ce = config.get("alpha_ce", 0.5)
    alpha_kl = config.get("alpha_kl", 0.5)
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(config["epochs"]):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            # 获取教师soft labels
            with torch.no_grad():
                teacher_logits = []
                for t in teachers:
                    teacher_logits.append(t(x))
                teacher_logits_stacked = torch.stack(teacher_logits, dim=1)
                teacher_soft = stacking(teacher_logits_stacked)
            
            optimizer.zero_grad()
            student_logits = student(x)
            
            # CE loss (hard labels)
            ce_loss = ce_criterion(student_logits, y)
            
            # KL loss (soft labels)
            student_soft = torch.log_softmax(student_logits / T, dim=-1)
            teacher_soft_prob = torch.softmax(teacher_soft / T, dim=-1)
            kl_loss = kl_criterion(student_soft, teacher_soft_prob) * (T * T)
            
            loss = alpha_ce * ce_loss + alpha_kl * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        # 验证
        student.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = student(x)
                preds = logits.argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
        
        if (epoch + 1) % 5 == 0:
            print(f"  KD Epoch {epoch+1}: val_acc={val_acc:.4f} (best={best_acc:.4f})")
    
    student.load_state_dict(best_state)
    return student, best_acc

# ============================================================================
# 主实验函数
# ============================================================================

def run_single_experiment(dataset_name: str, run_id: int, output_dir: Path, device):
    """运行单次完整实验"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Run: {run_id}")
    print(f"{'='*60}")
    
    results = {
        "dataset": dataset_name,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 加载数据
    print("Loading dataset...")
    train_loader, val_loader, test_loader, info = load_dataset(
        dataset_name, CONFIG["batch_size"], CONFIG["num_workers"]
    )
    num_classes = info["num_classes"]
    results["num_classes"] = num_classes
    results["train_samples"] = info.get("train_samples", "N/A")
    
    # 阶段1: 训练教师模型
    print("\n[Phase 1] Training Teachers...")
    teachers = []
    teacher_results = {}
    
    for teacher_name in CONFIG["teacher_combo"]:
        print(f"  Training {teacher_name}...")
        teacher = create_teacher_model(teacher_name, num_classes)
        teacher, val_acc = train_teacher(teacher, train_loader, val_loader, CONFIG["teacher"], device)
        test_metrics = evaluate_model(teacher, test_loader, device)
        teachers.append(teacher)
        teacher_results[teacher_name] = {
            "val_acc": val_acc,
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1_macro"],
            "params_m": count_parameters(teacher),
        }
        print(f"  {teacher_name}: test_acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1_macro']:.4f}")
    
    results["teachers"] = teacher_results
    
    # 阶段2: 训练Stacking
    print("\n[Phase 2] Training Stacking Ensemble...")
    stacking, stacking_val_acc = train_stacking(
        teachers, train_loader, val_loader, num_classes, CONFIG["stacking"], device
    )
    
    # 评估Stacking
    stacking.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            teacher_logits = []
            for t in teachers:
                teacher_logits.append(t(x))
            teacher_logits = torch.stack(teacher_logits, dim=1)
            logits = stacking(teacher_logits)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    stacking_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    stacking_f1 = f1_score(all_labels, all_preds, average='macro')
    results["stacking"] = {"test_acc": float(stacking_acc), "test_f1": float(stacking_f1)}
    print(f"  Stacking: test_acc={stacking_acc:.4f}, f1={stacking_f1:.4f}")
    
    # 阶段3: 知识蒸馏
    print("\n[Phase 3] Knowledge Distillation...")
    student = create_student_model(num_classes)
    student, student_val_acc = train_student_kd(
        student, teachers, stacking, train_loader, val_loader, CONFIG["student"], device
    )
    
    student_metrics = evaluate_model(student, test_loader, device)
    results["student"] = {
        "test_acc": student_metrics["accuracy"],
        "test_f1": student_metrics["f1_macro"],
        "test_precision": student_metrics["precision"],
        "test_recall": student_metrics["recall"],
        "params_m": count_parameters(student),
    }
    print(f"  Student: test_acc={student_metrics['accuracy']:.4f}, f1={student_metrics['f1_macro']:.4f}")
    
    # 保存结果
    run_dir = output_dir / dataset_name / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # 保存模型权重
    for i, (name, teacher) in enumerate(zip(CONFIG["teacher_combo"], teachers)):
        torch.save(teacher.state_dict(), run_dir / f"{name}_teacher.pth")
    torch.save(stacking.state_dict(), run_dir / "stacking.pth")
    torch.save(student.state_dict(), run_dir / "student.pth")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None, help="Specific datasets to run")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per dataset")
    parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    datasets = args.datasets or CONFIG["datasets"]
    num_runs = args.runs
    
    output_dir = Path(CONFIG["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dataset_name in datasets:
        for run_id in range(1, num_runs + 1):
            try:
                results = run_single_experiment(dataset_name, run_id, output_dir, device)
                all_results.append(results)
            except Exception as e:
                print(f"ERROR in {dataset_name} run {run_id}: {e}")
                import traceback
                traceback.print_exc()
    
    # 汇总结果
    summary = {}
    for dataset_name in datasets:
        dataset_results = [r for r in all_results if r["dataset"] == dataset_name]
        if dataset_results:
            accs = [r["student"]["test_acc"] for r in dataset_results]
            f1s = [r["student"]["test_f1"] for r in dataset_results]
            summary[dataset_name] = {
                "mean_acc": float(np.mean(accs)),
                "std_acc": float(np.std(accs)),
                "mean_f1": float(np.mean(f1s)),
                "std_f1": float(np.std(f1s)),
                "runs": len(dataset_results),
            }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    for ds, stats in summary.items():
        print(f"{ds}: Acc={stats['mean_acc']:.4f}±{stats['std_acc']:.4f}, F1={stats['mean_f1']:.4f}±{stats['std_f1']:.4f}")
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
