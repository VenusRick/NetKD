#!/usr/bin/env python3
"""
完整数据集实验 - 全量+不同比例(100%/75%/50%/25%)
================================================
使用最佳教师组合进行知识蒸馏，在所有数据集上运行
支持数据效率实验（不同训练数据比例）

用法:
    CUDA_VISIBLE_DEVICES=0,1,2 python scripts/run_complete_experiment.py
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import random

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

# ============================================================================
# 配置
# ============================================================================

CONFIG = {
    "datasets": [
        "ISCXVPN2016",
        "ISCXTor2016",
        "USTC-TFC2016",
        "CICIoT2022",
        "CrossPlatform-Android",
        "CrossPlatform-iOS",
    ],
    "data_ratios": [1.0, 0.75, 0.50, 0.25],  # 全量和不同比例
    "dataset_root": "/workspace/yqm/Dataset",
    "output_root": "/workspace/yqm/NetKD/results/complete_experiment",
    "num_runs": 3,
    "batch_size": 128,
    "num_workers": 8,
    # 教师训练配置
    "teacher": {
        "epochs": 25,
        "lr": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 2,
        "early_stop_patience": 5,
    },
    # Stacking配置
    "stacking": {
        "epochs": 12,
        "lr": 0.0005,
        "hidden_dim": 384,
    },
    # 学生蒸馏配置
    "student": {
        "epochs": 25,
        "lr": 0.0003,
        "temperature": 3.0,
        "alpha_ce": 0.5,
        "alpha_kl": 0.5,
    },
    "teacher_combo": ["densenet121", "mobilenetv3", "convnextv2_tiny"],
}

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def load_dataset_with_ratio(dataset_name: str, batch_size: int, num_workers: int, ratio: float = 1.0):
    """加载数据集，支持按比例采样"""
    from data_preprocessing.image_loader import quick_load_dataset
    
    train_loader, val_loader, test_loader, info = quick_load_dataset(
        dataset_name=dataset_name,
        dataset_root=CONFIG["dataset_root"],
        batch_size=batch_size,
        num_workers=num_workers,
        to_grayscale=True,
    )
    
    # 如果不是全量数据，进行采样
    if ratio < 1.0:
        train_dataset = train_loader.dataset
        n_samples = int(len(train_dataset) * ratio)
        indices = random.sample(range(len(train_dataset)), n_samples)
        sampled_dataset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            sampled_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        info["train_samples_used"] = n_samples
        info["data_ratio"] = ratio
    else:
        info["train_samples_used"] = info.get("train_samples", len(train_loader.dataset))
        info["data_ratio"] = 1.0
    
    return train_loader, val_loader, test_loader, info

def create_teacher_model(name: str, num_classes: int):
    """创建教师模型"""
    from models.teacher_models import (
        DenseNet121Teacher, MobileNetV3LargeTeacher, ConvNeXtV2TinyTeacher,
        ResNet50Teacher, EfficientNetV2STeacher
    )
    
    teacher_map = {
        "densenet121": DenseNet121Teacher,
        "mobilenetv3": MobileNetV3LargeTeacher,
        "convnextv2_tiny": ConvNeXtV2TinyTeacher,
        "resnet50": ResNet50Teacher,
        "efficientnetv2": EfficientNetV2STeacher,
    }
    
    if name not in teacher_map:
        raise ValueError(f"Unknown teacher: {name}")
    
    model = teacher_map[name](num_classes=num_classes, use_eca=True, pretrained=False)
    return model

def create_student_model(num_classes: int):
    """创建学生模型 (RepViT)"""
    from models.student_models_v2 import create_lightweight_student
    return create_lightweight_student("repvit_m0_9", num_classes=num_classes, in_channels=1)

class StackingEnsemble(nn.Module):
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(num_teachers * num_classes, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, teacher_logits):
        # teacher_logits: (B, num_teachers, num_classes)
        x = teacher_logits.flatten(1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)

def evaluate_model(model, test_loader, device) -> Dict:
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    return {
        "accuracy": float(np.mean(all_preds == all_labels)),
        "f1_macro": float(f1_score(all_labels, all_preds, average='macro')),
        "precision": float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        "recall": float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
    }

def train_teacher(model, train_loader, val_loader, config, device):
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    best_acc = 0.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(config["epochs"]):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config["early_stop_patience"]:
            print(f"    Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_acc

def train_stacking(teachers, train_loader, val_loader, num_classes, config, device):
    stacking = StackingEnsemble(len(teachers), num_classes, config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(stacking.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    for t in teachers:
        t.eval()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(config["epochs"]):
        stacking.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = torch.stack([t(x) for t in teachers], dim=1)
            
            optimizer.zero_grad()
            logits = stacking(teacher_logits)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
        
        # 验证
        stacking.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                teacher_logits = torch.stack([t(x) for t in teachers], dim=1)
                preds = stacking(teacher_logits).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in stacking.state_dict().items()}
    
    stacking.load_state_dict(best_state)
    return stacking, best_acc

def train_student_kd(student, teachers, stacking, train_loader, val_loader, config, device):
    student = student.to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"])
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    
    T = config["temperature"]
    alpha_ce = config["alpha_ce"]
    alpha_kl = config["alpha_kl"]
    
    for t in teachers:
        t.eval()
    stacking.eval()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(config["epochs"]):
        student.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                teacher_logits = torch.stack([t(x) for t in teachers], dim=1)
                teacher_soft = stacking(teacher_logits)
            
            optimizer.zero_grad()
            student_logits = student(x)
            
            ce_loss = ce_criterion(student_logits, y)
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
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = student(x).argmax(dim=-1)
                correct += (preds == y).sum().item()
                total += y.numel()
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in student.state_dict().items()}
    
    student.load_state_dict(best_state)
    return student, best_acc

def run_single_experiment(dataset_name: str, ratio: float, run_id: int, output_dir: Path, device):
    """运行单次实验"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Ratio: {int(ratio*100)}% | Run: {run_id}")
    print(f"{'='*60}")
    
    results = {
        "dataset": dataset_name,
        "data_ratio": ratio,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
    }
    
    # 加载数据
    print("Loading dataset...")
    train_loader, val_loader, test_loader, info = load_dataset_with_ratio(
        dataset_name, CONFIG["batch_size"], CONFIG["num_workers"], ratio
    )
    num_classes = info["num_classes"]
    results["num_classes"] = num_classes
    results["train_samples_used"] = info["train_samples_used"]
    
    # 阶段1: 训练教师
    print("\n[Phase 1] Training Teachers...")
    teachers = []
    teacher_results = {}
    
    for teacher_name in CONFIG["teacher_combo"]:
        print(f"  Training {teacher_name}...")
        try:
            teacher = create_teacher_model(teacher_name, num_classes)
            teacher, val_acc = train_teacher(teacher, train_loader, val_loader, CONFIG["teacher"], device)
            test_metrics = evaluate_model(teacher, test_loader, device)
            teachers.append(teacher)
            teacher_results[teacher_name] = {
                "val_acc": float(val_acc),
                "test_acc": test_metrics["accuracy"],
                "test_f1": test_metrics["f1_macro"],
            }
            print(f"    test_acc={test_metrics['accuracy']:.4f}, f1={test_metrics['f1_macro']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue
    
    results["teachers"] = teacher_results
    
    if len(teachers) == 0:
        print("No teachers trained successfully!")
        return results
    
    # 阶段2: Stacking
    print("\n[Phase 2] Training Stacking...")
    stacking, stacking_val = train_stacking(
        teachers, train_loader, val_loader, num_classes, CONFIG["stacking"], device
    )
    
    # 评估Stacking
    stacking.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            teacher_logits = torch.stack([t(x) for t in teachers], dim=1)
            preds = stacking(teacher_logits).argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    stacking_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    stacking_f1 = f1_score(all_labels, all_preds, average='macro')
    results["stacking"] = {"test_acc": float(stacking_acc), "test_f1": float(stacking_f1)}
    print(f"  Stacking: acc={stacking_acc:.4f}, f1={stacking_f1:.4f}")
    
    # 阶段3: 知识蒸馏
    print("\n[Phase 3] Knowledge Distillation...")
    student = create_student_model(num_classes)
    student, student_val = train_student_kd(
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
    print(f"  Student: acc={student_metrics['accuracy']:.4f}, f1={student_metrics['f1_macro']:.4f}")
    
    # 保存结果
    ratio_str = f"ratio_{int(ratio*100)}"
    run_dir = output_dir / dataset_name / ratio_str / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    torch.save(student.state_dict(), run_dir / "student.pth")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--ratios", nargs="+", type=float, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    datasets = args.datasets or CONFIG["datasets"]
    ratios = args.ratios or CONFIG["data_ratios"]
    num_runs = args.runs
    
    output_dir = Path(CONFIG["output_root"]) / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for dataset_name in datasets:
        for ratio in ratios:
            for run_id in range(1, num_runs + 1):
                try:
                    results = run_single_experiment(dataset_name, ratio, run_id, output_dir, device)
                    all_results.append(results)
                except Exception as e:
                    print(f"ERROR: {dataset_name} ratio={ratio} run={run_id}: {e}")
                    import traceback
                    traceback.print_exc()
    
    # 汇总
    summary = {}
    for ds in datasets:
        summary[ds] = {}
        for ratio in ratios:
            ds_results = [r for r in all_results if r["dataset"] == ds and r.get("data_ratio") == ratio]
            if ds_results and all("student" in r for r in ds_results):
                accs = [r["student"]["test_acc"] for r in ds_results if "student" in r]
                f1s = [r["student"]["test_f1"] for r in ds_results if "student" in r]
                if accs:
                    summary[ds][f"ratio_{int(ratio*100)}"] = {
                        "mean_acc": float(np.mean(accs)),
                        "std_acc": float(np.std(accs)),
                        "mean_f1": float(np.mean(f1s)),
                        "std_f1": float(np.std(f1s)),
                        "runs": len(accs),
                    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for ds, ratios_data in summary.items():
        print(f"\n{ds}:")
        for ratio_key, stats in ratios_data.items():
            print(f"  {ratio_key}: Acc={stats['mean_acc']:.4f}±{stats['std_acc']:.4f}, F1={stats['mean_f1']:.4f}")
    
    print(f"\nResults: {output_dir}")

if __name__ == "__main__":
    main()
