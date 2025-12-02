#!/usr/bin/env python3
"""
教师模型搜索完整实验脚本
========================
Phase 2: 教师微调 + ECA消融实验
Phase 3: Stacking组合实验
Phase 4: 结果分析与对比

Usage:
    python scripts/run_teacher_search_experiment.py --phase all --dataset ISCXVPN2016
    python scripts/run_teacher_search_experiment.py --phase 2 --dataset ISCXVPN2016
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================================
# 实验配置
# ============================================================================

EXPERIMENT_CONFIG = {
    # 微调配置
    "finetune": {
        "epochs": 50,
        "batch_size": 128,
        "base_lr": 0.001,
        "weight_decay": 0.0001,
        "warmup_epochs": 5,
        "label_smoothing": 0.05,
        "optimizer": "adamw",
        "scheduler": "cosine",
    },
    # Stacking配置
    "stacking": {
        "epochs": 30,
        "batch_size": 128,
        "base_lr": 0.0005,
        "weight_decay": 0.0001,
        "warmup_epochs": 3,
        "label_smoothing": 0.05,
        "hidden_dim": 384,
    },
    # 数据加载配置
    "dataloader": {
        "num_workers": 8,
        "pin_memory": True,
    },
    # 教师模型列表
    "teachers": {
        "resnet50": {"class": "ResNet50Teacher", "pretrained": True},
        "densenet121": {"class": "DenseNet121Teacher", "pretrained": True},
        "mobilenetv3": {"class": "MobileNetV3LargeTeacher", "pretrained": True},
        "convnextv2_tiny": {"class": "ConvNeXtV2TinyTeacher", "pretrained": True},
        "efficientnetv2_s": {"class": "EfficientNetV2STeacher", "pretrained": True},
    },
    # ECA消融实验配置
    "eca_ablation": {
        "teachers_to_test": ["resnet50", "densenet121", "mobilenetv3", "convnextv2_tiny", "efficientnetv2_s"],
    },
    # Stacking组合配置
    "stacking_sets": {
        "baseline": ["resnet50", "densenet121", "mobilenetv3"],
        "replace_resnet": ["convnextv2_tiny", "densenet121", "mobilenetv3"],
        "replace_mobilenet": ["resnet50", "densenet121", "efficientnetv2_s"],
        "new_trio": ["convnextv2_tiny", "densenet121", "efficientnetv2_s"],
        "quad_ensemble": ["resnet50", "densenet121", "convnextv2_tiny", "efficientnetv2_s"],
        "full_ensemble": ["resnet50", "densenet121", "mobilenetv3", "convnextv2_tiny", "efficientnetv2_s"],
    },
}

# ============================================================================
# 工具函数
# ============================================================================

def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def create_teacher_model(name: str, num_classes: int, use_eca: bool = False, pretrained: bool = True):
    """根据名称创建教师模型"""
    from models.teacher_models import (
        ResNet50Teacher, DenseNet121Teacher, MobileNetV3LargeTeacher,
        ConvNeXtV2TinyTeacher, EfficientNetV2STeacher
    )
    
    teacher_classes = {
        "resnet50": ResNet50Teacher,
        "densenet121": DenseNet121Teacher,
        "mobilenetv3": MobileNetV3LargeTeacher,
        "convnextv2_tiny": ConvNeXtV2TinyTeacher,
        "efficientnetv2_s": EfficientNetV2STeacher,
    }
    
    if name not in teacher_classes:
        raise ValueError(f"Unknown teacher: {name}")
    
    return teacher_classes[name](num_classes=num_classes, pretrained=pretrained, use_eca=use_eca)

def create_dataloaders(dataset_name: str, batch_size: int, num_workers: int = 8):
    """创建数据加载器"""
    from data_preprocessing import quick_load_dataset
    
    train_loader, val_loader, test_loader, info = quick_load_dataset(
        dataset_name=dataset_name,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    
    num_classes = info["num_classes"]
    print(f"Dataset: {dataset_name}")
    print(f"Train: {info.get('train_samples', 'N/A')}, Val: {info.get('val_samples', 'N/A')}, Test: {info.get('test_samples', 'N/A')}")
    print(f"Number of classes: {num_classes}")
    
    return train_loader, val_loader, test_loader, num_classes

def create_optimizer(model: nn.Module, config: dict):
    """创建优化器"""
    lr = config.get("base_lr", 0.001)
    weight_decay = config.get("weight_decay", 0.0001)
    opt_name = config.get("optimizer", "adamw").lower()
    
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

def create_scheduler(optimizer, config: dict, steps_per_epoch: int):
    """创建学习率调度器 (线性warmup + cosine decay)"""
    epochs = config.get("epochs", 50)
    warmup_epochs = config.get("warmup_epochs", 5)
    scheduler_type = config.get("scheduler", "cosine")
    
    total_steps = epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch
    
    if scheduler_type == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps - warmup_steps
        )
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, [warmup_scheduler, main_scheduler], [warmup_steps]
            )
        return main_scheduler
    else:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.1)

# ============================================================================
# 训练函数
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    """训练一个epoch"""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
    
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    
    for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

# ============================================================================
# Phase 2: 教师微调 + ECA消融
# ============================================================================

def run_teacher_finetune(
    teacher_name: str,
    dataset_name: str,
    output_dir: Path,
    device: torch.device,
    use_eca: bool = False,
    resume: str = None,
) -> dict:
    """微调单个教师模型"""
    config = EXPERIMENT_CONFIG["finetune"]
    dl_config = EXPERIMENT_CONFIG["dataloader"]
    
    print(f"\n{'='*60}")
    print(f"Fine-tuning Teacher: {teacher_name} (ECA={use_eca})")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        dataset_name, config["batch_size"], dl_config["num_workers"]
    )
    
    # 创建模型
    model = create_teacher_model(teacher_name, num_classes, use_eca=use_eca, pretrained=True)
    model = model.to(device)
    params = count_parameters(model)
    print(f"Model parameters: {params:.2f}M")
    
    # 创建优化器和调度器
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    scaler = GradScaler()
    
    # 训练记录
    suffix = "_eca" if use_eca else "_noeca"
    checkpoint_path = output_dir / f"{teacher_name}{suffix}_best.pth"
    
    result = {
        "teacher_name": teacher_name,
        "use_eca": use_eca,
        "params_millions": params,
        "train_losses": [],
        "val_losses": [],
        "val_accs": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
    }
    
    # 恢复训练
    start_epoch = 0
    best_val_acc = 0.0
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_val_acc = ckpt.get("best_val_acc", 0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    
    # 训练循环
    start_time = time.time()
    for epoch in range(start_epoch, config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, device
        )
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        result["train_losses"].append(train_loss)
        result["val_losses"].append(val_metrics["loss"])
        result["val_accs"].append(val_metrics["accuracy"])
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            result["best_epoch"] = epoch + 1
            result["best_val_acc"] = best_val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "teacher_name": teacher_name,
                "use_eca": use_eca,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_acc={best_val_acc:.4f})")
    
    result["training_time_seconds"] = time.time() - start_time
    result["checkpoint_path"] = str(checkpoint_path)
    
    # 测试集评估
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    result["test_accuracy"] = test_metrics["accuracy"]
    result["test_f1_macro"] = test_metrics["f1_macro"]
    result["test_f1_weighted"] = test_metrics["f1_weighted"]
    
    print(f"\n=== Final Results for {teacher_name} (ECA={use_eca}) ===")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"F1-macro: {test_metrics['f1_macro']:.4f}, F1-weighted: {test_metrics['f1_weighted']:.4f}")
    
    return result

def run_phase2_eca_ablation(dataset_name: str, output_dir: Path, device: torch.device):
    """Phase 2: 教师微调 + ECA消融实验"""
    print("\n" + "="*80)
    print("PHASE 2: Teacher Fine-tuning with ECA Ablation")
    print("="*80)
    
    output_dir = output_dir / "eca_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    teachers_to_test = EXPERIMENT_CONFIG["eca_ablation"]["teachers_to_test"]
    results = {}
    
    for teacher_name in teachers_to_test:
        # 不使用ECA
        result_noeca = run_teacher_finetune(
            teacher_name, dataset_name, output_dir, device, use_eca=False
        )
        results[f"{teacher_name}_noeca"] = result_noeca
        
        # 保存中间结果
        with open(output_dir / "eca_ablation_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 使用ECA
        result_eca = run_teacher_finetune(
            teacher_name, dataset_name, output_dir, device, use_eca=True
        )
        results[f"{teacher_name}_eca"] = result_eca
        
        # 保存中间结果
        with open(output_dir / "eca_ablation_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # 打印ECA效果对比
        print(f"\n>>> ECA Effect on {teacher_name}:")
        print(f"    Without ECA: {result_noeca['test_accuracy']:.4f}")
        print(f"    With ECA:    {result_eca['test_accuracy']:.4f}")
        print(f"    Improvement: {(result_eca['test_accuracy'] - result_noeca['test_accuracy'])*100:.2f}%")
    
    # 生成汇总表格
    print("\n" + "="*80)
    print("ECA Ablation Summary")
    print("="*80)
    print(f"{'Teacher':<20} {'No ECA':<12} {'With ECA':<12} {'Improvement':<12}")
    print("-"*60)
    
    for teacher_name in teachers_to_test:
        noeca_acc = results.get(f"{teacher_name}_noeca", {}).get("test_accuracy", 0)
        eca_acc = results.get(f"{teacher_name}_eca", {}).get("test_accuracy", 0)
        improvement = (eca_acc - noeca_acc) * 100
        print(f"{teacher_name:<20} {noeca_acc:.4f}       {eca_acc:.4f}       {improvement:+.2f}%")
    
    return results

# ============================================================================
# Phase 3: Stacking组合实验
# ============================================================================

def run_stacking_experiment(
    teacher_set_name: str,
    teacher_names: List[str],
    dataset_name: str,
    teacher_ckpts_dir: Path,
    output_dir: Path,
    device: torch.device,
    use_eca: bool = True,
) -> dict:
    """训练Stacking模型"""
    from models.teacher_models import DynamicStackingModel
    
    config = EXPERIMENT_CONFIG["stacking"]
    dl_config = EXPERIMENT_CONFIG["dataloader"]
    
    print(f"\n{'='*60}")
    print(f"Training Stacking: {teacher_set_name}")
    print(f"Teachers: {teacher_names}")
    print(f"{'='*60}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, num_classes = create_dataloaders(
        dataset_name, config["batch_size"], dl_config["num_workers"]
    )
    
    # 加载教师模型
    teachers = []
    suffix = "_eca" if use_eca else "_noeca"
    for name in teacher_names:
        model = create_teacher_model(name, num_classes, use_eca=use_eca, pretrained=True)
        ckpt_path = teacher_ckpts_dir / f"{name}{suffix}_best.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded {name} from {ckpt_path}")
        else:
            print(f"Warning: No checkpoint found for {name}, using pretrained weights")
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        teachers.append(model)
    
    # 创建Stacking模型
    stacking_model = DynamicStackingModel(
        num_teachers=len(teachers),
        num_classes=num_classes,
        hidden_dim=config["hidden_dim"]
    )
    stacking_model = stacking_model.to(device)
    
    # 训练配置
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        stacking_model.parameters(),
        lr=config["base_lr"],
        weight_decay=config["weight_decay"]
    )
    
    total_steps = config["epochs"] * len(train_loader)
    warmup_steps = config["warmup_epochs"] * len(train_loader)
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_steps])
    scaler = GradScaler()
    
    checkpoint_path = output_dir / f"stacking_{teacher_set_name}_best.pth"
    
    result = {
        "teacher_set_name": teacher_set_name,
        "teachers": teacher_names,
        "train_losses": [],
        "val_losses": [],
        "val_accs": [],
        "best_epoch": 0,
        "best_val_acc": 0.0,
    }
    
    best_val_acc = 0.0
    start_time = time.time()
    
    for epoch in range(config["epochs"]):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Training
        stacking_model.train()
        total_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc="Training Stacking", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast():
                teacher_logits = [t(inputs) for t in teachers]
                outputs = stacking_model(*teacher_logits)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{correct/total:.4f}"})
        
        train_loss = total_loss / total
        train_acc = correct / total
        
        # Validation
        stacking_model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                with autocast():
                    teacher_logits = [t(inputs) for t in teachers]
                    outputs = stacking_model(*teacher_logits)
                    loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += inputs.size(0)
        
        val_loss = total_loss / total
        val_acc = correct / total
        
        result["train_losses"].append(train_loss)
        result["val_losses"].append(val_loss)
        result["val_accs"].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            result["best_epoch"] = epoch + 1
            result["best_val_acc"] = best_val_acc
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": stacking_model.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": num_classes,
                "teacher_set_name": teacher_set_name,
                "teachers": teacher_names,
            }, checkpoint_path)
            print(f"✓ Saved best stacking model (val_acc={best_val_acc:.4f})")
    
    result["training_time_seconds"] = time.time() - start_time
    result["checkpoint_path"] = str(checkpoint_path)
    
    # 测试集评估
    ckpt = torch.load(checkpoint_path, map_location=device)
    stacking_model.load_state_dict(ckpt["model_state_dict"])
    stacking_model.eval()
    
    all_preds, all_labels = [], []
    total_loss, correct, total = 0.0, 0, 0
    all_teacher_preds = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                teacher_logits = [t(inputs) for t in teachers]
                outputs = stacking_model(*teacher_logits)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            # 收集教师预测
            batch_teacher_preds = [tl.argmax(dim=1).cpu().numpy() for tl in teacher_logits]
            all_teacher_preds.append(batch_teacher_preds)
    
    from sklearn.metrics import f1_score
    import numpy as np
    
    result["test_accuracy"] = correct / total
    result["test_f1_macro"] = f1_score(all_labels, all_preds, average='macro')
    result["test_f1_weighted"] = f1_score(all_labels, all_preds, average='weighted')
    
    # 计算教师多样性指标
    teacher_preds_np = [np.concatenate([b[i] for b in all_teacher_preds]) for i in range(len(teachers))]
    labels_np = np.array(all_labels)
    
    # 不一致率
    n_samples = len(labels_np)
    disagreements = 0
    for i in range(n_samples):
        preds_at_i = [tp[i] for tp in teacher_preds_np]
        if len(set(preds_at_i)) > 1:
            disagreements += 1
    result["disagreement_rate"] = disagreements / n_samples
    
    # Oracle准确率
    oracle_correct = 0
    for i in range(n_samples):
        if any(tp[i] == labels_np[i] for tp in teacher_preds_np):
            oracle_correct += 1
    result["oracle_accuracy"] = oracle_correct / n_samples
    
    print(f"\n=== Final Results for Stacking {teacher_set_name} ===")
    print(f"Teachers: {teacher_names}")
    print(f"Test Accuracy: {result['test_accuracy']:.4f}")
    print(f"F1-macro: {result['test_f1_macro']:.4f}, F1-weighted: {result['test_f1_weighted']:.4f}")
    print(f"Disagreement Rate: {result['disagreement_rate']:.4f}")
    print(f"Oracle Accuracy: {result['oracle_accuracy']:.4f}")
    
    return result

def run_phase3_stacking(dataset_name: str, output_dir: Path, device: torch.device):
    """Phase 3: Stacking组合实验"""
    print("\n" + "="*80)
    print("PHASE 3: Stacking Combination Experiments")
    print("="*80)
    
    stacking_output_dir = output_dir / "stacking"
    stacking_output_dir.mkdir(parents=True, exist_ok=True)
    
    teacher_ckpts_dir = output_dir / "eca_ablation"
    stacking_sets = EXPERIMENT_CONFIG["stacking_sets"]
    
    results = {}
    
    for set_name, teachers in stacking_sets.items():
        result = run_stacking_experiment(
            set_name, teachers, dataset_name, teacher_ckpts_dir, stacking_output_dir, device
        )
        results[set_name] = result
        
        # 保存中间结果
        with open(stacking_output_dir / "stacking_results.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印汇总
    print("\n" + "="*80)
    print("Stacking Experiments Summary")
    print("="*80)
    print(f"{'Teacher Set':<20} {'Teachers':<40} {'Test Acc':<10} {'F1-macro':<10}")
    print("-"*80)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]["test_accuracy"], reverse=True)
    for set_name, result in sorted_results:
        teachers_str = ", ".join(result["teachers"][:3]) + ("..." if len(result["teachers"]) > 3 else "")
        print(f"{set_name:<20} {teachers_str:<40} {result['test_accuracy']:.4f}     {result['test_f1_macro']:.4f}")
    
    return results

# ============================================================================
# Phase 4: 结果分析
# ============================================================================

def run_phase4_analysis(output_dir: Path):
    """Phase 4: 结果分析与对比"""
    print("\n" + "="*80)
    print("PHASE 4: Results Analysis")
    print("="*80)
    
    # 加载所有结果
    eca_results_path = output_dir / "eca_ablation" / "eca_ablation_results.json"
    stacking_results_path = output_dir / "stacking" / "stacking_results.json"
    
    eca_results = {}
    stacking_results = {}
    
    if eca_results_path.exists():
        with open(eca_results_path, "r") as f:
            eca_results = json.load(f)
    
    if stacking_results_path.exists():
        with open(stacking_results_path, "r") as f:
            stacking_results = json.load(f)
    
    # 生成Markdown报告
    report_lines = [
        "# 教师模型搜索实验报告",
        f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n",
        "## 1. ECA消融实验结果\n",
        "| 教师模型 | 无ECA准确率 | 有ECA准确率 | 提升幅度 |",
        "|----------|-------------|-------------|----------|",
    ]
    
    teachers = EXPERIMENT_CONFIG["eca_ablation"]["teachers_to_test"]
    for teacher in teachers:
        noeca = eca_results.get(f"{teacher}_noeca", {})
        eca = eca_results.get(f"{teacher}_eca", {})
        noeca_acc = noeca.get("test_accuracy", 0)
        eca_acc = eca.get("test_accuracy", 0)
        improvement = (eca_acc - noeca_acc) * 100
        report_lines.append(f"| {teacher} | {noeca_acc:.4f} | {eca_acc:.4f} | {improvement:+.2f}% |")
    
    report_lines.extend([
        "\n## 2. Stacking组合实验结果\n",
        "| 组合名称 | 教师列表 | 测试准确率 | F1-macro | 不一致率 | Oracle准确率 |",
        "|----------|----------|------------|----------|----------|--------------|",
    ])
    
    sorted_stacking = sorted(stacking_results.items(), key=lambda x: x[1].get("test_accuracy", 0), reverse=True)
    for set_name, result in sorted_stacking:
        teachers_str = ", ".join(result.get("teachers", []))
        test_acc = result.get("test_accuracy", 0)
        f1_macro = result.get("test_f1_macro", 0)
        disagreement = result.get("disagreement_rate", 0)
        oracle = result.get("oracle_accuracy", 0)
        report_lines.append(f"| {set_name} | {teachers_str} | {test_acc:.4f} | {f1_macro:.4f} | {disagreement:.4f} | {oracle:.4f} |")
    
    # 最佳推荐
    if sorted_stacking:
        best_set_name, best_result = sorted_stacking[0]
        report_lines.extend([
            "\n## 3. 推荐配置\n",
            f"**最佳教师组合**: {best_set_name}",
            f"- 教师列表: {', '.join(best_result.get('teachers', []))}",
            f"- 测试准确率: {best_result.get('test_accuracy', 0):.4f}",
            f"- F1-macro: {best_result.get('test_f1_macro', 0):.4f}",
        ])
    
    # 保存报告
    report_path = output_dir / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"Report saved to: {report_path}")
    print("\n".join(report_lines))

# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Teacher Search Experiment")
    parser.add_argument("--phase", type=str, default="all", choices=["all", "2", "3", "4"],
                        help="Which phase to run: all, 2 (finetune+ECA), 3 (stacking), 4 (analysis)")
    parser.add_argument("--dataset", type=str, default="ISCXVPN2016", help="Dataset name")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "results/teacher_search"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 保存配置
    with open(output_dir / "experiment_config.json", "w") as f:
        json.dump(EXPERIMENT_CONFIG, f, indent=2)
    
    if args.phase in ["all", "2"]:
        run_phase2_eca_ablation(args.dataset, output_dir, device)
    
    if args.phase in ["all", "3"]:
        run_phase3_stacking(args.dataset, output_dir, device)
    
    if args.phase in ["all", "4"]:
        run_phase4_analysis(output_dir)
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)

if __name__ == "__main__":
    main()
