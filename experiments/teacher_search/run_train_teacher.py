#!/usr/bin/env python3
"""Teacher Search - Train Single Teacher Model"""
import argparse
import time
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import yaml
from tqdm import tqdm

from models.teacher_registry import TeacherRegistry
from data_preprocessing.dataset import TrafficDataset
from experiments.teacher_search.result_schema import TrainingConfig, TeacherResult, ExperimentResult

def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    elif config.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
    return torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

def create_scheduler(optimizer, config: TrainingConfig, steps_per_epoch: int):
    total_steps = config.epochs * steps_per_epoch
    warmup_steps = config.warmup_epochs * steps_per_epoch
    if config.scheduler.lower() == "cosine":
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
            return torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, main_scheduler], [warmup_steps])
        return main_scheduler
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.epochs // 3, gamma=0.1)

def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler: scheduler.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += inputs.size(0)
    return total_loss / total, correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
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
    return total_loss / total, correct / total, f1_macro, f1_weighted

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def train_teacher(teacher_name: str, config_path: Path, data_dir: Path, output_dir: Path, device: str, resume: str = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    teacher_cfg = config["teacher_candidates"].get(teacher_name)
    if not teacher_cfg:
        raise ValueError(f"Teacher '{teacher_name}' not found in config")
    defaults = config.get("defaults", {})
    training_config = TrainingConfig(
        epochs=defaults.get("epochs", 50), batch_size=defaults.get("batch_size", 256),
        learning_rate=defaults.get("learning_rate", 0.001), weight_decay=defaults.get("weight_decay", 0.0001),
        optimizer=defaults.get("optimizer", "adamw"), scheduler=defaults.get("scheduler", "cosine"),
        label_smoothing=defaults.get("label_smoothing", 0.05), warmup_epochs=defaults.get("warmup_epochs", 5))
    
    train_dataset = TrafficDataset(root_dir=data_dir / "train", transform=None)
    val_dataset = TrafficDataset(root_dir=data_dir / "valid", transform=None)
    test_dataset = TrafficDataset(root_dir=data_dir / "test", transform=None)
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    model = TeacherRegistry.create_teacher(teacher_name, num_classes=num_classes,
        pretrained=teacher_cfg.get("pretrained", True), use_eca=teacher_cfg.get("use_eca", True))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    optimizer = create_optimizer(model, training_config)
    scheduler = create_scheduler(optimizer, training_config, len(train_loader))
    scaler = GradScaler()
    
    result = TeacherResult(name=teacher_name, class_name=teacher_cfg["class_name"],
        pretrained=teacher_cfg.get("pretrained", True), use_eca=teacher_cfg.get("use_eca", True),
        params_millions=count_parameters(model))
    
    start_epoch, best_val_acc = 0, 0.0
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"{teacher_name}_best.pth"
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, best_val_acc = ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    
    start_time = time.time()
    for epoch in range(start_epoch, training_config.epochs):
        print(f"\nEpoch {epoch+1}/{training_config.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            result.best_epoch, result.best_val_acc = epoch + 1, best_val_acc
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "best_val_acc": best_val_acc,
                        "num_classes": num_classes, "teacher_name": teacher_name}, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
    
    result.training_time_seconds = time.time() - start_time
    result.checkpoint_path = str(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, f1_macro, f1_weighted = evaluate(model, test_loader, criterion, device)
    result.test_accuracy, result.test_f1_macro, result.test_f1_weighted = test_acc, f1_macro, f1_weighted
    print(f"\n=== Final Results for {teacher_name} ===")
    print(f"Test Accuracy: {test_acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Train a single teacher model")
    parser.add_argument("--teacher", type=str, required=True, help="Teacher name from config")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/teachers.yaml"))
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "checkpoints/teacher_search"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-result", action="store_true", help="Save result to JSON")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_teacher(args.teacher, Path(args.config), Path(args.data_dir), output_dir, args.device, args.resume)
    
    if args.save_result:
        result_path = output_dir / f"{args.teacher}_result.json"
        import json
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved result to {result_path}")
    return result

if __name__ == "__main__":
    main()
