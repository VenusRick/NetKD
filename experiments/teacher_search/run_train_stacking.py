#!/usr/bin/env python3
"""Teacher Search - Train Stacking Model with Multiple Teachers"""
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
from models.teacher_models import DynamicStackingModel
from data_preprocessing.dataset import TrafficDataset
from experiments.teacher_search.result_schema import TrainingConfig, StackingResult
from experiments.teacher_search.metrics import compute_disagreement_rate, compute_oracle_accuracy, compute_diversity_score

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

def load_teacher_checkpoints(teacher_names, teacher_config, checkpoints_dir, num_classes, device):
    teachers = []
    for name in teacher_names:
        cfg = teacher_config[name]
        model = TeacherRegistry.create_teacher(name, num_classes=num_classes,
            pretrained=cfg.get("pretrained", True), use_eca=cfg.get("use_eca", True))
        ckpt_path = Path(checkpoints_dir) / f"{name}_best.pth"
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint for {name} from {ckpt_path}")
        else:
            print(f"Warning: No checkpoint found for {name} at {ckpt_path}, using pretrained weights")
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        teachers.append(model)
    return teachers

def train_one_epoch(stacking_model, teachers, loader, criterion, optimizer, scheduler, scaler, device):
    stacking_model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training Stacking", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            teacher_logits = []
            for teacher in teachers:
                with torch.no_grad():
                    teacher_logits.append(teacher(inputs))
            outputs = stacking_model(teacher_logits)
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

def evaluate(stacking_model, teachers, loader, criterion, device):
    stacking_model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_teacher_preds = [], [], []
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            with autocast():
                teacher_logits = [teacher(inputs) for teacher in teachers]
                outputs = stacking_model(teacher_logits)
                loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += inputs.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch_teacher_preds = [tl.argmax(dim=1).cpu().numpy() for tl in teacher_logits]
            all_teacher_preds.append(batch_teacher_preds)
    from sklearn.metrics import f1_score
    import numpy as np
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    teacher_preds_np = [np.concatenate([b[i] for b in all_teacher_preds]) for i in range(len(teachers))]
    labels_np = np.array(all_labels)
    disagreement = compute_disagreement_rate(teacher_preds_np)
    oracle_acc = compute_oracle_accuracy(teacher_preds_np, labels_np)
    diversity = compute_diversity_score(teacher_preds_np, labels_np)
    return total_loss / total, correct / total, f1_macro, f1_weighted, disagreement, oracle_acc, diversity

def train_stacking(teacher_set_name: str, config_path: Path, data_dir: Path, checkpoints_dir: Path, output_dir: Path, device: str, resume: str = None):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    teacher_set_cfg = config["teacher_sets"].get(teacher_set_name)
    if not teacher_set_cfg:
        raise ValueError(f"Teacher set '{teacher_set_name}' not found in config")
    teacher_names = teacher_set_cfg["teachers"]
    hidden_dim = teacher_set_cfg.get("stacking_hidden_dim", 384)
    defaults = config.get("defaults", {})
    stacking_defaults = config.get("stacking_defaults", defaults)
    training_config = TrainingConfig(
        epochs=stacking_defaults.get("epochs", 30), batch_size=stacking_defaults.get("batch_size", 256),
        learning_rate=stacking_defaults.get("learning_rate", 0.0005), weight_decay=stacking_defaults.get("weight_decay", 0.0001),
        optimizer=stacking_defaults.get("optimizer", "adamw"), scheduler=stacking_defaults.get("scheduler", "cosine"),
        label_smoothing=stacking_defaults.get("label_smoothing", 0.05), warmup_epochs=stacking_defaults.get("warmup_epochs", 3))
    
    train_dataset = TrafficDataset(root_dir=data_dir / "train", transform=None)
    val_dataset = TrafficDataset(root_dir=data_dir / "valid", transform=None)
    test_dataset = TrafficDataset(root_dir=data_dir / "test", transform=None)
    num_classes = len(train_dataset.classes)
    train_loader = DataLoader(train_dataset, batch_size=training_config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=training_config.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=training_config.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    
    teachers = load_teacher_checkpoints(teacher_names, config["teacher_candidates"], checkpoints_dir, num_classes, device)
    stacking_model = DynamicStackingModel(num_teachers=len(teachers), num_classes=num_classes, hidden_dim=hidden_dim)
    stacking_model = stacking_model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=training_config.label_smoothing)
    optimizer = create_optimizer(stacking_model, training_config)
    scheduler = create_scheduler(optimizer, training_config, len(train_loader))
    scaler = GradScaler()
    
    result = StackingResult(teacher_set_name=teacher_set_name, teachers=teacher_names, hidden_dim=hidden_dim)
    start_epoch, best_val_acc = 0, 0.0
    output_dir = Path(output_dir)
    checkpoint_path = output_dir / f"stacking_{teacher_set_name}_best.pth"
    if resume and Path(resume).exists():
        ckpt = torch.load(resume, map_location=device)
        stacking_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch, best_val_acc = ckpt.get("epoch", 0), ckpt.get("best_val_acc", 0)
        print(f"Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.4f}")
    
    start_time = time.time()
    for epoch in range(start_epoch, training_config.epochs):
        print(f"\nEpoch {epoch+1}/{training_config.epochs}")
        train_loss, train_acc = train_one_epoch(stacking_model, teachers, train_loader, criterion, optimizer, scheduler, scaler, device)
        val_loss, val_acc, _, _, _, _, _ = evaluate(stacking_model, teachers, val_loader, criterion, device)
        result.train_losses.append(train_loss)
        result.val_losses.append(val_loss)
        result.val_accs.append(val_acc)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            result.best_epoch, result.best_val_acc = epoch + 1, best_val_acc
            torch.save({"epoch": epoch + 1, "model_state_dict": stacking_model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(), "best_val_acc": best_val_acc,
                        "num_classes": num_classes, "teacher_set_name": teacher_set_name,
                        "teacher_names": teacher_names, "hidden_dim": hidden_dim}, checkpoint_path)
            print(f"Saved best stacking model to {checkpoint_path}")
    
    result.training_time_seconds = time.time() - start_time
    result.checkpoint_path = str(checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    stacking_model.load_state_dict(ckpt["model_state_dict"])
    _, test_acc, f1_macro, f1_weighted, disagreement, oracle_acc, diversity = evaluate(stacking_model, teachers, test_loader, criterion, device)
    result.test_accuracy, result.test_f1_macro, result.test_f1_weighted = test_acc, f1_macro, f1_weighted
    result.disagreement_rate, result.oracle_accuracy, result.diversity_score = disagreement, oracle_acc, diversity
    print(f"\n=== Final Results for Stacking {teacher_set_name} ===")
    print(f"Teachers: {teacher_names}")
    print(f"Test Accuracy: {test_acc:.4f}, F1-macro: {f1_macro:.4f}, F1-weighted: {f1_weighted:.4f}")
    print(f"Disagreement: {disagreement:.4f}, Oracle Acc: {oracle_acc:.4f}, Diversity: {diversity:.4f}")
    return result

def main():
    parser = argparse.ArgumentParser(description="Train stacking model with multiple teachers")
    parser.add_argument("--teacher-set", type=str, required=True, help="Teacher set name from config")
    parser.add_argument("--config", type=str, default=str(PROJECT_ROOT / "configs/teachers.yaml"))
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--checkpoints-dir", type=str, required=True, help="Path to teacher checkpoints")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "checkpoints/teacher_search"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--save-result", action="store_true", help="Save result to JSON")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result = train_stacking(args.teacher_set, Path(args.config), Path(args.data_dir), 
                            Path(args.checkpoints_dir), output_dir, args.device, args.resume)
    if args.save_result:
        result_path = output_dir / f"stacking_{args.teacher_set}_result.json"
        import json
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"Saved result to {result_path}")
    return result

if __name__ == "__main__":
    main()
