#!/usr/bin/env python3
"""
Extended KD Experiments with New Teacher and Student Models
- New Teachers: ConvNeXtV2-Small, EfficientNetV2-S, GhostNetV3, RepViT-M1
- New Students: GhostNet, GhostNetV3-0.75x, RepViT-M0.9
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset


# ============== New Model Registry ==============

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class TimmTeacher(nn.Module):
    """Generic teacher model using timm"""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.name = model_name
    
    def forward(self, x):
        return self.model(x)


class TimmStudent(nn.Module):
    """Generic lightweight student model using timm"""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        self.name = model_name
    
    def forward(self, x):
        return self.model(x)


# Teacher model configs (larger models with pretrained weights)
TEACHER_MODELS = {
    # ConvNeXt V2 family
    "convnextv2_tiny": {"timm_name": "convnextv2_tiny", "pretrained": True},
    "convnextv2_small": {"timm_name": "convnextv2_small", "pretrained": True},
    "convnextv2_nano": {"timm_name": "convnextv2_nano", "pretrained": True},
    # EfficientNet V2 family
    "efficientnetv2_s": {"timm_name": "efficientnetv2_s", "pretrained": True},
    "efficientnetv2_rw_s": {"timm_name": "efficientnetv2_rw_s", "pretrained": True},
    # GhostNet V3 family (teacher version - larger)
    "ghostnetv3_100": {"timm_name": "ghostnetv3_100", "pretrained": True},
    "ghostnetv3_130": {"timm_name": "ghostnetv3_130", "pretrained": True},
    # RepViT family (teacher version - larger)
    "repvit_m1_0": {"timm_name": "repvit_m1_0", "pretrained": True},
    "repvit_m1_1": {"timm_name": "repvit_m1_1", "pretrained": True},
}

# Student model configs (lightweight models)
STUDENT_MODELS = {
    # GhostNet family (lightweight)
    "ghostnet_050": {"timm_name": "ghostnet_050", "pretrained": False},
    "ghostnet_100": {"timm_name": "ghostnet_100", "pretrained": False},
    "ghostnetv3_050": {"timm_name": "ghostnetv3_050", "pretrained": False},
    # RepViT family (lightweight)
    "repvit_m0_9": {"timm_name": "repvit_m0_9", "pretrained": False},
    # MobileNet family
    "mobilenetv2_050": {"timm_name": "mobilenetv2_050", "pretrained": False},
    "mobilenetv3_small_050": {"timm_name": "mobilenetv3_small_050", "pretrained": False},
    # EfficientNet Lite
    "efficientnet_lite0": {"timm_name": "efficientnet_lite0", "pretrained": False},
}


def create_teacher(name: str, num_classes: int, device: torch.device):
    """Create and load a teacher model"""
    if name not in TEACHER_MODELS:
        raise ValueError(f"Unknown teacher: {name}. Available: {list(TEACHER_MODELS.keys())}")
    
    cfg = TEACHER_MODELS[name]
    model = TimmTeacher(cfg["timm_name"], num_classes, pretrained=cfg["pretrained"])
    model = model.to(device)
    params = count_parameters(model) / 1e6
    print(f"  ✓ {name} ({params:.2f}M params)")
    return model


def create_student(name: str, num_classes: int, device: torch.device):
    """Create a student model"""
    if name not in STUDENT_MODELS:
        raise ValueError(f"Unknown student: {name}. Available: {list(STUDENT_MODELS.keys())}")
    
    cfg = STUDENT_MODELS[name]
    model = TimmStudent(cfg["timm_name"], num_classes, pretrained=cfg["pretrained"])
    model = model.to(device)
    params = count_parameters(model) / 1e6
    print(f"  ✓ Student: {name} ({params:.2f}M params)")
    return model, params


# ============== Training Functions ==============

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return correct / total


@torch.no_grad()
def compute_f1(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    return f1_score(labels, preds, average='macro')


def train_teacher(model, loaders, epochs, device, output_dir, name):
    """Train/finetune a teacher model"""
    train_loader, val_loader, test_loader = loaders
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    warmup_steps = 3 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Teacher E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{logits.argmax(1).eq(y).float().mean():.3f}")
        
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }, output_dir / f"teacher_{name}_best.pth")
        
        if epoch % 10 == 0:
            print(f"E{epoch}: val_acc={val_acc:.4f} (best={best_val_acc:.4f}@E{best_epoch})")
    
    # Load best and evaluate
    ckpt = torch.load(output_dir / f"teacher_{name}_best.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc = evaluate(model, test_loader, device)
    test_f1 = compute_f1(model, test_loader, device)
    
    return {"name": name, "best_epoch": best_epoch, "val_acc": best_val_acc, "test_acc": test_acc, "test_f1": test_f1}


def train_student_kd(student, teachers, loaders, epochs, device, output_dir, name, 
                     temperature=3.0, alpha_ce=0.5, alpha_kl=0.5):
    """Train student with knowledge distillation from multiple teachers"""
    train_loader, val_loader, test_loader = loaders
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = epochs * len(train_loader)
    warmup_steps = 5 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    # Freeze teachers
    for t in teachers:
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    
    best_val_acc, best_epoch = 0, 0
    T = temperature
    
    for epoch in range(1, epochs + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                s_logits = student(x)
                
                # Teacher ensemble (average)
                with torch.no_grad():
                    t_logits_list = [t(x) for t in teachers]
                    t_logits = torch.stack(t_logits_list, dim=0).mean(dim=0)
                
                # CE Loss
                loss_ce = F.cross_entropy(s_logits, y)
                
                # KL Divergence Loss
                loss_kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T ** 2)
                
                loss = alpha_ce * loss_ce + alpha_kl * loss_kl
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{s_logits.argmax(1).eq(y).float().mean():.3f}")
        
        val_acc = evaluate(student, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": student.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc
            }, output_dir / f"student_{name}_best.pth")
        
        if epoch % 10 == 0:
            print(f"E{epoch}: val_acc={val_acc:.4f}")
    
    # Load best and evaluate
    ckpt = torch.load(output_dir / f"student_{name}_best.pth", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    test_acc = evaluate(student, test_loader, device)
    test_f1 = compute_f1(student, test_loader, device)
    
    return {"name": name, "best_epoch": best_epoch, "val_acc": best_val_acc, "test_acc": test_acc, "test_f1": test_f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train_teachers", "train_students", "full"], required=True)
    parser.add_argument("--teachers", nargs="+", default=["convnextv2_tiny", "efficientnetv2_s", "ghostnetv3_100"])
    parser.add_argument("--students", nargs="+", default=["ghostnet_100", "repvit_m0_9"])
    parser.add_argument("--teacher-epochs", type=int, default=30)
    parser.add_argument("--student-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="/workspace/yqm/Dataset")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    train_l, val_l, test_l, meta = quick_load_dataset(
        "ISCXVPN2016",
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        num_workers=8
    )
    num_classes = meta["num_classes"]
    print(f"Dataset loaded: {num_classes} classes")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {"teachers": [], "students": []}
    
    if args.mode in ["train_teachers", "full"]:
        print("\n" + "="*60)
        print("Phase 1: Training/Finetuning Teachers")
        print("="*60)
        
        for t_name in args.teachers:
            print(f"\nTraining teacher: {t_name}")
            try:
                teacher = create_teacher(t_name, num_classes, device)
                start_time = time.time()
                result = train_teacher(teacher, (train_l, val_l, test_l), args.teacher_epochs, device, output_dir, t_name)
                result["time_min"] = (time.time() - start_time) / 60
                result["params_m"] = count_parameters(teacher) / 1e6
                results["teachers"].append(result)
                print(f"✓ {t_name}: val={result['val_acc']:.4f}, test={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
            except Exception as e:
                print(f"✗ {t_name} failed: {e}")
    
    if args.mode in ["train_students", "full"]:
        print("\n" + "="*60)
        print("Phase 2: Training Students with KD")
        print("="*60)
        
        # Load teachers
        print("Loading teachers...")
        teachers = []
        for t_name in args.teachers:
            ckpt_path = output_dir / f"teacher_{t_name}_best.pth"
            if ckpt_path.exists():
                teacher = create_teacher(t_name, num_classes, device)
                ckpt = torch.load(ckpt_path, weights_only=False)
                teacher.load_state_dict(ckpt["model_state_dict"])
                teacher.eval()
                teachers.append(teacher)
            else:
                print(f"  ! Teacher checkpoint not found: {ckpt_path}")
                # Try to use pretrained
                teacher = create_teacher(t_name, num_classes, device)
                teacher.eval()
                teachers.append(teacher)
        
        # Train students
        for s_name in args.students:
            print(f"\nTraining student: {s_name}")
            try:
                student, params = create_student(s_name, num_classes, device)
                start_time = time.time()
                result = train_student_kd(student, teachers, (train_l, val_l, test_l), args.student_epochs, device, output_dir, s_name)
                result["time_min"] = (time.time() - start_time) / 60
                result["params_m"] = params
                results["students"].append(result)
                print(f"✓ {s_name}: val={result['val_acc']:.4f}, test={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
            except Exception as e:
                print(f"✗ {s_name} failed: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    
    if results["teachers"]:
        print("\nTeachers:")
        print(f"{'Model':<25} {'Params':<10} {'Val Acc':<10} {'Test Acc':<10} {'F1':<10} {'Time':<10}")
        print("-" * 75)
        for r in results["teachers"]:
            print(f"{r['name']:<25} {r['params_m']:.2f}M     {r['val_acc']*100:.2f}%     {r['test_acc']*100:.2f}%     {r['test_f1']*100:.2f}%  {r['time_min']:.1f}min")
    
    if results["students"]:
        print("\nStudents:")
        print(f"{'Model':<25} {'Params':<10} {'Val Acc':<10} {'Test Acc':<10} {'F1':<10} {'Time':<10}")
        print("-" * 75)
        for r in results["students"]:
            print(f"{r['name']:<25} {r['params_m']:.2f}M     {r['val_acc']*100:.2f}%     {r['test_acc']*100:.2f}%     {r['test_f1']*100:.2f}%  {r['time_min']:.1f}min")


if __name__ == "__main__":
    main()
