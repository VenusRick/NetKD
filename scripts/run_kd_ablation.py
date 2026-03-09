#!/usr/bin/env python3
"""KD Configuration Ablation Study"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset
from models.teacher_models import (DynamicStackingModel, ConvNeXtV2TinyTeacher,
                                    DenseNet121Teacher, MobileNetV3LargeTeacher)
from models.student_models_v2 import create_student_model

# KD配置
KD_CONFIGS = {
    "ce_only": {"alpha_ce": 1.0, "alpha_kl": 0.0, "alpha_rkl": 0.0, "use_sinkhorn": False},
    "ce_kl": {"alpha_ce": 0.5, "alpha_kl": 0.5, "alpha_rkl": 0.0, "use_sinkhorn": False},
    "ce_symmetric_kl": {"alpha_ce": 0.4, "alpha_kl": 0.3, "alpha_rkl": 0.3, "use_sinkhorn": False},
    "traffic_kd_full": {"alpha_ce": 0.3, "alpha_kl": 0.25, "alpha_rkl": 0.25, "use_sinkhorn": True, "alpha_sink": 0.2},
}

CONFIG = {
    "dataset": "ISCXVPN2016",
    "batch_size": 128,
    "num_workers": 8,
    "teachers": {
        "convnextv2_tiny": {"class": ConvNeXtV2TinyTeacher, "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/convnextv2_tiny_eca_best.pth"},
        "densenet121": {"class": DenseNet121Teacher, "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/densenet121_eca_best.pth"},
        "mobilenetv3": {"class": MobileNetV3LargeTeacher, "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/mobilenetv3_eca_best.pth"},
    },
    "stacking": {"checkpoint": "results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth", "hidden_dim": 384},
    "training": {"epochs": 100, "base_lr": 0.001, "weight_decay": 1e-4, "warmup_epochs": 5, "grad_clip": 1.0},
    "temperature": 3.0,
}

def load_teachers(config, num_classes, device):
    teachers = []
    for name, cfg in config["teachers"].items():
        model = cfg["class"](num_classes=num_classes, use_eca=cfg["use_eca"])
        ckpt = torch.load(PROJECT_ROOT / cfg["checkpoint"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        for p in model.parameters(): p.requires_grad = False
        teachers.append(model)
    return teachers

def load_stacking(config, num_teachers, num_classes, device):
    stacking = DynamicStackingModel(num_classes, num_teachers, config["stacking"]["hidden_dim"])
    ckpt = torch.load(PROJECT_ROOT / config["stacking"]["checkpoint"], map_location=device, weights_only=False)
    stacking.load_state_dict(ckpt["model_state_dict"])
    stacking = stacking.to(device).eval()
    for p in stacking.parameters(): p.requires_grad = False
    return stacking

def sinkhorn_loss(student_logits, teacher_logits, eps=0.1, n_iter=3):
    """Sinkhorn divergence loss"""
    s_probs = F.softmax(student_logits, dim=-1)
    t_probs = F.softmax(teacher_logits, dim=-1)
    cost = torch.cdist(s_probs, t_probs, p=2)
    K = torch.exp(-cost / eps)
    u = torch.ones_like(s_probs[:, 0])
    for _ in range(n_iter):
        v = 1.0 / (K.transpose(-2, -1) @ u.unsqueeze(-1) + 1e-8).squeeze(-1)
        u = 1.0 / (K @ v.unsqueeze(-1) + 1e-8).squeeze(-1)
    return (u.unsqueeze(-1) * K * cost * v.unsqueeze(-2)).sum(dim=(-2, -1)).mean()

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

def train_with_kd_config(student, teachers, stacking, loaders, kd_config, config, device, output_dir):
    train_loader, val_loader, test_loader = loaders
    cfg_t = config["training"]
    T = config["temperature"]
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg_t["base_lr"], weight_decay=cfg_t["weight_decay"])
    total_steps = cfg_t["epochs"] * len(train_loader)
    warmup_steps = cfg_t["warmup_epochs"] * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps: return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_val_acc, best_epoch = 0, 0
    alpha_ce = kd_config.get("alpha_ce", 0.5)
    alpha_kl = kd_config.get("alpha_kl", 0.5)
    alpha_rkl = kd_config.get("alpha_rkl", 0.0)
    use_sinkhorn = kd_config.get("use_sinkhorn", False)
    alpha_sink = kd_config.get("alpha_sink", 0.0)
    
    for epoch in range(1, cfg_t["epochs"] + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                s_logits = student(x)
                
                # Teacher ensemble prediction
                with torch.no_grad():
                    t_logits_list = [t(x) for t in teachers]
                    t_stack = torch.stack(t_logits_list, dim=1)
                    t_logits = stacking(t_stack)
                
                # CE Loss
                loss_ce = F.cross_entropy(s_logits, y)
                
                # Forward KL (student learns from teacher)
                loss_kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T ** 2) if alpha_kl > 0 else 0
                
                # Reverse KL
                loss_rkl = F.kl_div(
                    F.log_softmax(t_logits / T, dim=-1),
                    F.softmax(s_logits / T, dim=-1),
                    reduction='batchmean'
                ) * (T ** 2) if alpha_rkl > 0 else 0
                
                # Sinkhorn
                loss_sink = sinkhorn_loss(s_logits, t_logits) if use_sinkhorn else 0
                
                loss = alpha_ce * loss_ce + alpha_kl * loss_kl + alpha_rkl * loss_rkl + alpha_sink * loss_sink
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg_t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{s_logits.argmax(1).eq(y).float().mean():.3f}")
        
        val_acc = evaluate(student, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({"model_state_dict": student.state_dict(), "epoch": epoch, "val_acc": val_acc},
                       output_dir / "best_model.pth")
        
        if epoch % 10 == 0:
            print(f"E{epoch}: val_acc={val_acc:.4f}")
    
    # Load best and evaluate
    ckpt = torch.load(output_dir / "best_model.pth", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    test_acc = evaluate(student, test_loader, device)
    test_f1 = compute_f1(student, test_loader, device)
    
    return {"best_epoch": best_epoch, "best_val_acc": best_val_acc, "test_acc": test_acc, "test_f1": test_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", default="mobilenetv2")
    parser.add_argument("--kd-config", required=True, choices=list(KD_CONFIGS.keys()))
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"KD Config: {args.kd_config}")
    
    # Load data
    train_l, val_l, test_l, meta = quick_load_dataset(
        CONFIG["dataset"], dataset_root="/workspace/yqm/Dataset",
        batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"]
    )
    num_classes = meta["num_classes"]
    
    # Load teachers
    print("Loading teachers...")
    teachers = load_teachers(CONFIG, num_classes, device)
    stacking = load_stacking(CONFIG, len(teachers), num_classes, device)
    
    # Create student
    student = create_student_model(args.student, num_classes).to(device)
    
    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train
    start_time = time.time()
    results = train_with_kd_config(
        student, teachers, stacking,
        (train_l, val_l, test_l),
        KD_CONFIGS[args.kd_config],
        CONFIG, device, output_dir
    )
    elapsed = (time.time() - start_time) / 60
    
    # Save results
    results["kd_config"] = args.kd_config
    results["student"] = args.student
    results["elapsed_min"] = elapsed
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"KD Config: {args.kd_config}")
    print(f"Best Epoch: {results['best_epoch']}")
    print(f"Val Acc: {results['best_val_acc']*100:.2f}%")
    print(f"Test Acc: {results['test_acc']*100:.2f}%")
    print(f"Test F1: {results['test_f1']*100:.2f}%")
    print(f"Time: {elapsed:.1f} min")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
