#!/usr/bin/env python3
"""
Train lightweight student models using existing teacher checkpoints
Uses: ConvNeXtV2-Tiny + DenseNet121 + MobileNetV3-Large as teachers
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset
from models.teacher_models import (DynamicStackingModel, ConvNeXtV2TinyTeacher,
                                    DenseNet121Teacher, MobileNetV3LargeTeacher)
from models.student_models_v2 import create_lightweight_student

CONFIG = {
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
    "stacking": {
        "checkpoint": "results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth",
        "hidden_dim": 384
    },
    "training": {"epochs": 100, "base_lr": 0.001, "weight_decay": 1e-4, "warmup_epochs": 5},
    "temperature": 3.0,
}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def load_teachers(config, num_classes, device):
    teachers = []
    for name, cfg in config["teachers"].items():
        model = cfg["class"](num_classes=num_classes, use_eca=cfg["use_eca"])
        ckpt_path = PROJECT_ROOT / cfg["checkpoint"]
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"  ✓ {name} loaded from checkpoint")
        else:
            print(f"  ! {name} checkpoint not found, using random init")
        model = model.to(device).eval()
        for p in model.parameters(): p.requires_grad = False
        teachers.append(model)
    return teachers

def load_stacking(config, num_teachers, num_classes, device):
    stacking = DynamicStackingModel(num_classes, num_teachers, config["stacking"]["hidden_dim"])
    ckpt_path = PROJECT_ROOT / config["stacking"]["checkpoint"]
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        stacking.load_state_dict(ckpt["model_state_dict"])
        print("  ✓ stacking loaded")
    stacking = stacking.to(device).eval()
    for p in stacking.parameters(): p.requires_grad = False
    return stacking

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += model(x).argmax(1).eq(y).sum().item()
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

def train_student(student, teachers, stacking, loaders, config, device, output_dir, name):
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
    
    for epoch in range(1, cfg_t["epochs"] + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                s_logits = student(x)
                with torch.no_grad():
                    t_out = stacking(*[t(x) for t in teachers])
                
                loss_ce = F.cross_entropy(s_logits, y)
                loss_kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_out / T, dim=-1),
                    reduction='batchmean'
                ) * (T ** 2)
                loss = 0.5 * loss_ce + 0.5 * loss_kl
            
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
            best_val_acc, best_epoch = val_acc, epoch
            torch.save({"model_state_dict": student.state_dict(), "epoch": epoch, "val_acc": val_acc},
                       output_dir / f"student_{name}_best.pth")
        
        if epoch % 10 == 0 or epoch == cfg_t["epochs"]:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f}@E{best_epoch})")
    
    ckpt = torch.load(output_dir / f"student_{name}_best.pth", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    test_acc = evaluate(student, test_loader, device)
    test_f1 = compute_f1(student, test_loader, device)
    
    return {"name": name, "best_epoch": best_epoch, "val_acc": best_val_acc, "test_acc": test_acc, "test_f1": test_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--students", nargs="+", required=True,
                       choices=["ghostnet_050", "ghostnetv3_050", "repvit_m0_9", 
                               "mobilenetv2_050", "mobilenetv3_small_050"])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--dataset-root", type=str, default="/workspace/yqm/Dataset")
    args = parser.parse_args()
    
    device = torch.device("cuda")
    print(f"Device: {device}")
    
    # Load data
    train_l, val_l, test_l, meta = quick_load_dataset(
        "ISCXVPN2016", dataset_root=args.dataset_root,
        batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"]
    )
    num_classes = meta["num_classes"]
    print(f"Dataset: {num_classes} classes")
    
    # Load teachers
    print("\nLoading teachers...")
    teachers = load_teachers(CONFIG, num_classes, device)
    stacking = load_stacking(CONFIG, len(teachers), num_classes, device)
    
    CONFIG["training"]["epochs"] = args.epochs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for s_name in args.students:
        print(f"\n{'='*60}")
        print(f"Training: {s_name}")
        print(f"{'='*60}")
        
        student = create_lightweight_student(s_name, num_classes)
        student = student.to(device)
        params = count_params(student)
        print(f"  Params: {params:.2f}M")
        
        start = time.time()
        result = train_student(student, teachers, stacking, (train_l, val_l, test_l), CONFIG, device, output_dir, s_name)
        result["time_min"] = (time.time() - start) / 60
        result["params_m"] = params
        results.append(result)
        
        print(f"\n✓ {s_name}: val={result['val_acc']*100:.2f}% test={result['test_acc']*100:.2f}% F1={result['test_f1']*100:.2f}%")
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(f"{'Model':<25} {'Params':<10} {'Val':<10} {'Test':<10} {'F1':<10} {'Time':<10}")
    print("-"*70)
    for r in results:
        print(f"{r['name']:<25} {r['params_m']:.2f}M     {r['val_acc']*100:.2f}%     {r['test_acc']*100:.2f}%     {r['test_f1']*100:.2f}%  {r['time_min']:.1f}min")

if __name__ == "__main__":
    main()
