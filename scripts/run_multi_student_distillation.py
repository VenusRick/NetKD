#!/usr/bin/env python3
"""Multi-Student Distillation Experiment: Compare GhostNet, MobileNetV2, MobileNetV3-Small"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset
from models.teacher_models import (DynamicStackingModel, ConvNeXtV2TinyTeacher, 
                                    DenseNet121Teacher, MobileNetV3LargeTeacher)
from models.student_models_v2 import create_student_model, get_model_info

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
    "training": {"epochs": 100, "base_lr": 0.001, "weight_decay": 1e-4, "warmup_epochs": 5, "grad_clip": 1.0, "label_smoothing": 0.1},
    "distillation": {"temperature": 3.0, "alpha_ce": 0.5, "alpha_kl": 0.5},
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
        print(f"  ✓ {name}")
    return teachers


def load_stacking(config, num_teachers, num_classes, device):
    stacking = DynamicStackingModel(num_classes, num_teachers, config["stacking"]["hidden_dim"])
    ckpt = torch.load(PROJECT_ROOT / config["stacking"]["checkpoint"], map_location=device, weights_only=False)
    stacking.load_state_dict(ckpt["model_state_dict"])
    stacking = stacking.to(device).eval()
    for p in stacking.parameters(): p.requires_grad = False
    print("  ✓ stacking")
    return stacking


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss += F.cross_entropy(out, y).item() * x.size(0)
        correct += out.argmax(1).eq(y).sum().item()
        total += x.size(0)
    return loss / total, correct / total


@torch.no_grad()
def compute_f1(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    return f1_score(labels, preds, average='macro')


def train_student(student, teachers, stacking, loaders, config, device, output_dir, model_name):
    train_loader, val_loader, test_loader = loaders
    cfg_t, cfg_d = config["training"], config["distillation"]
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=cfg_t["base_lr"], weight_decay=cfg_t["weight_decay"])
    total_steps = cfg_t["epochs"] * len(train_loader)
    warmup_steps = cfg_t["warmup_epochs"] * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_val_acc, best_epoch = 0, 0
    ckpt_path = output_dir / f"student_{model_name}_best.pth"
    
    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"\n{'='*60}")
    print(f"Training {model_name} ({params:.2f}M) | Epochs: {cfg_t['epochs']} | T: {cfg_d['temperature']}")
    print(f"{'='*60}")
    
    start_time = time.time()
    step = 0
    
    for epoch in range(1, cfg_t["epochs"] + 1):
        student.train()
        total_loss, correct, total = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            with autocast():
                with torch.no_grad():
                    teacher_out = stacking(*[t(x) for t in teachers])
                student_out = student(x)
                
                # Distillation loss
                ce = F.cross_entropy(student_out, y, label_smoothing=cfg_t["label_smoothing"])
                T = cfg_d["temperature"]
                s_soft = F.log_softmax(student_out / T, dim=-1)
                t_soft = F.softmax(teacher_out / T, dim=-1)
                kl = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T ** 2)
                loss = cfg_d["alpha_ce"] * ce + cfg_d["alpha_kl"] * kl
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg_t["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            step += 1
            
            total_loss += loss.item() * x.size(0)
            correct += student_out.argmax(1).eq(y).sum().item()
            total += x.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.3f}", "acc": f"{correct/total:.3f}"})
        
        train_acc = correct / total
        _, val_acc = evaluate(student, val_loader, device)
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"E{epoch}: train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({"epoch": epoch, "model_state_dict": student.state_dict(), "val_acc": val_acc}, ckpt_path)
    
    # Load best model
    student.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
    _, test_acc = evaluate(student, test_loader, device)
    test_f1 = compute_f1(student, test_loader, device)
    train_time = time.time() - start_time
    
    print(f"\n✓ {model_name}: Best E{best_epoch} val={best_val_acc:.4f} | Test: acc={test_acc:.4f} F1={test_f1:.4f} | {train_time/60:.1f}min")
    
    return {"model": model_name, "params_m": params, "best_epoch": best_epoch, "best_val_acc": best_val_acc,
            "test_acc": test_acc, "test_f1": test_f1, "train_time_min": train_time/60}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="results/multi_student")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--students", type=str, default="ghostnet,mobilenetv2,mobilenetv3_small",
                        help="Comma-separated list of student models")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    CONFIG["training"]["epochs"] = args.epochs
    
    # Load dataset
    train_l, val_l, test_l, meta = quick_load_dataset(
        dataset_name=CONFIG["dataset"],
        dataset_root="/walnut_data/yqm/Dataset",
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"]
    )
    num_classes = meta["num_classes"]
    print(f"Dataset: {meta['dataset_name']} | Train: {meta['train_size']} | Val: {meta['val_size']} | Test: {meta['test_size']} | Classes: {num_classes}")
    
    # Load teachers
    print("\nLoading teachers...")
    teachers = load_teachers(CONFIG, num_classes, device)
    stacking = load_stacking(CONFIG, len(teachers), num_classes, device)
    
    # Train each student
    student_names = [s.strip() for s in args.students.split(",")]
    results = []
    
    for name in student_names:
        print(f"\n{'#'*60}")
        print(f"# Student: {name}")
        print(f"{'#'*60}")
        
        student = create_student_model(name, num_classes).to(device)
        result = train_student(student, teachers, stacking, (train_l, val_l, test_l), CONFIG, device, output_dir, name)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{'Model':<20} {'Params':>8} {'Val Acc':>10} {'Test Acc':>10} {'F1':>8} {'Time':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['model']:<20} {r['params_m']:>6.2f}M {r['best_val_acc']*100:>9.2f}% {r['test_acc']*100:>9.2f}% {r['test_f1']*100:>7.2f}% {r['train_time_min']:>6.1f}m")
    
    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
