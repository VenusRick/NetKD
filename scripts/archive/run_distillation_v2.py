#!/usr/bin/env python3
"""改进的知识蒸馏实验脚本 v2 - 数值稳定版"""

from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing import quick_load_dataset
from models.student_models_v2 import create_student_model, get_model_info
from models.teacher_models import ConvNeXtV2TinyTeacher, DenseNet121Teacher, MobileNetV3LargeTeacher, DynamicStackingModel

CONFIG = {
    "dataset": "ISCXVPN2016", "batch_size": 128, "num_workers": 8,
    "teachers": {
        "convnextv2_tiny": {"class": "ConvNeXtV2TinyTeacher", "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/convnextv2_tiny_eca_best.pth"},
        "densenet121": {"class": "DenseNet121Teacher", "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/densenet121_eca_best.pth"},
        "mobilenetv3": {"class": "MobileNetV3LargeTeacher", "use_eca": True,
            "checkpoint": "results/teacher_search_bs128/eca_ablation/mobilenetv3_eca_best.pth"},
    },
    "stacking": {"checkpoint": "results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth", "hidden_dim": 384},
    "student": {"epochs": 100, "base_lr": 0.001, "weight_decay": 1e-4, "warmup_epochs": 5, "grad_clip": 1.0},
    "distillation": {"mode": "kl", "temperature": 2.0, "alpha": 0.9},
}

def load_teachers(config, num_classes, device):
    teachers = []
    classes = {"ConvNeXtV2TinyTeacher": ConvNeXtV2TinyTeacher, "DenseNet121Teacher": DenseNet121Teacher, "MobileNetV3LargeTeacher": MobileNetV3LargeTeacher}
    for name, cfg in config["teachers"].items():
        model = classes[cfg["class"]](num_classes=num_classes, use_eca=cfg["use_eca"])
        ckpt_path = PROJECT_ROOT / cfg["checkpoint"]
        if ckpt_path.exists():
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False)["model_state_dict"])
            print(f"✓ {name}")
        model = model.to(device).eval()
        for p in model.parameters(): p.requires_grad = False
        teachers.append(model)
    return teachers

def load_stacking(config, num_teachers, num_classes, device):
    stacking = DynamicStackingModel(num_teachers=num_teachers, num_classes=num_classes, hidden_dim=config["stacking"]["hidden_dim"])
    ckpt = PROJECT_ROOT / config["stacking"]["checkpoint"]
    if ckpt.exists():
        stacking.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model_state_dict"])
        print("✓ stacking")
    stacking = stacking.to(device).eval()
    for p in stacking.parameters(): p.requires_grad = False
    return stacking

def distill_loss(s_logits, t_logits, labels, mode="kl", T=2.0, alpha=0.9):
    ce = F.cross_entropy(s_logits, labels)
    if mode == "ce": return ce, {"ce": ce.item(), "kl": 0, "total": ce.item()}
    
    s_soft = F.log_softmax(s_logits.clamp(-50,50) / T, dim=-1)
    t_soft = F.softmax(t_logits.clamp(-50,50) / T, dim=-1).clamp(min=1e-8)
    kl = F.kl_div(s_soft, t_soft, reduction="batchmean") * (T**2)
    if torch.isnan(kl): kl = ce * 0  # keep gradient graph
    
    total = kl if mode == "kl" else alpha*kl + (1-alpha)*ce
    return total, {"ce": ce.item(), "kl": kl.item(), "total": total.item()}

def evaluate(model, loader, device):
    model.eval()
    loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss += F.cross_entropy(out, y).item() * x.size(0)
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)
    return loss/total, correct/total

def compute_f1(model, loader, device):
    from sklearn.metrics import f1_score
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {"macro": f1_score(labels, preds, average="macro"), "weighted": f1_score(labels, preds, average="weighted")}

def train(student, teachers, stacking, loaders, config, device, out_dir, name):
    train_loader, val_loader, test_loader = loaders
    cfg_s, cfg_d = config["student"], config["distillation"]
    
    opt = torch.optim.AdamW(student.parameters(), lr=cfg_s["base_lr"], weight_decay=cfg_s["weight_decay"])
    total_steps = cfg_s["epochs"] * len(train_loader)
    warmup = cfg_s["warmup_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s+1)/warmup) if s < warmup else max(0.01, 0.5*(1+math.cos(math.pi*(s-warmup)/(total_steps-warmup)))))
    scaler = GradScaler()
    
    best_acc, best_epoch = 0, 0
    ckpt_path = out_dir / f"student_{name}_best.pth"
    print(f"\n=== Training {name} | Epochs: {cfg_s['epochs']} | T: {cfg_d['temperature']} ===")
    
    t0 = time.time()
    for epoch in range(1, cfg_s["epochs"]+1):
        student.train()
        loss_sum, correct, total = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"E{epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with autocast():
                with torch.no_grad():
                    t_out = stacking(*[t(x) for t in teachers])
                s_out = student(x)
                loss, _ = distill_loss(s_out, t_out, y, cfg_d["mode"], cfg_d["temperature"], cfg_d["alpha"])
            if torch.isnan(loss): continue
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg_s["grad_clip"])
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            loss_sum += loss.item()*x.size(0)
            correct += s_out.argmax(1).eq(y).sum().item()
            total += x.size(0)
        
        _, val_acc = evaluate(student, val_loader, device)
        if epoch % 10 == 0 or epoch == 1:
            print(f"E{epoch}: train_acc={correct/total:.4f}, val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            torch.save({"model_state_dict": student.state_dict()}, ckpt_path)
    
    student.load_state_dict(torch.load(ckpt_path, weights_only=False)["model_state_dict"])
    _, test_acc = evaluate(student, test_loader, device)
    f1 = compute_f1(student, test_loader, device)
    
    print(f"Done! Best E{best_epoch} val={best_acc:.4f} | Test: acc={test_acc:.4f} F1={f1['macro']:.4f} | {(time.time()-t0)/60:.1f}min")
    return {"test_accuracy": test_acc, "test_f1_macro": f1["macro"], "best_val_acc": best_acc, "best_epoch": best_epoch}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/distillation_v2")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--student", default="all", choices=["all","shufflenet","ghostnet","efficientnet_lite","mobilenetv3_small"])
    parser.add_argument("--mode", default="kl", choices=["ce","kl","both"])
    parser.add_argument("--temperature", type=float, default=2.0)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name()})" if device.type=="cuda" else ""))
    
    config = CONFIG.copy()
    config["student"]["epochs"] = args.epochs
    config["distillation"]["mode"] = args.mode
    config["distillation"]["temperature"] = args.temperature
    
    out_dir = PROJECT_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    
    train_l, val_l, test_l, meta = quick_load_dataset(dataset_name=config["dataset"], dataset_root="/walnut_data/yqm/Dataset", batch_size=config["batch_size"], num_workers=config["num_workers"])
    print(f"Dataset loaded: {meta}"); nc = meta["num_classes"]
    
    teachers = load_teachers(config, nc, device)
    stacking = load_stacking(config, len(teachers), nc, device)
    
    students = ["shufflenet","ghostnet","efficientnet_lite","mobilenetv3_small"] if args.student=="all" else [args.student]
    results = {}
    
    for st in students:
        model = create_student_model(st, nc).to(device)
        info = get_model_info(model)
        print(f"\n### {st} ({info['total_params_M']:.2f}M) ###")
        r = train(model, teachers, stacking, (train_l, val_l, test_l), config, device, out_dir, st)
        r["params_M"] = info["total_params_M"]
        results[st] = r
    
    with open(out_dir/"comparison_results.json", "w") as f: json.dump(results, f, indent=2)
    
    print("\n" + "="*60 + "\nFINAL RESULTS\n" + "="*60)
    for n, r in results.items():
        print(f"{n:<20} {r['params_M']:.2f}M  acc={r['test_accuracy']*100:.2f}%  F1={r['test_f1_macro']*100:.2f}%")

if __name__ == "__main__":
    main()
