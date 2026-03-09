#!/usr/bin/env python3
"""
消融实验脚本
使用已有的训练好的教师模型，进行：
1. 蒸馏温度消融 (T = 1, 2, 3, 4, 5, 10)
2. 损失权重消融 (alpha_ce = 0.3, 0.5, 0.7)
3. 不同教师组合对比
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime

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
from models.teacher_models import (
    DynamicStackingModel, ConvNeXtV2TinyTeacher,
    DenseNet121Teacher, MobileNetV3LargeTeacher, EfficientNetV2STeacher
)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


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


def load_existing_teachers(teacher_ckpts, num_classes, device):
    """加载已有的教师模型"""
    teachers = []
    teacher_classes = {
        "convnextv2_tiny": ConvNeXtV2TinyTeacher,
        "densenet121": DenseNet121Teacher,
        "mobilenetv3": MobileNetV3LargeTeacher,
        "efficientnetv2_s": EfficientNetV2STeacher,
    }
    
    for name, ckpt_path in teacher_ckpts.items():
        path = Path(ckpt_path)
        if not path.exists():
            print(f"  ! {name} checkpoint not found: {ckpt_path}")
            continue
        
        # 确定教师类
        for key, cls in teacher_classes.items():
            if key in name.lower():
                teacher = cls(num_classes=num_classes, use_eca=("eca" in name.lower()))
                break
        else:
            print(f"  ! Unknown teacher type: {name}")
            continue
        
        ckpt = torch.load(path, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"])
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teachers.append(teacher)
        print(f"  ✓ {name} loaded ({count_params(teacher):.2f}M)")
    
    return teachers


def train_student_kd(student, teachers, stacking, loaders, epochs, device, output_dir, name,
                     temperature=3.0, alpha_ce=0.5, alpha_kl=0.5):
    """知识蒸馏训练学生模型"""
    train_loader, val_loader, test_loader = loaders
    T = temperature
    
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
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                s_logits = student(x)
                
                with torch.no_grad():
                    if stacking is not None:
                        t_logits = [t(x) for t in teachers]
                        t_out = stacking(*t_logits)
                    else:
                        # 简单平均
                        t_logits = [t(x) for t in teachers]
                        t_out = torch.stack(t_logits, dim=0).mean(dim=0)
                
                loss_ce = F.cross_entropy(s_logits, y)
                loss_kl = F.kl_div(
                    F.log_softmax(s_logits / T, dim=-1),
                    F.softmax(t_out / T, dim=-1),
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
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f}@E{best_epoch})")
    
    # 加载最佳并测试
    ckpt = torch.load(output_dir / f"student_{name}_best.pth", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    test_acc = evaluate(student, test_loader, device)
    test_f1 = compute_f1(student, test_loader, device)
    
    return {
        "name": name,
        "best_epoch": best_epoch,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "temperature": temperature,
        "alpha_ce": alpha_ce,
        "alpha_kl": alpha_kl
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["temperature", "alpha", "all"], default="all")
    parser.add_argument("--student", type=str, default="repvit_m0_9")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="results/ablation_20251207")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载数据
    train_l, val_l, test_l, meta = quick_load_dataset(
        "ISCXVPN2016",
        dataset_root="/workspace/yqm/Dataset",
        batch_size=128,
        num_workers=8
    )
    num_classes = meta["num_classes"]
    print(f"Dataset: {num_classes} classes")
    
    # 定义已有的教师模型
    teacher_ckpts = {
        "convnextv2_tiny_eca": "results/teacher_search_bs128/eca_ablation/convnextv2_tiny_eca_best.pth",
        "densenet121_eca": "results/teacher_search_bs128/eca_ablation/densenet121_eca_best.pth",
        "mobilenetv3_eca": "results/teacher_search_bs128/eca_ablation/mobilenetv3_eca_best.pth",
    }
    
    # 加载教师
    print("\nLoading teachers...")
    teachers = load_existing_teachers(teacher_ckpts, num_classes, device)
    
    # 加载Stacking
    stacking_path = Path("results/teacher_search_bs128/stacking/stacking_replace_resnet_best.pth")
    stacking = DynamicStackingModel(num_classes, len(teachers), hidden_dim=384)
    if stacking_path.exists():
        ckpt = torch.load(stacking_path, map_location=device, weights_only=False)
        stacking.load_state_dict(ckpt["model_state_dict"])
        print("  ✓ Stacking loaded")
    stacking = stacking.to(device).eval()
    for p in stacking.parameters():
        p.requires_grad = False
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loaders = (train_l, val_l, test_l)
    all_results = {"temperature": {}, "alpha": {}}
    
    # 温度消融
    if args.experiment in ["temperature", "all"]:
        print("\n" + "="*60)
        print("实验1: 蒸馏温度消融")
        print("="*60)
        
        for T in [1, 2, 3, 4, 5, 10]:
            print(f"\n>>> Temperature = {T}")
            student = timm.create_model(args.student, pretrained=False, num_classes=num_classes, in_chans=1)
            student = student.to(device)
            
            start = time.time()
            result = train_student_kd(
                student, teachers, stacking, loaders, args.epochs, device, output_dir,
                f"T{T}_{args.student}", temperature=T, alpha_ce=0.5, alpha_kl=0.5
            )
            result["time_min"] = (time.time() - start) / 60
            all_results["temperature"][f"T={T}"] = result
            
            print(f"T={T}: test={result['test_acc']*100:.2f}% F1={result['test_f1']*100:.2f}%")
            
            del student
            torch.cuda.empty_cache()
    
    # Alpha消融
    if args.experiment in ["alpha", "all"]:
        print("\n" + "="*60)
        print("实验2: 损失权重消融")
        print("="*60)
        
        for alpha_ce in [0.3, 0.5, 0.7]:
            alpha_kl = 1 - alpha_ce
            print(f"\n>>> Alpha CE={alpha_ce}, KL={alpha_kl}")
            student = timm.create_model(args.student, pretrained=False, num_classes=num_classes, in_chans=1)
            student = student.to(device)
            
            start = time.time()
            result = train_student_kd(
                student, teachers, stacking, loaders, args.epochs, device, output_dir,
                f"alpha{alpha_ce}_{args.student}", temperature=3.0, alpha_ce=alpha_ce, alpha_kl=alpha_kl
            )
            result["time_min"] = (time.time() - start) / 60
            all_results["alpha"][f"CE={alpha_ce}"] = result
            
            print(f"Alpha CE={alpha_ce}: test={result['test_acc']*100:.2f}% F1={result['test_f1']*100:.2f}%")
            
            del student
            torch.cuda.empty_cache()
    
    # 保存结果
    with open(output_dir / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # 打印总结
    print("\n" + "="*70)
    print("消融实验结果总结")
    print("="*70)
    
    if all_results["temperature"]:
        print("\n温度消融:")
        print(f"{'Temperature':<15} {'Test Acc':<12} {'F1':<12} {'Time':<10}")
        print("-"*50)
        for key, r in all_results["temperature"].items():
            print(f"{key:<15} {r['test_acc']*100:.2f}%       {r['test_f1']*100:.2f}%     {r['time_min']:.1f}min")
        
        # 找最佳温度
        best_T = max(all_results["temperature"].items(), key=lambda x: x[1]["test_acc"])
        print(f"\n最佳温度: {best_T[0]} (test={best_T[1]['test_acc']*100:.2f}%)")
    
    if all_results["alpha"]:
        print("\n损失权重消融:")
        print(f"{'Alpha':<15} {'Test Acc':<12} {'F1':<12} {'Time':<10}")
        print("-"*50)
        for key, r in all_results["alpha"].items():
            print(f"{key:<15} {r['test_acc']*100:.2f}%       {r['test_f1']*100:.2f}%     {r['time_min']:.1f}min")
        
        # 找最佳alpha
        best_alpha = max(all_results["alpha"].items(), key=lambda x: x[1]["test_acc"])
        print(f"\n最佳权重: {best_alpha[0]} (test={best_alpha[1]['test_acc']*100:.2f}%)")
    
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
