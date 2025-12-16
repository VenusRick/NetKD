#!/usr/bin/env python3
"""
简化版 KD 实验 - 使用平均集成教师（不需要预训练模型）
仅用于完成 STEP 3 & 4 的 CE+KL 对比实验
"""
import argparse, json, math, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.data import Subset
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import numpy as np
import timm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset
from models.student_models_v2 import create_student_model_extended as create_student_model

def subsample_dataset(dataset, train_fraction, seed=42):
    if train_fraction >= 1.0:
        return dataset
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    subset_size = int(len(dataset) * train_fraction)
    return Subset(dataset, indices[:subset_size])

@torch.no_grad()
def evaluate_detailed(model, loader, device):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        x = x.to(device)
        outputs = model(x)
        preds.extend(outputs.argmax(1).cpu().tolist())
        labels.extend(y.tolist())
    
    acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    return {
        "accuracy": acc,
        "macro_f1": f1_score(labels, preds, average='macro'),
        "precision": precision_score(labels, preds, average='macro', zero_division=0),
        "recall": recall_score(labels, preds, average='macro', zero_division=0)
    }

def train_with_kd(student, teacher_models, train_loader, val_loader, test_loader,
                  use_kd, config, device, output_dir):
    train_cfg = config["training"]
    T = config.get("temperature", 3.0)
    
    optimizer = torch.optim.AdamW(student.parameters(),
                                  lr=train_cfg["base_lr"],
                                  weight_decay=train_cfg["weight_decay"])
    
    total_steps = train_cfg["epochs"] * len(train_loader)
    warmup_steps = train_cfg["warmup_epochs"] * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler('cuda')
    
    best_val_f1 = 0
    best_epoch = 0
    
    for epoch in range(1, train_cfg["epochs"] + 1):
        student.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"E{epoch}/{train_cfg['epochs']}", leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                student_logits = student(images)
                ce_loss = F.cross_entropy(student_logits, labels)
                
                if use_kd and teacher_models:
                    # 简单平均集成教师预测
                    with torch.no_grad():
                        teacher_logits_list = [t(images) for t in teacher_models]
                        teacher_logits = torch.stack(teacher_logits_list).mean(dim=0)
                    
                    # KL散度
                    student_soft = F.log_softmax(student_logits / T, dim=1)
                    teacher_soft = F.softmax(teacher_logits / T, dim=1)
                    kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (T * T)
                    
                    loss = 0.5 * ce_loss + 0.5 * kl_loss
                else:
                    loss = ce_loss
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), train_cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        # 每5个epoch评估
        if epoch % 5 == 0 or epoch == train_cfg["epochs"]:
            val_metrics = evaluate_detailed(student, val_loader, device)
            val_f1 = val_metrics["macro_f1"]
            
            print(f"  E{epoch}: loss={epoch_loss/len(train_loader):.4f}, "
                  f"val_acc={val_metrics['accuracy']:.4f}, val_f1={val_f1:.4f}")
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch = epoch
                torch.save(student.state_dict(), output_dir / "best_model.pth")
    
    # 加载最佳模型并测试
    student.load_state_dict(torch.load(output_dir / "best_model.pth", weights_only=True))
    test_metrics = evaluate_detailed(student, test_loader, device)
    
    print(f"\n🎉 训练完成! 最佳验证 F1: {best_val_f1:.4f} (epoch {best_epoch})")
    print(f"测试结果: acc={test_metrics['accuracy']:.4f}, f1={test_metrics['macro_f1']:.4f}")
    
    return {"best_val_f1": best_val_f1, "best_epoch": best_epoch, "test_metrics": test_metrics}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=str, required=True)
    parser.add_argument("--use_kd", action="store_true", help="Use KD (CE+KL)")
    parser.add_argument("--train_fraction", type=float, default=1.0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="results/kd_simple")
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}")
    print(f"使用设备: {device}")
    
    # 配置
    CONFIG = {
        "data_path": "/workspace/yqm/Dataset/ISCXVPN2016",
        "batch_size": 128,
        "num_workers": 8,
        "training": {"epochs": 100, "base_lr": 0.001, "weight_decay": 1e-4,
                     "warmup_epochs": 5, "grad_clip": 1.0},
        "temperature": 3.0
    }
    
    # 输出目录
    kd_tag = "ce_kl" if args.use_kd else "ce_only"
    output_dir = PROJECT_ROOT / args.output_dir / f"{args.student}_{kd_tag}_frac{args.train_fraction}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print(f"\n加载数据集 (训练数据比例: {args.train_fraction})...")
    train_loader, val_loader, test_loader, dataset_info = quick_load_dataset(
        CONFIG["data_path"], batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"]
    )
    
    if args.train_fraction < 1.0:
        train_dataset = train_loader.dataset
        subsampled = subsample_dataset(train_dataset, args.train_fraction)
        train_loader = torch.utils.data.DataLoader(
            subsampled, batch_size=CONFIG["batch_size"],
            shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True
        )
        print(f"训练集大小: {len(train_dataset)} -> {len(subsampled)}")
    
    num_classes = dataset_info["num_classes"]
    
    # 创建教师模型（如果使用KD）
    teacher_models = None
    if args.use_kd:
        print("\n创建教师模型（使用timm预训练）...")
        teacher_names = ['resnet50', 'densenet121', 'mobilenetv3_large_100']
        teacher_models = []
        for name in teacher_names:
            t = timm.create_model(name, pretrained=True, num_classes=num_classes, in_chans=1)
            t = t.to(device).eval()
            for p in t.parameters():
                p.requires_grad = False
            teacher_models.append(t)
        print(f"创建了 {len(teacher_models)} 个教师模型")
    
    # 创建学生
    print(f"\n创建学生模型: {args.student}")
    student = create_student_model(args.student, num_classes=num_classes, in_chans=1)
    student = student.to(device)
    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"学生模型参数量: {params:.2f}M")
    
    # 训练
    print(f"\n开始训练 (KD: {args.use_kd})...")
    results = train_with_kd(student, teacher_models, train_loader, val_loader, test_loader,
                            args.use_kd, CONFIG, device, output_dir)
    
    # 保存结果
    output_json = {
        "experiment_type": "kd_simple",
        "student_name": args.student,
        "kd_config_id": kd_tag,
        "train_fraction": args.train_fraction,
        "params_m": params,
        "test_acc": results["test_metrics"]["accuracy"],
        "macro_f1": results["test_metrics"]["macro_f1"],
        "precision": results["test_metrics"]["precision"],
        "recall": results["test_metrics"]["recall"],
        "best_val_f1": results["best_val_f1"],
        "best_epoch": results["best_epoch"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(output_json, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_dir}/metrics.json")

if __name__ == "__main__":
    main()
