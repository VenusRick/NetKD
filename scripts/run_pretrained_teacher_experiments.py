#!/usr/bin/env python3
"""
预训练教师模型增强实验
研究假设：预训练+微调的教师模型 → 更好的Stacking → 更好的学生模型

实验流程：
Phase A: 教师模型训练/微调对比
Phase B: Stacking集成对比  
Phase C: 学生模型蒸馏对比
Phase D: 消融实验
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
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import timm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset

# ============== 配置 ==============

EXPERIMENT_CONFIG = {
    "dataset": "ISCXVPN2016",
    "dataset_root": "/workspace/yqm/Dataset",
    "batch_size": 128,
    "num_workers": 8,
    
    # 教师模型配置
    "teachers": {
        # 从头训练 (Scratch)
        "convnextv2_tiny_scratch": {"model": "convnextv2_tiny", "pretrained": False, "epochs": 50},
        "densenet121_scratch": {"model": "densenet121", "pretrained": False, "epochs": 50},
        "efficientnetv2_s_scratch": {"model": "efficientnetv2_rw_s", "pretrained": False, "epochs": 50},
        
        # 预训练+微调 (Pretrained)
        "convnextv2_tiny_pretrained": {"model": "convnextv2_tiny", "pretrained": True, "epochs": 30},
        "densenet121_pretrained": {"model": "densenet121", "pretrained": True, "epochs": 30},
        "efficientnetv2_s_pretrained": {"model": "efficientnetv2_rw_s", "pretrained": True, "epochs": 30},
        
        # 更大的预训练模型
        "convnextv2_base_pretrained": {"model": "convnextv2_base", "pretrained": True, "epochs": 30},
        "efficientnetv2_m_pretrained": {"model": "efficientnetv2_m", "pretrained": True, "epochs": 30},
    },
    
    # 学生模型配置
    "students": {
        "repvit_m0_9": {"model": "repvit_m0_9"},
        "mobilenetv3_small_050": {"model": "mobilenetv3_small_050"},
        "ghostnet_050": {"model": "ghostnet_050"},
    },
    
    # 蒸馏配置
    "distillation": {
        "temperature": 3.0,
        "alpha_ce": 0.5,
        "alpha_kl": 0.5,
        "epochs": 100,
    }
}


# ============== 模型定义 ==============

class TeacherModel(nn.Module):
    """通用教师模型包装器"""
    def __init__(self, model_name: str, num_classes: int, pretrained: bool = True, in_channels: int = 1):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, 
                                       num_classes=num_classes, in_chans=in_channels)
        self.name = model_name
        self.pretrained = pretrained
        
    def forward(self, x):
        return self.model(x)


class DynamicStackingModel(nn.Module):
    """动态权重Stacking模型"""
    def __init__(self, num_classes: int, num_teachers: int, hidden_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.num_teachers = num_teachers
        
        # 动态权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(num_classes * num_teachers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_teachers),
            nn.Softmax(dim=-1)
        )
        
        # 特征融合
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * num_teachers, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, *teacher_logits):
        # teacher_logits: List of [B, C]
        concat = torch.cat(teacher_logits, dim=-1)  # [B, C*T]
        
        # 动态权重
        weights = self.weight_net(concat)  # [B, T]
        
        # 加权融合
        stacked = torch.stack(teacher_logits, dim=1)  # [B, T, C]
        weighted = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # [B, C]
        
        # 残差连接
        fusion_out = self.fusion(concat)
        
        return weighted + 0.1 * fusion_out


class StudentModel(nn.Module):
    """通用学生模型包装器"""
    def __init__(self, model_name: str, num_classes: int, in_channels: int = 1):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False,
                                       num_classes=num_classes, in_chans=in_channels)
        self.name = model_name
        
    def forward(self, x):
        return self.model(x)


# ============== 训练函数 ==============

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


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
def evaluate_detailed(model, loader, device):
    """详细评估，返回准确率、F1和预测结果"""
    model.eval()
    all_preds, all_labels = [], []
    for x, y in loader:
        preds = model(x.to(device)).argmax(1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(y.tolist())
    
    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return acc, f1, all_preds, all_labels


def train_teacher(model, loaders, epochs, device, output_dir, name, lr=1e-4):
    """训练教师模型"""
    train_loader, val_loader, test_loader = loaders
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
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
    history = {"train_acc": [], "val_acc": [], "train_loss": []}
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
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
            
            epoch_loss += loss.item() * x.size(0)
            epoch_correct += logits.argmax(1).eq(y).sum().item()
            epoch_total += x.size(0)
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{epoch_correct/epoch_total:.3f}")
        
        train_acc = epoch_correct / epoch_total
        val_acc = evaluate(model, val_loader, device)
        
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_loss"].append(epoch_loss / epoch_total)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "history": history
            }, output_dir / f"teacher_{name}_best.pth")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: train={train_acc:.4f} val={val_acc:.4f} (best={best_val_acc:.4f}@E{best_epoch})")
    
    # 加载最佳并测试
    ckpt = torch.load(output_dir / f"teacher_{name}_best.pth", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, test_f1, _, _ = evaluate_detailed(model, test_loader, device)
    
    return {
        "name": name,
        "best_epoch": best_epoch,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "params_m": count_params(model),
        "history": history
    }


def train_stacking(stacking, teachers, loaders, epochs, device, output_dir, name):
    """训练Stacking模型"""
    train_loader, val_loader, test_loader = loaders
    
    # 冻结教师
    for t in teachers:
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    
    optimizer = torch.optim.AdamW(stacking.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        stacking.train()
        pbar = tqdm(train_loader, desc=f"Stack E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                t_logits = [t(x) for t in teachers]
            
            out = stacking(*t_logits)
            loss = F.cross_entropy(out, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        
        scheduler.step()
        
        # 验证
        stacking.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                t_logits = [t(x) for t in teachers]
                out = stacking(*t_logits)
                correct += out.argmax(1).eq(y).sum().item()
                total += x.size(0)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": stacking.state_dict(),
                "val_acc": val_acc
            }, output_dir / f"stacking_{name}_best.pth")
        
        if epoch % 5 == 0:
            print(f"Stack E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f})")
    
    # 测试
    ckpt = torch.load(output_dir / f"stacking_{name}_best.pth", weights_only=False)
    stacking.load_state_dict(ckpt["model_state_dict"])
    
    stacking.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            t_logits = [t(x) for t in teachers]
            out = stacking(*t_logits)
            all_preds.extend(out.argmax(1).cpu().tolist())
            all_labels.extend(y.tolist())
    
    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return {"name": name, "val_acc": best_val_acc, "test_acc": test_acc, "test_f1": test_f1}


def train_student_kd(student, teachers, stacking, loaders, config, device, output_dir, name):
    """知识蒸馏训练学生模型"""
    train_loader, val_loader, test_loader = loaders
    cfg = config["distillation"]
    T = cfg["temperature"]
    alpha_ce = cfg["alpha_ce"]
    alpha_kl = cfg["alpha_kl"]
    epochs = cfg["epochs"]
    
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
    
    # 冻结教师和stacking
    for t in teachers:
        t.eval()
    stacking.eval()
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                s_logits = student(x)
                
                with torch.no_grad():
                    t_logits = [t(x) for t in teachers]
                    t_out = stacking(*t_logits)
                
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
            
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        
        val_acc = evaluate(student, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": student.state_dict(),
                "val_acc": val_acc
            }, output_dir / f"student_{name}_best.pth")
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f})")
    
    # 测试
    ckpt = torch.load(output_dir / f"student_{name}_best.pth", weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    test_acc, test_f1, _, _ = evaluate_detailed(student, test_loader, device)
    
    return {
        "name": name,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "params_m": count_params(student)
    }


# ============== 主实验函数 ==============

def run_phase_a(config, loaders, num_classes, device, output_dir):
    """Phase A: 教师模型训练对比"""
    print("\n" + "="*70)
    print("Phase A: 教师模型训练对比 (从头训练 vs 预训练+微调)")
    print("="*70)
    
    results = {}
    
    for name, cfg in config["teachers"].items():
        print(f"\n>>> Training: {name}")
        print(f"    Model: {cfg['model']}, Pretrained: {cfg['pretrained']}, Epochs: {cfg['epochs']}")
        
        try:
            model = TeacherModel(cfg["model"], num_classes, pretrained=cfg["pretrained"])
            model = model.to(device)
            print(f"    Params: {count_params(model):.2f}M")
            
            start_time = time.time()
            result = train_teacher(model, loaders, cfg["epochs"], device, output_dir, name)
            result["time_min"] = (time.time() - start_time) / 60
            results[name] = result
            
            print(f"    ✓ {name}: val={result['val_acc']*100:.2f}% test={result['test_acc']*100:.2f}% F1={result['test_f1']*100:.2f}%")
            
            # 释放显存
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    ✗ {name} failed: {e}")
            import traceback
            traceback.print_exc()
    
    return results


def run_phase_b(config, teacher_results, loaders, num_classes, device, output_dir):
    """Phase B: Stacking对比"""
    print("\n" + "="*70)
    print("Phase B: Stacking集成对比")
    print("="*70)
    
    results = {}
    
    # 定义不同的教师组合
    stacking_configs = {
        "scratch_trio": ["convnextv2_tiny_scratch", "densenet121_scratch", "efficientnetv2_s_scratch"],
        "pretrained_trio": ["convnextv2_tiny_pretrained", "densenet121_pretrained", "efficientnetv2_s_pretrained"],
        "pretrained_large": ["convnextv2_base_pretrained", "efficientnetv2_m_pretrained", "densenet121_pretrained"],
        "mixed": ["convnextv2_tiny_pretrained", "densenet121_scratch", "efficientnetv2_s_pretrained"],
    }
    
    for stack_name, teacher_names in stacking_configs.items():
        print(f"\n>>> Stacking: {stack_name}")
        print(f"    Teachers: {teacher_names}")
        
        # 检查所有教师是否都训练成功
        if not all(t in teacher_results for t in teacher_names):
            missing = [t for t in teacher_names if t not in teacher_results]
            print(f"    ! Skipping - missing teachers: {missing}")
            continue
        
        try:
            # 加载教师模型
            teachers = []
            for t_name in teacher_names:
                t_cfg = config["teachers"][t_name]
                teacher = TeacherModel(t_cfg["model"], num_classes, pretrained=False)
                ckpt = torch.load(output_dir / f"teacher_{t_name}_best.pth", weights_only=False)
                teacher.load_state_dict(ckpt["model_state_dict"])
                teacher = teacher.to(device).eval()
                teachers.append(teacher)
            
            # 创建并训练Stacking
            stacking = DynamicStackingModel(num_classes, len(teachers), hidden_dim=256)
            stacking = stacking.to(device)
            
            result = train_stacking(stacking, teachers, loaders, 20, device, output_dir, stack_name)
            results[stack_name] = result
            results[stack_name]["teachers"] = teacher_names
            
            print(f"    ✓ {stack_name}: val={result['val_acc']*100:.2f}% test={result['test_acc']*100:.2f}% F1={result['test_f1']*100:.2f}%")
            
            # 清理
            del teachers, stacking
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"    ✗ {stack_name} failed: {e}")
    
    return results


def run_phase_c(config, stacking_results, loaders, num_classes, device, output_dir):
    """Phase C: 学生模型蒸馏对比"""
    print("\n" + "="*70)
    print("Phase C: 学生模型蒸馏对比")
    print("="*70)
    
    results = {}
    
    # 找到最佳的Stacking配置
    best_stacking = max(stacking_results.items(), key=lambda x: x[1]["test_acc"])
    print(f"\n最佳Stacking配置: {best_stacking[0]} (test={best_stacking[1]['test_acc']*100:.2f}%)")
    
    # 为每个Stacking配置训练学生
    for stack_name, stack_result in stacking_results.items():
        print(f"\n>>> 使用 {stack_name} 训练学生模型")
        
        # 加载教师和Stacking
        teacher_names = stack_result["teachers"]
        teachers = []
        for t_name in teacher_names:
            t_cfg = config["teachers"][t_name]
            teacher = TeacherModel(t_cfg["model"], num_classes, pretrained=False)
            ckpt = torch.load(output_dir / f"teacher_{t_name}_best.pth", weights_only=False)
            teacher.load_state_dict(ckpt["model_state_dict"])
            teacher = teacher.to(device).eval()
            teachers.append(teacher)
        
        stacking = DynamicStackingModel(num_classes, len(teachers), hidden_dim=256)
        ckpt = torch.load(output_dir / f"stacking_{stack_name}_best.pth", weights_only=False)
        stacking.load_state_dict(ckpt["model_state_dict"])
        stacking = stacking.to(device).eval()
        
        results[stack_name] = {}
        
        for s_name, s_cfg in config["students"].items():
            print(f"\n    Training student: {s_name}")
            
            try:
                student = StudentModel(s_cfg["model"], num_classes)
                student = student.to(device)
                
                start_time = time.time()
                result = train_student_kd(student, teachers, stacking, loaders, config, device, output_dir, f"{stack_name}_{s_name}")
                result["time_min"] = (time.time() - start_time) / 60
                result["stacking"] = stack_name
                results[stack_name][s_name] = result
                
                print(f"    ✓ {s_name}: val={result['val_acc']*100:.2f}% test={result['test_acc']*100:.2f}% F1={result['test_f1']*100:.2f}%")
                
                del student
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"    ✗ {s_name} failed: {e}")
        
        del teachers, stacking
        torch.cuda.empty_cache()
    
    return results


def run_phase_d(config, best_stacking_name, loaders, num_classes, device, output_dir):
    """Phase D: 消融实验"""
    print("\n" + "="*70)
    print("Phase D: 消融实验")
    print("="*70)
    
    results = {"temperature": {}, "alpha": {}}
    
    # 使用最佳Stacking配置
    stack_result = json.load(open(output_dir / "stacking_results.json"))
    teacher_names = stack_result[best_stacking_name]["teachers"]
    
    # 加载教师和Stacking
    teachers = []
    for t_name in teacher_names:
        t_cfg = config["teachers"][t_name]
        teacher = TeacherModel(t_cfg["model"], num_classes, pretrained=False)
        ckpt = torch.load(output_dir / f"teacher_{t_name}_best.pth", weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"])
        teacher = teacher.to(device).eval()
        teachers.append(teacher)
    
    stacking = DynamicStackingModel(num_classes, len(teachers), hidden_dim=256)
    ckpt = torch.load(output_dir / f"stacking_{best_stacking_name}_best.pth", weights_only=False)
    stacking.load_state_dict(ckpt["model_state_dict"])
    stacking = stacking.to(device).eval()
    
    # D1: 温度消融
    print("\n>>> D1: 蒸馏温度消融")
    for T in [1, 2, 3, 4, 5, 10]:
        print(f"\n    Temperature: {T}")
        ablation_config = config.copy()
        ablation_config["distillation"] = config["distillation"].copy()
        ablation_config["distillation"]["temperature"] = T
        ablation_config["distillation"]["epochs"] = 50  # 缩短epoch
        
        student = StudentModel("repvit_m0_9", num_classes)
        student = student.to(device)
        
        result = train_student_kd(student, teachers, stacking, loaders, ablation_config, device, output_dir, f"ablation_T{T}")
        results["temperature"][f"T={T}"] = result
        print(f"    T={T}: test={result['test_acc']*100:.2f}%")
        
        del student
        torch.cuda.empty_cache()
    
    # D2: Alpha消融
    print("\n>>> D2: 损失权重消融")
    for alpha_ce in [0.3, 0.5, 0.7]:
        alpha_kl = 1 - alpha_ce
        print(f"\n    Alpha: CE={alpha_ce}, KL={alpha_kl}")
        
        ablation_config = config.copy()
        ablation_config["distillation"] = config["distillation"].copy()
        ablation_config["distillation"]["alpha_ce"] = alpha_ce
        ablation_config["distillation"]["alpha_kl"] = alpha_kl
        ablation_config["distillation"]["epochs"] = 50
        
        student = StudentModel("repvit_m0_9", num_classes)
        student = student.to(device)
        
        result = train_student_kd(student, teachers, stacking, loaders, ablation_config, device, output_dir, f"ablation_alpha{alpha_ce}")
        results["alpha"][f"CE={alpha_ce}"] = result
        print(f"    Alpha CE={alpha_ce}: test={result['test_acc']*100:.2f}%")
        
        del student
        torch.cuda.empty_cache()
    
    del teachers, stacking
    torch.cuda.empty_cache()
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["A", "B", "C", "D", "all"], default="all")
    parser.add_argument("--output-dir", type=str, default="results/pretrained_exp")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 加载数据
    config = EXPERIMENT_CONFIG
    train_l, val_l, test_l, meta = quick_load_dataset(
        config["dataset"],
        dataset_root=config["dataset_root"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )
    num_classes = meta["num_classes"]
    print(f"Dataset: {num_classes} classes")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loaders = (train_l, val_l, test_l)
    all_results = {}
    
    # Phase A
    if args.phase in ["A", "all"]:
        teacher_results = run_phase_a(config, loaders, num_classes, device, output_dir)
        all_results["teachers"] = teacher_results
        with open(output_dir / "teacher_results.json", "w") as f:
            # 移除不可序列化的history
            save_results = {k: {kk: vv for kk, vv in v.items() if kk != "history"} for k, v in teacher_results.items()}
            json.dump(save_results, f, indent=2)
    
    # Phase B
    if args.phase in ["B", "all"]:
        if "teachers" not in all_results:
            all_results["teachers"] = json.load(open(output_dir / "teacher_results.json"))
        stacking_results = run_phase_b(config, all_results["teachers"], loaders, num_classes, device, output_dir)
        all_results["stacking"] = stacking_results
        with open(output_dir / "stacking_results.json", "w") as f:
            json.dump(stacking_results, f, indent=2)
    
    # Phase C
    if args.phase in ["C", "all"]:
        if "stacking" not in all_results:
            all_results["stacking"] = json.load(open(output_dir / "stacking_results.json"))
        student_results = run_phase_c(config, all_results["stacking"], loaders, num_classes, device, output_dir)
        all_results["students"] = student_results
        with open(output_dir / "student_results.json", "w") as f:
            json.dump(student_results, f, indent=2)
    
    # Phase D
    if args.phase in ["D", "all"]:
        if "stacking" not in all_results:
            all_results["stacking"] = json.load(open(output_dir / "stacking_results.json"))
        best_stacking = max(all_results["stacking"].items(), key=lambda x: x[1]["test_acc"])[0]
        ablation_results = run_phase_d(config, best_stacking, loaders, num_classes, device, output_dir)
        all_results["ablation"] = ablation_results
        with open(output_dir / "ablation_results.json", "w") as f:
            json.dump(ablation_results, f, indent=2)
    
    # 打印最终总结
    print("\n" + "="*70)
    print("实验完成! 结果总结")
    print("="*70)
    
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
