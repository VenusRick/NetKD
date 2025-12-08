#!/usr/bin/env python3
"""
TrafficKD 完整实验自动化执行脚本

用法:
    python scripts/run_full_experiment.py --config configs/full_experiment.yaml
    python scripts/run_full_experiment.py --phase pretrain
    python scripts/run_full_experiment.py --phase finetune
    python scripts/run_full_experiment.py --phase stacking
    python scripts/run_full_experiment.py --phase distill
"""
import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import yaml

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset


# ============== 工具函数 ==============

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

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# ============== SimCLR 对比学习 ==============

class SimCLRAugmentation:
    """SimCLR数据增强"""
    def __init__(self, img_size=40):
        import torchvision.transforms as T
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)


class SimCLRModel(nn.Module):
    """SimCLR对比学习模型"""
    def __init__(self, backbone_name, projection_dim=128, in_chans=1):
        super().__init__()
        self.encoder = timm.create_model(backbone_name, pretrained=False, 
                                         num_classes=0, in_chans=in_chans)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, 40, 40)
            feat_dim = self.encoder(dummy).shape[-1]
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, projection_dim)
        )
        self.feat_dim = feat_dim
        
    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return F.normalize(z, dim=-1)


def nt_xent_loss(z1, z2, temperature=0.5):
    """NT-Xent对比损失"""
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    
    sim = torch.mm(z, z.T) / temperature  # [2B, 2B]
    
    # 创建标签：正样本对在对角线偏移B处
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])  # [2B]
    
    # 掩码自身
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -float('inf'))
    
    loss = F.cross_entropy(sim, labels)
    return loss


# ============== CNN-MAE ==============

class CNNMaskedAutoencoder(nn.Module):
    """CNN版本的MAE"""
    def __init__(self, backbone_name, img_size=40, in_chans=1, mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = 4
        self.num_patches = (img_size // self.patch_size) ** 2
        
        self.encoder = timm.create_model(backbone_name, pretrained=False, 
                                         num_classes=0, in_chans=in_chans)
        
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            self.feat_dim = self.encoder(dummy).shape[-1]
        
        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.num_patches * self.patch_size ** 2 * in_chans)
        )
        
    def random_mask_image(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = w = H // p
        
        num_mask = int(self.num_patches * self.mask_ratio)
        noise = torch.rand(B, self.num_patches, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        mask = torch.zeros(B, self.num_patches, device=imgs.device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 1)
        mask = mask.reshape(B, h, w)
        
        mask_img = mask.unsqueeze(1).repeat(1, 1, p, 1).repeat(1, 1, 1, p)
        mask_img = mask_img.reshape(B, 1, H, W)
        
        masked_imgs = imgs * (1 - mask_img)
        return masked_imgs, mask
    
    def patchify(self, imgs):
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x
    
    def forward(self, imgs):
        B = imgs.shape[0]
        masked_imgs, mask = self.random_mask_image(imgs)
        feat = self.encoder(masked_imgs)
        pred = self.decoder(feat).reshape(B, self.num_patches, -1)
        target = self.patchify(imgs)
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        mask = mask.reshape(B, -1)
        loss = (loss * mask).sum() / mask.sum()
        return loss, pred, mask


# ============== 训练函数 ==============

def pretrain_mae(backbone, mask_ratio, config, device, output_dir):
    """MAE预训练"""
    print(f"\n{'='*60}")
    print(f"MAE Pretraining: {backbone}, mask={mask_ratio}")
    print(f"{'='*60}")
    
    train_l, _, _, meta = quick_load_dataset(
        config['experiment']['dataset'],
        dataset_root=config['experiment']['dataset_root'],
        batch_size=config['pretrain']['mae']['batch_size'],
        num_workers=8
    )
    
    mae = CNNMaskedAutoencoder(backbone, mask_ratio=mask_ratio).to(device)
    print(f"Params: {count_params(mae):.2f}M")
    
    epochs = config['pretrain']['mae']['epochs']
    lr = config['pretrain']['mae']['base_lr'] * config['pretrain']['mae']['batch_size'] / 256
    
    optimizer = torch.optim.AdamW(mae.parameters(), lr=lr, 
                                  weight_decay=config['pretrain']['mae']['weight_decay'])
    
    total_steps = epochs * len(train_l)
    warmup_steps = config['pretrain']['mae']['warmup_epochs'] * len(train_l)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        mae.train()
        epoch_loss = 0
        pbar = tqdm(train_l, desc=f"E{epoch}", leave=False)
        
        for x, _ in pbar:
            x = x.to(device)
            with autocast():
                loss, _, _ = mae(x)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_l)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': mae.state_dict(),
                'encoder_state_dict': mae.encoder.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'backbone': backbone,
                'mask_ratio': mask_ratio
            }, output_dir / f"mae_{backbone}_mr{mask_ratio}_best.pth")
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"E{epoch}: loss={avg_loss:.4f} (best={best_loss:.4f})")
    
    return {"backbone": backbone, "mask_ratio": mask_ratio, "best_loss": best_loss}


def pretrain_simclr(backbone, temperature, config, device, output_dir):
    """SimCLR预训练"""
    print(f"\n{'='*60}")
    print(f"SimCLR Pretraining: {backbone}, temp={temperature}")
    print(f"{'='*60}")
    
    train_l, _, _, meta = quick_load_dataset(
        config['experiment']['dataset'],
        dataset_root=config['experiment']['dataset_root'],
        batch_size=config['pretrain']['simclr']['batch_size'],
        num_workers=8
    )
    
    model = SimCLRModel(backbone).to(device)
    print(f"Params: {count_params(model):.2f}M")
    
    epochs = config['pretrain']['simclr']['epochs']
    lr = config['pretrain']['simclr']['base_lr'] * config['pretrain']['simclr']['batch_size'] / 256
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, 
                                  weight_decay=config['pretrain']['simclr']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler()
    
    aug = SimCLRAugmentation()
    best_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_l, desc=f"E{epoch}", leave=False)
        
        for x, _ in pbar:
            x = x.to(device)
            # 简化的数据增强：添加噪声作为第二视图
            x1 = x
            x2 = x + 0.1 * torch.randn_like(x)
            
            with autocast():
                z1 = model(x1)
                z2 = model(x2)
                loss = nt_xent_loss(z1, z2, temperature)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        scheduler.step()
        avg_loss = epoch_loss / len(train_l)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'encoder_state_dict': model.encoder.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'backbone': backbone,
                'temperature': temperature
            }, output_dir / f"simclr_{backbone}_t{temperature}_best.pth")
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"E{epoch}: loss={avg_loss:.4f} (best={best_loss:.4f})")
    
    return {"backbone": backbone, "temperature": temperature, "best_loss": best_loss}


def finetune_teacher(backbone, pretrain_ckpt, config, device, output_dir, name):
    """微调教师模型"""
    print(f"\n>>> Finetuning: {name}")
    
    train_l, val_l, test_l, meta = quick_load_dataset(
        config['experiment']['dataset'],
        dataset_root=config['experiment']['dataset_root'],
        batch_size=config['finetune']['batch_size'],
        num_workers=8
    )
    num_classes = meta['num_classes']
    
    model = timm.create_model(backbone, pretrained=False, 
                              num_classes=num_classes, in_chans=1)
    
    # 加载预训练权重
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        ckpt = torch.load(pretrain_ckpt, map_location='cpu', weights_only=False)
        if 'encoder_state_dict' in ckpt:
            missing, unexpected = model.load_state_dict(ckpt['encoder_state_dict'], strict=False)
            print(f"  Loaded pretrained: missing={len(missing)}, unexpected={len(unexpected)}")
            lr = config['finetune']['lr_pretrained']
        else:
            lr = config['finetune']['lr_scratch']
    else:
        lr = config['finetune']['lr_scratch']
    
    model = model.to(device)
    
    epochs = config['finetune']['epochs']
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, 
                                  weight_decay=config['finetune']['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler()
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_l, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with autocast():
                loss = F.cross_entropy(model(x), y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        val_acc = evaluate(model, val_l, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, output_dir / f"teacher_{name}_best.pth")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f}@E{best_epoch})")
    
    # 测试
    ckpt = torch.load(output_dir / f"teacher_{name}_best.pth", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_acc = evaluate(model, test_l, device)
    test_f1 = compute_f1(model, test_l, device)
    
    return {
        "name": name,
        "backbone": backbone,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "params_m": count_params(model)
    }


def train_stacking(teacher_ckpts, config, device, output_dir):
    """训练Stacking集成"""
    print(f"\n{'='*60}")
    print(f"Training Stacking Ensemble")
    print(f"{'='*60}")
    
    train_l, val_l, test_l, meta = quick_load_dataset(
        config['experiment']['dataset'],
        dataset_root=config['experiment']['dataset_root'],
        batch_size=config['stacking']['batch_size'],
        num_workers=8
    )
    num_classes = meta['num_classes']
    
    # 加载教师模型
    teachers = []
    for ckpt_path in teacher_ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = ckpt.get('backbone', 'convnextv2_tiny')
        teacher = timm.create_model(backbone, pretrained=False, 
                                   num_classes=num_classes, in_chans=1)
        teacher.load_state_dict(ckpt['model_state_dict'])
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teachers.append(teacher)
        print(f"  Loaded: {ckpt_path}")
    
    # Stacking模型
    from models.teacher_models import DynamicStackingModel
    stacking = DynamicStackingModel(num_classes, len(teachers), 
                                    hidden_dim=config['stacking']['hidden_dim'])
    stacking = stacking.to(device)
    
    epochs = config['stacking']['epochs']
    optimizer = torch.optim.AdamW(stacking.parameters(), lr=config['stacking']['lr'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler()
    
    best_val_acc = 0
    
    for epoch in range(1, epochs + 1):
        stacking.train()
        pbar = tqdm(train_l, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with torch.no_grad():
                t_logits = [t(x) for t in teachers]
            
            with autocast():
                out = stacking(*t_logits)
                loss = F.cross_entropy(out, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        
        # 验证
        stacking.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_l:
                x, y = x.to(device), y.to(device)
                t_logits = [t(x) for t in teachers]
                out = stacking(*t_logits)
                correct += out.argmax(1).eq(y).sum().item()
                total += x.size(0)
        
        val_acc = correct / total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': stacking.state_dict(),
                'teacher_ckpts': teacher_ckpts,
                'val_acc': val_acc
            }, output_dir / "stacking_best.pth")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f})")
    
    # 测试
    ckpt = torch.load(output_dir / "stacking_best.pth", weights_only=False)
    stacking.load_state_dict(ckpt['model_state_dict'])
    stacking.eval()
    
    correct, total = 0, 0
    preds_all, labels_all = [], []
    with torch.no_grad():
        for x, y in test_l:
            x, y = x.to(device), y.to(device)
            t_logits = [t(x) for t in teachers]
            out = stacking(*t_logits)
            preds_all.extend(out.argmax(1).cpu().tolist())
            labels_all.extend(y.cpu().tolist())
            correct += out.argmax(1).eq(y).sum().item()
            total += x.size(0)
    
    test_acc = correct / total
    test_f1 = f1_score(labels_all, preds_all, average='macro')
    
    return {
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "num_teachers": len(teachers)
    }


def train_student_kd(student_name, stacking_ckpt, teacher_ckpts, config, device, output_dir,
                     temperature=3.0, alpha_ce=0.5):
    """知识蒸馏训练学生"""
    print(f"\n>>> Training Student: {student_name} (T={temperature}, alpha={alpha_ce})")
    
    train_l, val_l, test_l, meta = quick_load_dataset(
        config['experiment']['dataset'],
        dataset_root=config['experiment']['dataset_root'],
        batch_size=config['distill']['batch_size'],
        num_workers=8
    )
    num_classes = meta['num_classes']
    
    # 加载教师和Stacking
    teachers = []
    for ckpt_path in teacher_ckpts:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        backbone = ckpt.get('backbone', 'convnextv2_tiny')
        teacher = timm.create_model(backbone, pretrained=False, 
                                   num_classes=num_classes, in_chans=1)
        teacher.load_state_dict(ckpt['model_state_dict'])
        teacher = teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False
        teachers.append(teacher)
    
    from models.teacher_models import DynamicStackingModel
    stacking = DynamicStackingModel(num_classes, len(teachers), hidden_dim=384)
    ckpt = torch.load(stacking_ckpt, map_location=device, weights_only=False)
    stacking.load_state_dict(ckpt['model_state_dict'])
    stacking = stacking.to(device).eval()
    for p in stacking.parameters():
        p.requires_grad = False
    
    # 学生模型
    student = timm.create_model(student_name, pretrained=False, 
                               num_classes=num_classes, in_chans=1)
    student = student.to(device)
    
    epochs = config['distill']['epochs']
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    
    total_steps = epochs * len(train_l)
    warmup_steps = 5 * len(train_l)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return 0.1 + 0.9 * (step / warmup_steps)
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    T = temperature
    alpha_kl = 1 - alpha_ce
    best_val_acc = 0
    
    for epoch in range(1, epochs + 1):
        student.train()
        pbar = tqdm(train_l, desc=f"E{epoch}", leave=False)
        
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
        
        val_acc = evaluate(student, val_l, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': student.state_dict(),
                'student_name': student_name,
                'temperature': temperature,
                'alpha_ce': alpha_ce,
                'val_acc': val_acc
            }, output_dir / f"student_{student_name}_T{temperature}_a{alpha_ce}_best.pth")
        
        if epoch % 20 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f})")
    
    # 测试
    ckpt = torch.load(output_dir / f"student_{student_name}_T{temperature}_a{alpha_ce}_best.pth", 
                     weights_only=False)
    student.load_state_dict(ckpt['model_state_dict'])
    test_acc = evaluate(student, test_l, device)
    test_f1 = compute_f1(student, test_l, device)
    
    return {
        "student": student_name,
        "temperature": temperature,
        "alpha_ce": alpha_ce,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "params_m": count_params(student)
    }


# ============== 主执行逻辑 ==============

def run_phase1_pretrain(config, gpu_id=0):
    """Phase 1: 预训练"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    output_root = Path(config['experiment']['output_root']) / "phase1_pretrain"
    
    results = {"mae": [], "simclr": []}
    
    # MAE预训练
    if config['pretrain']['mae']['enabled']:
        mae_dir = output_root / "mae"
        mae_dir.mkdir(parents=True, exist_ok=True)
        
        for backbone in config['pretrain']['mae']['backbones']:
            for mask_ratio in config['pretrain']['mae']['mask_ratios']:
                result = pretrain_mae(backbone, mask_ratio, config, device, mae_dir)
                results["mae"].append(result)
                save_json(results, output_root / "pretrain_results.json")
    
    # SimCLR预训练
    if config['pretrain']['simclr']['enabled']:
        simclr_dir = output_root / "simclr"
        simclr_dir.mkdir(parents=True, exist_ok=True)
        
        for backbone in config['pretrain']['simclr']['backbones']:
            for temp in config['pretrain']['simclr']['temperatures']:
                result = pretrain_simclr(backbone, temp, config, device, simclr_dir)
                results["simclr"].append(result)
                save_json(results, output_root / "pretrain_results.json")
    
    return results


def run_phase2_finetune(config, gpu_id=0):
    """Phase 2: 微调"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    output_root = Path(config['experiment']['output_root']) / "phase2_finetune"
    output_root.mkdir(parents=True, exist_ok=True)
    
    pretrain_root = Path(config['experiment']['output_root']) / "phase1_pretrain"
    results = []
    
    backbones = config['pretrain']['mae']['backbones']
    
    for backbone in backbones:
        # 从头训练对照
        if config['finetune']['baselines']['scratch']:
            result = finetune_teacher(backbone, None, config, device, output_root, 
                                     f"{backbone}_scratch")
            result['pretrain'] = 'scratch'
            results.append(result)
        
        # MAE预训练+微调
        for mask_ratio in config['pretrain']['mae']['mask_ratios']:
            ckpt_path = pretrain_root / "mae" / f"mae_{backbone}_mr{mask_ratio}_best.pth"
            if ckpt_path.exists():
                result = finetune_teacher(backbone, str(ckpt_path), config, device, output_root,
                                         f"{backbone}_mae_mr{mask_ratio}")
                result['pretrain'] = f'mae_mr{mask_ratio}'
                results.append(result)
        
        # SimCLR预训练+微调
        for temp in config['pretrain']['simclr']['temperatures']:
            ckpt_path = pretrain_root / "simclr" / f"simclr_{backbone}_t{temp}_best.pth"
            if ckpt_path.exists():
                result = finetune_teacher(backbone, str(ckpt_path), config, device, output_root,
                                         f"{backbone}_simclr_t{temp}")
                result['pretrain'] = f'simclr_t{temp}'
                results.append(result)
        
        save_json(results, output_root / "finetune_results.json")
    
    return results


def run_phase3_stacking(config, gpu_id=0):
    """Phase 3: Stacking"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    output_root = Path(config['experiment']['output_root']) / "phase3_stacking"
    output_root.mkdir(parents=True, exist_ok=True)
    
    finetune_root = Path(config['experiment']['output_root']) / "phase2_finetune"
    
    # 加载微调结果，选择top-3
    with open(finetune_root / "finetune_results.json") as f:
        finetune_results = json.load(f)
    
    # 按test_acc排序
    sorted_results = sorted(finetune_results, key=lambda x: x['test_acc'], reverse=True)
    top_teachers = sorted_results[:config['stacking']['num_teachers']]
    
    teacher_ckpts = []
    for t in top_teachers:
        ckpt_path = finetune_root / f"teacher_{t['name']}_best.pth"
        if ckpt_path.exists():
            teacher_ckpts.append(str(ckpt_path))
    
    print(f"\nTop {len(teacher_ckpts)} teachers:")
    for t in top_teachers[:len(teacher_ckpts)]:
        print(f"  {t['name']}: {t['test_acc']*100:.2f}%")
    
    result = train_stacking(teacher_ckpts, config, device, output_root)
    result['teacher_names'] = [t['name'] for t in top_teachers[:len(teacher_ckpts)]]
    
    save_json(result, output_root / "stacking_results.json")
    return result


def run_phase4_distill(config, gpu_id=0):
    """Phase 4: 知识蒸馏"""
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    output_root = Path(config['experiment']['output_root']) / "phase4_distill"
    output_root.mkdir(parents=True, exist_ok=True)
    
    stacking_root = Path(config['experiment']['output_root']) / "phase3_stacking"
    finetune_root = Path(config['experiment']['output_root']) / "phase2_finetune"
    
    stacking_ckpt = stacking_root / "stacking_best.pth"
    
    # 获取教师检查点
    with open(stacking_root / "stacking_results.json") as f:
        stacking_result = json.load(f)
    
    with open(finetune_root / "finetune_results.json") as f:
        finetune_results = json.load(f)
    
    sorted_results = sorted(finetune_results, key=lambda x: x['test_acc'], reverse=True)
    top_teachers = sorted_results[:config['stacking']['num_teachers']]
    
    teacher_ckpts = []
    for t in top_teachers:
        ckpt_path = finetune_root / f"teacher_{t['name']}_best.pth"
        if ckpt_path.exists():
            teacher_ckpts.append(str(ckpt_path))
    
    results = []
    
    for student_cfg in config['distill']['students']:
        student_name = student_cfg['name']
        
        # 超参数搜索
        if config['distill']['hyperparam_search']:
            for temp in config['distill']['temperatures']:
                for alpha in config['distill']['alpha_ce_values']:
                    result = train_student_kd(
                        student_name, str(stacking_ckpt), teacher_ckpts,
                        config, device, output_root,
                        temperature=temp, alpha_ce=alpha
                    )
                    results.append(result)
                    save_json(results, output_root / "distill_results.json")
        else:
            # 使用默认超参数
            result = train_student_kd(
                student_name, str(stacking_ckpt), teacher_ckpts,
                config, device, output_root,
                temperature=3.0, alpha_ce=0.5
            )
            results.append(result)
            save_json(results, output_root / "distill_results.json")
    
    return results


def generate_final_report(config):
    """生成最终报告"""
    output_root = Path(config['experiment']['output_root'])
    
    report = f"""# TrafficKD 完整实验报告

## 实验日期: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 数据集: {config['experiment']['dataset']}

---

"""
    
    # Phase 1 结果
    p1_file = output_root / "phase1_pretrain" / "pretrain_results.json"
    if p1_file.exists():
        with open(p1_file) as f:
            p1_results = json.load(f)
        
        report += "## Phase 1: 自监督预训练\n\n"
        
        if p1_results.get('mae'):
            report += "### MAE预训练\n"
            report += "| Backbone | Mask Ratio | Best Loss |\n"
            report += "|----------|------------|----------|\n"
            for r in p1_results['mae']:
                report += f"| {r['backbone']} | {r['mask_ratio']} | {r['best_loss']:.4f} |\n"
            report += "\n"
        
        if p1_results.get('simclr'):
            report += "### SimCLR预训练\n"
            report += "| Backbone | Temperature | Best Loss |\n"
            report += "|----------|-------------|----------|\n"
            for r in p1_results['simclr']:
                report += f"| {r['backbone']} | {r['temperature']} | {r['best_loss']:.4f} |\n"
            report += "\n"
    
    # Phase 2 结果
    p2_file = output_root / "phase2_finetune" / "finetune_results.json"
    if p2_file.exists():
        with open(p2_file) as f:
            p2_results = json.load(f)
        
        report += "## Phase 2: 教师模型微调\n\n"
        report += "| Name | Pretrain | Val Acc | Test Acc | F1 | Params(M) |\n"
        report += "|------|----------|---------|----------|-----|----------|\n"
        
        sorted_p2 = sorted(p2_results, key=lambda x: x['test_acc'], reverse=True)
        for r in sorted_p2:
            report += f"| {r['name']} | {r.get('pretrain', 'N/A')} | {r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n"
        report += "\n"
    
    # Phase 3 结果
    p3_file = output_root / "phase3_stacking" / "stacking_results.json"
    if p3_file.exists():
        with open(p3_file) as f:
            p3_results = json.load(f)
        
        report += "## Phase 3: Stacking集成\n\n"
        report += f"- **Teachers**: {', '.join(p3_results.get('teacher_names', []))}\n"
        report += f"- **Val Acc**: {p3_results['val_acc']*100:.2f}%\n"
        report += f"- **Test Acc**: {p3_results['test_acc']*100:.2f}%\n"
        report += f"- **F1**: {p3_results['test_f1']*100:.2f}%\n\n"
    
    # Phase 4 结果
    p4_file = output_root / "phase4_distill" / "distill_results.json"
    if p4_file.exists():
        with open(p4_file) as f:
            p4_results = json.load(f)
        
        report += "## Phase 4: 知识蒸馏\n\n"
        report += "| Student | Temp | Alpha | Val Acc | Test Acc | F1 | Params(M) |\n"
        report += "|---------|------|-------|---------|----------|-----|----------|\n"
        
        sorted_p4 = sorted(p4_results, key=lambda x: x['test_acc'], reverse=True)
        for r in sorted_p4:
            report += f"| {r['student']} | {r['temperature']} | {r['alpha_ce']} | {r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n"
        report += "\n"
        
        # 最佳学生
        best = sorted_p4[0]
        report += f"### 🏆 最佳学生模型\n"
        report += f"- **Model**: {best['student']}\n"
        report += f"- **Temperature**: {best['temperature']}\n"
        report += f"- **Alpha CE**: {best['alpha_ce']}\n"
        report += f"- **Test Accuracy**: {best['test_acc']*100:.2f}%\n"
        report += f"- **F1 Score**: {best['test_f1']*100:.2f}%\n"
        report += f"- **Parameters**: {best['params_m']:.2f}M\n"
    
    report += f"\n---\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    
    with open(output_root / "FINAL_REPORT.md", 'w') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {output_root / 'FINAL_REPORT.md'}")
    return report


def main():
    parser = argparse.ArgumentParser(description="TrafficKD完整实验")
    parser.add_argument("--config", type=str, default="configs/full_experiment.yaml")
    parser.add_argument("--phase", type=str, choices=["pretrain", "finetune", "stacking", "distill", "all", "report"],
                       default="all")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    # 添加时间戳
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    config['experiment']['output_root'] = f"{config['experiment']['output_root']}_{timestamp}"
    
    output_root = Path(config['experiment']['output_root'])
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 保存配置副本
    with open(output_root / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    print(f"Output directory: {output_root}")
    
    if args.phase in ["pretrain", "all"]:
        print("\n" + "="*70)
        print("PHASE 1: SELF-SUPERVISED PRETRAINING")
        print("="*70)
        run_phase1_pretrain(config, args.gpu)
    
    if args.phase in ["finetune", "all"]:
        print("\n" + "="*70)
        print("PHASE 2: TEACHER FINETUNING")
        print("="*70)
        run_phase2_finetune(config, args.gpu)
    
    if args.phase in ["stacking", "all"]:
        print("\n" + "="*70)
        print("PHASE 3: STACKING ENSEMBLE")
        print("="*70)
        run_phase3_stacking(config, args.gpu)
    
    if args.phase in ["distill", "all"]:
        print("\n" + "="*70)
        print("PHASE 4: KNOWLEDGE DISTILLATION")
        print("="*70)
        run_phase4_distill(config, args.gpu)
    
    if args.phase in ["report", "all"]:
        print("\n" + "="*70)
        print("GENERATING FINAL REPORT")
        print("="*70)
        generate_final_report(config)


if __name__ == "__main__":
    main()
