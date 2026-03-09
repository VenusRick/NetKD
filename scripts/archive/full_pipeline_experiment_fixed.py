#!/usr/bin/env python3
"""
TrafficKD 完整实验流程 - 修复版
修复MAE预训练的图像尺寸问题 (40x40)
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

# ============================================================================
# 全局配置
# ============================================================================
CONFIG = {
    'data_root': '/workspace/yqm/Dataset',
    'batch_size': 256,
    'num_workers': 8,
    'num_classes': 7,
    'img_size': 40,  # 实际图像尺寸
    
    'pretrain': {
        'mae_epochs': 100,
        'mae_batch_size': 512,
        'mae_lr': 0.01,
        'mae_mask_ratios': [0.75, 0.8],
        'contrastive_epochs': 100,
        'contrastive_batch_size': 512,
        'contrastive_lr': 0.01,
        'contrastive_temperature': 0.5,
    },
    
    'finetune': {
        'epochs': 50,
        'lr': 0.001,
        'weight_decay': 1e-4,
    },
    
    'stacking': {
        'epochs': 30,
        'hidden_dim': 384,
        'lr': 0.001,
    },
    
    'distill': {
        'epochs': 100,
        'temperatures': [3, 4, 5],
        'alphas': [0.3, 0.5, 0.7],
        'lr': 0.001,
    },
    
    'new_teachers': ['convnextv2_tiny', 'efficientnetv2_rw_s', 'ghostnetv2_100'],
    'students': ['ghostnet_100', 'mobilenetv3_small_075', 'repvit_m0_9', 'mobilenetv2_050'],
}

# ============================================================================
# 工具函数
# ============================================================================
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def log_msg(msg: str, log_file: Path = None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted, flush=True)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')

def save_json(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def get_data_loaders(config: Dict, log_file: Path = None):
    from data_preprocessing.image_loader import quick_load_dataset
    log_msg("Loading dataset...", log_file)
    train_loader, val_loader, test_loader, meta = quick_load_dataset(
        dataset_name="ISCXVPN2016",
        dataset_root=config['data_root'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    log_msg(f"Dataset loaded: train={len(train_loader.dataset)}, "
            f"val={len(val_loader.dataset)}, test={len(test_loader.dataset)}", log_file)
    return train_loader, val_loader, test_loader, meta

def get_pretrain_loader(config: Dict, batch_size: int):
    from data_preprocessing.image_loader import quick_load_dataset
    train_loader, _, _, _ = quick_load_dataset(
        dataset_name="ISCXVPN2016",
        dataset_root=config['data_root'],
        batch_size=batch_size,
        num_workers=config['num_workers']
    )
    return train_loader

# ============================================================================
# ECA模块
# ============================================================================
class ECALayer(nn.Module):
    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

def add_eca_to_model(model: nn.Module) -> nn.Module:
    for name, module in model.named_children():
        if isinstance(module, nn.Sequential):
            new_modules = []
            for sub_name, sub_module in module.named_children():
                new_modules.append(sub_module)
                if isinstance(sub_module, nn.Conv2d) and sub_module.out_channels >= 32:
                    new_modules.append(ECALayer(sub_module.out_channels))
            setattr(model, name, nn.Sequential(*new_modules))
        else:
            add_eca_to_model(module)
    return model

# ============================================================================
# MAE预训练 (修复版 - 支持40x40图像)
# ============================================================================
class MAEEncoder(nn.Module):
    def __init__(self, backbone_name: str, mask_ratio: float = 0.75, img_size: int = 40):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, 
                                          num_classes=0, in_chans=1)
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, img_size, img_size)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[-1]
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = x.shape
        patch_size = 8
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        num_keep = int(num_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        
        mask = torch.ones(B, num_patches, device=x.device)
        mask.scatter_(1, ids_keep, 0)
        mask = mask.reshape(B, 1, num_patches_h, num_patches_w)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        x_masked = x * (1 - mask)
        return x_masked, mask
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked, mask = self.random_masking(x)
        features = self.backbone(x_masked)
        return features, mask

class MAEDecoder(nn.Module):
    def __init__(self, feat_dim: int, out_channels: int = 1, img_size: int = 40):
        super().__init__()
        self.img_size = img_size
        self.out_channels = out_channels
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size * out_channels),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class MAEModel(nn.Module):
    def __init__(self, backbone_name: str, mask_ratio: float = 0.75, img_size: int = 40):
        super().__init__()
        self.encoder = MAEEncoder(backbone_name, mask_ratio, img_size)
        self.decoder = MAEDecoder(self.encoder.feat_dim, img_size=img_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, mask = self.encoder(x)
        reconstructed = self.decoder(features)
        return reconstructed, mask, features
    
    def get_encoder_state_dict(self) -> Dict:
        return self.encoder.backbone.state_dict()

def train_mae(backbone_name: str, mask_ratio: float, config: Dict, 
              gpu_id: int, output_dir: Path, log_file: Path) -> Dict:
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"{backbone_name}_mae_mask{mask_ratio}"
    log_msg(f"[MAE] Starting pretraining: {tag}", log_file)
    
    model = MAEModel(backbone_name, mask_ratio, img_size=config['img_size']).to(device)
    train_loader = get_pretrain_loader(config, config['pretrain']['mae_batch_size'])
    
    optimizer = optim.AdamW(model.parameters(), lr=config['pretrain']['mae_lr'], weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['pretrain']['mae_epochs'])
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(config['pretrain']['mae_epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x = batch[0].to(device)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                reconstructed, mask, _ = model(x)
                loss = F.mse_loss(reconstructed * mask, x * mask)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({
                'encoder_state': model.get_encoder_state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_msg(f"  [MAE] Epoch {epoch+1}/{config['pretrain']['mae_epochs']}, Loss: {avg_loss:.6f}", log_file)
    
    log_msg(f"[MAE] Completed {tag}: best_loss={best_loss:.6f}", log_file)
    return {'backbone': backbone_name, 'mask_ratio': mask_ratio, 'best_loss': best_loss, 
            'losses': losses, 'checkpoint': str(ckpt_path)}

# ============================================================================
# 对比学习
# ============================================================================
class ContrastiveAugmentation:
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.augment(x), self.augment(x)
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        if random.random() > 0.5:
            h, w = x.shape[-2:]
            crop_size = int(random.uniform(0.8, 1.0) * min(h, w))
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            x = x[..., top:top+crop_size, left:left+crop_size]
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-1])
        if random.random() > 0.5:
            x = x + torch.randn_like(x) * 0.1
        return x

class ContrastiveModel(nn.Module):
    def __init__(self, backbone_name: str, proj_dim: int = 128, img_size: int = 40):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, in_chans=1)
        with torch.no_grad():
            feat = self.backbone(torch.randn(1, 1, img_size, img_size))
            self.feat_dim = feat.shape[-1]
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        projections = F.normalize(self.projector(features), dim=1)
        return features, projections
    
    def get_backbone_state_dict(self) -> Dict:
        return self.backbone.state_dict()

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    mask = ~torch.eye(2*B, dtype=bool, device=z.device)
    sim = sim.masked_select(mask).view(2*B, -1)
    return F.cross_entropy(sim, labels)

def train_contrastive(backbone_name: str, config: Dict, gpu_id: int,
                      output_dir: Path, log_file: Path) -> Dict:
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"{backbone_name}_simclr"
    log_msg(f"[SimCLR] Starting pretraining: {tag}", log_file)
    
    model = ContrastiveModel(backbone_name, img_size=config['img_size']).to(device)
    augmenter = ContrastiveAugmentation()
    train_loader = get_pretrain_loader(config, config['pretrain']['contrastive_batch_size'])
    
    optimizer = optim.AdamW(model.parameters(), lr=config['pretrain']['contrastive_lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['pretrain']['contrastive_epochs'])
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(config['pretrain']['contrastive_epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x = batch[0].to(device)
            view1, view2 = augmenter(x)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss = nt_xent_loss(z1, z2, config['pretrain']['contrastive_temperature'])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({
                'backbone_state': model.get_backbone_state_dict(),
                'epoch': epoch,
                'loss': avg_loss
            }, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_msg(f"  [SimCLR] Epoch {epoch+1}/{config['pretrain']['contrastive_epochs']}, Loss: {avg_loss:.4f}", log_file)
    
    log_msg(f"[SimCLR] Completed {tag}: best_loss={best_loss:.4f}", log_file)
    return {'backbone': backbone_name, 'method': 'simclr', 'best_loss': best_loss, 
            'losses': losses, 'checkpoint': str(ckpt_path)}

# ============================================================================
# 教师模型微调
# ============================================================================
def finetune_teacher(backbone_name: str, pretrain_ckpt: Optional[str], 
                     use_eca: bool, config: Dict, train_loader: DataLoader,
                     val_loader: DataLoader, test_loader: DataLoader,
                     gpu_id: int, output_dir: Path, log_file: Path) -> Tuple[Dict, nn.Module]:
    device = torch.device(f'cuda:{gpu_id}')
    
    if pretrain_ckpt:
        method = 'mae' if 'mae' in pretrain_ckpt else ('simclr' if 'simclr' in pretrain_ckpt else 'pretrained')
    else:
        method = 'scratch'
    
    eca_tag = '_eca' if use_eca else ''
    tag = f"{backbone_name}_{method}{eca_tag}"
    log_msg(f"[Finetune] Starting: {tag}", log_file)
    
    model = timm.create_model(backbone_name, pretrained=False, num_classes=config['num_classes'], in_chans=1)
    
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        ckpt = torch.load(pretrain_ckpt, map_location='cpu')
        state_dict = ckpt.get('encoder_state') or ckpt.get('backbone_state') or ckpt
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        log_msg(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} weights from {pretrain_ckpt}", log_file)
    
    if use_eca:
        model = add_eca_to_model(model)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config['finetune']['lr'], weight_decay=config['finetune']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['finetune']['epochs'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler('cuda')
    
    best_val_acc = 0
    train_losses, val_accs = [], []
    
    for epoch in range(config['finetune']['epochs']):
        model.train()
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                out = model(x)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
        
        train_losses.append(epoch_loss / len(train_loader))
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        val_accs.append(val_acc)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 5 == 0:
            log_msg(f"  [Finetune] Epoch {epoch+1}/{config['finetune']['epochs']}, Loss: {train_losses[-1]:.4f}, Val: {val_acc*100:.2f}%", log_file)
    
    model.load_state_dict(torch.load(ckpt_path)['model_state'])
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    result = {
        'name': tag, 'backbone': backbone_name, 'method': method, 'use_eca': use_eca,
        'val_acc': best_val_acc, 'test_acc': test_acc, 'test_f1': test_f1,
        'params_m': count_parameters(model), 'checkpoint': str(ckpt_path)
    }
    log_msg(f"[Finetune] Completed {tag}: Val={best_val_acc*100:.2f}%, Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    return result, model

# ============================================================================
# Stacking
# ============================================================================
class StackingHead(nn.Module):
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_teachers * num_classes, hidden_dim),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, *teacher_logits):
        return self.fc(torch.cat(teacher_logits, dim=1))

def train_stacking(teacher_models: List[nn.Module], teacher_names: List[str],
                   config: Dict, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, gpu_id: int, output_dir: Path,
                   log_file: Path) -> Tuple[Dict, nn.Module]:
    device = torch.device(f'cuda:{gpu_id}')
    log_msg(f"[Stacking] Training with teachers: {teacher_names}", log_file)
    
    for t in teacher_models:
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    
    stacking = StackingHead(len(teacher_models), config['num_classes'], config['stacking']['hidden_dim']).to(device)
    optimizer = optim.AdamW(stacking.parameters(), lr=config['stacking']['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['stacking']['epochs'])
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    for epoch in range(config['stacking']['epochs']):
        stacking.train()
        for batch in tqdm(train_loader, desc=f"Stacking E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
            optimizer.zero_grad()
            loss = criterion(stacking(*teacher_logits), y)
            loss.backward()
            optimizer.step()
        
        stacking.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                teacher_logits = [t(x) for t in teacher_models]
                correct += (stacking(*teacher_logits).argmax(1) == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "stacking_best.pth"
            torch.save({'stacking_state': stacking.state_dict(), 'val_acc': val_acc}, ckpt_path)
    
    stacking.load_state_dict(torch.load(ckpt_path)['stacking_state'])
    stacking.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            teacher_logits = [t(x) for t in teacher_models]
            all_preds.extend(stacking(*teacher_logits).argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    log_msg(f"[Stacking] Completed: Val={best_val_acc*100:.2f}%, Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    return {'teacher_names': teacher_names, 'val_acc': best_val_acc, 'test_acc': test_acc, 
            'test_f1': test_f1, 'checkpoint': str(ckpt_path)}, stacking

# ============================================================================
# 知识蒸馏
# ============================================================================
def distill_student(student_name: str, teacher_models: List[nn.Module], stacking: nn.Module,
                    temperature: float, alpha_ce: float, config: Dict, train_loader: DataLoader,
                    val_loader: DataLoader, test_loader: DataLoader, gpu_id: int,
                    output_dir: Path, log_file: Path) -> Dict:
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"{student_name}_T{temperature}_a{alpha_ce}"
    log_msg(f"[Distill] Starting: {tag}", log_file)
    
    student = timm.create_model(student_name, pretrained=False, num_classes=config['num_classes'], in_chans=1).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=config['distill']['lr'], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['distill']['epochs'])
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    scaler = GradScaler('cuda')
    
    alpha_kd = 1.0 - alpha_ce
    best_val_acc = 0
    
    for epoch in range(config['distill']['epochs']):
        student.train()
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
                teacher_out = stacking(*teacher_logits)
                teacher_soft = F.softmax(teacher_out / temperature, dim=1)
            
            optimizer.zero_grad()
            with autocast('cuda'):
                student_out = student(x)
                student_soft = F.log_softmax(student_out / temperature, dim=1)
                ce_loss = ce_criterion(student_out, y)
                kd_loss = kl_criterion(student_soft, teacher_soft) * (temperature ** 2)
                loss = alpha_ce * ce_loss + alpha_kd * kd_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                correct += (student(x).argmax(1) == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        scheduler.step()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': student.state_dict(), 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_msg(f"  [Distill] Epoch {epoch+1}/{config['distill']['epochs']}, Val: {val_acc*100:.2f}%", log_file)
    
    student.load_state_dict(torch.load(ckpt_path)['model_state'])
    student.eval()
    all_preds, all_labels = [], []
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            torch.cuda.synchronize()
            start = time.time()
            out = student(x)
            torch.cuda.synchronize()
            inference_times.append((time.time() - start) / x.size(0) * 1000)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    result = {
        'name': tag, 'student': student_name, 'temperature': temperature, 'alpha_ce': alpha_ce,
        'val_acc': best_val_acc, 'test_acc': test_acc, 'test_f1': test_f1,
        'params_m': count_parameters(student), 'inference_ms': np.mean(inference_times),
        'checkpoint': str(ckpt_path)
    }
    log_msg(f"[Distill] Completed {tag}: Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, Params={result['params_m']:.2f}M", log_file)
    return result

# ============================================================================
# 生成报告
# ============================================================================
def generate_report(results: Dict, output_dir: Path, log_file: Path):
    report = [f"# TrafficKD 完整实验报告\n\n生成时间: {datetime.now()}\n\n"]
    
    if results.get('pretrain'):
        report.append("## 1. 预训练方法对比\n\n| 方法 | 最终Loss |\n|---|---|\n")
        for name, r in results['pretrain'].items():
            report.append(f"| {name} | {r.get('best_loss', 'N/A'):.6f} |\n")
        report.append("\n")
    
    report.append("## 2. 教师模型对比\n\n| 模型 | 方法 | ECA | Val Acc | Test Acc | F1 | Params(M) |\n|---|---|---|---|---|---|---|\n")
    teacher_data = []
    for name, r in sorted(results.get('teachers', {}).items(), key=lambda x: x[1].get('test_acc', 0), reverse=True):
        teacher_data.append(r)
        report.append(f"| {r['backbone']} | {r['method']} | {r['use_eca']} | {r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n")
    
    if results.get('stacking', {}).get('main'):
        r = results['stacking']['main']
        report.append(f"\n## 3. Stacking集成\n\n- 教师: {', '.join(r['teacher_names'])}\n- Test Acc: {r['test_acc']*100:.2f}%\n- F1: {r['test_f1']*100:.2f}%\n\n")
    
    report.append("## 4. 学生模型对比\n\n| 学生 | T | α | Test Acc | F1 | Params(M) | Infer(ms) |\n|---|---|---|---|---|---|---|\n")
    student_data = []
    for name, r in sorted(results.get('students', {}).items(), key=lambda x: x[1].get('test_acc', 0), reverse=True):
        student_data.append(r)
        report.append(f"| {r['student']} | {r['temperature']} | {r['alpha_ce']} | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} | {r.get('inference_ms', 0):.2f} |\n")
    
    with open(output_dir / "FINAL_REPORT.md", 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    log_msg(f"Report saved to {output_dir / 'FINAL_REPORT.md'}", log_file)

# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--skip_pretrain', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    
    set_seed(42)
    timestamp = get_timestamp()
    output_dir = Path(args.output_dir) if args.output_dir else Path(f"results/full_pipeline_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment.log"
    
    log_msg("=" * 80, log_file)
    log_msg("TrafficKD 完整实验流程 (修复版)", log_file)
    log_msg(f"Output: {output_dir}, GPU: {args.gpu}", log_file)
    log_msg("=" * 80, log_file)
    
    save_json(CONFIG, output_dir / "config.json")
    train_loader, val_loader, test_loader, meta = get_data_loaders(CONFIG, log_file)
    
    all_results = {'pretrain': {}, 'teachers': {}, 'stacking': {}, 'students': {}}
    
    # Phase 1: 预训练
    if not args.skip_pretrain:
        log_msg("\n" + "=" * 80 + "\nPHASE 1: 预训练\n" + "=" * 80, log_file)
        pretrain_dir = output_dir / "pretrain"
        pretrain_dir.mkdir(exist_ok=True)
        
        test_backbone = 'convnextv2_tiny'
        for mask_ratio in CONFIG['pretrain']['mae_mask_ratios']:
            result = train_mae(test_backbone, mask_ratio, CONFIG, args.gpu, pretrain_dir, log_file)
            all_results['pretrain'][f'mae_mask{mask_ratio}'] = result
            save_json(all_results, output_dir / "results.json")
        
        result = train_contrastive(test_backbone, CONFIG, args.gpu, pretrain_dir, log_file)
        all_results['pretrain']['simclr'] = result
        save_json(all_results, output_dir / "results.json")
    
    # Phase 2: 教师训练
    log_msg("\n" + "=" * 80 + "\nPHASE 2: 教师训练\n" + "=" * 80, log_file)
    teacher_dir = output_dir / "teachers"
    teacher_dir.mkdir(exist_ok=True)
    
    teacher_models, teacher_names = [], []
    for backbone in CONFIG['new_teachers']:
        result_scratch, _ = finetune_teacher(backbone, None, False, CONFIG, train_loader, val_loader, test_loader, args.gpu, teacher_dir, log_file)
        all_results['teachers'][f'{backbone}_scratch'] = result_scratch
        
        result_eca, model_eca = finetune_teacher(backbone, None, True, CONFIG, train_loader, val_loader, test_loader, args.gpu, teacher_dir, log_file)
        all_results['teachers'][f'{backbone}_scratch_eca'] = result_eca
        
        if not args.skip_pretrain:
            mae_ckpt = str(output_dir / f"pretrain/{backbone}_mae_mask0.75_best.pth")
            if os.path.exists(mae_ckpt):
                result, model = finetune_teacher(backbone, mae_ckpt, True, CONFIG, train_loader, val_loader, test_loader, args.gpu, teacher_dir, log_file)
                all_results['teachers'][f'{backbone}_mae_eca'] = result
        
        teacher_models.append(model_eca)
        teacher_names.append(f'{backbone}_scratch_eca')
        save_json(all_results, output_dir / "results.json")
    
    # Phase 3: Stacking
    log_msg("\n" + "=" * 80 + "\nPHASE 3: Stacking\n" + "=" * 80, log_file)
    stacking_dir = output_dir / "stacking"
    stacking_dir.mkdir(exist_ok=True)
    stacking_result, stacking = train_stacking(teacher_models, teacher_names, CONFIG, train_loader, val_loader, test_loader, args.gpu, stacking_dir, log_file)
    all_results['stacking']['main'] = stacking_result
    save_json(all_results, output_dir / "results.json")
    
    # Phase 4: 蒸馏
    log_msg("\n" + "=" * 80 + "\nPHASE 4: 知识蒸馏\n" + "=" * 80, log_file)
    student_dir = output_dir / "students"
    student_dir.mkdir(exist_ok=True)
    
    for student_name in CONFIG['students']:
        for temp in CONFIG['distill']['temperatures']:
            for alpha in CONFIG['distill']['alphas']:
                result = distill_student(student_name, teacher_models, stacking, temp, alpha, CONFIG, train_loader, val_loader, test_loader, args.gpu, student_dir, log_file)
                all_results['students'][result['name']] = result
                save_json(all_results, output_dir / "results.json")
    
    generate_report(all_results, output_dir, log_file)
    log_msg("\n" + "=" * 80 + "\n实验完成!\n" + "=" * 80, log_file)

if __name__ == "__main__":
    main()
