#!/usr/bin/env python3
"""
========================================================================
TrafficKD 完整实验流程 - 预训练 vs 直接微调对比实验
========================================================================

实验设计:
=========

Phase 1: 预训练方法对比 (在3个教师模型上测试)
    - 直接微调 (baseline)
    - MAE预训练 + 微调 (掩码率0.75, 0.8)
    - 对比学习预训练 (SimCLR) + 微调
    
Phase 2: 全量教师模型训练 (选择最优方法)
    - convnextv2_tiny (+ECA)
    - efficientnetv2_rw_s (+ECA)
    - ghostnetv2_100 (+ECA)
    
Phase 3: Stacking集成
    - 使用预训练教师 vs 直接微调教师
    
Phase 4: 知识蒸馏到学生模型
    - ghostnet_100
    - mobilenetv3_small_075
    - repvit_m0_9
    - mobilenetv2_050
    
Phase 5: 学生模型评估
    - FLOPs, 参数量, 推理时间
    - 帕累托前沿分析

作者: AI Assistant
日期: 2024-12-08
========================================================================
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
from collections import OrderedDict

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

# ============================================================================
# 全局配置
# ============================================================================
CONFIG = {
    'data_root': '/workspace/yqm/Dataset',
    'batch_size': 256,
    'num_workers': 8,
    'num_classes': 7,
    
    # 预训练配置
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
    
    # 微调配置
    'finetune': {
        'epochs': 50,
        'lr': 0.001,
        'weight_decay': 1e-4,
        'warmup_epochs': 5,
    },
    
    # Stacking配置
    'stacking': {
        'epochs': 30,
        'hidden_dim': 384,
        'lr': 0.001,
    },
    
    # 蒸馏配置
    'distill': {
        'epochs': 100,
        'temperatures': [3, 4, 5],
        'alphas': [0.3, 0.5, 0.7],
        'lr': 0.001,
    },
    
    # 新教师模型列表
    'new_teachers': [
        'convnextv2_tiny',
        'efficientnetv2_rw_s', 
        'ghostnetv2_100',
    ],
    
    # 学生模型列表
    'students': [
        'ghostnet_100',
        'mobilenetv3_small_075',
        'repvit_m0_9',
        'mobilenetv2_050',
    ],
}

# ============================================================================
# 工具函数
# ============================================================================
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def log_msg(msg: str, log_file: Path = None):
    """打印并记录日志"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')

def save_json(data: Dict, path: Path):
    """保存JSON文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def count_parameters(model: nn.Module) -> float:
    """计算模型参数量(M)"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_flops(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 64, 64)) -> float:
    """估算FLOPs(G) - 使用thop或手动估算"""
    try:
        from thop import profile
        x = torch.randn(input_size).cuda()
        flops, _ = profile(model, inputs=(x,), verbose=False)
        return flops / 1e9
    except:
        # 简单估算
        params = count_parameters(model)
        return params * 2  # 粗略估算

def measure_inference_time(model: nn.Module, input_size: Tuple[int, ...] = (1, 1, 64, 64), 
                           num_runs: int = 100) -> float:
    """测量推理时间(ms)"""
    model.eval()
    x = torch.randn(input_size).cuda()
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(x)
    torch.cuda.synchronize()
    
    return (time.time() - start) / num_runs * 1000

# ============================================================================
# 数据加载
# ============================================================================
def get_data_loaders(config: Dict, log_file: Path = None):
    """加载数据集"""
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
    """获取预训练数据加载器(无标签)"""
    from data_preprocessing.image_loader import quick_load_dataset
    
    train_loader, _, _, _ = quick_load_dataset(
        dataset_name="ISCXVPN2016",
        dataset_root=config['data_root'],
        batch_size=batch_size,
        num_workers=config['num_workers']
    )
    return train_loader

# ============================================================================
# ECA注意力模块
# ============================================================================
class ECALayer(nn.Module):
    """Efficient Channel Attention"""
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
    """为模型添加ECA注意力"""
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
# MAE预训练
# ============================================================================
class MAEEncoder(nn.Module):
    """MAE编码器 - 支持CNN backbone"""
    def __init__(self, backbone_name: str, mask_ratio: float = 0.75):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, 
                                          num_classes=0, in_chans=1)
        self.mask_ratio = mask_ratio
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, 64, 64)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[-1]
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """随机掩码"""
        B, C, H, W = x.shape
        
        # 创建patch网格
        patch_size = 8
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # 随机选择要保留的patches
        num_keep = int(num_patches * (1 - self.mask_ratio))
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :num_keep]
        
        # 创建mask
        mask = torch.ones(B, num_patches, device=x.device)
        mask.scatter_(1, ids_keep, 0)
        mask = mask.reshape(B, 1, num_patches_h, num_patches_w)
        mask = F.interpolate(mask, size=(H, W), mode='nearest')
        
        # 应用mask
        x_masked = x * (1 - mask)
        
        return x_masked, mask
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_masked, mask = self.random_masking(x)
        features = self.backbone(x_masked)
        return features, mask

class MAEDecoder(nn.Module):
    """MAE解码器"""
    def __init__(self, feat_dim: int, out_channels: int = 1, img_size: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, img_size * img_size * out_channels),
        )
        self.out_channels = out_channels
        self.img_size = img_size
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.decoder(x)
        x = x.view(-1, self.out_channels, self.img_size, self.img_size)
        return x

class MAEModel(nn.Module):
    """完整MAE模型"""
    def __init__(self, backbone_name: str, mask_ratio: float = 0.75):
        super().__init__()
        self.encoder = MAEEncoder(backbone_name, mask_ratio)
        self.decoder = MAEDecoder(self.encoder.feat_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features, mask = self.encoder(x)
        reconstructed = self.decoder(features)
        return reconstructed, mask, features
    
    def get_encoder_state_dict(self) -> Dict:
        """获取编码器权重"""
        return self.encoder.backbone.state_dict()

def train_mae(backbone_name: str, mask_ratio: float, config: Dict, 
              gpu_id: int, output_dir: Path, log_file: Path) -> Dict:
    """MAE预训练"""
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"{backbone_name}_mae_mask{mask_ratio}"
    log_msg(f"[MAE] Starting pretraining: {tag}", log_file)
    
    # 创建模型
    model = MAEModel(backbone_name, mask_ratio).to(device)
    
    # 获取数据
    train_loader = get_pretrain_loader(config, config['pretrain']['mae_batch_size'])
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['pretrain']['mae_lr'],
                            weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['pretrain']['mae_epochs'])
    scaler = GradScaler()
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(config['pretrain']['mae_epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x = batch[0].to(device)
            
            optimizer.zero_grad()
            with autocast():
                reconstructed, mask, _ = model(x)
                # 只计算被掩码区域的重建损失
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
            log_msg(f"  [MAE] Epoch {epoch+1}/{config['pretrain']['mae_epochs']}, "
                    f"Loss: {avg_loss:.6f}", log_file)
    
    result = {
        'backbone': backbone_name,
        'mask_ratio': mask_ratio,
        'best_loss': best_loss,
        'final_loss': losses[-1],
        'losses': losses,
        'checkpoint': str(ckpt_path)
    }
    
    log_msg(f"[MAE] Completed {tag}: best_loss={best_loss:.6f}", log_file)
    return result

# ============================================================================
# 对比学习预训练 (SimCLR风格)
# ============================================================================
class ContrastiveAugmentation:
    """对比学习数据增强"""
    def __init__(self):
        pass
    
    def __call__(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """生成两个增强视图"""
        view1 = self.augment(x)
        view2 = self.augment(x)
        return view1, view2
    
    def augment(self, x: torch.Tensor) -> torch.Tensor:
        """单个增强"""
        # 随机裁剪和调整大小
        if random.random() > 0.5:
            h, w = x.shape[-2:]
            crop_size = int(random.uniform(0.8, 1.0) * min(h, w))
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)
            x = x[..., top:top+crop_size, left:left+crop_size]
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)
        
        # 随机水平翻转
        if random.random() > 0.5:
            x = torch.flip(x, dims=[-1])
        
        # 随机噪声
        if random.random() > 0.5:
            noise = torch.randn_like(x) * 0.1
            x = x + noise
        
        return x

class ContrastiveModel(nn.Module):
    """SimCLR风格对比学习模型"""
    def __init__(self, backbone_name: str, proj_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False,
                                          num_classes=0, in_chans=1)
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, 64, 64)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[-1]
        
        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, proj_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        projections = self.projector(features)
        projections = F.normalize(projections, dim=1)
        return features, projections
    
    def get_backbone_state_dict(self) -> Dict:
        return self.backbone.state_dict()

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent对比损失"""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    
    sim = torch.mm(z, z.t()) / temperature
    
    # 创建标签
    labels = torch.cat([torch.arange(B) + B, torch.arange(B)]).to(z.device)
    
    # 移除对角线
    mask = ~torch.eye(2*B, dtype=bool, device=z.device)
    sim = sim.masked_select(mask).view(2*B, -1)
    
    return F.cross_entropy(sim, labels)

def train_contrastive(backbone_name: str, config: Dict, gpu_id: int,
                      output_dir: Path, log_file: Path) -> Dict:
    """对比学习预训练"""
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"{backbone_name}_simclr"
    log_msg(f"[SimCLR] Starting pretraining: {tag}", log_file)
    
    model = ContrastiveModel(backbone_name).to(device)
    augmenter = ContrastiveAugmentation()
    
    train_loader = get_pretrain_loader(config, config['pretrain']['contrastive_batch_size'])
    
    optimizer = optim.AdamW(model.parameters(), lr=config['pretrain']['contrastive_lr'],
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['pretrain']['contrastive_epochs'])
    scaler = GradScaler()
    
    best_loss = float('inf')
    losses = []
    
    for epoch in range(config['pretrain']['contrastive_epochs']):
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x = batch[0].to(device)
            
            # 生成两个视图
            view1, view2 = augmenter(x)
            
            optimizer.zero_grad()
            with autocast():
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
            log_msg(f"  [SimCLR] Epoch {epoch+1}/{config['pretrain']['contrastive_epochs']}, "
                    f"Loss: {avg_loss:.4f}", log_file)
    
    result = {
        'backbone': backbone_name,
        'method': 'simclr',
        'best_loss': best_loss,
        'final_loss': losses[-1],
        'losses': losses,
        'checkpoint': str(ckpt_path)
    }
    
    log_msg(f"[SimCLR] Completed {tag}: best_loss={best_loss:.4f}", log_file)
    return result

# ============================================================================
# 教师模型微调
# ============================================================================
def finetune_teacher(backbone_name: str, pretrain_ckpt: Optional[str], 
                     use_eca: bool, config: Dict, train_loader: DataLoader,
                     val_loader: DataLoader, test_loader: DataLoader,
                     gpu_id: int, output_dir: Path, log_file: Path) -> Dict:
    """教师模型微调"""
    device = torch.device(f'cuda:{gpu_id}')
    
    # 确定tag
    if pretrain_ckpt:
        if 'mae' in pretrain_ckpt:
            method = 'mae'
        elif 'simclr' in pretrain_ckpt:
            method = 'simclr'
        else:
            method = 'pretrained'
    else:
        method = 'scratch'
    
    eca_tag = '_eca' if use_eca else ''
    tag = f"{backbone_name}_{method}{eca_tag}"
    log_msg(f"[Finetune] Starting: {tag}", log_file)
    
    # 创建模型
    model = timm.create_model(backbone_name, pretrained=False, 
                              num_classes=config['num_classes'], in_chans=1)
    
    # 加载预训练权重
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        ckpt = torch.load(pretrain_ckpt, map_location='cpu')
        if 'encoder_state' in ckpt:
            state_dict = ckpt['encoder_state']
        elif 'backbone_state' in ckpt:
            state_dict = ckpt['backbone_state']
        else:
            state_dict = ckpt
        
        # 只加载匹配的权重
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        log_msg(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} weights from {pretrain_ckpt}", log_file)
    
    # 添加ECA
    if use_eca:
        model = add_eca_to_model(model)
    
    model = model.to(device)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=config['finetune']['lr'],
                            weight_decay=config['finetune']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['finetune']['epochs'])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    scaler = GradScaler()
    
    best_val_acc = 0
    train_losses = []
    val_accs = []
    
    for epoch in range(config['finetune']['epochs']):
        # 训练
        model.train()
        epoch_loss = 0
        
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            
            optimizer.zero_grad()
            with autocast():
                out = model(x)
                loss = criterion(out, y)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        val_accs.append(val_acc)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({
                'model_state': model.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, ckpt_path)
        
        if (epoch + 1) % 5 == 0:
            log_msg(f"  [Finetune] Epoch {epoch+1}/{config['finetune']['epochs']}, "
                    f"Loss: {avg_loss:.4f}, Val: {val_acc*100:.2f}%", log_file)
    
    # 测试
    model.load_state_dict(torch.load(ckpt_path)['model_state'])
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_precision = precision_score(all_labels, all_preds, average='macro')
    test_recall = recall_score(all_labels, all_preds, average='macro')
    
    result = {
        'name': tag,
        'backbone': backbone_name,
        'method': method,
        'use_eca': use_eca,
        'pretrain_ckpt': pretrain_ckpt,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'params_m': count_parameters(model),
        'train_losses': train_losses,
        'val_accs': val_accs,
        'checkpoint': str(ckpt_path)
    }
    
    log_msg(f"[Finetune] Completed {tag}: Val={best_val_acc*100:.2f}%, "
            f"Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    
    return result, model

# ============================================================================
# Stacking集成
# ============================================================================
class StackingHead(nn.Module):
    """Stacking集成头"""
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_teachers * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, *teacher_logits):
        x = torch.cat(teacher_logits, dim=1)
        return self.fc(x)

def train_stacking(teacher_models: List[nn.Module], teacher_names: List[str],
                   config: Dict, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, gpu_id: int, output_dir: Path,
                   log_file: Path) -> Tuple[Dict, nn.Module]:
    """训练Stacking"""
    device = torch.device(f'cuda:{gpu_id}')
    log_msg(f"[Stacking] Training with teachers: {teacher_names}", log_file)
    
    # 冻结教师
    for t in teacher_models:
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    
    stacking = StackingHead(len(teacher_models), config['num_classes'],
                            config['stacking']['hidden_dim']).to(device)
    
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
            out = stacking(*teacher_logits)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # 验证
        stacking.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                teacher_logits = [t(x) for t in teacher_models]
                out = stacking(*teacher_logits)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "stacking_best.pth"
            torch.save({'stacking_state': stacking.state_dict(), 'val_acc': val_acc}, ckpt_path)
    
    # 测试
    stacking.load_state_dict(torch.load(ckpt_path)['stacking_state'])
    stacking.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            teacher_logits = [t(x) for t in teacher_models]
            out = stacking(*teacher_logits)
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    result = {
        'teacher_names': teacher_names,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'checkpoint': str(ckpt_path)
    }
    
    log_msg(f"[Stacking] Completed: Val={best_val_acc*100:.2f}%, "
            f"Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    
    return result, stacking

# ============================================================================
# 知识蒸馏
# ============================================================================
def distill_student(student_name: str, teacher_models: List[nn.Module],
                    stacking: nn.Module, temperature: float, alpha_ce: float,
                    config: Dict, train_loader: DataLoader, val_loader: DataLoader,
                    test_loader: DataLoader, gpu_id: int, output_dir: Path,
                    log_file: Path) -> Dict:
    """知识蒸馏训练学生模型"""
    device = torch.device(f'cuda:{gpu_id}')
    tag = f"{student_name}_T{temperature}_a{alpha_ce}"
    log_msg(f"[Distill] Starting: {tag}", log_file)
    
    # 创建学生模型
    student = timm.create_model(student_name, pretrained=False,
                                num_classes=config['num_classes'], in_chans=1).to(device)
    
    optimizer = optim.AdamW(student.parameters(), lr=config['distill']['lr'],
                            weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, config['distill']['epochs'])
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    scaler = GradScaler()
    
    alpha_kd = 1.0 - alpha_ce
    best_val_acc = 0
    
    for epoch in range(config['distill']['epochs']):
        student.train()
        
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            
            # 获取教师输出
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
                teacher_out = stacking(*teacher_logits)
                teacher_soft = F.softmax(teacher_out / temperature, dim=1)
            
            optimizer.zero_grad()
            with autocast():
                student_out = student(x)
                student_soft = F.log_softmax(student_out / temperature, dim=1)
                
                ce_loss = ce_criterion(student_out, y)
                kd_loss = kl_criterion(student_soft, teacher_soft) * (temperature ** 2)
                loss = alpha_ce * ce_loss + alpha_kd * kd_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # 验证
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = student(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': student.state_dict(), 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_msg(f"  [Distill] Epoch {epoch+1}/{config['distill']['epochs']}, "
                    f"Val: {val_acc*100:.2f}%", log_file)
    
    # 测试
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
            
            pred = out.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    # 计算模型指标
    params_m = count_parameters(student)
    flops_g = count_flops(student)
    avg_inference_ms = np.mean(inference_times)
    
    result = {
        'name': tag,
        'student': student_name,
        'temperature': temperature,
        'alpha_ce': alpha_ce,
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'params_m': params_m,
        'flops_g': flops_g,
        'inference_ms': avg_inference_ms,
        'checkpoint': str(ckpt_path)
    }
    
    log_msg(f"[Distill] Completed {tag}: Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, "
            f"Params={params_m:.2f}M, FLOPs={flops_g:.2f}G", log_file)
    
    return result

# ============================================================================
# 主实验流程
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='TrafficKD Full Pipeline Experiment')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--skip_pretrain', action='store_true', help='Skip pretraining phase')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    timestamp = get_timestamp()
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(f"results/full_pipeline_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "experiment.log"
    
    log_msg("=" * 80, log_file)
    log_msg("TrafficKD 完整实验流程", log_file)
    log_msg(f"Output: {output_dir}", log_file)
    log_msg(f"GPU: {args.gpu}", log_file)
    log_msg("=" * 80, log_file)
    
    # 保存配置
    save_json(CONFIG, output_dir / "config.json")
    
    # 加载数据
    train_loader, val_loader, test_loader, meta = get_data_loaders(CONFIG, log_file)
    
    all_results = {
        'pretrain': {},
        'teachers': {},
        'stacking': {},
        'students': {}
    }
    
    # ========================================================================
    # Phase 1: 预训练方法对比
    # ========================================================================
    if not args.skip_pretrain:
        log_msg("\n" + "=" * 80, log_file)
        log_msg("PHASE 1: 预训练方法对比", log_file)
        log_msg("=" * 80, log_file)
        
        pretrain_dir = output_dir / "pretrain"
        pretrain_dir.mkdir(exist_ok=True)
        
        # 选择一个模型进行预训练对比实验
        test_backbone = 'convnextv2_tiny'
        
        # MAE预训练 - 不同掩码率
        for mask_ratio in CONFIG['pretrain']['mae_mask_ratios']:
            result = train_mae(test_backbone, mask_ratio, CONFIG, args.gpu,
                              pretrain_dir, log_file)
            all_results['pretrain'][f'mae_mask{mask_ratio}'] = result
            save_json(all_results, output_dir / "results.json")
        
        # 对比学习预训练
        result = train_contrastive(test_backbone, CONFIG, args.gpu, pretrain_dir, log_file)
        all_results['pretrain']['simclr'] = result
        save_json(all_results, output_dir / "results.json")
    
    # ========================================================================
    # Phase 2: 教师模型训练 (预训练 vs 直接微调对比)
    # ========================================================================
    log_msg("\n" + "=" * 80, log_file)
    log_msg("PHASE 2: 教师模型训练", log_file)
    log_msg("=" * 80, log_file)
    
    teacher_dir = output_dir / "teachers"
    teacher_dir.mkdir(exist_ok=True)
    
    teacher_models = []
    teacher_names = []
    
    for backbone in CONFIG['new_teachers']:
        # 方案1: 直接微调 (无预训练)
        result_scratch, _ = finetune_teacher(
            backbone, None, False, CONFIG,
            train_loader, val_loader, test_loader,
            args.gpu, teacher_dir, log_file
        )
        all_results['teachers'][f'{backbone}_scratch'] = result_scratch
        
        # 方案2: 直接微调 + ECA
        result_scratch_eca, model_scratch_eca = finetune_teacher(
            backbone, None, True, CONFIG,
            train_loader, val_loader, test_loader,
            args.gpu, teacher_dir, log_file
        )
        all_results['teachers'][f'{backbone}_scratch_eca'] = result_scratch_eca
        
        # 如果有预训练checkpoint，也测试预训练+微调
        if not args.skip_pretrain:
            # MAE预训练 + 微调
            mae_ckpt = str(output_dir / f"pretrain/{backbone}_mae_mask0.75_best.pth")
            if os.path.exists(mae_ckpt):
                result_mae, _ = finetune_teacher(
                    backbone, mae_ckpt, False, CONFIG,
                    train_loader, val_loader, test_loader,
                    args.gpu, teacher_dir, log_file
                )
                all_results['teachers'][f'{backbone}_mae'] = result_mae
                
                # MAE + ECA
                result_mae_eca, model_mae_eca = finetune_teacher(
                    backbone, mae_ckpt, True, CONFIG,
                    train_loader, val_loader, test_loader,
                    args.gpu, teacher_dir, log_file
                )
                all_results['teachers'][f'{backbone}_mae_eca'] = result_mae_eca
            
            # SimCLR预训练 + 微调
            simclr_ckpt = str(output_dir / f"pretrain/{backbone}_simclr_best.pth")
            if os.path.exists(simclr_ckpt):
                result_simclr, _ = finetune_teacher(
                    backbone, simclr_ckpt, False, CONFIG,
                    train_loader, val_loader, test_loader,
                    args.gpu, teacher_dir, log_file
                )
                all_results['teachers'][f'{backbone}_simclr'] = result_simclr
        
        # 选择最佳教师模型用于后续实验
        teacher_models.append(model_scratch_eca)
        teacher_names.append(f'{backbone}_scratch_eca')
        
        save_json(all_results, output_dir / "results.json")
    
    # ========================================================================
    # Phase 3: Stacking集成
    # ========================================================================
    log_msg("\n" + "=" * 80, log_file)
    log_msg("PHASE 3: Stacking集成", log_file)
    log_msg("=" * 80, log_file)
    
    stacking_dir = output_dir / "stacking"
    stacking_dir.mkdir(exist_ok=True)
    
    stacking_result, stacking_model = train_stacking(
        teacher_models, teacher_names, CONFIG,
        train_loader, val_loader, test_loader,
        args.gpu, stacking_dir, log_file
    )
    all_results['stacking']['main'] = stacking_result
    save_json(all_results, output_dir / "results.json")
    
    # ========================================================================
    # Phase 4: 知识蒸馏学生模型
    # ========================================================================
    log_msg("\n" + "=" * 80, log_file)
    log_msg("PHASE 4: 知识蒸馏学生模型", log_file)
    log_msg("=" * 80, log_file)
    
    student_dir = output_dir / "students"
    student_dir.mkdir(exist_ok=True)
    
    for student_name in CONFIG['students']:
        for temp in CONFIG['distill']['temperatures']:
            for alpha in CONFIG['distill']['alphas']:
                result = distill_student(
                    student_name, teacher_models, stacking_model,
                    temp, alpha, CONFIG,
                    train_loader, val_loader, test_loader,
                    args.gpu, student_dir, log_file
                )
                all_results['students'][result['name']] = result
                save_json(all_results, output_dir / "results.json")
    
    # ========================================================================
    # Phase 5: 生成最终报告
    # ========================================================================
    log_msg("\n" + "=" * 80, log_file)
    log_msg("PHASE 5: 生成报告", log_file)
    log_msg("=" * 80, log_file)
    
    generate_report(all_results, output_dir, log_file)
    
    log_msg("\n" + "=" * 80, log_file)
    log_msg("实验完成!", log_file)
    log_msg(f"结果保存在: {output_dir}", log_file)
    log_msg("=" * 80, log_file)

def generate_report(results: Dict, output_dir: Path, log_file: Path):
    """生成实验报告"""
    report = []
    report.append("# TrafficKD 完整实验报告\n")
    report.append(f"生成时间: {datetime.now()}\n\n")
    
    # 预训练对比
    if results.get('pretrain'):
        report.append("## 1. 预训练方法对比\n\n")
        report.append("| 方法 | 最终Loss |\n|---|---|\n")
        for name, r in results['pretrain'].items():
            report.append(f"| {name} | {r.get('final_loss', 'N/A'):.6f} |\n")
        report.append("\n")
    
    # 教师模型对比
    report.append("## 2. 教师模型对比\n\n")
    report.append("| 模型 | 方法 | ECA | Val Acc | Test Acc | F1 | Params(M) |\n")
    report.append("|---|---|---|---|---|---|---|\n")
    
    teacher_data = []
    for name, r in sorted(results.get('teachers', {}).items(), 
                          key=lambda x: x[1].get('test_acc', 0), reverse=True):
        teacher_data.append(r)
        report.append(f"| {r['backbone']} | {r['method']} | {r['use_eca']} | "
                     f"{r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | "
                     f"{r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n")
    report.append("\n")
    
    # Stacking结果
    if results.get('stacking', {}).get('main'):
        r = results['stacking']['main']
        report.append("## 3. Stacking集成结果\n\n")
        report.append(f"- 教师模型: {', '.join(r['teacher_names'])}\n")
        report.append(f"- Val Acc: {r['val_acc']*100:.2f}%\n")
        report.append(f"- Test Acc: {r['test_acc']*100:.2f}%\n")
        report.append(f"- F1: {r['test_f1']*100:.2f}%\n\n")
    
    # 学生模型对比
    report.append("## 4. 学生模型对比\n\n")
    report.append("| 学生 | T | α | Test Acc | F1 | Params(M) | FLOPs(G) | Infer(ms) |\n")
    report.append("|---|---|---|---|---|---|---|---|\n")
    
    student_data = []
    for name, r in sorted(results.get('students', {}).items(),
                          key=lambda x: x[1].get('test_acc', 0), reverse=True):
        student_data.append(r)
        report.append(f"| {r['student']} | {r['temperature']} | {r['alpha_ce']} | "
                     f"{r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | "
                     f"{r['params_m']:.2f} | {r.get('flops_g', 0):.2f} | "
                     f"{r.get('inference_ms', 0):.2f} |\n")
    report.append("\n")
    
    # 最佳模型总结
    report.append("## 5. 最佳模型总结\n\n")
    
    if teacher_data:
        best_teacher = max(teacher_data, key=lambda x: x.get('test_acc', 0))
        report.append(f"**最佳教师模型**: {best_teacher['name']}\n")
        report.append(f"- Test Acc: {best_teacher['test_acc']*100:.2f}%\n")
        report.append(f"- F1: {best_teacher['test_f1']*100:.2f}%\n\n")
    
    if student_data:
        best_student = max(student_data, key=lambda x: x.get('test_acc', 0))
        report.append(f"**最佳学生模型**: {best_student['name']}\n")
        report.append(f"- Test Acc: {best_student['test_acc']*100:.2f}%\n")
        report.append(f"- F1: {best_student['test_f1']*100:.2f}%\n")
        report.append(f"- Params: {best_student['params_m']:.2f}M\n")
        report.append(f"- FLOPs: {best_student.get('flops_g', 0):.2f}G\n")
        report.append(f"- Inference: {best_student.get('inference_ms', 0):.2f}ms\n\n")
    
    # 帕累托分析
    report.append("## 6. 学生模型帕累托分析\n\n")
    report.append("以下是在精度-效率权衡上表现最佳的模型:\n\n")
    
    if student_data:
        # 简单的帕累托筛选
        pareto_front = []
        for s in student_data:
            dominated = False
            for other in student_data:
                if (other['test_acc'] >= s['test_acc'] and 
                    other['params_m'] <= s['params_m'] and
                    (other['test_acc'] > s['test_acc'] or other['params_m'] < s['params_m'])):
                    dominated = True
                    break
            if not dominated:
                pareto_front.append(s)
        
        report.append("| 模型 | Test Acc | Params(M) | 效率分数 |\n|---|---|---|---|\n")
        for s in pareto_front:
            efficiency = s['test_acc'] / (s['params_m'] + 0.1)
            report.append(f"| {s['name']} | {s['test_acc']*100:.2f}% | "
                         f"{s['params_m']:.2f} | {efficiency:.4f} |\n")
    
    # 保存报告
    report_path = output_dir / "FINAL_REPORT.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(''.join(report))
    
    log_msg(f"Report saved to {report_path}", log_file)

if __name__ == "__main__":
    main()
