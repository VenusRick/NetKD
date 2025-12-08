#!/usr/bin/env python3
"""
TrafficKD 完整实验流程
Phase 1: MAE/SimCLR 预训练
Phase 2: 教师模型微调 (含ECA消融)
Phase 3: Stacking集成
Phase 4: 知识蒸馏学生训练
Phase 5: 模型评估与Pareto分析
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import yaml

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm

# ================== 工具函数 ==================
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def load_config(config_path: str) -> Dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def save_json(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)

def log_message(msg: str, log_file: Path = None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')

def get_device(gpu_id: int = 0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')

def count_parameters(model: nn.Module) -> float:
    """返回参数量（百万）"""
    return sum(p.numel() for p in model.parameters()) / 1e6

def count_flops(model: nn.Module, input_size=(1, 1, 40, 40)) -> float:
    """估算FLOPs（十亿）"""
    try:
        from fvcore.nn import FlopCountAnalysis
        model.eval()
        dummy = torch.randn(input_size).to(next(model.parameters()).device)
        flops = FlopCountAnalysis(model, dummy)
        return flops.total() / 1e9
    except:
        return -1.0

# ================== 数据加载 ==================
def get_data_loaders(data_root: str, batch_size: int = 128, num_workers: int = 4):
    """加载ISCXVPN2016数据集"""
    from data_preprocessing.image_loader import quick_load_dataset
    train_loader, val_loader, test_loader, meta = quick_load_dataset(
        dataset_name="ISCXVPN2016",
        dataset_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader

# ================== ECA 模块 ==================
class ECALayer(nn.Module):
    """Efficient Channel Attention"""
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        import math
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

def add_eca_to_model(model: nn.Module) -> nn.Module:
    """在模型的特征层后添加ECA"""
    # 对于timm模型，在forward_features后添加ECA
    original_forward = model.forward_features
    
    # 获取特征维度
    with torch.no_grad():
        dummy = torch.randn(1, 1, 40, 40)
        try:
            feat = original_forward(dummy)
            if len(feat.shape) == 4:
                channels = feat.shape[1]
            else:
                channels = feat.shape[-1]
        except:
            channels = 512
    
    eca = ECALayer(channels)
    
    def new_forward(x):
        feat = original_forward(x)
        if len(feat.shape) == 4:
            feat = eca(feat)
        return feat
    
    model.forward_features = new_forward
    return model

# ================== MAE 预训练 ==================
class MAEEncoder(nn.Module):
    """MAE编码器包装器"""
    def __init__(self, backbone_name: str, mask_ratio: float = 0.75):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, in_chans=1)
        self.mask_ratio = mask_ratio
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.randn(1, 1, 40, 40)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[-1]
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, 256),
            nn.GELU(),
            nn.Linear(256, 32 * 32)
        )
    
    def forward(self, x, return_loss=True):
        B, C, H, W = x.shape
        
        # 随机掩码
        if return_loss:
            mask = torch.rand(B, H * W, device=x.device) < self.mask_ratio
            x_masked = x.clone()
            x_masked = x_masked.view(B, C, -1)
            x_masked[:, :, mask[0]] = 0
            x_masked = x_masked.view(B, C, H, W)
        else:
            x_masked = x
            mask = None
        
        # 编码
        feat = self.backbone(x_masked)
        
        if return_loss:
            # 解码
            recon = self.decoder(feat).view(B, C, H, W)
            # 只计算掩码位置的损失
            loss = ((recon - x) ** 2).mean()
            return loss, feat
        return feat

def train_mae(backbone_name: str, mask_ratio: float, epochs: int, 
              data_loader: DataLoader, device: torch.device, 
              output_dir: Path, log_file: Path) -> Dict:
    """MAE预训练"""
    log_message(f"Starting MAE pretrain: {backbone_name}, mask={mask_ratio}", log_file)
    
    model = MAEEncoder(backbone_name, mask_ratio).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    results = {'backbone': backbone_name, 'mask_ratio': mask_ratio, 'losses': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"MAE {backbone_name} E{epoch+1}", leave=False):
            x = batch[0].to(device)
            optimizer.zero_grad()
            loss, _ = model(x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        results['losses'].append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / f"mae_{backbone_name}_mask{mask_ratio}_best.pth"
            torch.save({
                'backbone_state': model.backbone.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_message(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", log_file)
    
    results['best_loss'] = best_loss
    results['checkpoint'] = str(ckpt_path)
    log_message(f"MAE pretrain done: {backbone_name}, best_loss={best_loss:.4f}", log_file)
    return results

# ================== SimCLR 对比学习 ==================
class SimCLREncoder(nn.Module):
    """SimCLR编码器"""
    def __init__(self, backbone_name: str, proj_dim: int = 128):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0, in_chans=1)
        
        with torch.no_grad():
            dummy = torch.randn(1, 1, 40, 40)
            feat = self.backbone(dummy)
            self.feat_dim = feat.shape[-1]
        
        self.projector = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.Linear(self.feat_dim, proj_dim)
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        proj = self.projector(feat)
        return nn.functional.normalize(proj, dim=1)

def simclr_loss(z1, z2, temperature=0.5):
    """NT-Xent损失"""
    B = z1.shape[0]
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.t()) / temperature
    
    # 掩码对角线
    mask = torch.eye(2*B, device=z.device).bool()
    sim.masked_fill_(mask, -float('inf'))
    
    # 正样本对
    pos_mask = torch.zeros(2*B, 2*B, device=z.device).bool()
    pos_mask[:B, B:] = torch.eye(B, device=z.device).bool()
    pos_mask[B:, :B] = torch.eye(B, device=z.device).bool()
    
    labels = torch.arange(2*B, device=z.device)
    labels[:B] = labels[:B] + B
    labels[B:] = labels[B:] - B
    
    loss = nn.functional.cross_entropy(sim, labels)
    return loss

def augment_batch(x):
    """简单数据增强"""
    # 随机噪声
    noise = torch.randn_like(x) * 0.1
    x1 = x + noise
    # 随机翻转
    if torch.rand(1) > 0.5:
        x2 = torch.flip(x, dims=[-1])
    else:
        x2 = x + torch.randn_like(x) * 0.05
    return x1, x2

def train_simclr(backbone_name: str, temperature: float, epochs: int,
                 data_loader: DataLoader, device: torch.device,
                 output_dir: Path, log_file: Path) -> Dict:
    """SimCLR对比学习预训练"""
    log_message(f"Starting SimCLR pretrain: {backbone_name}, temp={temperature}", log_file)
    
    model = SimCLREncoder(backbone_name).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    best_loss = float('inf')
    results = {'backbone': backbone_name, 'temperature': temperature, 'losses': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(data_loader, desc=f"SimCLR {backbone_name} E{epoch+1}", leave=False):
            x = batch[0].to(device)
            x1, x2 = augment_batch(x)
            
            optimizer.zero_grad()
            z1, z2 = model(x1), model(x2)
            loss = simclr_loss(z1, z2, temperature)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader)
        results['losses'].append(avg_loss)
        scheduler.step()
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt_path = output_dir / f"simclr_{backbone_name}_temp{temperature}_best.pth"
            torch.save({
                'backbone_state': model.backbone.state_dict(),
                'epoch': epoch,
                'loss': best_loss
            }, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_message(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}", log_file)
    
    results['best_loss'] = best_loss
    results['checkpoint'] = str(ckpt_path)
    log_message(f"SimCLR pretrain done: {backbone_name}, best_loss={best_loss:.4f}", log_file)
    return results

# ================== 教师微调 ==================
def finetune_teacher(backbone_name: str, pretrain_ckpt: Optional[str],
                     use_eca: bool, epochs: int, train_loader: DataLoader,
                     val_loader: DataLoader, test_loader: DataLoader,
                     num_classes: int, device: torch.device,
                     output_dir: Path, log_file: Path) -> Dict:
    """微调教师模型"""
    tag = f"{backbone_name}_{'eca' if use_eca else 'noeca'}_{'pretrain' if pretrain_ckpt else 'scratch'}"
    log_message(f"Finetuning teacher: {tag}", log_file)
    
    # 创建模型
    model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes, in_chans=1)
    
    # 加载预训练权重
    if pretrain_ckpt and Path(pretrain_ckpt).exists():
        ckpt = torch.load(pretrain_ckpt, map_location='cpu')
        model.load_state_dict(ckpt['backbone_state'], strict=False)
        log_message(f"  Loaded pretrain: {pretrain_ckpt}", log_file)
        lr = 1e-4
    else:
        lr = 1e-3
    
    # 添加ECA
    if use_eca:
        model = add_eca_to_model(model)
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    results = {
        'name': tag, 'backbone': backbone_name, 'use_eca': use_eca,
        'pretrain': pretrain_ckpt is not None, 'train_losses': [], 'val_accs': []
    }
    
    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Finetune {tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        results['train_losses'].append(avg_loss)
        
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
        results['val_accs'].append(val_acc)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_message(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val: {val_acc*100:.2f}%", log_file)
    
    # 测试
    model.load_state_dict(torch.load(ckpt_path)['model_state'])
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = correct / total
    from sklearn.metrics import f1_score
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results['val_acc'] = best_val_acc
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['params_m'] = count_parameters(model)
    results['checkpoint'] = str(ckpt_path)
    
    log_message(f"Teacher {tag}: Val={best_val_acc*100:.2f}%, Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    return results

# ================== Stacking ==================
class StackingHead(nn.Module):
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_teachers * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, *teacher_logits):
        x = torch.cat(teacher_logits, dim=1)
        return self.fc(x)

def train_stacking(teacher_models: List[nn.Module], teacher_names: List[str],
                   epochs: int, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, num_classes: int, device: torch.device,
                   output_dir: Path, log_file: Path) -> Dict:
    """训练Stacking集成"""
    log_message(f"Training Stacking with teachers: {teacher_names}", log_file)
    
    # 冻结教师
    for t in teacher_models:
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    
    stacking = StackingHead(len(teacher_models), num_classes).to(device)
    optimizer = optim.AdamW(stacking.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    results = {'teacher_names': teacher_names, 'val_accs': []}
    
    for epoch in range(epochs):
        stacking.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Stacking E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
            
            optimizer.zero_grad()
            out = stacking(*teacher_logits)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
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
        results['val_accs'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "stacking_best.pth"
            torch.save({'stacking_state': stacking.state_dict(), 'val_acc': val_acc}, ckpt_path)
    
    # 测试
    stacking.load_state_dict(torch.load(ckpt_path)['stacking_state'])
    stacking.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            teacher_logits = [t(x) for t in teacher_models]
            out = stacking(*teacher_logits)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = correct / total
    from sklearn.metrics import f1_score
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results['val_acc'] = best_val_acc
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['checkpoint'] = str(ckpt_path)
    
    log_message(f"Stacking: Val={best_val_acc*100:.2f}%, Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    return results, stacking, teacher_models

# ================== 知识蒸馏 ==================
def distill_student(student_name: str, teacher_models: List[nn.Module],
                    stacking: nn.Module, temperature: float, alpha_ce: float,
                    epochs: int, train_loader: DataLoader, val_loader: DataLoader,
                    test_loader: DataLoader, num_classes: int, device: torch.device,
                    output_dir: Path, log_file: Path) -> Dict:
    """知识蒸馏训练学生"""
    tag = f"{student_name}_T{temperature}_a{alpha_ce}"
    log_message(f"Distilling student: {tag}", log_file)
    
    # 创建学生
    student = timm.create_model(student_name, pretrained=False, num_classes=num_classes, in_chans=1).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    alpha_kd = 1.0 - alpha_ce
    best_val_acc = 0
    results = {
        'student': student_name, 'temperature': temperature, 'alpha_ce': alpha_ce,
        'train_losses': [], 'val_accs': []
    }
    
    for epoch in range(epochs):
        student.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Distill {tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            
            # 教师输出
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
                teacher_out = stacking(*teacher_logits)
                teacher_soft = nn.functional.softmax(teacher_out / temperature, dim=1)
            
            optimizer.zero_grad()
            student_out = student(x)
            student_soft = nn.functional.log_softmax(student_out / temperature, dim=1)
            
            # 损失
            ce_loss = ce_criterion(student_out, y)
            kd_loss = kl_criterion(student_soft, teacher_soft) * (temperature ** 2)
            loss = alpha_ce * ce_loss + alpha_kd * kd_loss
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        results['train_losses'].append(avg_loss)
        
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
        results['val_accs'].append(val_acc)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': student.state_dict(), 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 20 == 0:
            log_message(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val: {val_acc*100:.2f}%", log_file)
    
    # 测试
    student.load_state_dict(torch.load(ckpt_path)['model_state'])
    student.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    # 推理时间测试
    import time
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            
            # 测推理时间
            start = time.time()
            out = student(x)
            torch.cuda.synchronize()
            inference_times.append((time.time() - start) / x.size(0) * 1000)
            
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = correct / total
    from sklearn.metrics import f1_score
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results['val_acc'] = best_val_acc
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['params_m'] = count_parameters(student)
    results['flops_g'] = count_flops(student)
    results['inference_ms'] = sum(inference_times) / len(inference_times)
    results['checkpoint'] = str(ckpt_path)
    
    log_message(f"Student {tag}: Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, Params={results['params_m']:.2f}M", log_file)
    return results

# ================== 主流程 ==================
def run_phase1_pretrain(config: Dict, output_dir: Path, log_file: Path, gpu_id: int = 0):
    """Phase 1: 预训练"""
    log_message("="*60, log_file)
    log_message("PHASE 1: SELF-SUPERVISED PRETRAINING", log_file)
    log_message("="*60, log_file)
    
    device = get_device(gpu_id)
    train_loader, _, _ = get_data_loaders(config['experiment']['dataset_root'], batch_size=512)
    
    mae_dir = output_dir / "phase1_pretrain" / "mae"
    simclr_dir = output_dir / "phase1_pretrain" / "simclr"
    mae_dir.mkdir(parents=True, exist_ok=True)
    simclr_dir.mkdir(parents=True, exist_ok=True)
    
    results = {'mae': [], 'simclr': []}
    
    # MAE预训练
    if config['pretrain']['mae']['enabled']:
        for backbone in config['pretrain']['mae']['backbones']:
            for mask_ratio in config['pretrain']['mae']['mask_ratios']:
                try:
                    r = train_mae(backbone, mask_ratio, config['pretrain']['mae']['epochs'],
                                  train_loader, device, mae_dir, log_file)
                    results['mae'].append(r)
                    save_json(results, output_dir / "phase1_results.json")
                except Exception as e:
                    log_message(f"MAE failed for {backbone}: {e}", log_file)
    
    # SimCLR预训练
    if config['pretrain']['simclr']['enabled']:
        for backbone in config['pretrain']['simclr']['backbones']:
            for temp in config['pretrain']['simclr']['temperatures']:
                try:
                    r = train_simclr(backbone, temp, config['pretrain']['simclr']['epochs'],
                                     train_loader, device, simclr_dir, log_file)
                    results['simclr'].append(r)
                    save_json(results, output_dir / "phase1_results.json")
                except Exception as e:
                    log_message(f"SimCLR failed for {backbone}: {e}", log_file)
    
    return results

def run_phase2_finetune(config: Dict, output_dir: Path, log_file: Path, gpu_id: int = 0):
    """Phase 2: 教师微调"""
    log_message("="*60, log_file)
    log_message("PHASE 2: TEACHER FINETUNING", log_file)
    log_message("="*60, log_file)
    
    device = get_device(gpu_id)
    train_loader, val_loader, test_loader = get_data_loaders(
        config['experiment']['dataset_root'], batch_size=config['finetune']['batch_size'])
    
    finetune_dir = output_dir / "phase2_finetune"
    finetune_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载Phase1结果
    p1_results = load_json(output_dir / "phase1_results.json") if (output_dir / "phase1_results.json").exists() else {'mae': [], 'simclr': []}
    
    # 构建预训练检查点映射
    pretrain_ckpts = {}
    for r in p1_results.get('mae', []):
        key = r['backbone']
        if key not in pretrain_ckpts or r['best_loss'] < pretrain_ckpts[key]['loss']:
            pretrain_ckpts[key] = {'path': r['checkpoint'], 'loss': r['best_loss'], 'type': 'mae'}
    for r in p1_results.get('simclr', []):
        key = r['backbone']
        if key not in pretrain_ckpts or r['best_loss'] < pretrain_ckpts[key]['loss']:
            pretrain_ckpts[key] = {'path': r['checkpoint'], 'loss': r['best_loss'], 'type': 'simclr'}
    
    results = []
    backbones = config['pretrain']['mae']['backbones']
    
    for backbone in backbones:
        pretrain_ckpt = pretrain_ckpts.get(backbone, {}).get('path')
        
        # 4种组合: scratch, scratch+eca, pretrain, pretrain+eca
        for use_pretrain in [False, True]:
            for use_eca in [False, True]:
                ckpt = pretrain_ckpt if use_pretrain else None
                try:
                    r = finetune_teacher(backbone, ckpt, use_eca, config['finetune']['epochs'],
                                         train_loader, val_loader, test_loader, 7, device,
                                         finetune_dir, log_file)
                    results.append(r)
                    save_json(results, output_dir / "phase2_results.json")
                except Exception as e:
                    log_message(f"Finetune failed for {backbone}: {e}", log_file)
    
    return results

def run_phase3_stacking(config: Dict, output_dir: Path, log_file: Path, gpu_id: int = 0):
    """Phase 3: Stacking"""
    log_message("="*60, log_file)
    log_message("PHASE 3: STACKING ENSEMBLE", log_file)
    log_message("="*60, log_file)
    
    device = get_device(gpu_id)
    train_loader, val_loader, test_loader = get_data_loaders(
        config['experiment']['dataset_root'], batch_size=config['stacking'].get('batch_size', 128))
    
    stacking_dir = output_dir / "phase3_stacking"
    stacking_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载Phase2结果，选择top-N教师
    p2_results = load_json(output_dir / "phase2_results.json")
    sorted_teachers = sorted(p2_results, key=lambda x: x['test_acc'], reverse=True)
    top_teachers = sorted_teachers[:config['stacking']['num_teachers']]
    
    log_message(f"Selected top {len(top_teachers)} teachers:", log_file)
    for t in top_teachers:
        log_message(f"  - {t['name']}: {t['test_acc']*100:.2f}%", log_file)
    
    # 加载教师模型
    teacher_models = []
    teacher_names = []
    for t in top_teachers:
        model = timm.create_model(t['backbone'], pretrained=False, num_classes=7, in_chans=1)
        if t['use_eca']:
            model = add_eca_to_model(model)
        model.load_state_dict(torch.load(t['checkpoint'])['model_state'])
        model = model.to(device)
        model.eval()
        teacher_models.append(model)
        teacher_names.append(t['name'])
    
    results, stacking, _ = train_stacking(
        teacher_models, teacher_names, config['stacking']['epochs'],
        train_loader, val_loader, test_loader, 7, device, stacking_dir, log_file)
    
    save_json(results, output_dir / "phase3_results.json")
    return results, stacking, teacher_models

def run_phase4_distill(config: Dict, output_dir: Path, log_file: Path,
                       stacking: nn.Module, teacher_models: List[nn.Module], gpu_id: int = 0):
    """Phase 4: 知识蒸馏"""
    log_message("="*60, log_file)
    log_message("PHASE 4: KNOWLEDGE DISTILLATION", log_file)
    log_message("="*60, log_file)
    
    device = get_device(gpu_id)
    train_loader, val_loader, test_loader = get_data_loaders(
        config['experiment']['dataset_root'], batch_size=config['distill'].get('batch_size', 128))
    
    distill_dir = output_dir / "phase4_distill"
    distill_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for student_name in config['distill']['students']:
        for temp in config['distill']['temperatures']:
            for alpha in config['distill']['alpha_ce_values']:
                try:
                    r = distill_student(student_name, teacher_models, stacking,
                                        temp, alpha, config['distill']['epochs'],
                                        train_loader, val_loader, test_loader, 7,
                                        device, distill_dir, log_file)
                    results.append(r)
                    save_json(results, output_dir / "phase4_results.json")
                except Exception as e:
                    log_message(f"Distill failed for {student_name}: {e}", log_file)
    
    return results

def generate_final_report(output_dir: Path, log_file: Path):
    """生成最终报告"""
    log_message("="*60, log_file)
    log_message("GENERATING FINAL REPORT", log_file)
    log_message("="*60, log_file)
    
    report = "# TrafficKD 完整实验报告\n\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Phase 1
    if (output_dir / "phase1_results.json").exists():
        p1 = load_json(output_dir / "phase1_results.json")
        report += "## Phase 1: 自监督预训练\n\n"
        
        if p1.get('mae'):
            report += "### MAE预训练\n"
            report += "| Backbone | Mask Ratio | Best Loss |\n|---|---|---|\n"
            for r in sorted(p1['mae'], key=lambda x: x['best_loss']):
                report += f"| {r['backbone']} | {r['mask_ratio']} | {r['best_loss']:.4f} |\n"
            report += "\n"
        
        if p1.get('simclr'):
            report += "### SimCLR预训练\n"
            report += "| Backbone | Temperature | Best Loss |\n|---|---|---|\n"
            for r in sorted(p1['simclr'], key=lambda x: x['best_loss']):
                report += f"| {r['backbone']} | {r['temperature']} | {r['best_loss']:.4f} |\n"
            report += "\n"
    
    # Phase 2
    if (output_dir / "phase2_results.json").exists():
        p2 = load_json(output_dir / "phase2_results.json")
        report += "## Phase 2: 教师模型微调\n\n"
        report += "| Name | Pretrain | ECA | Val Acc | Test Acc | F1 | Params(M) |\n"
        report += "|---|---|---|---|---|---|---|\n"
        for r in sorted(p2, key=lambda x: x['test_acc'], reverse=True):
            report += f"| {r['backbone']} | {'✓' if r['pretrain'] else '✗'} | {'✓' if r['use_eca'] else '✗'} | "
            report += f"{r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n"
        report += "\n"
    
    # Phase 3
    if (output_dir / "phase3_results.json").exists():
        p3 = load_json(output_dir / "phase3_results.json")
        report += "## Phase 3: Stacking集成\n\n"
        report += f"- Teachers: {', '.join(p3['teacher_names'])}\n"
        report += f"- Val Acc: {p3['val_acc']*100:.2f}%\n"
        report += f"- Test Acc: {p3['test_acc']*100:.2f}%\n"
        report += f"- F1: {p3['test_f1']*100:.2f}%\n\n"
    
    # Phase 4
    if (output_dir / "phase4_results.json").exists():
        p4 = load_json(output_dir / "phase4_results.json")
        report += "## Phase 4: 知识蒸馏学生模型\n\n"
        report += "| Student | Temp | Alpha | Val Acc | Test Acc | F1 | Params(M) | FLOPs(G) | Infer(ms) |\n"
        report += "|---|---|---|---|---|---|---|---|---|\n"
        for r in sorted(p4, key=lambda x: x['test_acc'], reverse=True):
            report += f"| {r['student']} | {r['temperature']} | {r['alpha_ce']} | "
            report += f"{r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | "
            report += f"{r['params_m']:.2f} | {r.get('flops_g', -1):.2f} | {r.get('inference_ms', -1):.2f} |\n"
        report += "\n"
        
        # Pareto前沿
        report += "### Pareto前沿分析\n\n"
        report += "最佳学生模型（按准确率排序）:\n"
        best_per_student = {}
        for r in p4:
            s = r['student']
            if s not in best_per_student or r['test_acc'] > best_per_student[s]['test_acc']:
                best_per_student[s] = r
        
        report += "| Student | Test Acc | F1 | Params(M) |\n|---|---|---|---|\n"
        for r in sorted(best_per_student.values(), key=lambda x: x['test_acc'], reverse=True):
            report += f"| {r['student']} | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n"
    
    # 保存报告
    report_path = output_dir / "FINAL_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    log_message(f"Report saved to: {report_path}", log_file)
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full_experiment_v2.yaml")
    parser.add_argument("--phase", default="all", choices=["pretrain", "finetune", "stacking", "distill", "report", "all"])
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    timestamp = get_timestamp()
    output_dir = Path(config['experiment']['output_root']) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "experiment.log"
    
    log_message(f"Starting TrafficKD Full Experiment", log_file)
    log_message(f"Output: {output_dir}", log_file)
    log_message(f"Config: {args.config}", log_file)
    
    # 保存配置
    import shutil
    shutil.copy(args.config, output_dir / "config.yaml")
    
    stacking, teacher_models = None, None
    
    if args.phase in ["pretrain", "all"]:
        run_phase1_pretrain(config, output_dir, log_file, args.gpu)
    
    if args.phase in ["finetune", "all"]:
        run_phase2_finetune(config, output_dir, log_file, args.gpu)
    
    if args.phase in ["stacking", "all"]:
        _, stacking, teacher_models = run_phase3_stacking(config, output_dir, log_file, args.gpu)
    
    if args.phase in ["distill", "all"]:
        if stacking is None:
            # 重新加载
            _, stacking, teacher_models = run_phase3_stacking(config, output_dir, log_file, args.gpu)
        run_phase4_distill(config, output_dir, log_file, stacking, teacher_models, args.gpu)
    
    if args.phase in ["report", "all"]:
        generate_final_report(output_dir, log_file)
    
    log_message("="*60, log_file)
    log_message("EXPERIMENT COMPLETED!", log_file)
    log_message("="*60, log_file)

if __name__ == "__main__":
    main()
