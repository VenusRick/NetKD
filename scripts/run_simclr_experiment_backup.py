#!/usr/bin/env python3
"""
SimCLR对比学习预训练实验脚本
在GPU 1上运行
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
import torchvision.transforms as T

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data_preprocessing.image_loader import quick_load_dataset

# =========== SimCLR组件 ===========
class SimCLRDataset(Dataset):
    """SimCLR数据增强包装器"""
    def __init__(self, base_dataset, img_size=40):
        self.base_dataset = base_dataset
        self.transform = T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(),
            T.RandomApply([T.GaussianBlur(kernel_size=3)], p=0.5),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
        ])
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        return self.transform(img), self.transform(img), label

class SimCLRModel(nn.Module):
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
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.mm(z, z.T) / temperature
    labels = torch.arange(batch_size, device=z.device)
    labels = torch.cat([labels + batch_size, labels])
    mask = torch.eye(2 * batch_size, device=z.device).bool()
    sim.masked_fill_(mask, -float('inf'))
    return F.cross_entropy(sim, labels)

def pretrain_simclr(backbone, config, device, output_dir):
    """SimCLR预训练"""
    print(f"\n>>> SimCLR预训练: {backbone}")
    
    os.makedirs(output_dir, exist_ok=True)
    train_ds, val_ds, test_ds, _ = quick_load_dataset(config['data_path'])
    simclr_ds = SimCLRDataset(train_ds)
    loader = DataLoader(simclr_ds, batch_size=config['batch_size'], 
                       shuffle=True, num_workers=8, pin_memory=True)
    
    model = SimCLRModel(backbone, in_chans=1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.05)
    scaler = GradScaler()
    
    best_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"SimCLR E{epoch+1}")
        for x1, x2, _ in pbar:
            x1, x2 = x1.to(device), x2.to(device)
            optimizer.zero_grad()
            with autocast():
                z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss(z1, z2, temperature=config['temperature'])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(loader)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.encoder.state_dict(), 
                      f"{output_dir}/{backbone}_simclr_encoder.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"  E{epoch+1}: loss={avg_loss:.4f} (best={best_loss:.4f})")
    
    return best_loss

def finetune_from_simclr(backbone, pretrain_path, config, device, output_dir):
    """使用SimCLR预训练权重微调"""
    print(f"\n>>> 微调SimCLR: {backbone}")
    
    train_ds, val_ds, test_ds, _ = quick_load_dataset(config['data_path'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], 
                             shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'], 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'], 
                            num_workers=4, pin_memory=True)
    
    model = timm.create_model(backbone, pretrained=False, 
                             num_classes=config['num_classes'], in_chans=1)
    
    # 加载预训练encoder权重
    if pretrain_path and os.path.exists(pretrain_path):
        state_dict = torch.load(pretrain_path, map_location='cpu')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"  Loaded SimCLR: missing={len(missing)}, unexpected={len(unexpected)}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['finetune_lr'], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['finetune_epochs'])
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    best_val, best_state = 0, None
    for epoch in range(config['finetune_epochs']):
        model.train()
        for x, y in tqdm(train_loader, desc=f"FT E{epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += model(x).argmax(1).eq(y).sum().item()
                total += x.size(0)
        val_acc = correct / total
        
        if val_acc > best_val:
            best_val = val_acc
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 10 == 0:
            print(f"  E{epoch+1}: val={val_acc:.4f} (best={best_val:.4f})")
    
    # 测试
    model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(model(x.to(device)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    
    test_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    test_f1 = f1_score(labels, preds, average='macro')
    
    torch.save(best_state, f"{output_dir}/{backbone}_simclr_finetuned.pth")
    
    return {'val_acc': best_val, 'test_acc': test_acc, 'test_f1': test_f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/simclr_experiment_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'data_path': '/workspace/yqm/Dataset/ISCXVPN2016',
        'batch_size': 256,
        'lr': 0.01,
        'epochs': 100,
        'temperature': 0.5,
        'finetune_lr': 1e-3,
        'finetune_epochs': 50,
        'num_classes': 12,
    }
    
    backbones = ['efficientnetv2_rw_s', 'convnextv2_tiny', 'mobilenetv3_large_100']
    results = []
    
    for backbone in backbones:
        pretrain_dir = f'{output_dir}/pretrain'
        os.makedirs(pretrain_dir, exist_ok=True)
        
        # 1. SimCLR预训练
        pretrain_loss = pretrain_simclr(backbone, config, device, pretrain_dir)
        
        # 2. 微调
        pretrain_path = f'{pretrain_dir}/{backbone}_simclr_encoder.pth'
        finetune_results = finetune_from_simclr(backbone, pretrain_path, config, device, output_dir)
        
        # 3. 对比：直接训练(无预训练)
        scratch_results = finetune_from_simclr(backbone, None, config, device, output_dir)
        
        results.append({
            'backbone': backbone,
            'simclr_pretrain_loss': pretrain_loss,
            'simclr_finetune': finetune_results,
            'scratch': scratch_results,
            'improvement': finetune_results['test_acc'] - scratch_results['test_acc']
        })
        
        print(f"\n=== {backbone} ===")
        print(f"  SimCLR+FT: {finetune_results['test_acc']:.4f}")
        print(f"  Scratch:   {scratch_results['test_acc']:.4f}")
        print(f"  Δ: {finetune_results['test_acc'] - scratch_results['test_acc']:+.4f}")
    
    # 保存结果
    with open(f'{output_dir}/simclr_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ SimCLR实验完成，结果保存至: {output_dir}")

if __name__ == '__main__':
    main()
