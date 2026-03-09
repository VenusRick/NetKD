#!/usr/bin/env python3
"""
MAE风格自监督预训练 + 微调实验

实验设计：
1. 预训练阶段：使用掩码自编码器在ISCXVPN2016上自监督预训练
   - Batch size: 512
   - Epochs: 100
   - 基础学习率: 0.01 (线性缩放)
   - 优化器: AdamW
   - 掩码率: 0.9 (也测试0.8)

2. 微调阶段：在预训练权重基础上微调分类任务
   - Epochs: 50
   - 学习率: 1e-4 (微调用较小学习率)

3. 对比实验：
   - 预训练+微调 vs 从头训练
   - 掩码率0.9 vs 0.8
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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_preprocessing.image_loader import quick_load_dataset


# ============== MAE模型定义 ==============

class PatchEmbed(nn.Module):
    """将图像分割为patches并嵌入"""
    def __init__(self, img_size=40, patch_size=4, in_chans=1, embed_dim=256):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 10x10 = 100 patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: [B, C, H, W] -> [B, num_patches, embed_dim]
        x = self.proj(x)  # [B, embed_dim, H/p, W/p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MAEEncoder(nn.Module):
    """MAE编码器"""
    def __init__(self, img_size=40, patch_size=4, in_chans=1, embed_dim=256, 
                 depth=6, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
    def forward(self, x, mask=None):
        # Patch embedding
        x = self.patch_embed(x)  # [B, N, D]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 如果有mask，只保留未masked的patches
        if mask is not None:
            # mask: [B, N], True表示保留
            B, N, D = x.shape
            x = x[mask].view(B, -1, D)
        
        # Transformer编码
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        return x


class MAEDecoder(nn.Module):
    """MAE解码器"""
    def __init__(self, num_patches=100, patch_size=4, in_chans=1, 
                 embed_dim=256, decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        # 投影到解码器维度
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # 位置编码
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        # Transformer blocks
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # 预测patch像素
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans)
        
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
    def forward(self, x, ids_restore):
        # x: 编码器输出 [B, N_visible, D]
        # ids_restore: 用于恢复patch顺序 [B, N]
        
        x = self.decoder_embed(x)  # [B, N_visible, decoder_D]
        
        B, N_vis, D = x.shape
        N = self.num_patches
        
        # 添加mask tokens
        mask_tokens = self.mask_token.repeat(B, N - N_vis, 1)
        
        # 恢复顺序
        x_full = torch.zeros(B, N, D, device=x.device, dtype=x.dtype)
        
        # 使用ids_restore恢复位置
        for i in range(B):
            vis_idx = ids_restore[i, :N_vis]
            mask_idx = ids_restore[i, N_vis:]
            x_full[i, vis_idx] = x[i]
            x_full[i, mask_idx] = mask_tokens[i]
        
        # 添加位置编码
        x_full = x_full + self.decoder_pos_embed
        
        # 解码
        for block in self.decoder_blocks:
            x_full = block(x_full)
        
        x_full = self.decoder_norm(x_full)
        x_full = self.decoder_pred(x_full)  # [B, N, patch_size^2 * C]
        
        return x_full


class MAE(nn.Module):
    """Masked Autoencoder"""
    def __init__(self, img_size=40, patch_size=4, in_chans=1, 
                 embed_dim=256, depth=6, num_heads=8,
                 decoder_embed_dim=128, decoder_depth=4, decoder_num_heads=4,
                 mask_ratio=0.9):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans
        
        self.encoder = MAEEncoder(img_size, patch_size, in_chans, embed_dim, depth, num_heads)
        self.decoder = MAEDecoder(
            self.encoder.num_patches, patch_size, in_chans,
            embed_dim, decoder_embed_dim, decoder_depth, decoder_num_heads
        )
        self.num_patches = self.encoder.num_patches
        
    def random_masking(self, x):
        """随机掩码patches"""
        B, N, D = x.shape
        num_keep = int(N * (1 - self.mask_ratio))
        
        # 随机打乱
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # 保留前num_keep个
        ids_keep = ids_shuffle[:, :num_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        # 生成mask
        mask = torch.ones([B, N], device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore, ids_keep
    
    def patchify(self, imgs):
        """将图像转为patches"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x
    
    def unpatchify(self, x):
        """将patches还原为图像"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        B = x.shape[0]
        C = self.in_chans
        
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, h * p, w * p)
        return x
    
    def forward(self, imgs):
        # Patch embedding
        x = self.encoder.patch_embed(imgs)
        x = x + self.encoder.pos_embed
        
        # 随机掩码
        x, mask, ids_restore, ids_keep = self.random_masking(x)
        
        # 编码
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)
        
        # 解码
        pred = self.decoder(x, ids_restore)
        
        # 计算loss (只计算masked部分)
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask


class MAETeacher(nn.Module):
    """使用MAE预训练的教师模型"""
    def __init__(self, backbone_name, num_classes, img_size=40, in_chans=1, 
                 embed_dim=256, depth=6, num_heads=8):
        super().__init__()
        self.backbone_name = backbone_name
        
        # MAE编码器 (预训练后只用编码器)
        self.encoder = MAEEncoder(img_size, 4, in_chans, embed_dim, depth, num_heads)
        
        # 分类头
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes)
        )
        
    def forward(self, x):
        # 编码
        x = self.encoder(x)  # [B, N, D]
        # 池化并分类
        x = x.transpose(1, 2)  # [B, D, N]
        x = self.head(x)
        return x
    
    def load_pretrained(self, mae_checkpoint):
        """加载MAE预训练权重"""
        ckpt = torch.load(mae_checkpoint, map_location='cpu', weights_only=False)
        encoder_state = {}
        for k, v in ckpt['model_state_dict'].items():
            if k.startswith('encoder.'):
                encoder_state[k.replace('encoder.', '')] = v
        self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"  ✓ Loaded MAE pretrained weights from {mae_checkpoint}")


# ============== CNN-MAE变体 (适配CNN backbone) ==============

class CNNMaskedAutoencoder(nn.Module):
    """CNN版本的MAE，适用于ConvNeXt, DenseNet等"""
    def __init__(self, backbone_name, img_size=40, in_chans=1, mask_ratio=0.9):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = 4  # 将图像分为10x10个patch
        self.num_patches = (img_size // self.patch_size) ** 2
        
        # 编码器 (使用timm backbone)
        self.encoder = timm.create_model(backbone_name, pretrained=False, 
                                         num_classes=0, in_chans=in_chans)
        
        # 获取特征维度
        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            feat = self.encoder(dummy)
            self.feat_dim = feat.shape[-1]
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.num_patches * self.patch_size ** 2 * in_chans)
        )
        
    def random_mask_image(self, imgs):
        """随机掩码图像块"""
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = w = H // p
        
        # 创建mask
        num_mask = int(self.num_patches * self.mask_ratio)
        noise = torch.rand(B, self.num_patches, device=imgs.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        
        # 生成patch级别的mask
        mask = torch.zeros(B, self.num_patches, device=imgs.device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], 1)
        mask = mask.reshape(B, h, w)
        
        # 扩展到图像大小
        mask_img = mask.unsqueeze(1).repeat(1, 1, p, 1).repeat(1, 1, 1, p)
        mask_img = mask_img.reshape(B, 1, H, W)
        
        # 应用mask (masked区域设为0或均值)
        masked_imgs = imgs * (1 - mask_img)
        
        return masked_imgs, mask
    
    def patchify(self, imgs):
        """将图像转为patches"""
        B, C, H, W = imgs.shape
        p = self.patch_size
        h = w = H // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x
    
    def forward(self, imgs):
        B = imgs.shape[0]
        
        # 随机掩码
        masked_imgs, mask = self.random_mask_image(imgs)
        
        # 编码
        feat = self.encoder(masked_imgs)  # [B, feat_dim]
        
        # 解码重建
        pred = self.decoder(feat)  # [B, N * p^2 * C]
        pred = pred.reshape(B, self.num_patches, -1)
        
        # 计算重建loss
        target = self.patchify(imgs)
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [B, N]
        mask = mask.reshape(B, -1)
        loss = (loss * mask).sum() / mask.sum()
        
        return loss, pred, mask


# ============== 训练函数 ==============

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


def pretrain_mae(mae_model, train_loader, epochs, device, output_dir, name, 
                 base_lr=0.01, batch_size=512):
    """MAE预训练"""
    print(f"\n{'='*60}")
    print(f"MAE Pretraining: {name}")
    print(f"Mask ratio: {mae_model.mask_ratio}, Epochs: {epochs}, BS: {batch_size}")
    print(f"{'='*60}")
    
    # 线性学习率缩放
    lr = base_lr * batch_size / 256
    
    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.95))
    
    # Cosine学习率调度
    total_steps = epochs * len(train_loader)
    warmup_steps = 10 * len(train_loader)
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler()
    
    best_loss = float('inf')
    history = []
    
    for epoch in range(1, epochs + 1):
        mae_model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, _ in pbar:
            x = x.to(device)
            
            with autocast():
                loss, _, _ = mae_model(x)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        history.append(avg_loss)
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'model_state_dict': mae_model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'mask_ratio': mae_model.mask_ratio
            }, output_dir / f"mae_{name}_best.pth")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: loss={avg_loss:.4f} (best={best_loss:.4f}) lr={scheduler.get_last_lr()[0]:.6f}")
    
    return {"name": name, "best_loss": best_loss, "history": history}


def finetune_teacher(teacher, loaders, epochs, device, output_dir, name, lr=1e-4):
    """微调教师模型"""
    print(f"\n>>> Finetuning: {name}")
    
    train_loader, val_loader, test_loader = loaders
    
    optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler()
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        teacher.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                logits = teacher(x)
                loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.3f}")
        
        scheduler.step()
        
        val_acc = evaluate(teacher, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': teacher.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc
            }, output_dir / f"teacher_{name}_best.pth")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f}@E{best_epoch})")
    
    # 测试
    ckpt = torch.load(output_dir / f"teacher_{name}_best.pth", weights_only=False)
    teacher.load_state_dict(ckpt['model_state_dict'])
    test_acc = evaluate(teacher, test_loader, device)
    test_f1 = compute_f1(teacher, test_loader, device)
    
    return {
        "name": name,
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "best_epoch": best_epoch
    }


def train_from_scratch(backbone_name, loaders, epochs, num_classes, device, output_dir, in_chans=1):
    """从头训练（对照组）"""
    print(f"\n>>> Training from scratch: {backbone_name}")
    
    train_loader, val_loader, test_loader = loaders
    
    model = timm.create_model(backbone_name, pretrained=False, 
                              num_classes=num_classes, in_chans=in_chans)
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    scaler = GradScaler()
    
    best_val_acc, best_epoch = 0, 0
    
    for epoch in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"E{epoch}", leave=False)
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        
        val_acc = evaluate(model, val_loader, device)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc
            }, output_dir / f"teacher_{backbone_name}_scratch_best.pth")
        
        if epoch % 10 == 0 or epoch == epochs:
            print(f"E{epoch}: val={val_acc:.4f} (best={best_val_acc:.4f})")
    
    # 测试
    ckpt = torch.load(output_dir / f"teacher_{backbone_name}_scratch_best.pth", weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_acc = evaluate(model, test_loader, device)
    test_f1 = compute_f1(model, test_loader, device)
    
    return {
        "name": f"{backbone_name}_scratch",
        "val_acc": best_val_acc,
        "test_acc": test_acc,
        "test_f1": test_f1
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["pretrain", "finetune", "scratch", "all"], default="all")
    parser.add_argument("--backbone", type=str, default="convnextv2_tiny")
    parser.add_argument("--mask-ratio", type=float, default=0.9)
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="results/mae_pretrain")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Backbone: {args.backbone}, Mask ratio: {args.mask_ratio}")
    
    # 加载数据
    train_l, val_l, test_l, meta = quick_load_dataset(
        "ISCXVPN2016",
        dataset_root="/workspace/yqm/Dataset",
        batch_size=args.batch_size,
        num_workers=8
    )
    num_classes = meta["num_classes"]
    print(f"Dataset: {num_classes} classes, BS={args.batch_size}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    loaders = (train_l, val_l, test_l)
    results = {}
    
    # Phase 1: MAE预训练
    if args.phase in ["pretrain", "all"]:
        print("\n" + "="*70)
        print("Phase 1: MAE Self-supervised Pretraining")
        print("="*70)
        
        mae = CNNMaskedAutoencoder(args.backbone, mask_ratio=args.mask_ratio)
        mae = mae.to(device)
        print(f"MAE Params: {count_params(mae):.2f}M")
        
        pretrain_result = pretrain_mae(
            mae, train_l, args.pretrain_epochs, device, output_dir,
            f"{args.backbone}_mr{args.mask_ratio}", 
            base_lr=0.01, batch_size=args.batch_size
        )
        results["pretrain"] = pretrain_result
    
    # Phase 2: 微调
    if args.phase in ["finetune", "all"]:
        print("\n" + "="*70)
        print("Phase 2: Finetuning with Pretrained Weights")
        print("="*70)
        
        # 创建教师模型并加载预训练权重
        teacher = timm.create_model(args.backbone, pretrained=False, 
                                    num_classes=num_classes, in_chans=1)
        
        # 加载MAE预训练的编码器权重
        mae_ckpt_path = output_dir / f"mae_{args.backbone}_mr{args.mask_ratio}_best.pth"
        if mae_ckpt_path.exists():
            ckpt = torch.load(mae_ckpt_path, map_location='cpu', weights_only=False)
            # 加载编码器权重
            encoder_state = {}
            for k, v in ckpt['model_state_dict'].items():
                if k.startswith('encoder.'):
                    new_k = k.replace('encoder.', '')
                    encoder_state[new_k] = v
            
            # 尝试加载（忽略不匹配的key）
            missing, unexpected = teacher.load_state_dict(encoder_state, strict=False)
            print(f"  Loaded pretrained weights, missing: {len(missing)}, unexpected: {len(unexpected)}")
        
        teacher = teacher.to(device)
        
        # 重新加载数据（较小batch用于微调）
        train_l_ft, val_l_ft, test_l_ft, _ = quick_load_dataset(
            "ISCXVPN2016",
            dataset_root="/workspace/yqm/Dataset",
            batch_size=128,
            num_workers=8
        )
        
        finetune_result = finetune_teacher(
            teacher, (train_l_ft, val_l_ft, test_l_ft), 
            args.finetune_epochs, device, output_dir,
            f"{args.backbone}_pretrained_mr{args.mask_ratio}"
        )
        results["pretrained_finetune"] = finetune_result
    
    # Phase 3: 从头训练对照
    if args.phase in ["scratch", "all"]:
        print("\n" + "="*70)
        print("Phase 3: Training from Scratch (Baseline)")
        print("="*70)
        
        # 重新加载数据
        train_l_sc, val_l_sc, test_l_sc, _ = quick_load_dataset(
            "ISCXVPN2016",
            dataset_root="/workspace/yqm/Dataset",
            batch_size=128,
            num_workers=8
        )
        
        scratch_result = train_from_scratch(
            args.backbone, (train_l_sc, val_l_sc, test_l_sc),
            args.finetune_epochs, num_classes, device, output_dir
        )
        results["scratch"] = scratch_result
    
    # 保存结果
    with open(output_dir / f"results_{args.backbone}_mr{args.mask_ratio}.json", "w") as f:
        # 移除不可序列化的history
        save_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                save_results[k] = {kk: vv for kk, vv in v.items() if kk != "history"}
            else:
                save_results[k] = v
        json.dump(save_results, f, indent=2)
    
    # 打印对比结果
    print("\n" + "="*70)
    print("实验结果对比")
    print("="*70)
    print(f"{'Method':<35} {'Val Acc':<12} {'Test Acc':<12} {'F1':<12}")
    print("-"*70)
    
    for key, res in results.items():
        if isinstance(res, dict) and 'test_acc' in res:
            print(f"{res.get('name', key):<35} {res['val_acc']*100:.2f}%       {res['test_acc']*100:.2f}%      {res['test_f1']*100:.2f}%")
    
    print(f"\n结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
