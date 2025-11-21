#!/usr/bin/env python3
"""
MAE-style Self-Supervised Pretraining Driver

This script implements Masked Autoencoder (MAE) style self-supervised pretraining
for the NetKD project on unlabeled traffic flow image data.

Key features:
- Batch size: 512
- Epochs: 200
- Base learning rate: 0.001 with linear scaling
- Optimizer: AdamW
- Mask ratio: 0.8
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessing import quick_load_dataset
from training.engine import pretrain_one_epoch
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from models import StudentNet


class SimpleArgs:
    """Simple argparse-like object for pretrain_one_epoch compatibility"""

    def __init__(
        self,
        mask_ratio=0.8,
        accum_iter=1,
        save_steps_freq=1000,
        output_dir='./checkpoints/pretrain',
        lr=1e-3,
        epochs=200,
        warmup_epochs=10,
        min_lr=1e-6,
    ):
        self.mask_ratio = mask_ratio
        self.accum_iter = accum_iter
        self.save_steps_freq = save_steps_freq
        self.output_dir = output_dir
        # Fields consumed by util.lr_sched.adjust_learning_rate
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr


class MAEWrapper(nn.Module):
    """Wrapper to make StudentNet compatible with MAE-style pretraining
    
    This adds a simple reconstruction head for masked autoencoding.
    """
    def __init__(self, base_model, patch_size=4, in_channels=1):
        super().__init__()
        self.base_model = base_model
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Simple reconstruction head (decoder)
        # Assuming feature dimension from backbone is 1024 (ShuffleNet output)
        self.decoder = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, patch_size * patch_size * in_channels)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, mask_ratio=0.9):
        """
        Forward pass with masking
        
        Args:
            x: input images [B, C, H, W]
            mask_ratio: ratio of patches to mask
            
        Returns:
            loss: reconstruction loss
            pred: predicted patches
            mask: binary mask
        """
        B, C, H, W = x.shape
        
        # Simple patching (divide image into patches)
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Reshape to patches [B, num_patches, patch_size*patch_size*C]
        x_patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x_patches = x_patches.contiguous().view(B, C, num_patches, self.patch_size * self.patch_size)
        x_patches = x_patches.permute(0, 2, 1, 3).contiguous().view(B, num_patches, -1)
        
        # Random masking
        num_masked = int(mask_ratio * num_patches)
        noise = torch.rand(B, num_patches, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Create mask (1 = keep, 0 = masked)
        mask = torch.ones([B, num_patches], device=x.device)
        mask[:, :num_masked] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Keep only unmasked patches for encoding
        ids_keep = ids_shuffle[:, num_masked:]
        x_masked = torch.gather(x_patches, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x_patches.shape[2]))
        
        # Reshape back for backbone input (approximate reconstruction)
        # For simplicity, we'll use the full image with masked patches zeroed out
        x_input = x.clone()
        mask_img = mask.view(B, num_patches_h, num_patches_w).unsqueeze(1)
        mask_img = mask_img.repeat_interleave(self.patch_size, dim=2).repeat_interleave(self.patch_size, dim=3)
        x_input = x_input * mask_img
        
        # Encode (through backbone)
        features = self.base_model.backbone(x_input)  # [B, C, H', W']
        features = self.global_pool(features).flatten(1)  # [B, C]
        
        # Decode
        pred_patches = self.decoder(features)  # [B, patch_size*patch_size*C]
        pred_patches = pred_patches.unsqueeze(1).repeat(1, num_patches, 1)  # Simple broadcast
        
        # Compute loss only on masked patches
        target = x_patches
        loss = (pred_patches - target) ** 2
        loss = (loss * (1 - mask.unsqueeze(-1))).sum() / (1 - mask).sum()
        
        return loss, pred_patches, mask


def get_args_parser():
    parser = argparse.ArgumentParser('MAE Self-Supervised Pretraining', add_help=False)
    
    # Dataset parameters
    parser.add_argument('--dataset', default='ISCXVPN2016', type=str,
                        choices=['ISCXVPN2016', 'ISCXTor2016', 'USTC-TFC2016', 
                                'CrossPlatform-Android', 'CrossPlatform-iOS', 'CICIoT2022'],
                        help='Dataset name for pretraining')
    parser.add_argument('--data_root', default='/walnut_data/yqm/Dataset', type=str,
                        help='Dataset root path')
    
    # Training parameters (as specified)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='Batch size per GPU (default: 512)')
    parser.add_argument('--epochs', default=200, type=int,
                        help='Number of pretraining epochs (default: 200)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Base learning rate (default: 0.001)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    
    # MAE parameters
    parser.add_argument('--mask_ratio', default=0.8, type=float,
                        help='Masking ratio (default: 0.8)')
    
    # Optimizer parameters
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Epochs to warmup LR')
    parser.add_argument('--min_lr', type=float, default=1e-6,
                        help='Lower lr bound for cyclic schedulers')
    
    # Other parameters
    parser.add_argument('--output_dir', default='./checkpoints/pretrain',
                        help='Path to save checkpoints')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers')
    parser.add_argument('--log_dir', default='./runs/pretrain',
                        help='Path for tensorboard logs')
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    parser.add_argument('--save_freq', default=10, type=int,
                        help='Save checkpoint every N epochs')
    
    # Dataset split parameters (allow disabling val/test by setting to 0)
    parser.add_argument('--val_ratio', type=float, default=0.0,
                        help='Validation set ratio (0 to disable)')
    parser.add_argument('--test_ratio', type=float, default=0.0,
                        help='Test set ratio (0 to disable)')
    
    return parser


def main(args):
    print(f"\n{'='*70}")
    print("MAE Self-Supervised Pretraining for NetKD")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Data root: {args.data_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Mask ratio: {args.mask_ratio}")
    print(f"Optimizer: AdamW")
    print(f"Output directory: {args.output_dir}")
    print(f"{'='*70}\n")
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset (we only need training data for pretraining, no labels needed)
    print("\nLoading dataset...")
    train_loader, val_loader, test_loader, metadata = quick_load_dataset(
        dataset_name=args.dataset,
        dataset_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    print(f"Dataset loaded!")
    print(f"  Training samples: {metadata['train_size']}")
    print(f"  Image size: {metadata['image_channels']}x{metadata['image_height']}x{metadata['image_width']}")
    print(f"  Number of classes: {metadata['num_classes']}")
    
    # Create model
    print("\nInitializing model...")
    base_model = StudentNet(num_classes=metadata['num_classes'])
    model = MAEWrapper(
        base_model=base_model,
        patch_size=4,  # 40x40 image -> 10x10 = 100 patches with patch_size=4
        in_channels=metadata['image_channels']
    )
    model = model.to(device)
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_parameters / 1e6:.2f}M")
    
    # Setup optimizer with linear learning rate scaling (baseline batch size 512)
    lr_scaled = args.lr * (args.batch_size / 512.0)
    print(f"Learning rate (scaled): {lr_scaled:.6f} (base {args.lr} * {args.batch_size}/512)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_scaled,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Loss scaler for mixed precision
    loss_scaler = NativeScaler()
    amp_autocast = torch.cuda.amp.autocast if device.type == 'cuda' else torch.cpu.amp.autocast
    
    # Tensorboard logger
    log_writer = SummaryWriter(log_dir=args.log_dir)
    
    # Simple args object for pretrain_one_epoch
    simple_args = SimpleArgs(
        mask_ratio=args.mask_ratio,
        accum_iter=1,
        save_steps_freq=1000,
        output_dir=args.output_dir,
        lr=lr_scaled,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
    )
    
    print(f"\n{'='*70}")
    print("Starting pretraining...")
    print(f"{'='*70}\n")
    
    # Training loop
    for epoch in range(args.epochs):
        # Train one epoch
        train_stats = pretrain_one_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_scaler=loss_scaler,
            amp_autocast=amp_autocast,
            log_writer=log_writer,
            model_without_ddp=model,
            args=simple_args
        )
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0 or epoch == args.epochs - 1:
            checkpoint_path = Path(args.output_dir) / f'pretrain_epoch_{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Log to tensorboard
        if log_writer is not None:
            log_writer.add_scalar('train/loss_epoch', train_stats['loss'], epoch)
            log_writer.add_scalar('train/lr_epoch', train_stats['lr'], epoch)
    
    # Save final backbone only (for easy loading in fine-tuning)
    final_backbone_path = Path(args.output_dir) / 'backbone_pretrained.pth'
    torch.save(model.base_model.backbone.state_dict(), final_backbone_path)
    print(f"\n{'='*70}")
    print(f"✅ Pretraining completed!")
    print(f"✅ Final backbone saved: {final_backbone_path}")
    print(f"{'='*70}\n")
    
    log_writer.close()


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    main(args)
