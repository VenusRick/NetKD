#!/usr/bin/env python3
"""
学生模型知识蒸馏训练脚本 (Student KD Training)

支持完整的 TrafficKD 损失函数: CE + FKL + RKL + Sinkhorn
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import yaml

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from models.student_registry import StudentRegistry, KDConfig
from models.teacher_registry import TeacherRegistry
from models.student_models_v2 import create_student_model_v2, get_model_info
from data_preprocessing.image_loader import get_dataset_loaders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrafficKDLoss(nn.Module):
    """TrafficKD 完整损失函数
    
    L = α_ce * L_CE + α_fkl * L_FKL + α_rkl * L_RKL + α_sk * L_Sinkhorn
    """
    
    def __init__(self, kd_config: KDConfig, num_classes: int):
        super().__init__()
        self.config = kd_config
        self.num_classes = num_classes
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """计算完整损失"""
        losses = {}
        total_loss = torch.tensor(0.0, device=student_logits.device)
        
        T = self.config.temperature
        
        # 1. Cross-Entropy Loss (hard labels)
        if self.config.use_ce and self.config.alpha_ce > 0:
            ce_loss = self.ce_loss(student_logits, labels)
            losses['ce'] = ce_loss
            total_loss = total_loss + self.config.alpha_ce * ce_loss
        
        # 2. Forward KL Divergence: KL(P_teacher || P_student)
        if self.config.use_fkl and self.config.alpha_fkl > 0:
            teacher_probs = F.softmax(teacher_logits / T, dim=-1)
            student_log_probs = F.log_softmax(student_logits / T, dim=-1)
            fkl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (T * T)
            losses['fkl'] = fkl_loss
            total_loss = total_loss + self.config.alpha_fkl * fkl_loss
        
        # 3. Reverse KL Divergence: KL(P_student || P_teacher)
        if self.config.use_rkl and self.config.alpha_rkl > 0:
            student_probs = F.softmax(student_logits / T, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits / T, dim=-1)
            rkl_loss = F.kl_div(teacher_log_probs, student_probs, reduction='batchmean') * (T * T)
            losses['rkl'] = rkl_loss
            total_loss = total_loss + self.config.alpha_rkl * rkl_loss
        
        # 4. Sinkhorn Divergence (简化版本)
        if self.config.use_sinkhorn and self.config.alpha_sinkhorn > 0:
            sinkhorn_loss = self._sinkhorn_divergence(
                F.softmax(student_logits / T, dim=-1),
                F.softmax(teacher_logits / T, dim=-1),
            )
            losses['sinkhorn'] = sinkhorn_loss
            total_loss = total_loss + self.config.alpha_sinkhorn * sinkhorn_loss
        
        losses['total'] = total_loss
        return losses
    
    def _sinkhorn_divergence(
        self,
        p: torch.Tensor,
        q: torch.Tensor,
        n_iters: int = 10,
        eps: float = 0.1,
    ) -> torch.Tensor:
        """简化的 Sinkhorn 散度计算"""
        # 使用 L2 距离作为代价矩阵的近似
        diff = p - q
        return (diff * diff).sum(dim=-1).mean()


def load_teachers(
    teacher_registry: TeacherRegistry,
    teacher_set_id: str,
    num_classes: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> Dict[str, nn.Module]:
    """加载教师模型"""
    teacher_set = teacher_registry.get_teacher_set(teacher_set_id)
    teachers = {}
    
    for teacher_name in teacher_set.teachers:
        model = teacher_registry.build_teacher(teacher_name, num_classes)
        
        # 尝试加载检查点
        ckpt_path = checkpoint_dir / f"{teacher_name}_best.pth"
        if ckpt_path.exists():
            state_dict = torch.load(ckpt_path, map_location=device)
            if 'model_state_dict' in state_dict:
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                model.load_state_dict(state_dict)
            logger.info(f"Loaded teacher checkpoint: {ckpt_path}")
        else:
            logger.warning(f"Teacher checkpoint not found: {ckpt_path}")
        
        model = model.to(device)
        model.eval()
        teachers[teacher_name] = model
    
    return teachers


def get_teacher_ensemble_logits(
    teachers: Dict[str, nn.Module],
    images: torch.Tensor,
) -> torch.Tensor:
    """获取教师集成的 logits (平均)"""
    with torch.no_grad():
        all_logits = []
        for teacher in teachers.values():
            logits = teacher(images)
            all_logits.append(logits)
        
        # 简单平均
        ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)
    
    return ensemble_logits


def train_one_epoch(
    student: nn.Module,
    teachers: Dict[str, nn.Module],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: TrafficKDLoss,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """训练一个 epoch"""
    student.train()
    
    total_loss = 0.0
    loss_components = {'ce': 0.0, 'fkl': 0.0, 'rkl': 0.0, 'sinkhorn': 0.0}
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # 获取教师集成 logits
        teacher_logits = get_teacher_ensemble_logits(teachers, images)
        
        if scaler is not None:
            with autocast():
                student_logits = student(images)
                losses = criterion(student_logits, teacher_logits, labels)
            
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            student_logits = student(images)
            losses = criterion(student_logits, teacher_logits, labels)
            
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
        
        total_loss += losses['total'].item()
        for key in loss_components:
            if key in losses:
                loss_components[key] += losses[key].item()
        
        _, predicted = student_logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    n_batches = len(train_loader)
    metrics = {
        'loss': total_loss / n_batches,
        'accuracy': 100.0 * correct / total,
    }
    for key in loss_components:
        metrics[f'loss_{key}'] = loss_components[key] / n_batches
    
    return metrics


@torch.no_grad()
def evaluate(
    student: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """评估模型"""
    student.eval()
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = student(images)
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = 100.0 * correct / total
    
    # 计算 F1 分数
    from sklearn.metrics import f1_score
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
    }


def main():
    parser = argparse.ArgumentParser(description="Student KD Training")
    parser.add_argument("--students_config", type=str, default="configs/students.yaml")
    parser.add_argument("--teachers_config", type=str, default="configs/teachers.yaml")
    parser.add_argument("--student_name", type=str, required=True)
    parser.add_argument("--teacher_set_id", type=str, required=True)
    parser.add_argument("--kd_config_id", type=str, default="traffic_kd_full")
    parser.add_argument("--dataset", type=str, default="ISCXVPN2016")
    parser.add_argument("--dataset_root", type=str, default="/workspace/yqm/Dataset")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--output_dir", type=str, default="results/student_kd")
    parser.add_argument("--teacher_ckpt_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 加载配置
    student_registry = StudentRegistry(args.students_config)
    teacher_registry = TeacherRegistry(args.teachers_config)
    
    kd_config = student_registry.get_kd_config(args.kd_config_id)
    logger.info(f"KD Config: {kd_config.display_name}")
    
    # 加载数据
    dataset_path = Path(args.dataset_root) / args.dataset
    train_loader, val_loader, test_loader, num_classes = get_dataset_loaders(
        dataset_path,
        batch_size=args.batch_size,
        num_workers=4,
    )
    logger.info(f"Dataset: {args.dataset}, num_classes: {num_classes}")
    
    # 创建学生模型
    student = create_student_model_v2(
        model_type=args.student_name,
        num_classes=num_classes,
        in_channels=1,
    )
    student = student.to(device)
    model_info = get_model_info(student)
    logger.info(f"Student: {args.student_name}, params: {model_info['total_params_M']:.2f}M")
    
    # 加载教师模型
    teachers = load_teachers(
        teacher_registry,
        args.teacher_set_id,
        num_classes,
        Path(args.teacher_ckpt_dir),
        device,
    )
    logger.info(f"Loaded {len(teachers)} teachers: {list(teachers.keys())}")
    
    # 创建损失函数和优化器
    criterion = TrafficKDLoss(kd_config, num_classes)
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )
    scaler = GradScaler() if device.type == 'cuda' else None
    
    # 创建输出目录
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / args.student_name / args.teacher_set_id / args.kd_config_id / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练循环
    best_val_acc = 0.0
    history = []
    
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            student, teachers, train_loader, optimizer, criterion,
            device, scaler, grad_clip=1.0,
        )
        val_metrics = evaluate(student, val_loader, device)
        scheduler.step()
        
        logger.info(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.2f}%, "
            f"Val Acc: {val_metrics['accuracy']:.2f}%"
        )
        
        history.append({
            'epoch': epoch,
            'train': train_metrics,
            'val': val_metrics,
            'lr': optimizer.param_groups[0]['lr'],
        })
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_metrics['accuracy'],
            }, output_dir / "best_model.pth")
    
    # 最终测试
    student.load_state_dict(torch.load(output_dir / "best_model.pth")['model_state_dict'])
    test_metrics = evaluate(student, test_loader, device)
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.2f}%")
    
    # 保存结果
    results = {
        'student_name': args.student_name,
        'teacher_set_id': args.teacher_set_id,
        'kd_config_id': args.kd_config_id,
        'dataset': args.dataset,
        'num_params_m': model_info['total_params_M'],
        'best_val_accuracy': best_val_acc,
        'test_metrics': test_metrics,
        'history': history,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'kd_config': vars(kd_config),
        },
    }
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")
    return results


if __name__ == "__main__":
    main()
