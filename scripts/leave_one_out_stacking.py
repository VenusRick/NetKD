#!/usr/bin/env python3
"""
Leave-One-Out分析：评估每个教师对Stacking集成的贡献

通过移除每个教师并重新训练Stacking，评估性能下降来衡量贡献度
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict
import argparse
from pathlib import Path


class StackingEnsemble(nn.Module):
    """简化的Stacking模型"""
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        self.fc1 = nn.Linear(num_teachers * num_classes, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, teacher_logits):
        # teacher_logits: (B, num_teachers, num_classes)
        x = teacher_logits.flatten(1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.fc2(x)


def train_stacking(teacher_logits_list: List[torch.Tensor], 
                   labels: torch.Tensor,
                   num_classes: int,
                   epochs: int = 12,
                   lr: float = 0.0005,
                   device: str = 'cuda') -> nn.Module:
    """
    训练Stacking模型
    
    Args:
        teacher_logits_list: List of teacher logits, each of shape (N, C)
        labels: Ground truth labels of shape (N,)
        num_classes: Number of classes
        epochs: Training epochs
        lr: Learning rate
        device: Device to train on
    
    Returns:
        Trained stacking model
    """
    num_teachers = len(teacher_logits_list)
    n_samples = len(labels)
    
    # 堆叠教师输出: (N, num_teachers, C)
    stacked_logits = torch.stack(teacher_logits_list, dim=1).to(device)
    labels = labels.to(device)
    
    # 创建数据集
    dataset = torch.utils.data.TensorDataset(stacked_logits, labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    
    # 初始化模型
    model = StackingEnsemble(num_teachers, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_logits, batch_labels in loader:
            optimizer.zero_grad()
            output = model(batch_logits)
            loss = criterion(output, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    return model


def evaluate_stacking(model: nn.Module, 
                     teacher_logits_list: List[torch.Tensor],
                     labels: torch.Tensor,
                     device: str = 'cuda') -> float:
    """
    评估Stacking模型准确率
    """
    model.eval()
    stacked_logits = torch.stack(teacher_logits_list, dim=1).to(device)
    labels = labels.to(device)
    
    with torch.no_grad():
        predictions = model(stacked_logits).argmax(dim=-1)
        accuracy = (predictions == labels).float().mean().item()
    
    return accuracy


def leave_one_out_analysis(teacher_models: List[nn.Module],
                           teacher_names: List[str],
                           train_loader,
                           test_loader,
                           num_classes: int,
                           device: str = 'cuda') -> Dict:
    """
    执行Leave-One-Out分析
    
    Args:
        teacher_models: List of teacher models
        teacher_names: Names of teachers
        train_loader: Training data loader
        test_loader: Test data loader
        num_classes: Number of classes
        device: Device to use
    
    Returns:
        Dictionary of results
    """
    n_teachers = len(teacher_models)
    
    # 1. 收集所有教师的输出
    print("📊 收集教师模型输出...")
    teacher_train_logits = [[] for _ in range(n_teachers)]
    teacher_test_logits = [[] for _ in range(n_teachers)]
    train_labels, test_labels = [], []
    
    for model in teacher_models:
        model.eval()
        model.to(device)
    
    with torch.no_grad():
        # 训练集
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            if len(train_labels) == 0 or len(train_labels[-1]) < len(train_loader.dataset):
                train_labels.append(y)
            
            for i, model in enumerate(teacher_models):
                logits = model(x)
                teacher_train_logits[i].append(logits.cpu())
        
        # 测试集
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if len(test_labels) == 0 or len(test_labels[-1]) < len(test_loader.dataset):
                test_labels.append(y)
            
            for i, model in enumerate(teacher_models):
                logits = model(x)
                teacher_test_logits[i].append(logits.cpu())
    
    # 合并
    train_labels = torch.cat(train_labels)
    test_labels = torch.cat(test_labels)
    teacher_train_logits = [torch.cat(logits) for logits in teacher_train_logits]
    teacher_test_logits = [torch.cat(logits) for logits in teacher_test_logits]
    
    # 2. 训练完整Stacking (baseline)
    print("\n🏗️  训练完整Stacking (baseline)...")
    full_stacking = train_stacking(teacher_train_logits, train_labels, num_classes, device=device)
    full_acc = evaluate_stacking(full_stacking, teacher_test_logits, test_labels, device=device)
    print(f"  完整Stacking准确率: {full_acc:.4f}")
    
    # 3. Leave-One-Out分析
    results = {
        'full_accuracy': full_acc,
        'teachers': {}
    }
    
    print("\n�� Leave-One-Out分析:")
    print("-" * 80)
    
    for i in range(n_teachers):
        teacher_name = teacher_names[i]
        print(f"\n  移除: {teacher_name}")
        
        # 移除第i个教师
        remaining_indices = [j for j in range(n_teachers) if j != i]
        remaining_train_logits = [teacher_train_logits[j] for j in remaining_indices]
        remaining_test_logits = [teacher_test_logits[j] for j in remaining_indices]
        
        # 训练新的Stacking
        loo_stacking = train_stacking(remaining_train_logits, train_labels, num_classes, device=device)
        loo_acc = evaluate_stacking(loo_stacking, remaining_test_logits, test_labels, device=device)
        
        # 计算贡献度
        contribution = full_acc - loo_acc
        
        results['teachers'][teacher_name] = {
            'loo_accuracy': loo_acc,
            'contribution': contribution,
            'relative_contribution': contribution / full_acc if full_acc > 0 else 0
        }
        
        print(f"    剩余Stacking准确率: {loo_acc:.4f}")
        print(f"    贡献度: {contribution:+.4f} ({contribution/full_acc*100:+.2f}%)")
    
    return results


def print_loo_report(results: Dict):
    """打印Leave-One-Out分析报告"""
    print("\n" + "="*80)
    print("Leave-One-Out Stacking 贡献度分析")
    print("="*80)
    
    print(f"\n🎯 完整Stacking准确率: {results['full_accuracy']:.4f}")
    
    print(f"\n👥 各教师贡献度排名:")
    print("-" * 80)
    print(f"{'教师':<25} {'移除后准确率':<15} {'贡献度':<15} {'相对贡献':<15}")
    print("-" * 80)
    
    # 按贡献度排序
    sorted_teachers = sorted(
        results['teachers'].items(),
        key=lambda x: x[1]['contribution'],
        reverse=True
    )
    
    for name, metrics in sorted_teachers:
        print(f"{name:<25} "
              f"{metrics['loo_accuracy']:<15.4f} "
              f"{metrics['contribution']:+14.4f} "
              f"{metrics['relative_contribution']*100:+14.2f}%")
    
    print("\n💡 解释:")
    print("  - 贡献度 > 0: 移除该教师会降低性能，教师有价值")
    print("  - 贡献度 < 0: 移除该教师会提升性能，教师可能有害")
    print("  - 贡献度 ≈ 0: 教师可有可无")
    print("="*80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-One-Out Stacking Analysis")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory with teacher checkpoints")
    parser.add_argument("--dataset", type=str, default="ISCXVPN2016", help="Dataset name")
    parser.add_argument("--dataset_root", type=str, default="/workspace/yqm/Dataset", help="Dataset root")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    
    args = parser.parse_args()
    
    print("🚀 Leave-One-Out Stacking分析")
    print(f"  数据集: {args.dataset}")
    print(f"  检查点目录: {args.checkpoint_dir}")
    
    # TODO: 实现完整的命令行接口
    print("\n⚠️  请在Python脚本中使用leave_one_out_analysis函数")
