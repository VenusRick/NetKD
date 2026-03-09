#!/usr/bin/env python3
"""
实验1: 学生模型基线实验 (Student Baseline without KD)

直接使用未经训练的学生模型在ISCXVPN2016数据集上训练，
不使用知识蒸馏，作为对比基线。

支持的学生模型:
- MobileNetV2
- MobileNetV3-Small
- MobileNetV3-Large
- GhostNet
- EfficientNet-Lite0
- RepVit-M0.9
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import timm
from torch.cuda.amp import autocast, GradScaler

# 学生模型配置
STUDENT_CONFIGS = {
    'mobilenetv2': {
        'timm_name': 'mobilenetv2_100',
        'display_name': 'MobileNetV2',
    },
    'mobilenetv3_small': {
        'timm_name': 'mobilenetv3_small_100',
        'display_name': 'MobileNetV3-Small',
    },
    'mobilenetv3_large': {
        'timm_name': 'mobilenetv3_large_100',
        'display_name': 'MobileNetV3-Large',
    },
    'ghostnet': {
        'timm_name': 'ghostnet_100',
        'display_name': 'GhostNet',
    },
    'efficientnet_lite0': {
        'timm_name': 'efficientnet_lite0',
        'display_name': 'EfficientNet-Lite0',
    },
    'repvit_m0_9': {
        'timm_name': 'repvit_m0_9',
        'display_name': 'RepVit-M0.9',
    },
}

def create_student_model(model_key: str, num_classes: int, in_channels: int = 1) -> nn.Module:
    """创建学生模型"""
    cfg = STUDENT_CONFIGS[model_key]
    model = timm.create_model(
        cfg['timm_name'],
        pretrained=False,  # 不使用预训练权重
        num_classes=num_classes,
        in_chans=in_channels,
    )
    return model

def get_data_loaders(data_root: str, batch_size: int = 128, num_workers: int = 4):
    """获取数据加载器"""
    # 灰度图像转换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])
    
    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'valid')
    test_dir = os.path.join(data_root, 'test')
    
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    
    num_classes = len(train_dataset.classes)
    print(f'Classes: {train_dataset.classes}')
    print(f'Train samples: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')
    
    return train_loader, val_loader, test_loader, num_classes

def train_epoch(model, train_loader, criterion, optimizer, scaler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, data_loader, device):
    """评估模型"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def train_student(
    model_key: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    num_classes: int,
    device: torch.device,
    output_dir: str,
    epochs: int = 50,
    lr: float = 0.001,
):
    """训练单个学生模型"""
    cfg = STUDENT_CONFIGS[model_key]
    model = create_student_model(model_key, num_classes).to(device)
    
    # 计算参数量
    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f'\nTraining {cfg["display_name"]}')
    print(f'  Parameters: {params/1e6:.2f}M (trainable: {trainable/1e6:.2f}M)')
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_val_acc = 0.0
    best_state = None
    history = {'train_loss': [], 'train_acc': [], 'val_acc': []}
    
    start_time = time.time()
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, device)
        val_acc = evaluate(model, val_loader, device)
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f'  Epoch {epoch+1}/{epochs}: Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
    
    training_time = time.time() - start_time
    
    # 加载最佳模型并测试
    if best_state:
        model.load_state_dict(best_state)
    test_acc = evaluate(model, test_loader, device)
    
    print(f'  Best Val Acc: {best_val_acc:.2f}%, Test Acc: {test_acc:.2f}%')
    print(f'  Training Time: {training_time/60:.1f} minutes')
    
    # 保存模型和结果
    model_dir = os.path.join(output_dir, model_key)
    os.makedirs(model_dir, exist_ok=True)
    
    torch.save(best_state, os.path.join(model_dir, 'best_model.pth'))
    
    result = {
        'model_name': cfg['display_name'],
        'model_key': model_key,
        'parameters': params,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'training_time_seconds': training_time,
        'epochs': epochs,
        'learning_rate': lr,
        'history': history,
    }
    
    with open(os.path.join(model_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def run_baseline_experiment(
    data_root: str,
    output_dir: str,
    gpu_id: int = 0,
    epochs: int = 50,
    batch_size: int = 128,
):
    """运行所有学生模型的基线实验"""
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    print('Loading data...')
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        data_root, batch_size=batch_size
    )
    print(f'Number of classes: {num_classes}')
    
    results = {}
    
    for model_key in STUDENT_CONFIGS.keys():
        try:
            result = train_student(
                model_key, train_loader, val_loader, test_loader,
                num_classes, device, output_dir, epochs
            )
            results[model_key] = result
        except Exception as e:
            print(f'Error training {model_key}: {e}')
            results[model_key] = {'error': str(e)}
        
        # 清理GPU内存
        torch.cuda.empty_cache()
    
    # 保存汇总结果
    summary_path = os.path.join(output_dir, 'baseline_results.json')
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = generate_report(results)
    report_path = os.path.join(output_dir, 'baseline_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f'\nResults saved to: {summary_path}')
    print(f'Report saved to: {report_path}')
    
    return results

def generate_report(results: dict) -> str:
    """生成Markdown报告"""
    report = []
    report.append('# 学生模型基线实验报告')
    report.append(f'\n生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    report.append('## 实验概述')
    report.append('\n本实验直接训练学生模型（无知识蒸馏），作为对比基线。\n')
    report.append('数据集: ISCXVPN2016\n')
    
    report.append('## 实验结果\n')
    report.append('| 模型 | 参数量 | 最佳验证准确率 | 测试准确率 | 训练时间 |')
    report.append('|------|--------|----------------|------------|----------|')
    
    # 按测试准确率排序
    valid_results = [(k, v) for k, v in results.items() if 'error' not in v]
    sorted_results = sorted(valid_results, key=lambda x: x[1]['test_acc'], reverse=True)
    
    for model_key, data in sorted_results:
        params = data['parameters'] / 1e6
        val_acc = data['best_val_acc']
        test_acc = data['test_acc']
        time_min = data['training_time_seconds'] / 60
        report.append(f'| {data["model_name"]} | {params:.2f}M | {val_acc:.2f}% | {test_acc:.2f}% | {time_min:.1f}min |')
    
    # 记录错误
    error_results = [(k, v) for k, v in results.items() if 'error' in v]
    if error_results:
        report.append('\n## 训练失败的模型\n')
        for model_key, data in error_results:
            report.append(f'- {model_key}: {data["error"]}')
    
    report.append('\n## 结论\n')
    if sorted_results:
        best = sorted_results[0]
        report.append(f'最佳模型: {best[1]["model_name"]} (测试准确率: {best[1]["test_acc"]:.2f}%)')
    
    return '\n'.join(report)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Student Baseline Experiment')
    parser.add_argument('--data-root', type=str, 
                        default='/workspace/yqm/Dataset/ISCXVPN2016',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str,
                        default='results/student_baseline',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    
    args = parser.parse_args()
    
    run_baseline_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
