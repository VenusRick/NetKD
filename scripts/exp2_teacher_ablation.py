#!/usr/bin/env python3
"""
实验2: 教师模型消融实验 (Teacher Ablation Study)

基于最优的3教师组合 (DenseNet121-ECA, MobileNetV3-Large-ECA, ConvNeXtV2-Tiny-ECA),
测试不同教师组合的性能：
1. 完整组合 (3个教师)
2. 删除任意一个教师 (3种组合)
3. 只保留一个教师 (3种组合)

总共7种组合的对比实验。
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

from models.teacher_models import (
    DenseNet121Teacher,
    MobileNetV3LargeTeacher,
    ConvNeXtV2TinyTeacher,
)
from torch.cuda.amp import autocast, GradScaler

# 教师模型配置
TEACHER_CONFIGS = {
    'densenet121': {
        'class': DenseNet121Teacher,
        'name': 'DenseNet121-ECA',
    },
    'mobilenetv3': {
        'class': MobileNetV3LargeTeacher,
        'name': 'MobileNetV3-Large-ECA',
    },
    'convnextv2': {
        'class': ConvNeXtV2TinyTeacher,
        'name': 'ConvNeXtV2-Tiny-ECA',
    },
}

# 消融实验组合
ABLATION_COMBINATIONS = {
    'full_3teachers': ['densenet121', 'mobilenetv3', 'convnextv2'],
    'remove_densenet': ['mobilenetv3', 'convnextv2'],
    'remove_mobilenet': ['densenet121', 'convnextv2'],
    'remove_convnext': ['densenet121', 'mobilenetv3'],
    'only_densenet': ['densenet121'],
    'only_mobilenet': ['mobilenetv3'],
    'only_convnext': ['convnextv2'],
}

def create_teacher(key: str, num_classes: int, device: torch.device) -> nn.Module:
    """创建教师模型"""
    cfg = TEACHER_CONFIGS[key]
    model = cfg['class'](num_classes=num_classes, pretrained=True, use_eca=True)
    return model.to(device)

def get_data_loaders(data_root: str, batch_size: int = 64, num_workers: int = 4):
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

def train_teacher(model, train_loader, val_loader, device, epochs=30, lr=0.001):
    """训练单个教师模型"""
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_acc = 0.0
    best_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
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
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        scheduler.step()
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        
        print(f'  Epoch {epoch+1}/{epochs}: Train Acc={train_acc:.2f}%, Val Acc={val_acc:.2f}%')
    
    if best_state:
        model.load_state_dict(best_state)
    
    return model, best_acc

def evaluate_ensemble(teachers: list, test_loader, device):
    """评估教师集成的性能"""
    for teacher in teachers:
        teacher.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 平均集成
            outputs = None
            for teacher in teachers:
                logits = teacher(images)
                probs = torch.softmax(logits, dim=1)
                if outputs is None:
                    outputs = probs
                else:
                    outputs += probs
            
            outputs /= len(teachers)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return 100. * correct / total

def run_ablation_experiment(
    data_root: str,
    output_dir: str,
    gpu_id: int = 0,
    epochs: int = 30,
    batch_size: int = 64,
):
    """运行消融实验"""
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
    
    for combo_name, teacher_keys in ABLATION_COMBINATIONS.items():
        print(f'\n{"="*60}')
        print(f'Running combination: {combo_name}')
        print(f'Teachers: {[TEACHER_CONFIGS[k]["name"] for k in teacher_keys]}')
        print(f'{"="*60}')
        
        start_time = time.time()
        teachers = []
        teacher_results = {}
        
        # 训练每个教师
        for key in teacher_keys:
            cfg = TEACHER_CONFIGS[key]
            print(f'\nTraining {cfg["name"]}...')
            
            model = create_teacher(key, num_classes, device)
            model, val_acc = train_teacher(model, train_loader, val_loader, device, epochs)
            teachers.append(model)
            teacher_results[key] = {'val_acc': val_acc}
            print(f'{cfg["name"]} - Best Val Acc: {val_acc:.2f}%')
        
        # 评估集成性能
        print('\nEvaluating ensemble...')
        ensemble_acc = evaluate_ensemble(teachers, test_loader, device)
        
        elapsed = time.time() - start_time
        
        results[combo_name] = {
            'teachers': [TEACHER_CONFIGS[k]['name'] for k in teacher_keys],
            'num_teachers': len(teacher_keys),
            'teacher_results': teacher_results,
            'ensemble_test_acc': ensemble_acc,
            'training_time_seconds': elapsed,
        }
        
        print(f'\n{combo_name} Results:')
        print(f'  Ensemble Test Accuracy: {ensemble_acc:.2f}%')
        print(f'  Training Time: {elapsed/60:.1f} minutes')
        
        # 保存模型
        combo_dir = os.path.join(output_dir, combo_name)
        os.makedirs(combo_dir, exist_ok=True)
        for key, model in zip(teacher_keys, teachers):
            torch.save(model.state_dict(), os.path.join(combo_dir, f'{key}.pth'))
        
        # 清理GPU内存
        del teachers
        torch.cuda.empty_cache()
    
    # 保存结果
    results_path = os.path.join(output_dir, 'ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # 生成报告
    report = generate_report(results)
    report_path = os.path.join(output_dir, 'ablation_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f'\nResults saved to: {results_path}')
    print(f'Report saved to: {report_path}')
    
    return results

def generate_report(results: dict) -> str:
    """生成Markdown报告"""
    report = []
    report.append('# 教师模型消融实验报告')
    report.append(f'\n生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    
    report.append('## 实验概述')
    report.append('\n本实验探究不同教师模型组合对知识蒸馏性能的影响。\n')
    report.append('最优教师组合: DenseNet121-ECA + MobileNetV3-Large-ECA + ConvNeXtV2-Tiny-ECA\n')
    
    report.append('## 实验结果\n')
    report.append('| 组合名称 | 教师数量 | 教师模型 | 集成测试准确率 | 训练时间 |')
    report.append('|----------|----------|----------|----------------|----------|')
    
    sorted_results = sorted(results.items(), key=lambda x: x[1]['ensemble_test_acc'], reverse=True)
    
    for combo_name, data in sorted_results:
        teachers_str = ', '.join(data['teachers'])
        acc = data['ensemble_test_acc']
        time_min = data['training_time_seconds'] / 60
        report.append(f'| {combo_name} | {data["num_teachers"]} | {teachers_str} | {acc:.2f}% | {time_min:.1f}min |')
    
    report.append('\n## 消融分析\n')
    
    # 找到完整组合的结果
    full_acc = results.get('full_3teachers', {}).get('ensemble_test_acc', 0)
    
    report.append('### 删除单个教师的影响\n')
    for combo in ['remove_densenet', 'remove_mobilenet', 'remove_convnext']:
        if combo in results:
            acc = results[combo]['ensemble_test_acc']
            diff = acc - full_acc
            removed = combo.replace('remove_', '').replace('densenet', 'DenseNet121').replace('mobilenet', 'MobileNetV3').replace('convnext', 'ConvNeXtV2')
            report.append(f'- 删除 {removed}: {acc:.2f}% (相比完整组合 {diff:+.2f}%)')
    
    report.append('\n### 单个教师的性能\n')
    for combo in ['only_densenet', 'only_mobilenet', 'only_convnext']:
        if combo in results:
            acc = results[combo]['ensemble_test_acc']
            teacher_name = results[combo]['teachers'][0]
            report.append(f'- {teacher_name}: {acc:.2f}%')
    
    report.append('\n## 结论\n')
    report.append('(实验完成后根据结果填写)')
    
    return '\n'.join(report)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Teacher Ablation Study')
    parser.add_argument('--data-root', type=str, 
                        default='/workspace/yqm/Dataset/ISCXVPN2016',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str,
                        default='results/teacher_ablation',
                        help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--epochs', type=int, default=30, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    
    args = parser.parse_args()
    
    run_ablation_experiment(
        data_root=args.data_root,
        output_dir=args.output_dir,
        gpu_id=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
