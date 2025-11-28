"""
注意力机制消融实验
对比 Agent Attention, CBAM, ECA, SimAM 和 None (无注意力) 的性能
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import json
import sys
import time
from tqdm import tqdm

from data_preprocessing.image_loader import quick_load_dataset
from models.student_model_flexible import FlexibleStudentNet


def train_epoch(model, loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return total_loss / len(loader), correct / total


def run_experiment(attention_type, args):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"实验: {attention_type.upper()}")
    print(f"{'='*80}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader, meta = quick_load_dataset(
        dataset_name=args['dataset'],
        dataset_root=args['data_root'],
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],
        pin_memory=True,
        val_ratio=0.15,
        test_ratio=0.15
    )
    num_classes = meta['num_classes']
    print(f"✓ 数据集加载完成: 类别数={num_classes}")
    
    # 创建模型
    print(f"\n创建模型 (注意力类型: {attention_type})...")
    model = FlexibleStudentNet(
        num_classes=num_classes,
        attention_type=attention_type
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args['lr'],
        weight_decay=args['weight_decay']
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args['epochs']
    )
    
    # 训练
    print(f"\n开始训练 ({args['epochs']} epochs)...")
    best_acc = 0
    best_epoch = 0
    history = []
    
    start_time = time.time()
    
    for epoch in range(args['epochs']):
        print(f"\nEpoch {epoch+1}/{args['epochs']}")
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        })
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            output_dir = Path(args['output_dir']) / f"attention_{attention_type}"
            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), output_dir / "best_model.pth")
            print(f"✓ 保存最佳模型 (Acc: {best_acc*100:.2f}%)")
    
    training_time = time.time() - start_time
    
    print(f"\n{'='*80}")
    print(f"训练完成!")
    print(f"最佳验证准确率: {best_acc*100:.2f}% (Epoch {best_epoch})")
    print(f"训练耗时: {training_time:.2f}s ({training_time/60:.2f}min)")
    print(f"{'='*80}")
    
    # 保存结果
    results = {
        'attention_type': attention_type,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'training_time': training_time,
        'config': args,
        'history': history
    }
    
    output_dir = Path(args['output_dir']) / f"attention_{attention_type}"
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    # 实验配置
    config = {
        'dataset': 'ISCXVPN2016',
        'data_root': '/walnut_data/yqm/Dataset',
        'batch_size': 256,
        'num_workers': 8,
        'epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'output_dir': 'checkpoints/attention_ablation'
    }
    
    # 注意力机制列表
    attention_types = ['agent', 'cbam', 'eca', 'simam', 'none']
    
    # 如果命令行指定了特定的注意力类型
    if len(sys.argv) > 1:
        attention_type = sys.argv[1].lower()
        if attention_type not in attention_types:
            print(f"错误: 未知的注意力类型 '{attention_type}'")
            print(f"可用类型: {attention_types}")
            sys.exit(1)
        attention_types = [attention_type]
    
    print("="*80)
    print("注意力机制消融实验")
    print("="*80)
    print(f"数据集: {config['dataset']}")
    print(f"Batch Size: {config['batch_size']}")
    print(f"Epochs: {config['epochs']}")
    print(f"学习率: {config['lr']}")
    print(f"待测试的注意力机制: {attention_types}")
    print("="*80)
    
    # 运行所有实验
    all_results = {}
    for attention_type in attention_types:
        results = run_experiment(attention_type, config)
        all_results[attention_type] = results
    
    # 保存汇总结果
    summary_path = Path(config['output_dir']) / "summary_results.json"
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("所有实验完成!")
    print(f"汇总结果保存至: {summary_path}")
    print(f"{'='*80}")
    
    # 打印对比表格
    print("\n实验结果对比:")
    print(f"{'注意力类型':<15} {'最佳准确率':<12} {'参数量':<15} {'训练时间(分钟)':<15}")
    print("-"*60)
    for att_type, res in all_results.items():
        print(f"{att_type:<15} {res['best_acc']*100:>6.2f}%      "
              f"{res['total_params']:>10,}    {res['training_time']/60:>8.2f}")
