"""
修复版注意力消融实验 - 针对Agent Attention调整训练策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import sys
from pathlib import Path

from data_preprocessing.image_loader import quick_load_dataset
from models.student_model_flexible import FlexibleStudentNet


def get_optimizer_and_scheduler(model, attention_type, lr, weight_decay, epochs):
    """根据注意力类型返回优化器和调度器"""
    
    # Agent Attention需要特殊处理
    if attention_type == 'agent':
        # 为Agent Attention使用更低的学习率和warmup
        optimizer = optim.AdamW(model.parameters(), lr=lr*0.1, weight_decay=weight_decay)
        
        # Warmup + Cosine Annealing
        def warmup_cosine(epoch):
            warmup_epochs = 10
            if epoch < warmup_epochs:
                return (epoch + 1) / warmup_epochs
            else:
                progress = (epoch - warmup_epochs) / (epochs - warmup_epochs)
                return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_cosine)
    else:
        # 其他机制使用标准配置
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    return optimizer, scheduler


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
        
        # Gradient clipping for Agent Attention
        loss.backward()
        if hasattr(model, 'attention_type') and model.attention_type == 'agent':
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 为Agent Attention使用更小的batch size
    batch_size = args['batch_size'] // 2 if attention_type == 'agent' else args['batch_size']
    print(f"Batch Size: {batch_size} (Agent使用减半的batch size)")
    
    # 加载数据
    print("\n加载数据集...")
    train_loader, val_loader, test_loader, meta = quick_load_dataset(
        dataset_name=args['dataset'],
        dataset_root=args['data_root'],
        batch_size=batch_size,
        num_workers=args['num_workers'],
        pin_memory=True,
        val_ratio=0.15,
        test_ratio=0.15
    )
    num_classes = meta['num_classes']
    print(f"类别数: {num_classes}")
    
    # 创建模型
    print(f"\n创建模型: FlexibleStudentNet with {attention_type} attention...")
    model = FlexibleStudentNet(
        num_classes=num_classes,
        attention_type=attention_type,
        in_channels=1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, attention_type, args['lr'], args['weight_decay'], args['epochs']
    )
    
    # 训练
    print(f"\n开始训练 ({args['epochs']} epochs)...")
    best_acc = 0
    best_epoch = 0
    history = []
    start_time = time.time()
    
    for epoch in range(1, args['epochs'] + 1):
        print(f"\nEpoch {epoch}/{args['epochs']}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        print(f"LR: {current_lr:.6f}")
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': current_lr
        })
        
        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            
            output_dir = Path(args['output_dir']) / f'attention_{attention_type}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = output_dir / 'best_model.pth'
            torch.save(model.state_dict(), model_path)
            print(f"✓ 保存最佳模型 (Val Acc: {best_acc*100:.2f}%)")
    
    training_time = time.time() - start_time
    print(f"\n训练完成! 用时: {training_time/60:.2f}分钟")
    print(f"最佳验证准确率: {best_acc*100:.2f}% (Epoch {best_epoch})")
    
    # 保存结果
    output_dir = Path(args['output_dir']) / f'attention_{attention_type}'
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
    
    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == '__main__':
    config = {
        'dataset': 'ISCXVPN2016',
        'data_root': '/walnut_data/yqm/Dataset',
        'batch_size': 256,
        'num_workers': 8,
        'epochs': 50,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'output_dir': 'checkpoints/attention_ablation_fixed'
    }
    
    # 只重新训练Agent Attention
    attention_type = 'agent'
    
    if len(sys.argv) > 1:
        attention_type = sys.argv[1].lower()
    
    print("="*80)
    print("注意力机制消融实验 (修复版)")
    print("="*80)
    print(f"数据集: {config['dataset']}")
    print(f"基础Batch Size: {config['batch_size']} (Agent会自动减半)")
    print(f"Epochs: {config['epochs']}")
    print(f"学习率: {config['lr']} (Agent会自动降至0.1x)")
    print(f"注意力类型: {attention_type}")
    print("="*80)
    
    results = run_experiment(attention_type, config)
    
    print("\n" + "="*80)
    print("实验完成!")
    print(f"最佳准确率: {results['best_acc']*100:.2f}%")
    print(f"训练时间: {results['training_time']/60:.2f}分钟")
    print(f"结果保存在: {config['output_dir']}/attention_{attention_type}/")
    print("="*80)
