#!/usr/bin/env python3
"""
全流程知识蒸馏实验脚本 - 简洁版
教师: DenseNet121-ECA (最佳), ConvNeXtV2-Tiny, MobileNetV3-Large
学生: RepViT-M0.9 (最佳), GhostNet-100, MobileNetV3-Small
"""
import sys
sys.path.insert(0, '/workspace/yqm/NetKD')

import argparse
import json
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from data_preprocessing.image_loader import quick_load_dataset

# 配置
TEACHER_EPOCHS = 30
STUDENT_EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3
EARLY_STOP_PATIENCE = 10

def create_grayscale_model(name, num_classes, in_channels=1):
    """创建灰度图模型"""
    import timm
    
    # 教师模型配置
    teacher_configs = {
        'densenet121': {'model': 'densenet121', 'features': 1024},
        'convnextv2_tiny': {'model': 'convnextv2_tiny', 'features': 768},
        'mobilenetv3_large': {'model': 'mobilenetv3_large_100', 'features': 1280},
    }
    
    # 学生模型配置
    student_configs = {
        'repvit_m09': {'model': 'repvit_m0_9', 'features': 384},
        'ghostnet_100': {'model': 'ghostnet_100', 'features': 1280},
        'mobilenetv3_small': {'model': 'mobilenetv3_small_100', 'features': 1024},
    }
    
    configs = {**teacher_configs, **student_configs}
    
    if name not in configs:
        raise ValueError(f"Unknown model: {name}")
    
    cfg = configs[name]
    model = timm.create_model(cfg['model'], pretrained=False, num_classes=0)
    
    # 修改第一层接受灰度图
    first_conv = None
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            first_conv = (n, m)
            break
    
    if first_conv:
        name_parts = first_conv[0].split('.')
        old_conv = first_conv[1]
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels, 
            old_conv.kernel_size, old_conv.stride, 
            old_conv.padding, bias=old_conv.bias is not None
        )
        # 初始化: 平均RGB权重
        if old_conv.in_channels == 3:
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        
        # 设置新卷积层
        parent = model
        for part in name_parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, name_parts[-1], new_conv)
    
    # 添加分类头
    model.classifier = nn.Linear(cfg['features'], num_classes)
    
    return model

def train_model(model, train_loader, val_loader, epochs, device, lr=LR, patience=EARLY_STOP_PATIENCE, 
                kd_teacher=None, kd_alpha=0.3, kd_temp=3.0):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    best_state = None
    no_improve = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x)
            
            loss = criterion(logits, y)
            
            # KD loss
            if kd_teacher is not None:
                kd_teacher.eval()
                with torch.no_grad():
                    teacher_logits = kd_teacher(x)
                kd_loss = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=1),
                    F.softmax(teacher_logits / kd_temp, dim=1),
                    reduction='batchmean'
                ) * (kd_temp ** 2)
                loss = (1 - kd_alpha) * loss + kd_alpha * kd_loss
            
            loss.backward()
            optimizer.step()
        
        scheduler.step()
        
        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f} (best={best_val_acc:.4f})")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    return model, best_val_acc

def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    correct = total = 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().tolist())
            all_labels.extend(y.cpu().tolist())
    
    from sklearn.metrics import f1_score
    acc = correct / total
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return acc, f1

def run_experiment(dataset_name, device, output_dir):
    """运行单个数据集的全流程实验"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | GPU: {device}")
    print(f"{'='*60}")
    
    # 加载数据
    train_ds, val_ds, test_ds, num_classes = quick_load_dataset(
        dataset_name, '/workspace/yqm/Dataset', test_ratio=0.1, val_ratio=0.1
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=4)
    
    print(f"Classes: {num_classes}, Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
    
    results = {'dataset': dataset_name, 'num_classes': num_classes, 'teachers': {}, 'students': {}}
    
    # Phase 1: 训练教师模型
    print("\n[Phase 1] Training Teachers...")
    teachers = ['densenet121', 'convnextv2_tiny', 'mobilenetv3_large']
    teacher_models = {}
    
    for t_name in teachers:
        print(f"  Training {t_name}...")
        model = create_grayscale_model(t_name, num_classes)
        model, val_acc = train_model(model, train_loader, val_loader, TEACHER_EPOCHS, device)
        test_acc, f1 = evaluate(model, test_loader, device)
        print(f"  {t_name}: test_acc={test_acc:.4f}, f1={f1:.4f}")
        
        teacher_models[t_name] = model
        results['teachers'][t_name] = {'test_acc': test_acc, 'f1': f1}
    
    # 选择最佳教师
    best_teacher_name = max(results['teachers'], key=lambda x: results['teachers'][x]['test_acc'])
    best_teacher = teacher_models[best_teacher_name]
    print(f"\nBest teacher: {best_teacher_name} ({results['teachers'][best_teacher_name]['test_acc']:.4f})")
    
    # Phase 2: 知识蒸馏训练学生
    print("\n[Phase 2] Training Students with KD...")
    students = ['repvit_m09', 'ghostnet_100', 'mobilenetv3_small']
    
    for s_name in students:
        print(f"  Training {s_name} (KD from {best_teacher_name})...")
        model = create_grayscale_model(s_name, num_classes)
        model, val_acc = train_model(
            model, train_loader, val_loader, STUDENT_EPOCHS, device,
            kd_teacher=best_teacher, kd_alpha=0.3, kd_temp=3.0
        )
        test_acc, f1 = evaluate(model, test_loader, device)
        print(f"  {s_name}: test_acc={test_acc:.4f}, f1={f1:.4f}")
        
        results['students'][s_name] = {'test_acc': test_acc, 'f1': f1, 'teacher': best_teacher_name}
    
    # 保存结果
    save_path = os.path.join(output_dir, f'{dataset_name}_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {save_path}")
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    args = parser.parse_args()
    
    device = f'cuda:{args.gpu}'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'/workspace/yqm/NetKD/results/full_pipeline_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    for dataset in args.datasets:
        try:
            results = run_experiment(dataset, device, output_dir)
            all_results[dataset] = results
        except Exception as e:
            print(f"ERROR in {dataset}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"Results: {output_dir}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
