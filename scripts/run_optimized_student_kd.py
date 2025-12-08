#!/usr/bin/env python3
"""
优化版学生蒸馏实验
- 基于收敛分析：30 epochs 足够
- 并行 3 GPU
"""
import os
import sys
import argparse
import json
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import timm
from tqdm import tqdm

sys.path.insert(0, '/workspace/yqm/NetKD')
from data_preprocessing.image_loader import quick_load_dataset

def train_student_kd(student_name, teacher_paths, config, device, output_dir):
    """学生蒸馏训练"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    train_loader, val_loader, test_loader, info = quick_load_dataset(
        config['data_path'], batch_size=config['batch_size'], num_workers=8)
    
    # 创建学生模型
    student = timm.create_model(student_name, pretrained=False, 
                                num_classes=info['num_classes'], in_chans=1)
    student = student.to(device)
    
    # 加载教师模型
    teachers = []
    for tpath in teacher_paths:
        if os.path.exists(tpath):
            t = torch.load(tpath, map_location=device)
            if isinstance(t, dict) and 'model' in t:
                t = t['model']
            t.eval()
            teachers.append(t)
    
    if not teachers:
        print(f"警告: 无有效教师模型，使用纯CE训练")
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()
    
    T = config['temperature']
    alpha = config['alpha']
    
    best_val = 0
    results = {'student': student_name, 'epochs': [], 'best_val': 0, 'test_acc': 0}
    
    for epoch in range(1, config['epochs'] + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f'E{epoch}', leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                s_logits = student(images)
                
                # 硬标签损失
                ce_loss = F.cross_entropy(s_logits, labels)
                
                # 软标签损失 (如果有教师)
                if teachers:
                    with torch.no_grad():
                        t_logits = torch.stack([t(images) for t in teachers]).mean(0)
                    
                    soft_loss = F.kl_div(
                        F.log_softmax(s_logits / T, dim=1),
                        F.softmax(t_logits / T, dim=1),
                        reduction='batchmean'
                    ) * (T * T)
                    
                    loss = alpha * soft_loss + (1 - alpha) * ce_loss
                else:
                    loss = ce_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        scheduler.step()
        
        # 验证 (每5轮或最后5轮)
        if epoch % 5 == 0 or epoch > config['epochs'] - 5:
            student.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = student(images)
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
            val_acc = correct / total
            results['epochs'].append({'epoch': epoch, 'val_acc': val_acc})
            
            if val_acc > best_val:
                best_val = val_acc
                torch.save(student.state_dict(), f'{output_dir}/{student_name}_best.pth')
            
            print(f'  E{epoch}: val={val_acc:.4f} (best={best_val:.4f})')
    
    # 测试
    student.load_state_dict(torch.load(f'{output_dir}/{student_name}_best.pth'))
    student.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = student(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    test_acc = correct / total
    
    results['best_val'] = best_val
    results['test_acc'] = test_acc
    
    print(f'\n=== {student_name} ===')
    print(f'  Best Val: {best_val:.4f}')
    print(f'  Test Acc: {test_acc:.4f}')
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--student', type=str, required=True)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    config = {
        'data_path': '/workspace/yqm/Dataset/ISCXVPN2016',
        'batch_size': 64,
        'epochs': 30,  # 优化: 30 epochs 足够
        'lr': 1e-4,
        'temperature': 3.0,
        'alpha': 0.3,
    }
    
    # 教师模型路径
    teacher_paths = [
        'checkpoints/densenet121_teacher.pth',
        'checkpoints/efficientnetv2_teacher.pth',
        'checkpoints/mobilenetv3_teacher.pth',
    ]
    
    output_dir = f'results/student_kd_{timestamp}'
    
    print(f'\n>>> 学生蒸馏: {args.student} on GPU {args.gpu}')
    print(f'>>> 配置: epochs={config["epochs"]}, T={config["temperature"]}, α={config["alpha"]}')
    
    results = train_student_kd(args.student, teacher_paths, config, device, output_dir)
    
    # 保存结果
    with open(f'{output_dir}/{args.student}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n✅ 完成! 结果保存至: {output_dir}')

if __name__ == '__main__':
    main()
