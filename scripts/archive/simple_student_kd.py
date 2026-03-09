#!/usr/bin/env python3
"""简单版学生KD训练 - 纯CE训练(无教师)"""
import os, sys, argparse, json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import timm
from tqdm import tqdm

sys.path.insert(0, '/workspace/yqm/NetKD')
from data_preprocessing.image_loader import quick_load_dataset

def train(student_name, config, device, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    train_loader, val_loader, test_loader, info = quick_load_dataset(
        config['data_path'], batch_size=config['batch_size'], num_workers=8)
    
    # 创建学生模型 (随机初始化)
    student = timm.create_model(student_name, pretrained=False, 
                                num_classes=info['num_classes'], in_chans=1)
    student = student.to(device)
    
    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"模型参数量: {params:.2f}M")
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
    scaler = GradScaler()
    
    best_val = 0
    for epoch in range(1, config['epochs'] + 1):
        student.train()
        pbar = tqdm(train_loader, desc=f'E{epoch}', leave=False)
        
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits = student(images)
                loss = F.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            pbar.set_postfix(loss=f'{loss.item():.4f}')
        
        scheduler.step()
        
        # 验证 (每5轮)
        if epoch % 5 == 0:
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
    
    print(f'\n=== {student_name} ===')
    print(f'  Params: {params:.2f}M')
    print(f'  Best Val: {best_val:.4f}')
    print(f'  Test Acc: {test_acc:.4f}')
    
    return {'student': student_name, 'params_m': params, 'best_val': best_val, 'test_acc': test_acc}

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
        'epochs': 30,
        'lr': 1e-3,
    }
    
    output_dir = f'results/student_ce_{timestamp}'
    print(f'\n>>> CE训练: {args.student} on GPU {args.gpu}')
    
    results = train(args.student, config, device, output_dir)
    
    with open(f'{output_dir}/{args.student}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'\n✅ 完成! 结果: {output_dir}')

if __name__ == '__main__':
    main()
