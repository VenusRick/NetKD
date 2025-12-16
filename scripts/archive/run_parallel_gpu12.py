#!/usr/bin/env python3
"""
并行实验脚本 - GPU 1和2
使用现有训练好的教师模型进行学生蒸馏
"""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data_preprocessing.image_loader import quick_load_dataset

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def load_teacher(path, backbone, device, num_classes=12):
    """加载教师模型"""
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes, in_chans=1)
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    return model.to(device).eval()

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_student(student_name, teacher, train_loader, val_loader, test_loader, 
                  temp, alpha, epochs, device, output_dir):
    """训练学生模型"""
    student = timm.create_model(student_name, pretrained=False, num_classes=12, in_chans=1).to(device)
    params = count_params(student)
    
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()
    
    best_val, best_state = 0, None
    for epoch in range(epochs):
        student.train()
        for x, y in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                teacher_logits = teacher(x)
            optimizer.zero_grad()
            with autocast():
                student_logits = student(x)
                loss = distillation_loss(student_logits, teacher_logits, y, temp, alpha)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                correct += student(x).argmax(1).eq(y).sum().item()
                total += x.size(0)
        val_acc = correct / total
        
        if val_acc > best_val:
            best_val = val_acc
            best_state = student.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"  E{epoch+1}: val={val_acc:.4f} (best={best_val:.4f})")
    
    student.load_state_dict(best_state)
    student.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            preds.extend(student(x.to(device)).argmax(1).cpu().tolist())
            labels.extend(y.tolist())
    
    test_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    test_f1 = f1_score(labels, preds, average='macro')
    
    return {
        'student': student_name,
        'temperature': temp,
        'alpha': alpha,
        'val_acc': best_val,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'params_m': params
    }, best_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=1)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/parallel_gpu{args.gpu}_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    train_loader, val_loader, test_loader, _ = quick_load_dataset('/workspace/yqm/Dataset/ISCXVPN2016')
    
    # 学生模型列表 - 根据GPU分配不同模型
    if args.gpu == 1:
        students = ['ghostnet_100', 'mobilenetv3_small_075', 'efficientnet_lite0']
        teacher_path = 'results/full_experiment_20251208_1045/phase2_finetune/teacher_efficientnetv2_rw_s_scratch_best.pth'
        teacher_backbone = 'efficientnetv2_rw_s'
    else:  # gpu == 2
        students = ['mobilenetv2_050', 'shufflenetv2_x0_5', 'mnasnet_050']
        teacher_path = 'results/full_experiment_20251208_1045/phase2_finetune/teacher_mobilenetv3_large_100_scratch_best.pth'
        teacher_backbone = 'mobilenetv3_large_100'
    
    # 加载教师
    print(f"加载教师模型: {teacher_backbone}")
    teacher = load_teacher(teacher_path, teacher_backbone, device)
    
    all_results = []
    for student_name in students:
        print(f"\n{'='*50}")
        print(f"训练学生模型: {student_name}")
        print('='*50)
        
        for temp in [3, 4, 5]:
            for alpha in [0.3, 0.5, 0.7]:
                print(f"\n>>> {student_name} T={temp} α={alpha}")
                try:
                    result, state = train_student(
                        student_name, teacher, train_loader, val_loader, test_loader,
                        temp, alpha, 100, device, output_dir
                    )
                    all_results.append(result)
                    print(f"  结果: test={result['test_acc']:.4f}, F1={result['test_f1']:.4f}")
                    torch.save(state, f"{output_dir}/{student_name}_T{temp}_a{alpha}.pth")
                except Exception as e:
                    print(f"错误: {e}")
    
    with open(f'{output_dir}/results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ GPU{args.gpu}实验完成，结果保存至: {output_dir}")

if __name__ == '__main__':
    main()
