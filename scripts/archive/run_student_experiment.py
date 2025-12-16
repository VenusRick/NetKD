#!/usr/bin/env python3
"""
学生模型蒸馏实验脚本 - 更多轻量级学生模型
在GPU 2上运行
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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import timm
from fvcore.nn import FlopCountAnalysis

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from data_preprocessing.image_loader import quick_load_dataset

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6

def count_flops(model, input_size=(1, 1, 40, 40)):
    model.eval()
    x = torch.zeros(input_size).to(next(model.parameters()).device)
    flops = FlopCountAnalysis(model, x)
    return flops.total() / 1e9  # GFLOPs

def load_stacking_teacher(teacher_paths, device):
    """加载Stacking教师模型"""
    teachers = []
    for path in teacher_paths:
        if os.path.exists(path):
            # 从路径推断backbone名称
            if 'convnext' in path.lower():
                backbone = 'convnextv2_tiny'
            elif 'efficient' in path.lower():
                backbone = 'efficientnetv2_rw_s'
            elif 'mobilenet' in path.lower():
                backbone = 'mobilenetv3_large_100'
            else:
                continue
            
            model = timm.create_model(backbone, pretrained=False, num_classes=12, in_chans=1)
            model.load_state_dict(torch.load(path, map_location='cpu'))
            model = model.to(device).eval()
            teachers.append(model)
            print(f"  Loaded teacher: {backbone}")
    return teachers

@torch.no_grad()
def get_teacher_logits(teachers, x):
    """获取教师模型的平均logits"""
    logits = []
    for t in teachers:
        logits.append(t(x))
    return torch.stack(logits).mean(0)

def distillation_loss(student_logits, teacher_logits, labels, temperature, alpha):
    """知识蒸馏损失"""
    soft_loss = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean'
    ) * (temperature ** 2)
    
    hard_loss = F.cross_entropy(student_logits, labels)
    return alpha * hard_loss + (1 - alpha) * soft_loss

def train_student(student_name, teachers, config, device, output_dir):
    """训练学生模型"""
    train_ds, val_ds, test_ds, _ = quick_load_dataset(config['data_path'])
    train_loader = DataLoader(train_ds, batch_size=config['batch_size'],
                             shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'],
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config['batch_size'],
                            num_workers=4, pin_memory=True)
    
    results = []
    
    for temp in config['temperatures']:
        for alpha in config['alphas']:
            print(f"\n>>> {student_name} T={temp} α={alpha}")
            
            # 创建学生模型
            student = timm.create_model(student_name, pretrained=False,
                                       num_classes=12, in_chans=1).to(device)
            
            params = count_params(student)
            try:
                flops = count_flops(student)
            except:
                flops = -1
            
            optimizer = torch.optim.AdamW(student.parameters(), lr=config['lr'], weight_decay=0.05)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'])
            scaler = GradScaler()
            
            best_val, best_state = 0, None
            for epoch in range(config['epochs']):
                student.train()
                for x, y in tqdm(train_loader, desc=f"E{epoch+1}", leave=False):
                    x, y = x.to(device), y.to(device)
                    with torch.no_grad():
                        teacher_logits = get_teacher_logits(teachers, x)
                    
                    optimizer.zero_grad()
                    with autocast():
                        student_logits = student(x)
                        loss = distillation_loss(student_logits, teacher_logits, y, temp, alpha)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                scheduler.step()
                
                # 验证
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
            
            # 测试
            student.load_state_dict(best_state)
            student.eval()
            preds, labels = [], []
            with torch.no_grad():
                for x, y in test_loader:
                    preds.extend(student(x.to(device)).argmax(1).cpu().tolist())
                    labels.extend(y.tolist())
            
            test_acc = sum(p == l for p, l in zip(preds, labels)) / len(labels)
            test_f1 = f1_score(labels, preds, average='macro')
            
            result = {
                'student': student_name,
                'temperature': temp,
                'alpha': alpha,
                'val_acc': best_val,
                'test_acc': test_acc,
                'test_f1': test_f1,
                'params_m': params,
                'flops_g': flops
            }
            results.append(result)
            
            print(f"  结果: test={test_acc:.4f}, F1={test_f1:.4f}, params={params:.2f}M")
            
            # 保存最佳模型
            torch.save(best_state, f"{output_dir}/{student_name}_T{temp}_a{alpha}.pth")
    
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/student_experiment_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    config = {
        'data_path': '/workspace/yqm/Dataset/ISCXVPN2016',
        'batch_size': 128,
        'lr': 1e-3,
        'epochs': 100,
        'temperatures': [3, 4, 5],
        'alphas': [0.3, 0.5, 0.7],
    }
    
    # 轻量级学生模型列表
    students = [
        'ghostnet_100',            # GhostNet 1.0x
        'mobilenetv3_small_075',   # MobileNetV3-Small 0.75x
        'mobilenetv2_050',         # MobileNetV2 0.5x
        'efficientnet_lite0',      # EfficientNet-Lite0
        'shufflenetv2_x0_5',       # ShuffleNetV2 0.5x
        'mnasnet_050',             # MNASNet 0.5x
    ]
    
    # 查找最新的教师模型
    teacher_dirs = [
        'results/full_experiment_20251208_1045/phase2_finetune',
        'results/full_pipeline_20251208_004257/gpu1_scratch/teachers',
    ]
    
    teacher_paths = []
    for tdir in teacher_dirs:
        if os.path.exists(tdir):
            for f in os.listdir(tdir):
                if f.endswith('.pth') and 'scratch' in f:
                    teacher_paths.append(os.path.join(tdir, f))
    
    print(f"找到 {len(teacher_paths)} 个教师模型")
    
    # 加载教师模型
    teachers = load_stacking_teacher(teacher_paths[:3], device)  # 最多使用3个教师
    
    if not teachers:
        print("错误：未找到有效的教师模型")
        return
    
    all_results = []
    for student_name in students:
        print(f"\n{'='*50}")
        print(f"训练学生模型: {student_name}")
        print('='*50)
        
        try:
            results = train_student(student_name, teachers, config, device, output_dir)
            all_results.extend(results)
        except Exception as e:
            print(f"错误: {student_name} - {e}")
            continue
    
    # 保存所有结果
    with open(f'{output_dir}/student_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成帕累托前沿分析
    print("\n" + "="*60)
    print("帕累托前沿分析 (按test_acc排序)")
    print("="*60)
    sorted_results = sorted(all_results, key=lambda x: x['test_acc'], reverse=True)
    for i, r in enumerate(sorted_results[:15], 1):
        print(f"{i:2d}. {r['student']:25s} T={r['temperature']} α={r['alpha']:.1f} "
              f"test={r['test_acc']:.4f} params={r['params_m']:.2f}M")
    
    print(f"\n✅ 学生模型实验完成，结果保存至: {output_dir}")

if __name__ == '__main__':
    main()
