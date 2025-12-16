#!/usr/bin/env python3
"""
TrafficKD 教师-学生全流程实验
Phase 1: 教师模型微调 (含ECA消融)
Phase 2: Stacking集成
Phase 3: 知识蒸馏学生训练
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import timm
from tqdm import tqdm
from sklearn.metrics import f1_score
import math

# ================== 工具函数 ==================
def get_timestamp():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_json(data: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)

def log_message(msg: str, log_file: Path = None):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted = f"[{timestamp}] {msg}"
    print(formatted)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(formatted + '\n')

def get_device(gpu_id: int = 0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_id}')
    return torch.device('cpu')

def count_parameters(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6

def get_data_loaders(data_root: str, batch_size: int = 128, num_workers: int = 4):
    from data_preprocessing.image_loader import quick_load_dataset
    train_loader, val_loader, test_loader, meta = quick_load_dataset(
        dataset_name="ISCXVPN2016",
        dataset_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader

# ================== ECA 模块 ==================
class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        return x * self.sigmoid(y)

# ================== 教师微调 ==================
def train_teacher(backbone_name: str, use_eca: bool, epochs: int,
                  train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader,
                  num_classes: int, device: torch.device,
                  output_dir: Path, log_file: Path) -> Dict:
    tag = f"{backbone_name}_{'eca' if use_eca else 'base'}"
    log_message(f"Training teacher: {tag}", log_file)
    
    model = timm.create_model(backbone_name, pretrained=False, num_classes=num_classes, in_chans=1)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    results = {'name': tag, 'backbone': backbone_name, 'use_eca': use_eca, 'train_losses': [], 'val_accs': []}
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        results['train_losses'].append(avg_loss)
        
        # 验证
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        results['val_accs'].append(val_acc)
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': model.state_dict(), 'epoch': epoch, 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 5 == 0:
            log_message(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Val: {val_acc*100:.2f}%", log_file)
    
    # 测试
    model.load_state_dict(torch.load(ckpt_path)['model_state'])
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results['val_acc'] = best_val_acc
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['params_m'] = count_parameters(model)
    results['checkpoint'] = str(ckpt_path)
    
    log_message(f"Teacher {tag}: Val={best_val_acc*100:.2f}%, Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    return results, model

# ================== Stacking ==================
class StackingHead(nn.Module):
    def __init__(self, num_teachers: int, num_classes: int, hidden_dim: int = 384):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_teachers * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, *teacher_logits):
        x = torch.cat(teacher_logits, dim=1)
        return self.fc(x)

def train_stacking(teacher_models: List[nn.Module], teacher_names: List[str],
                   epochs: int, train_loader: DataLoader, val_loader: DataLoader,
                   test_loader: DataLoader, num_classes: int, device: torch.device,
                   output_dir: Path, log_file: Path) -> Dict:
    log_message(f"Training Stacking with teachers: {teacher_names}", log_file)
    
    for t in teacher_models:
        t.eval()
        for p in t.parameters():
            p.requires_grad = False
    
    stacking = StackingHead(len(teacher_models), num_classes).to(device)
    optimizer = optim.AdamW(stacking.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0
    results = {'teacher_names': teacher_names, 'val_accs': []}
    
    for epoch in range(epochs):
        stacking.train()
        for batch in tqdm(train_loader, desc=f"Stacking E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
            optimizer.zero_grad()
            out = stacking(*teacher_logits)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        stacking.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                teacher_logits = [t(x) for t in teacher_models]
                out = stacking(*teacher_logits)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        results['val_accs'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / "stacking_best.pth"
            torch.save({'stacking_state': stacking.state_dict(), 'val_acc': val_acc}, ckpt_path)
    
    # 测试
    stacking.load_state_dict(torch.load(ckpt_path)['stacking_state'])
    stacking.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            teacher_logits = [t(x) for t in teacher_models]
            out = stacking(*teacher_logits)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results['val_acc'] = best_val_acc
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['checkpoint'] = str(ckpt_path)
    
    log_message(f"Stacking: Val={best_val_acc*100:.2f}%, Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%", log_file)
    return results, stacking

# ================== 知识蒸馏 ==================
def distill_student(student_name: str, teacher_models: List[nn.Module],
                    stacking: nn.Module, temperature: float, alpha_ce: float,
                    epochs: int, train_loader: DataLoader, val_loader: DataLoader,
                    test_loader: DataLoader, num_classes: int, device: torch.device,
                    output_dir: Path, log_file: Path) -> Dict:
    tag = f"{student_name}_T{temperature}_a{alpha_ce}"
    log_message(f"Distilling student: {tag}", log_file)
    
    student = timm.create_model(student_name, pretrained=False, num_classes=num_classes, in_chans=1).to(device)
    optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    ce_criterion = nn.CrossEntropyLoss()
    kl_criterion = nn.KLDivLoss(reduction='batchmean')
    
    alpha_kd = 1.0 - alpha_ce
    best_val_acc = 0
    results = {'student': student_name, 'temperature': temperature, 'alpha_ce': alpha_ce}
    
    for epoch in range(epochs):
        student.train()
        for batch in tqdm(train_loader, desc=f"{tag} E{epoch+1}", leave=False):
            x, y = batch[0].to(device), batch[1].to(device)
            
            with torch.no_grad():
                teacher_logits = [t(x) for t in teacher_models]
                teacher_out = stacking(*teacher_logits)
                teacher_soft = nn.functional.softmax(teacher_out / temperature, dim=1)
            
            optimizer.zero_grad()
            student_out = student(x)
            student_soft = nn.functional.log_softmax(student_out / temperature, dim=1)
            
            ce_loss = ce_criterion(student_out, y)
            kd_loss = kl_criterion(student_soft, teacher_soft) * (temperature ** 2)
            loss = alpha_ce * ce_loss + alpha_kd * kd_loss
            
            loss.backward()
            optimizer.step()
        
        student.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                out = student(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        
        val_acc = correct / total
        scheduler.step()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = output_dir / f"{tag}_best.pth"
            torch.save({'model_state': student.state_dict(), 'val_acc': val_acc}, ckpt_path)
        
        if (epoch + 1) % 10 == 0:
            log_message(f"  Epoch {epoch+1}/{epochs}, Val: {val_acc*100:.2f}%", log_file)
    
    # 测试
    student.load_state_dict(torch.load(ckpt_path)['model_state'])
    student.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    inference_times = []
    
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            start = time.time()
            out = student(x)
            torch.cuda.synchronize()
            inference_times.append((time.time() - start) / x.size(0) * 1000)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    
    test_acc = correct / total
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    
    results['val_acc'] = best_val_acc
    results['test_acc'] = test_acc
    results['test_f1'] = test_f1
    results['params_m'] = count_parameters(student)
    results['inference_ms'] = sum(inference_times) / len(inference_times)
    results['checkpoint'] = str(ckpt_path)
    
    log_message(f"Student {tag}: Test={test_acc*100:.2f}%, F1={test_f1*100:.2f}%, Params={results['params_m']:.2f}M", log_file)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--teacher_epochs", type=int, default=30)
    parser.add_argument("--student_epochs", type=int, default=50)
    args = parser.parse_args()
    
    timestamp = get_timestamp()
    output_dir = Path(f"results/teacher_student_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "experiment.log"
    
    device = get_device(args.gpu)
    log_message(f"Starting experiment on GPU {args.gpu}", log_file)
    
    # 加载数据
    train_loader, val_loader, test_loader = get_data_loaders("/workspace/yqm/Dataset", batch_size=128, num_workers=4)
    log_message("Data loaded", log_file)
    
    # Phase 1: 教师模型
    log_message("="*60, log_file)
    log_message("PHASE 1: TEACHER TRAINING", log_file)
    log_message("="*60, log_file)
    
    teacher_backbones = ['convnextv2_tiny', 'efficientnetv2_rw_s', 'ghostnetv2_100']
    teacher_results = []
    teacher_models = []
    teacher_names = []
    
    for backbone in teacher_backbones:
        result, model = train_teacher(backbone, False, args.teacher_epochs,
                                       train_loader, val_loader, test_loader,
                                       7, device, output_dir / "teachers", log_file)
        teacher_results.append(result)
        teacher_models.append(model)
        teacher_names.append(result['name'])
    
    save_json(teacher_results, output_dir / "teacher_results.json")
    
    # Phase 2: Stacking
    log_message("="*60, log_file)
    log_message("PHASE 2: STACKING", log_file)
    log_message("="*60, log_file)
    
    stacking_result, stacking = train_stacking(teacher_models, teacher_names, 20,
                                                train_loader, val_loader, test_loader,
                                                7, device, output_dir / "stacking", log_file)
    save_json(stacking_result, output_dir / "stacking_results.json")
    
    # Phase 3: 知识蒸馏
    log_message("="*60, log_file)
    log_message("PHASE 3: KNOWLEDGE DISTILLATION", log_file)
    log_message("="*60, log_file)
    
    student_names = ['repvit_m0_9', 'ghostnet_100', 'mobilenetv3_small_075', 'mobilenetv2_050']
    temps = [3, 4]
    alphas = [0.3, 0.5]
    
    student_results = []
    for student_name in student_names:
        for temp in temps:
            for alpha in alphas:
                result = distill_student(student_name, teacher_models, stacking,
                                         temp, alpha, args.student_epochs,
                                         train_loader, val_loader, test_loader,
                                         7, device, output_dir / "students", log_file)
                student_results.append(result)
                save_json(student_results, output_dir / "student_results.json")
    
    # 生成报告
    log_message("="*60, log_file)
    log_message("GENERATING REPORT", log_file)
    log_message("="*60, log_file)
    
    report = f"# TrafficKD 实验报告\n\n生成时间: {datetime.now()}\n\n"
    report += "## 教师模型\n\n| Model | Val Acc | Test Acc | F1 | Params(M) |\n|---|---|---|---|---|\n"
    for r in sorted(teacher_results, key=lambda x: x['test_acc'], reverse=True):
        report += f"| {r['name']} | {r['val_acc']*100:.2f}% | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} |\n"
    
    report += f"\n## Stacking\n\n- Teachers: {', '.join(stacking_result['teacher_names'])}\n"
    report += f"- Test Acc: {stacking_result['test_acc']*100:.2f}%\n- F1: {stacking_result['test_f1']*100:.2f}%\n\n"
    
    report += "## 学生模型\n\n| Student | Temp | Alpha | Test Acc | F1 | Params(M) | Infer(ms) |\n|---|---|---|---|---|---|---|\n"
    for r in sorted(student_results, key=lambda x: x['test_acc'], reverse=True):
        report += f"| {r['student']} | {r['temperature']} | {r['alpha_ce']} | {r['test_acc']*100:.2f}% | {r['test_f1']*100:.2f}% | {r['params_m']:.2f} | {r['inference_ms']:.2f} |\n"
    
    with open(output_dir / "REPORT.md", 'w') as f:
        f.write(report)
    
    log_message(f"Report saved to {output_dir / 'REPORT.md'}", log_file)
    log_message("EXPERIMENT COMPLETED!", log_file)

if __name__ == "__main__":
    main()
