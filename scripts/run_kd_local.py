"""使用本地教师模型进行CE+KL知识蒸馏"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append('/workspace/yqm/NetKD')
from src.data import get_dataloaders

TEACHER_PATHS = {
    'efficientnetv2_rw_s': 'results/teacher_search_bs128/efficientnetv2_rw_s/best_model.pt',
    'convnextv2_tiny': 'results/teacher_search_bs128/convnextv2_tiny/best_model.pt',
    'mobilenetv3_large_100': 'results/teacher_search_bs128/mobilenetv3_large_100/best_model.pt',
}

def load_teacher(name, path, device):
    print(f"Loading teacher: {name}")
    model = timm.create_model(name, pretrained=False, num_classes=5)
    ckpt = torch.load(path, map_location='cpu')
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()
    return model

def kd_loss(student_logits, teacher_logits, labels, alpha, temperature):
    ce_loss = F.cross_entropy(student_logits, labels)
    soft_targets = F.softmax(teacher_logits / temperature, dim=1)
    soft_student = F.log_softmax(student_logits / temperature, dim=1)
    kl_loss = F.kl_div(soft_student, soft_targets, reduction='batchmean') * (temperature ** 2)
    return alpha * ce_loss + (1 - alpha) * kl_loss

def train_epoch(student, teacher, loader, optimizer, device, alpha, temperature):
    student.train()
    total_loss = 0
    all_preds, all_labels = [], []
    
    for images, labels in tqdm(loader, desc='Training'):
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            teacher_logits = teacher(images)
        student_logits = student(images)
        loss = kd_loss(student_logits, teacher_logits, labels, alpha, temperature)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = student_logits.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='macro')
    return total_loss / len(loader), f1

def validate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Validating'):
            images = images.to(device)
            preds = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return f1_score(all_labels, all_preds, average='macro')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher', required=True)
    parser.add_argument('--student', required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=4.0)
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu}')
    print(f"\nCE+KL KD: {args.teacher} -> {args.student}")
    
    teacher = load_teacher(args.teacher, TEACHER_PATHS[args.teacher], device)
    student = timm.create_model(args.student, pretrained=True, num_classes=5).to(device)
    
    train_loader, val_loader, _ = get_dataloaders(batch_size=64, num_workers=4)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    save_dir = Path(f'results/kd_local/{args.teacher}_to_{args.student}')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_f1 = train_epoch(student, teacher, train_loader, optimizer, device, args.alpha, args.temperature)
        val_f1 = validate(student, val_loader, device)
        scheduler.step()
        print(f"Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model_state_dict': student.state_dict(), 'val_f1': val_f1}, save_dir / 'best_model.pt')
            print(f"Saved! Best F1: {best_f1:.4f}")
    
    print(f"\nDone! Best: {best_f1:.4f}")
    with open('results/kd_local_results.csv', 'a') as f:
        f.write(f"{args.teacher},{args.student},{args.alpha},{args.temperature},{best_f1:.4f}\n")
