"""
完整优化流程: 使用预训练教师 → Stacking → 学生(固定权重 vs 自适应权重)
目标: 学生模型测试精度 ≥ 98.5%
"""
import os, sys, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import numpy as np

# GrayscaleImageFolder
class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert("L")
        if self.transform: sample = self.transform(sample)
        return sample, target

# 导入模块
from models.teacher_models import ResNet50Teacher, MobileNetV3LargeTeacher, DenseNet121Teacher
from models.student_model import StudentNet
from training.adaptive_distillation import AdaptiveWeightedDistillation, FixedWeightDistillation, StackingEnsemble
from training.evaluation import evaluate_model

def setup_data():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(0.3), transforms.RandomRotation(10),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5])
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    
    train_dataset = GrayscaleImageFolder("../Dataset/ISCXVPN2016/train", transform=train_transform)
    val_dataset = GrayscaleImageFolder("../Dataset/ISCXVPN2016/valid", transform=test_transform)
    test_dataset = GrayscaleImageFolder("../Dataset/ISCXVPN2016/test", transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader, train_dataset

def load_pretrained_teachers(device):
    """加载已有的高精度预训练教师模型"""
    print("\n" + "="*80)
    print("加载预训练教师模型")
    print("="*80)
    
    teachers = []
    teacher_configs = [
        ("ResNet50+ECA", ResNet50Teacher, "checkpoints/resnet50_teacher.pth"),
        ("MobileNetV3+ECA", MobileNetV3LargeTeacher, "checkpoints/mbv3_teacher.pth"),
        ("DenseNet121", DenseNet121Teacher, "checkpoints/densenet121_teacher.pth")
    ]
    
    for name, model_class, path in teacher_configs:
        print(f"\n加载 {name}...")
        model = model_class(num_classes=7, use_eca=False).to(device)
        checkpoint = torch.load(path)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        print(f"  ✅ {name} 加载成功")
        teachers.append(model)
    
    return teachers

def train_stacking(teacher_models, train_loader, val_loader, test_loader, num_classes, device, epochs=30):
    print(f"\n{'='*80}\n训练Stacking集成\n{'='*80}")
    
    def get_predictions(loader):
        all_preds, all_labels = [], []
        for teacher in teacher_models:
            teacher.eval()
            teacher_preds = []
            with torch.no_grad():
                for images, labels in loader:
                    outputs = teacher(images.to(device))
                    probs = torch.softmax(outputs, dim=1)
                    teacher_preds.append(probs.cpu())
                    if len(all_labels) == 0: all_labels.append(labels)
            teacher_preds = torch.cat(teacher_preds, dim=0)
            all_preds.append(teacher_preds)
        stacked = torch.cat(all_preds, dim=1)
        return stacked, torch.cat(all_labels, dim=0)
    
    print("生成教师预测...")
    train_preds, train_labels = get_predictions(train_loader)
    val_preds, val_labels = get_predictions(val_loader)
    test_preds, test_labels = get_predictions(test_loader)
    
    input_dim = len(teacher_models) * num_classes
    stacking_model = StackingEnsemble(input_dim=input_dim, hidden_dim=64, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(stacking_model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    save_path = "checkpoints/stacking_model_optimized.pth"
    train_dataset = torch.utils.data.TensorDataset(train_preds, train_labels)
    train_loader_stacking = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    for epoch in range(epochs):
        stacking_model.train()
        correct, total = 0, 0
        for preds, labels in train_loader_stacking:
            preds, labels = preds.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = stacking_model(preds)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()
        
        stacking_model.eval()
        with torch.no_grad():
            val_outputs = stacking_model(val_preds.to(device))
            _, val_predicted = val_outputs.max(1)
            val_acc = 100. * val_predicted.eq(val_labels.to(device)).sum().item() / len(val_labels)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(stacking_model.state_dict(), save_path)
    
    stacking_model.load_state_dict(torch.load(save_path))
    stacking_model.eval()
    with torch.no_grad():
        test_outputs = stacking_model(test_preds.to(device))
        _, test_predicted = test_outputs.max(1)
        test_acc = 100. * test_predicted.eq(test_labels.to(device)).sum().item() / len(test_labels)
    
    print(f"\nStacking 最终: 验证={best_val_acc:.2f}% | 测试={test_acc:.2f}%")
    return stacking_model, test_acc

def train_student_fixed(student_model, teacher_models, train_loader, val_loader, test_loader, device, epochs=100):
    print(f"\n{'='*80}\n训练学生 - 固定权重\n{'='*80}")
    student_model = student_model.to(device)
    for t in teacher_models: t.eval()
    
    distillation = FixedWeightDistillation(temperature=4.0, alpha=0.7, beta=0.3)
    optimizer = optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    save_path = "checkpoints/student_fixed_weights.pth"
    
    for epoch in range(epochs):
        student_model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = [t(images) for t in teacher_models]
            student_logits = student_model(images)
            loss = distillation(student_logits, teacher_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()
        val_acc_raw, _, _ = evaluate_model(student_model, val_loader, device)
        val_acc = val_acc_raw * 100
        
        if (epoch + 1) % 10 == 0:
            test_acc_raw, _, _ = evaluate_model(student_model, test_loader, device)
            test_acc = test_acc_raw * 100
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), save_path)
    
    student_model.load_state_dict(torch.load(save_path))
    final_test_acc_raw, _, class_acc = evaluate_model(student_model, test_loader, device)
    final_test_acc = final_test_acc_raw * 100
    print(f"\n学生(固定) 最终: 验证={best_val_acc:.2f}% | 测试={final_test_acc:.2f}%")
    return student_model, final_test_acc

def train_student_adaptive(student_model, teacher_models, train_loader, val_loader, test_loader, device, epochs=100):
    print(f"\n{'='*80}\n训练学生 - 自适应权重\n{'='*80}")
    student_model = student_model.to(device)
    for t in teacher_models: t.eval()
    
    distillation = AdaptiveWeightedDistillation(temperature=4.0, alpha=0.7, beta=0.3, adaptation_rate=0.1, min_weight=0.1)
    optimizer = optim.AdamW(student_model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    teacher_weights = torch.ones(len(teacher_models)) / len(teacher_models)
    
    best_val_acc = 0.0
    save_path = "checkpoints/student_adaptive_weights.pth"
    
    for epoch in range(epochs):
        student_model.train()
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                teacher_logits = [t(images) for t in teacher_models]
            student_logits = student_model(images)
            loss, teacher_weights, _ = distillation(student_logits, teacher_logits, labels, teacher_weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = student_logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * correct / total
        scheduler.step()
        val_acc_raw, _, _ = evaluate_model(student_model, val_loader, device)
        val_acc = val_acc_raw * 100
        
        if (epoch + 1) % 10 == 0:
            test_acc_raw, _, _ = evaluate_model(student_model, test_loader, device)
            test_acc = test_acc_raw * 100
            print(f"Epoch {epoch+1}/{epochs} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Test: {test_acc:.2f}%")
            print(f"  权重: {teacher_weights.cpu().numpy()}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student_model.state_dict(), save_path)
    
    student_model.load_state_dict(torch.load(save_path))
    final_test_acc_raw, _, class_acc = evaluate_model(student_model, test_loader, device)
    final_test_acc = final_test_acc_raw * 100
    print(f"\n学生(自适应) 最终: 验证={best_val_acc:.2f}% | 测试={final_test_acc:.2f}%")
    print(f"最终权重: {teacher_weights.cpu().numpy()}")
    return student_model, final_test_acc, teacher_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    print("\n" + "="*80)
    print("阶段1: 数据准备")
    print("="*80)
    train_loader, val_loader, test_loader, train_dataset = setup_data()
    print(f"训练集: {len(train_dataset)}")
    
    print("\n" + "="*80)
    print("阶段2: 加载预训练教师模型")
    print("="*80)
    teacher_models = load_pretrained_teachers(device)
    
    # 评估教师精度
    print("\n" + "="*80)
    print("验证教师模型精度")
    print("="*80)
    for i, (teacher, name) in enumerate(zip(teacher_models, ["ResNet50+ECA", "MobileNetV3+ECA", "DenseNet121"])):
        test_acc_raw, _, _ = evaluate_model(teacher, test_loader, device)
        test_acc = test_acc_raw * 100
        print(f"{name}: {test_acc:.2f}%")
    
    print("\n" + "="*80)
    print("阶段3: 训练Stacking集成")
    print("="*80)
    _, stacking_acc = train_stacking(teacher_models, train_loader, val_loader, test_loader, 7, device, epochs=30)
    
    print("\n" + "="*80)
    print("阶段4: 学生(固定)")
    print("="*80)
    student_fixed = StudentNet(num_classes=7)
    _, fixed_acc = train_student_fixed(student_fixed, teacher_models, train_loader, val_loader, test_loader, device, epochs=100)
    
    print("\n" + "="*80)
    print("阶段5: 学生(自适应)")
    print("="*80)
    student_adaptive = StudentNet(num_classes=7)
    _, adaptive_acc, final_weights = train_student_adaptive(student_adaptive, teacher_models, train_loader, val_loader, test_loader, device, epochs=100)
    
    print(f"\n{'='*80}\n最终对比\n{'='*80}")
    print(f"Stacking:     {stacking_acc:.2f}%")
    print(f"学生(固定):   {fixed_acc:.2f}%")
    print(f"学生(自适应): {adaptive_acc:.2f}%")
    print(f"改进: {adaptive_acc - fixed_acc:+.2f}%")
    print(f"权重: {final_weights.cpu().numpy()}")
    print(f"目标: {'✅' if adaptive_acc >= 98.5 else '❌'} (目标≥98.5%)")

if __name__ == "__main__":
    main()
