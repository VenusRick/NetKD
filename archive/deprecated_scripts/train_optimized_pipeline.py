"""
完整优化管道: 从教师模型到学生模型
目标: 整体精度最高,学生模型达到98.5%+
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import time
import json
import numpy as np
from models.teacher_models import ResNet50Teacher, MobileNetV3LargeTeacher, DenseNet121Teacher
from training.focal_loss import FocalLoss, get_class_weights

class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('L')
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

def train_teacher_with_focal(model_name, model, train_loader, val_loader, test_loader, 
                             device, class_weights, epochs=50):
    """使用Focal Loss训练教师模型"""
    print("="*80)
    print(f"🚀 训练 {model_name} (Focal Loss, {epochs} epochs)")
    print("="*80)
    
    # Focal Loss with class weights
    criterion = FocalLoss(alpha=class_weights.to(device), gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    best_test_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        # Test evaluation every 5 epochs
        if (epoch + 1) % 5 == 0:
            test_correct, test_total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    test_total += targets.size(0)
                    test_correct += predicted.eq(targets).sum().item()
            test_acc = 100. * test_correct / test_total
            print(f"Epoch {epoch+1:2d}/{epochs} - Train: {train_acc:.2f}% Val: {val_acc:.2f}% Test: {test_acc:.2f}%")
            best_test_acc = max(best_test_acc, test_acc)
        else:
            print(f"Epoch {epoch+1:2d}/{epochs} - Train: {train_acc:.2f}% Val: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f'checkpoints/teacher_finetune/{model_name.lower().replace(" ", "_").replace("+", "")}_optimized.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc / 100,
                'test_accuracy': best_test_acc / 100
            }, save_path)
        
        scheduler.step()
    
    # Final test
    save_path = f'checkpoints/teacher_finetune/{model_name.lower().replace(" ", "_").replace("+", "")}_optimized.pth'
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct, test_total = 0, 0
    class_correct, class_total = [0]*7, [0]*7
    confusion_matrix = np.zeros((7, 7), dtype=int)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            for i in range(targets.size(0)):
                label, pred = targets[i].item(), predicted[i].item()
                class_total[label] += 1
                class_correct[label] += (pred == label)
                confusion_matrix[label][pred] += 1
    
    test_acc = 100. * test_correct / test_total
    elapsed = time.time() - start_time
    
    print(f"\n最终结果:")
    print(f"  最佳验证精度: {best_val_acc:.2f}%")
    print(f"  测试精度: {test_acc:.2f}%")
    print(f"  训练时间: {elapsed/60:.1f} 分钟")
    
    class_names = ['browsing', 'chat', 'email', 'ftp', 'p2p', 'streaming', 'voip']
    print(f"\n各类别精度:")
    for i in range(7):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            emoji = "✅" if acc >= 95 else "⚠️" if acc >= 90 else "❌"
            print(f"  Class {i} ({class_names[i]:<12}): {acc:.2f}% {emoji}")
    
    # Save results
    checkpoint['test_accuracy'] = test_acc / 100
    checkpoint['confusion_matrix'] = confusion_matrix.tolist()
    checkpoint['training_time'] = elapsed
    torch.save(checkpoint, save_path)
    
    print(f"\n✅ {model_name} 训练完成\n")
    return test_acc

if __name__ == "__main__":
    print("="*80)
    print("🎯 目标: 整体精度最高,学生模型98.5%+")
    print("="*80)
    
    # Data preparation
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = GrayscaleImageFolder('../Dataset/ISCXVPN2016/train', transform=transform)
    val_dataset = GrayscaleImageFolder('../Dataset/ISCXVPN2016/valid', transform=transform)
    test_dataset = GrayscaleImageFolder('../Dataset/ISCXVPN2016/test', transform=transform)
    
    print(f"\n训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")
    print(f"测试集: {len(test_dataset)} 样本\n")
    
    # Calculate class weights
    print("计算类别权重...")
    class_weights = get_class_weights(train_dataset, num_classes=7)
    print("类别权重:", class_weights.numpy())
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}\n")
    
    # Train teachers
    teachers = [
        ('ResNet50+ECA', ResNet50Teacher(num_classes=7, use_eca=True)),
        ('MobileNetV3Large+ECA', MobileNetV3LargeTeacher(num_classes=7, use_eca=True)),
        ('DenseNet121', DenseNet121Teacher(num_classes=7, use_eca=False))
    ]
    
    results = []
    for model_name, model in teachers:
        model = model.to(device)
        test_acc = train_teacher_with_focal(model_name, model, train_loader, val_loader, 
                                           test_loader, device, class_weights, epochs=50)
        results.append((model_name, test_acc))
    
    print("\n" + "="*80)
    print("📊 教师模型训练总结")
    print("="*80)
    for name, acc in results:
        status = "✅" if acc >= 98.5 else "⚠️"
        print(f"{name:<25}: {acc:.2f}% {status}")
    
    print("\n下一步: 训练Stacking和学生模型")
    print("="*80)

