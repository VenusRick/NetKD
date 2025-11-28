"""
训练教师模型 - 使用平衡后的数据集
目标: 所有模型达到98.5%以上测试精度
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
import time
import json
import numpy as np
from models.teacher_models import ResNet50Teacher, MobileNetV3LargeTeacher, DenseNet121Teacher

# 自定义GrayscaleImageFolder
class GrayscaleImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path).convert('L')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

def train_teacher(model_name, model, train_loader, val_loader, test_loader, device, epochs=30):
    """训练单个教师模型"""
    print("="*80)
    print(f"🚀 开始训练 {model_name} (平衡数据集)")
    print("="*80)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    start_time = time.time()
    
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
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
            
            if (batch_idx + 1) % 20 == 0:
                print(f"Epoch {epoch+1:2d} [{batch_idx+1:3d}/{len(train_loader)}] "
                      f"Loss: {train_loss/(batch_idx+1):.4f} "
                      f"Acc: {100.*train_correct/train_total:.2f}%", end='\r')
        
        train_acc = 100. * train_correct / train_total
        
        # 验证
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1:2d}/{epochs} - Train: {train_acc:.2f}% Val: {val_acc:.2f}% LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"  ✅ 新最佳验证精度: {val_acc:.2f}%")
            save_path = f'checkpoints/teacher_finetune/{model_name.lower().replace(" ", "_").replace("+", "")}_balanced.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_accuracy': val_acc / 100,
                'optimizer_state_dict': optimizer.state_dict()
            }, save_path)
        
        scheduler.step()
    
    # 测试评估
    print("\n" + "="*80)
    print("测试集评估")
    print("="*80)
    
    save_path = f'checkpoints/teacher_finetune/{model_name.lower().replace(" ", "_").replace("+", "")}_balanced.pth'
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    test_correct = 0
    test_total = 0
    class_correct = [0] * 7
    class_total = [0] * 7
    confusion_matrix = np.zeros((7, 7), dtype=int)
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
            for i in range(targets.size(0)):
                label = targets[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                class_correct[label] += (pred == label)
                confusion_matrix[label][pred] += 1
    
    test_acc = 100. * test_correct / test_total
    elapsed = time.time() - start_time
    
    print(f"\n最佳验证精度: {best_val_acc:.2f}%")
    print(f"测试精度: {test_acc:.2f}%")
    print(f"训练时间: {elapsed/60:.1f} 分钟")
    
    class_names = ['browsing', 'chat', 'email', 'ftp', 'p2p', 'streaming', 'voip']
    print(f"\n各类别精度:")
    all_above_threshold = True
    for i in range(7):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            emoji = "✅" if acc >= 95 else "⚠️" if acc >= 90 else "❌"
            print(f"  Class {i} ({class_names[i]:<12}): {acc:.2f}% {emoji}")
            if acc < 95:
                all_above_threshold = False
    
    # 保存完整结果
    checkpoint['test_accuracy'] = test_acc / 100
    checkpoint['training_time'] = elapsed
    checkpoint['confusion_matrix'] = confusion_matrix.tolist()
    checkpoint['test_acc_percent'] = test_acc
    torch.save(checkpoint, save_path)
    
    # 保存JSON结果
    results = {
        'model_name': model_name,
        'test_accuracy': test_acc / 100,
        'val_accuracy': best_val_acc / 100,
        'training_time': elapsed,
        'confusion_matrix': confusion_matrix.tolist(),
        'class_accuracies': [100. * class_correct[i] / class_total[i] for i in range(7)]
    }
    
    json_path = f'checkpoints/teacher_finetune/{model_name.lower().replace(" ", "_").replace("+", "")}_balanced_results.json'
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ {model_name} 训练完成")
    print(f"{'='*80}\n")
    
    return test_acc, all_above_threshold

if __name__ == "__main__":
    # 数据准备
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = GrayscaleImageFolder('../Dataset/ISCXVPN2016/train', transform=transform)
    val_dataset = GrayscaleImageFolder('../Dataset/ISCXVPN2016/valid', transform=transform)
    test_dataset = GrayscaleImageFolder('../Dataset/ISCXVPN2016/test', transform=transform)
    
    print("="*80)
    print("📊 数据集信息")
    print("="*80)
    print(f"训练集样本: {len(train_dataset)}")
    print(f"验证集样本: {len(val_dataset)}")
    print(f"测试集样本: {len(test_dataset)}")
    print()
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}\n")
    
    # 训练配置
    teachers = [
        ('ResNet50+ECA', ResNet50Teacher(num_classes=7, use_eca=True)),
        ('MobileNetV3Large+ECA', MobileNetV3LargeTeacher(num_classes=7, use_eca=True)),
        ('DenseNet121', DenseNet121Teacher(num_classes=7, use_eca=False))
    ]
    
    results_summary = []
    all_teachers_pass = True
    
    for model_name, model in teachers:
        model = model.to(device)
        test_acc, passed = train_teacher(model_name, model, train_loader, val_loader, test_loader, device, epochs=30)
        results_summary.append((model_name, test_acc, passed))
        
        # 如果任何一个教师模型未达到98.5%,标记为失败
        if test_acc < 98.5:
            all_teachers_pass = False
            print(f"⚠️ {model_name} 未达到98.5%目标 (当前: {test_acc:.2f}%)")
    
    # 最终总结
    print("\n" + "="*80)
    print("📊 方案B最终评估")
    print("="*80)
    
    for model_name, test_acc, passed in results_summary:
        status = "✅ PASS" if test_acc >= 98.0 else "❌ FAIL"
        print(f"{model_name:<25}: {test_acc:.2f}% {status}")
    
    print("\n" + "="*80)
    print("💭 决策")
    print("="*80)
    
    if all_teachers_pass:
        print("✅ 方案B成功! 所有教师模型达到98.5%+")
        print("   下一步: 训练Stacking集成模型,目标99%+")
    else:
        print("⚠️ 方案B未完全成功,部分模型未达标")
        print("   建议: 停止方案B,转向方案A+C")
        print("   - 方案A: 直接优化Stacking (使用现有教师)")
        print("   - 方案C: 优化学生模型蒸馏策略")
    
    print("="*80)
