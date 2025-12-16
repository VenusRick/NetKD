"""评估最佳学生模型在所有数据集上的性能"""
import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import json
import csv
from datetime import datetime

# 配置
DATASETS = {
    'CICIoT2022': '/workspace/yqm/Dataset/CICIoT2022',
    'CrossPlatform-Android': '/workspace/yqm/Dataset/CrossPlatform-Android',
    'CrossPlatform-iOS': '/workspace/yqm/Dataset/CrossPlatform-iOS',
    'ISCXTor2016': '/workspace/yqm/Dataset/ISCXTor2016',
    'ISCXVPN2016': '/workspace/yqm/Dataset/ISCXVPN2016',
    'USTC-TFC2016': '/workspace/yqm/Dataset/USTC-TFC2016',
}

BEST_MODELS = {
    'repvit_m0_9': 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac1.0/best_model.pth',
    'mobilenetv3_small_100': 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac1.0/best_model.pth',
}

def count_parameters(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def load_model(model_name, checkpoint_path, num_classes, device):
    print(f"\n加载模型: {model_name}")
    print(f"检查点: {checkpoint_path}")
    
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Val F1 (训练时): {checkpoint.get('val_f1', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    params = count_parameters(model)
    print(f"参数量: {params:.2f}M")
    return model, params

def load_dataset(dataset_path):
    """加载数据集 - 需要根据实际格式调整"""
    from torch.utils.data import DataLoader, Dataset
    from torchvision import transforms
    from PIL import Image
    import os
    
    class TrafficDataset(Dataset):
        def __init__(self, root_dir, split='test', transform=None):
            self.root_dir = Path(root_dir) / split
            self.transform = transform
            self.samples = []
            self.labels = []
            self.class_names = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(self.root_dir / d)])
            self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
            
            for class_name in self.class_names:
                class_dir = self.root_dir / class_name
                if class_dir.is_dir():
                    for img_path in class_dir.glob('*.png'):
                        self.samples.append(str(img_path))
                        self.labels.append(self.class_to_idx[class_name])
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            img_path = self.samples[idx]
            label = self.labels[idx]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TrafficDataset(dataset_path, split='test', transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    return loader, len(dataset.class_names), dataset.class_names

def evaluate_model(model, loader, device):
    """评估模型"""
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='评估中'):
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(f'results/cross_dataset_evaluation_{timestamp}')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 对每个模型评估所有数据集
    for model_name, checkpoint_path in BEST_MODELS.items():
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name}")
        print(f"{'='*60}")
        
        for dataset_name, dataset_path in DATASETS.items():
            print(f"\n数据集: {dataset_name}")
            print(f"路径: {dataset_path}")
            
            try:
                # 加载数据集
                loader, num_classes, class_names = load_dataset(dataset_path)
                print(f"类别数: {num_classes}")
                print(f"测试样本数: {len(loader.dataset)}")
                
                # 加载模型
                model, params = load_model(model_name, checkpoint_path, num_classes, device)
                
                # 评估
                metrics = evaluate_model(model, loader, device)
                
                # 记录结果
                result = {
                    'model': model_name,
                    'dataset': dataset_name,
                    'parameters_M': params,
                    'num_classes': num_classes,
                    'test_samples': len(loader.dataset),
                    **metrics
                }
                all_results.append(result)
                
                print(f"\n结果:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 (Macro): {metrics['f1_macro']:.4f}")
                print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                import traceback
                traceback.print_exc()
    
    # 保存结果
    csv_path = results_dir / 'results.csv'
    with open(csv_path, 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    json_path = results_dir / 'results.json'
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n✅ 结果已保存:")
    print(f"  - {csv_path}")
    print(f"  - {json_path}")
    
    # 生成Markdown报告
    md_path = results_dir / 'REPORT.md'
    with open(md_path, 'w') as f:
        f.write(f"# 跨数据集评估报告\n\n")
        f.write(f"**评估时间**: {timestamp}\n\n")
        f.write(f"## 模型\n\n")
        for model_name in BEST_MODELS.keys():
            model_results = [r for r in all_results if r['model'] == model_name]
            if model_results:
                f.write(f"### {model_name} ({model_results[0]['parameters_M']:.2f}M参数)\n\n")
                f.write("| 数据集 | 类别数 | 样本数 | Accuracy | F1-Macro | F1-Weighted | Precision | Recall |\n")
                f.write("|--------|--------|--------|----------|----------|-------------|-----------|--------|\n")
                for r in model_results:
                    f.write(f"| {r['dataset']} | {r['num_classes']} | {r['test_samples']} | "
                           f"{r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['f1_weighted']:.4f} | "
                           f"{r['precision']:.4f} | {r['recall']:.4f} |\n")
                f.write("\n")
    
    print(f"  - {md_path}")
    print(f"\n🎉 评估完成!")

if __name__ == '__main__':
    main()
