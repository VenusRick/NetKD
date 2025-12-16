"""只评估repvit_m0_9在所有数据集上的性能"""
import torch
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
import csv
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

DATASETS = {
    'ISCXVPN2016': '/workspace/yqm/Dataset/ISCXVPN2016',
    'CICIoT2022': '/workspace/yqm/Dataset/CICIoT2022',
    'CrossPlatform-Android': '/workspace/yqm/Dataset/CrossPlatform-Android',
    'CrossPlatform-iOS': '/workspace/yqm/Dataset/CrossPlatform-iOS',
    'ISCXTor2016': '/workspace/yqm/Dataset/ISCXTor2016',
    'USTC-TFC2016': '/workspace/yqm/Dataset/USTC-TFC2016',
}

MODEL_PATH = 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac1.0/best_model.pth'

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

def evaluate():
    device = torch.device('cuda:0')
    print(f"Using device: {device}\n")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = []
    
    for ds_name, ds_path in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*60}")
        
        dataset = TrafficDataset(ds_path, split='test', transform=transform)
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
        num_classes = len(dataset.class_names)
        
        print(f"Classes: {num_classes}, Test samples: {len(dataset)}")
        
        # Load model
        model = timm.create_model('repvit_m0_9', pretrained=False, num_classes=num_classes)
        ckpt = torch.load(MODEL_PATH, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        model.eval()
        
        # Evaluate
        all_preds, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(loader, desc='Evaluating'):
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        
        result = {
            'dataset': ds_name,
            'num_classes': num_classes,
            'test_samples': len(dataset),
            'accuracy': acc,
            'f1_macro': f1,
            'precision': prec,
            'recall': rec,
        }
        results.append(result)
        
        print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")
    
    # Save results
    out_dir = Path('results/final_cross_dataset_evaluation')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create Markdown report
    with open(out_dir / 'CROSS_DATASET_REPORT.md', 'w') as f:
        f.write("# RepVit-M0.9 跨数据集评估报告\n\n")
        f.write("**模型**: repvit_m0_9 (4.72M参数)\n\n")
        f.write("| 数据集 | 类别数 | 测试样本数 | Accuracy | F1-Macro | Precision | Recall |\n")
        f.write("|--------|--------|------------|----------|----------|-----------|--------|\n")
        for r in results:
            f.write(f"| {r['dataset']} | {r['num_classes']} | {r['test_samples']} | "
                   f"{r['accuracy']:.4f} | {r['f1_macro']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |\n")
        
        f.write(f"\n## 统计总结\n\n")
        avg_acc = np.mean([r['accuracy'] for r in results])
        avg_f1 = np.mean([r['f1_macro'] for r in results])
        f.write(f"- **平均准确率**: {avg_acc:.4f}\n")
        f.write(f"- **平均F1分数**: {avg_f1:.4f}\n")
        f.write(f"- **总测试样本数**: {sum(r['test_samples'] for r in results)}\n")
    
    print(f"\n✅ 结果保存至: {out_dir}/")
    print(f"  - results.csv")
    print(f"  - results.json")
    print(f"  - CROSS_DATASET_REPORT.md")

if __name__ == '__main__':
    evaluate()
