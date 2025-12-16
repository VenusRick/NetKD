"""跨数据集评估 - 灰度图版本"""
import torch
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json, csv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import sys

# 6个数据集
DATASETS = {
    'ISCXVPN2016': '/workspace/yqm/Dataset/ISCXVPN2016',
    'CICIoT2022': '/workspace/yqm/Dataset/CICIoT2022',
    'CrossPlatform-Android': '/workspace/yqm/Dataset/CrossPlatform-Android',
    'CrossPlatform-iOS': '/workspace/yqm/Dataset/CrossPlatform-iOS',
    'ISCXTor2016': '/workspace/yqm/Dataset/ISCXTor2016',
    'USTC-TFC2016': '/workspace/yqm/Dataset/USTC-TFC2016',
}

# 最佳3个学生模型
BEST_MODELS = {
    'repvit_m0_9': 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac1.0/best_model.pth',
    'mobilenetv3_small_050': 'results/kd_ablation_20251208_191348/mobilenetv3_small_050_ce_only_frac1.0/best_model.pth',
    'mobilenetv3_small_075': 'results/kd_ablation_20251208_191348/mobilenetv3_small_075_ce_only_frac1.0/best_model.pth',
}

class TrafficDataset(Dataset):
    """灰度图数据集"""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples, self.labels = [], []
        
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset not found: {self.root_dir}")
            
        self.class_names = sorted([d for d in os.listdir(self.root_dir) 
                                   if os.path.isdir(self.root_dir / d)])
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
        # 保持灰度图，不转RGB
        image = Image.open(self.samples[idx]).convert('L')
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def load_model_grayscale(model_name, ckpt_path, num_classes, device):
    """加载灰度图模型 - 1通道输入"""
    # 创建1通道输入的模型
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes, in_chans=1)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # 移除model.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '') if k.startswith('model.') else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict)
    return model.to(device)

def evaluate_model_on_dataset(model_name, model_path, ds_name, ds_path, device, run_id):
    """评估单个模型在单个数据集上的性能"""
    
    # 灰度图变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # 灰度图归一化
    ])
    
    try:
        dataset = TrafficDataset(ds_path, 'test', transform)
    except FileNotFoundError as e:
        print(f"  ⚠️ {ds_name}: 数据集不存在")
        return None
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)
    num_classes = len(dataset.class_names)
    
    print(f"  Classes: {num_classes}, Samples: {len(dataset)}")
    
    # 加载模型
    try:
        model = load_model_grayscale(model_name, model_path, num_classes, device)
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return None
    
    model.eval()
    
    # 评估
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f'  Evaluating', leave=False):
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return {
        'model': model_name,
        'dataset': ds_name,
        'run': run_id,
        'num_classes': num_classes,
        'test_samples': len(dataset),
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}\n")
    
    all_results = []
    
    # 对每个模型
    for model_name, model_path in BEST_MODELS.items():
        print(f"\n{'='*70}")
        print(f"🔧 Model: {model_name}")
        print(f"{'='*70}")
        
        if not os.path.exists(model_path):
            print(f"  ⚠️ 模型文件不存在: {model_path}")
            continue
        
        # 对每个数据集
        for ds_name, ds_path in DATASETS.items():
            print(f"\n  📊 Dataset: {ds_name}")
            
            # 跑3轮
            for run_id in range(1, 4):
                print(f"    Run {run_id}/3:")
                result = evaluate_model_on_dataset(
                    model_name, model_path, ds_name, ds_path, device, run_id
                )
                
                if result:
                    all_results.append(result)
                    print(f"    ✅ Acc:{result['accuracy']:.4f} F1:{result['f1_macro']:.4f}")
    
    # 保存结果
    out_dir = Path('results/cross_dataset_final')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'all_results.csv', 'w', newline='') as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    
    with open(out_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成汇总报告
    print(f"\n{'='*70}")
    print("📊 结果汇总")
    print(f"{'='*70}")
    
    # 按模型和数据集分组统计
    from collections import defaultdict
    stats = defaultdict(list)
    for r in all_results:
        key = (r['model'], r['dataset'])
        stats[key].append(r['accuracy'])
    
    print(f"\n| Model | Dataset | Acc (mean±std) | Runs |")
    print(f"|-------|---------|----------------|------|")
    for (model, ds), accs in sorted(stats.items()):
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        print(f"| {model} | {ds} | {mean_acc:.4f}±{std_acc:.4f} | {len(accs)} |")
    
    print(f"\n✅ Results saved to: {out_dir}/")

if __name__ == '__main__':
    main()
