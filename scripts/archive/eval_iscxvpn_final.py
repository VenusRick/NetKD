"""ISCXVPN2016数据集评估 - 最佳学生模型3轮测试"""
import torch
import timm
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import json, csv
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# 数据集
DATASET_PATH = '/workspace/yqm/Dataset/ISCXVPN2016'
NUM_CLASSES = 7

# 最佳模型 (针对ISCXVPN2016训练的)
BEST_MODELS = {
    'repvit_m0_9_frac1.0': 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac1.0/best_model.pth',
    'repvit_m0_9_frac0.5': 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac0.5/best_model.pth',
    'repvit_m0_9_frac0.2': 'results/kd_ablation_20251208_191348/repvit_m0_9_ce_only_frac0.2/best_model.pth',
    'mobilenetv3_small_frac1.0': 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac1.0/best_model.pth',
    'mobilenetv3_small_frac0.5': 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac0.5/best_model.pth',
    'mobilenetv3_small_frac0.2': 'results/kd_ablation_20251208_191348/mobilenetv3_small_ce_only_frac0.2/best_model.pth',
}

class TrafficDataset(Dataset):
    """灰度图数据集"""
    def __init__(self, root_dir, split='test', transform=None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.samples, self.labels = [], []
        
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
        image = Image.open(self.samples[idx]).convert('L')  # 灰度图
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

def get_model_arch(model_name):
    """从模型名称获取架构"""
    if 'repvit' in model_name:
        return 'repvit_m0_9'
    elif 'mobilenetv3' in model_name:
        return 'mobilenetv3_small_100'
    return model_name

def load_model(model_name, ckpt_path, device):
    """正确加载模型 - 先创建架构，再加载权重"""
    arch = get_model_arch(model_name)
    
    # 创建1通道输入、7类输出的模型
    model = timm.create_model(arch, pretrained=False, num_classes=NUM_CLASSES, in_chans=1)
    
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt.get('model_state_dict', ckpt)
    
    # 移除model.前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace('model.', '') if k.startswith('model.') else k
        new_state_dict[new_key] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    return model.to(device)

def evaluate(model, loader, device):
    """评估模型"""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating', leave=False):
            outputs = model(images.to(device))
            all_preds.extend(outputs.argmax(1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    return {
        'accuracy': accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro'),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
    }

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    print(f"📊 Dataset: ISCXVPN2016 ({NUM_CLASSES} classes)\n")
    
    # 数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # 加载数据集
    dataset = TrafficDataset(DATASET_PATH, 'test', transform)
    print(f"Test samples: {len(dataset)}")
    print(f"Classes: {dataset.class_names}\n")
    
    all_results = []
    
    for model_name, model_path in BEST_MODELS.items():
        print(f"\n{'='*60}")
        print(f"🔧 Model: {model_name}")
        print(f"{'='*60}")
        
        if not os.path.exists(model_path):
            print(f"  ⚠️ 模型文件不存在")
            continue
        
        # 加载模型
        try:
            model = load_model(model_name, model_path, device)
        except Exception as e:
            print(f"  ❌ 模型加载失败: {e}")
            continue
        
        # 3轮测试
        for run_id in range(1, 4):
            # 每轮创建新的DataLoader（shuffle=True产生不同顺序）
            loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
            
            metrics = evaluate(model, loader, device)
            
            result = {
                'model': model_name,
                'run': run_id,
                **metrics
            }
            all_results.append(result)
            
            print(f"  Run {run_id}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    # 保存结果
    out_dir = Path('results/iscxvpn_final_eval')
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / 'results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    
    with open(out_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成汇总
    print(f"\n{'='*60}")
    print("📊 结果汇总 (3轮平均)")
    print(f"{'='*60}\n")
    
    from collections import defaultdict
    stats = defaultdict(list)
    for r in all_results:
        stats[r['model']].append((r['accuracy'], r['f1_macro']))
    
    print(f"| Model | Accuracy (mean±std) | F1-Macro (mean±std) |")
    print(f"|-------|---------------------|---------------------|")
    for model, runs in stats.items():
        accs = [r[0] for r in runs]
        f1s = [r[1] for r in runs]
        print(f"| {model} | {np.mean(accs):.4f}±{np.std(accs):.4f} | {np.mean(f1s):.4f}±{np.std(f1s):.4f} |")
    
    print(f"\n✅ Results saved to: {out_dir}/")

if __name__ == '__main__':
    main()
